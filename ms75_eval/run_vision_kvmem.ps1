# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - vision POSITION acceptance with the user's three
# images, each asked for ITS OWN ground truth.
#
# WHY THE FILLER MUST EXCEED THE DEVICE POOL
#   The pool holds B+R tokens per lane. Any prompt shorter than that is fully
#   resident, so an image it contains is never evicted and the probe proves only
#   that full attention works - it says nothing about KVMem. A 32K filler against
#   a 38400 pool is all resident and proves nothing. The
#   filler here defaults ABOVE the pool, so the image's pages must be evicted and
#   re-selected. kvmem_window.cpp:583 adds media pages to the MANDATORY set,
#   which is exactly the mechanism these arms turn into evidence.
#
# GROUND TRUTH (fixed test fixtures, 2026-10-15)
#   1.png  a green sticky note in the middle of a pile, carrying a black star
#   2.png  the digit 8 somewhere in a landscape picture
#   3.png  the password 7429 shown on a person
#   An image PASSES only when every position returns its own ground truth; the
#   run PASSES only when all three images pass at all four positions.
#
# WHY -KvDtype DEFAULTS TO rk8v4
#   Measured on this card (2026-10-15): with bf16 KV the long profile's device
#   reservation is 3.18 GiB against 2.75 GiB available after weights, so a 1M
#   context with this pool cannot start at all; rk8v4 brings the same pool to
#   1.50 GiB and leaves 1.09 GiB free. Vision residency needs that room.
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# Verdicts come from HTTP responses, never ExitCode. The child argument
# array is NOT named $args (that is a PowerShell automatic variable).
# ============================================================================
param(
    [string]$Model = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [string[]]$Images = @(
        (Join-Path $PSScriptRoot '..\media\1.png'),
        (Join-Path $PSScriptRoot '..\media\2.png'),
        (Join-Path $PSScriptRoot '..\media\3.png')
    ),
    [string[]]$Labels = @('sticky-note', 'landscape-digit', 'person-code'),
    [string[]]$Expect = @('star', '8', '7429'),
    [string[]]$Positions = @('baseline', 'front', 'middle', 'back'),
    [int]$FillerTokens = 51200,
    [int]$Ctx = 1048576,
    [string]$KvDtype = 'rk8v4',
    [switch]$GdnFp16,
    [string]$Mtp = 'off',
    [int]$B = 32768,
    [int]$R = 5632,
    [int]$HostMib = 49152,
    [string]$DiskPath = (Join-Path $PSScriptRoot '..\Cache\kvmem'),
    [int]$DiskMiB = 102400,
    [int]$Sessions = 1,
    [int]$Sink = 4096,
    [int]$Recent = 1024,
    [int]$AskMax = 256,
    [int]$BasePort = 18400,
    [int]$ReadySec = 900,
    [int]$GenTimeout = 3600,
    [string]$Tag = 'visionpos',
    [string]$BuildDir = (Join-Path $PSScriptRoot '..\build')
)

$ErrorActionPreference = 'Stop'

if ($Images.Count -ne $Expect.Count -or $Images.Count -ne $Labels.Count) {
    throw "Images / Labels / Expect must be the same length"
}

# 布局自适应：构建机用 <根>\build\apps\；只拿发布包时用 <根>\engine\
if (-not (Test-Path -LiteralPath $BuildDir)) {
    $BuildDir = Join-Path (Split-Path -Parent $PSScriptRoot) 'Cache\judge'
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
}
$exe = Join-Path $BuildDir 'apps\ninfer-serve.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    $packExe = Join-Path (Split-Path -Parent $PSScriptRoot) 'engine\ninfer-serve.exe'
    if (Test-Path -LiteralPath $packExe) { $exe = $packExe }
}
foreach ($p in @($exe, $Model) + $Images) { if (-not (Test-Path -LiteralPath $p)) { throw "MISSING: $p" } }

# One question per image, so a position-dependent failure shows up as that
# image's own ground truth disappearing instead of a generic character count.
$questions = @(
    'Look at the image. In the middle of the pile of items there is a sticky note. Reply with exactly two lines. Line 1: NOTE-COLOR: <the colour of the note>. Line 2: NOTE-SHAPE: <the single shape drawn on it>.',
    'Look at the landscape picture. A single digit appears in it. Reply with exactly that digit and nothing else.',
    'Look at the image of the person. A 4-digit code is shown on it. Reply with exactly those four digits and nothing else.'
)

$b64s = @()
foreach ($img in $Images) { $b64s += [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($img)) }

$pool = $B + $R
Write-Output ("exe sha256 = {0}" -f (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)
Write-Output ("exe mtime  = {0}" -f (Get-Item -LiteralPath $exe).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
Write-Output ("images     = {0}" -f (($Labels | ForEach-Object { $_ }) -join ', '))
Write-Output ("expect     = {0}" -f ($Expect -join ' | '))
Write-Output ("kv-dtype   = {0}   mtp={1}   gdn-fp16={2}" -f $KvDtype, $Mtp, $GdnFp16)
Write-Output ("pool       = B {0} + R {1} = {2} tokens/lane; filler {3} tokens ({4:N2}x pool)" -f `
    $B, $R, $pool, $FillerTokens, ($FillerTokens / $pool))
if ($FillerTokens -le $pool) {
    Write-Output 'WARNING: filler <= pool, so every image stays resident and this run does NOT test eviction/re-selection.'
}
Write-Output ''

function Stop-Orphans {
    Get-Process -Name 'ninfer-serve', 'ninfer-perplexity', 'ninfer' -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.Id -Force }
    Start-Sleep -Seconds 5
}

function Build-Filler {
    param([int]$Tokens, [string]$Salt)
    $base = 'The quick brown fox jumps over the lazy dog. A bounded KV window keeps part of the history on the device and streams the rest from the Host tier. '
    $sb = [System.Text.StringBuilder]::new()
    $i = 0
    while ($sb.Length -lt ($Tokens * 4)) {
        [void]$sb.Append('f'); [void]$sb.Append($Salt); [void]$sb.Append('#'); [void]$sb.Append($i)
        [void]$sb.Append(' '); [void]$sb.Append($base)
        $i++
    }
    return $sb.ToString()
}

# Places image i at the requested position inside the (optionally long) text.
function Build-Content {
    param([int]$Index, [string]$Where, [string]$Question, [string]$Filler)
    $img = @{ type = 'image_url'; image_url = @{ url = "data:image/png;base64,$($b64s[$Index])" } }
    if ($Where -eq 'baseline') { return @(@{ type = 'text'; text = $Question }, $img) }
    if ($Where -eq 'front')    { return @($img, @{ type = 'text'; text = $Filler + ' ' + $Question }) }
    if ($Where -eq 'back')     { return @(@{ type = 'text'; text = $Filler + ' ' + $Question }, $img) }
    $half = [int]($Filler.Length / 2)
    return @(@{ type = 'text'; text = $Filler.Substring(0, $half) }, $img,
             @{ type = 'text'; text = $Filler.Substring($half) + ' ' + $Question })
}

function Get-DoneLines {
    param([string]$LogPath)
    return @(Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue |
             Select-String -Pattern 'req#\d+ done')
}

$arm = '{0}_{1}' -f $Tag, $KvDtype
$errLog = Join-Path $BuildDir ("$arm.server.err.log")
$outLog = Join-Path $BuildDir ("$arm.server.out.log")
Remove-Item -LiteralPath $errLog, $outLog -ErrorAction SilentlyContinue
Stop-Orphans

$childArgs = @($Model, '--host', '127.0.0.1', '--port', "$BasePort", '--device-profile', 'off',
               '--max-context', "$Ctx", '--max-concurrency', '1',
               '--kvmem-budget', "$B", '--kvmem-gen-reserve', "$R",
               '--kvmem-sink-tokens', "$Sink", '--kvmem-recent-tokens', "$Recent",
               '--kvmem-host-mib', "$HostMib", '--kvmem-sessions', "$Sessions",
               '--kvmem-disk-path', "$DiskPath", '--kvmem-disk-mib', "$DiskMiB",
               '--vision', '--vision-residency', 'resident', '--log-level', 'info')
if ($KvDtype -ne '') { $childArgs += @('--kv-dtype', $KvDtype) }
if ($GdnFp16) { $childArgs += '--gdn-state-fp16' }
if ($Mtp -eq 'mtp') { $childArgs += @('--spec', 'mtp', '--draft-tokens', '3') }
Write-Output ('flags = ' + ($childArgs -join ' '))

$t0 = Get-Date
$proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog -ArgumentList $childArgs

$baseUrl  = "http://127.0.0.1:$BasePort"
$ready    = $false
$deadline = (Get-Date).AddSeconds($ReadySec)
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) { break }
    try { [void](Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 3); $ready = $true; break }
    catch { Start-Sleep -Seconds 5 }
}
$readySec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

if (-not $ready) {
    Write-Output ("NOT-READY after {0}s" -f $readySec)
    Get-Content -LiteralPath $errLog -ErrorAction SilentlyContinue |
        Select-String -Pattern 'KVMem rejected|vision|FATAL|requested Engine runtime reservation|it needs|can hold about' |
        Select-Object -Last 6 | ForEach-Object { Write-Output ('  ' + $_.Line.Trim()) }
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Write-Output 'RESULT: NOT-READY'
    exit 1
}

Write-Output ("ready in {0}s" -f $readySec)
$capText = @(Get-Content -LiteralPath $errLog -ErrorAction SilentlyContinue) |
           Select-String -Pattern 'capacity \|' | Select-Object -Last 1
if ($capText) { Write-Output ('  ' + $capText.Line.Trim()) }
Write-Output ('  gpu = ' + ((& nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader) -join ' '))

$modelId = 'local'
try {
    $m = Invoke-RestMethod -Uri "$baseUrl/v1/models" -TimeoutSec 10
    if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
} catch { }
$chatUrl = "$baseUrl/v1/chat/completions"

$rows = @()
for ($i = 0; $i -lt $Images.Count; $i++) {
    foreach ($where in $Positions) {
        $filler = if ($where -eq 'baseline') { '' } else { Build-Filler -Tokens $FillerTokens -Salt ('v' + $i + $where) }
        $body = @{
            model = $modelId
            messages = @(@{ role = 'user'; content = (Build-Content -Index $i -Where $where -Question $questions[$i] -Filler $filler) })
            max_tokens = $AskMax; stream = $false; temperature = 0.0
        } | ConvertTo-Json -Depth 12

        $answer = ''; $finish = 'n/a'; $wall = 0; $hit = 'MISS'; $note = ''
        try {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            # NOT $r: PowerShell variable names are case-insensitive, so $r IS the [int]
            # parameter -R, and the response object would be cast to Int32 (measured
            # 2026-10-16: the first vision run failed every arm that way).
            $resp = Invoke-RestMethod -Uri $chatUrl -Method Post -ContentType 'application/json' `
                    -Body $body -TimeoutSec $GenTimeout
            $sw.Stop()
            $wall = [math]::Round($sw.Elapsed.TotalSeconds, 1)
            $message = $resp.choices[0].message
            $answer = ([string]$message.content -replace "\s+", ' ').Trim()
            $finish = [string]$resp.choices[0].finish_reason
            # Chinese answers are common; match the ground truth in either script.
            # "star" may come back in either script; the non-ASCII form is written as
            # a \uXXXX regex escape so the judgement logic itself stays ASCII-only.
            $pat = [regex]::Escape($Expect[$i])
            if ($Expect[$i] -eq 'star') { $pat = 'star|\u661f' }
            if ($answer -match $pat) { $hit = 'HIT' }
        } catch {
            $note = $_.Exception.Message
            Write-Output ('  {0,-16} {1,-9} FAILED: {2}' -f $Labels[$i], $where, $note)
        }
        if ($hit -eq 'MISS' -and $answer.Length -gt 160) { $answer = $answer.Substring(0, 160) }
        Write-Output ('  {0,-16} {1,-9} {2,-4} wall={3,7:N1}s finish={4,-8} answer=[{5}]' -f `
            $Labels[$i], $where, $hit, $wall, $finish, $answer)
        $rows += [pscustomobject]@{
            image = $Labels[$i]; expect = $Expect[$i]; position = $where
            hit = $hit; wall_s = $wall; finish = $finish; answer = $answer
        }
    }
}

$loadJson = 'n/a'
try {
    $load = Invoke-RestMethod -Uri "$baseUrl/v1/load" -TimeoutSec 30
    $loadJson = ($load.kvmem | ConvertTo-Json -Compress)
    Write-Output ('/v1/load kvmem = ' + $loadJson)
} catch { }

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
Start-Sleep -Seconds 3
Stop-Orphans

Write-Output '================ VISION POSITION SUMMARY ================'
$rows | Format-Table image, position, hit, wall_s, finish -AutoSize | Out-String | Write-Output

$misses = @($rows | Where-Object { $_.hit -ne 'HIT' })
$csv = Join-Path $BuildDir ($arm + '_summary.csv')
Remove-Item -LiteralPath $csv -ErrorAction SilentlyContinue
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
Write-Output ('summary csv: ' + $csv)
foreach ($img in $Labels) {
    $mine = @($rows | Where-Object { $_.image -eq $img })
    Write-Output ('  {0,-16} {1}/{2} positions returned the ground truth' -f $img,
                  @($mine | Where-Object { $_.hit -eq 'HIT' }).Count, $mine.Count)
}
if ($misses.Count -eq 0) {
    Write-Output 'RESULT: ALL-PASS (every image answered by its own ground truth at every position)'
} else {
    Write-Output ('RESULT: HAS-FAILURE ({0} of {1} arms missed)' -f $misses.Count, $rows.Count)
}
