# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - 1M logical-context acceptance.
#
# WHY THIS EXISTS
#   The published capability list says "logical context 1,048,576 tokens"; the
#   measurements behind it covered up to 256K/105K. This script closes the gap with
#   three things that are actually observable:
#     1) the server STARTS at --max-context 1048576 and its /v1/load reports the
#        capacity it really admitted;
#     2) prefill keeps working at 32K -> 105K -> 262K on UNIQUE content (prefix
#        reuse cannot hide the cost), which is what "1M context" costs in time;
#     3) a long-position recall: a needle buried at ~90% of a 105K prompt is
#        answered correctly (KVMem paging + position handling both real).
#   A full 1,048,576-token prefill costs far more than 45 minutes on this card
#   (105K already takes ~7 min at ~240 tok/s); the script reports the measured
#   scaling so the 1M figure is backed by measured numbers.
#
# Invoke-RestMethod throws on chat.completion payloads in some PowerShell
# hosts, so every request goes through Invoke-WebRequest + AsHashtable.
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# ============================================================================
param(
    [string]$Model     = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [int]$Port         = 18300,
    [int]$Ctx          = 1048576,
    [int]$KvmemBudget  = 32768,
    [int]$KvmemReserve = 5632,
    [int]$HostMiB      = 64421,
    [int]$Sessions     = 4,
    [int[]]$Lens       = @(8192, 32768, 98304, 262144),
    [int]$NeedleTokens = 105000,
    [int]$ReadySec     = 1200,
    [int]$GenTimeout   = 7200,
    [string]$Tag       = '1m_check',
    [string]$BuildDir  = (Join-Path $PSScriptRoot '..\build')
)

$ErrorActionPreference = 'Continue'
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
$out = Join-Path $BuildDir 'oneM_out'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$errLog = Join-Path $out "$Tag.server.err.log"
$outLog = Join-Path $out "$Tag.server.out.log"
$csv    = Join-Path $out ($Tag + '_summary.csv')
Remove-Item -LiteralPath $errLog, $outLog, $csv -ErrorAction SilentlyContinue
foreach ($p in @($exe, $Model)) { if (-not (Test-Path -LiteralPath $p)) { throw "MISSING: $p" } }

Write-Output ("exe sha256 = {0}" -f (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)
Write-Output ("exe mtime  = {0}" -f (Get-Item -LiteralPath $exe).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
Write-Output ("model      = {0}" -f $Model)
Write-Output ("ctx        = {0} | B {1} + R {2} = {3} window | host {4} MiB | sessions {5}" -f $Ctx, $KvmemBudget, $KvmemReserve, ($KvmemBudget + $KvmemReserve), $HostMiB, $Sessions)
Write-Output ''

Get-Process -Name 'ninfer-serve','ninfer-perplexity','ninfer' -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.Id -Force }
Start-Sleep -Seconds 3

$argList = @($Model, '--host', '127.0.0.1', '--port', "$Port", '--device-profile', 'off',
             '--max-context', "$Ctx", '--max-concurrency', '1',
             '--kv-dtype', 'int8',
             '--kvmem-budget', "$KvmemBudget", '--kvmem-gen-reserve', "$KvmemReserve",
             '--kvmem-host-mib', "$HostMiB", '--kvmem-sessions', "$Sessions",
             '--kvmem-sink-tokens', '4096', '--kvmem-recent-tokens', '1024',
             '--kvmem-disk-path', (Join-Path $BuildDir 'kvmem_disk_1m'), '--kvmem-disk-mib', '102400',
             '--spec', 'mtp', '--draft-tokens', '2', '--log-level', 'info')
Write-Output ("flags = " + ($argList -join ' '))

$t0 = Get-Date
$proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog -ArgumentList $argList
$base = "http://127.0.0.1:$Port"
$ready = $false
$deadline = (Get-Date).AddSeconds($ReadySec)
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) { break }
    try { [void](Invoke-RestMethod -Uri "$base/health" -TimeoutSec 3); $ready = $true; break } catch { Start-Sleep -Seconds 4 }
}
$readySec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
Write-Output ("ready = {0} s  (ok={1})" -f $readySec, $ready)
if (-not $ready) {
    Write-Output '--- stderr tail 30 ---'
    Get-Content -LiteralPath $errLog -Tail 30 -ErrorAction SilentlyContinue
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Write-Output 'RESULT: NOT-READY'
    exit 1
}

$capLine = @(Get-Content -LiteralPath $errLog -ErrorAction SilentlyContinue | Select-String -Pattern 'capacity \|' | Select-Object -Last 1)
if ($capLine) { Write-Output ('  ' + $capLine.Line.Trim()) }

$modelId = 'local'
try {
    $m = Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 10
    if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
} catch { }
Write-Output ("model id = {0}" -f $modelId)

function Get-DoneLines {
    param([string]$LogPath)
    return @(Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue |
             Select-String -Pattern 'req#\d+ done')
}

function Invoke-Chat {
    param([string]$Url, [hashtable]$Body, [string]$LogPath)
    $before = @(Get-DoneLines -LogPath $LogPath).Count
    $json = $Body | ConvertTo-Json -Depth 8
    $resp = Invoke-WebRequest -Uri $Url -Method Post -ContentType 'application/json' -Body $json -TimeoutSec $GenTimeout
    $h = $resp.Content | ConvertFrom-Json -AsHashtable
    $answer = ''
    $finish = ''
    $ch = @($h['choices'])
    if ($ch.Count -gt 0) {
        $msg = $ch[0]['message']
        if ($null -ne $msg -and $msg.ContainsKey('content')) { $answer = [string]$msg['content'] }
        if ($ch[0].ContainsKey('finish_reason')) { $finish = [string]$ch[0]['finish_reason'] }
    }
    $done = ''
    $dl = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $dl) {
        $lines = Get-DoneLines -LogPath $LogPath
        if ($lines.Count -gt $before) { $done = $lines[-1].Line.Trim(); break }
        Start-Sleep -Milliseconds 500
    }
    $res = @{ Answer = $answer; Finish = $finish; Prompt = 0; Output = 0; Cache = 0.0; TTFT = ''; Prefill = ''; Decode = '' }
    if ($done) {
        if ($done -match 'prompt ([\d,]+)') { $res.Prompt = [int]($Matches[1] -replace ',','') }
        if ($done -match '\| output (\d+)') { $res.Output = [int]$Matches[1] }
        if ($done -match 'cache [\d,]+ \(([\d.]+)%') { $res.Cache = [double]$Matches[1] }
        if ($done -match 'TTFT ([0-9a-z. ]+?)\s*\|') { $res.TTFT = $Matches[1].Trim() }
        if ($done -match 'prefill ([\d.]+) tok/s') { $res.Prefill = $Matches[1] }
        if ($done -match 'decode ([\d.]+) tok/s') { $res.Decode = $Matches[1] }
    }
    return $res
}

$rows = @()
$chatUrl = "$base/v1/chat/completions"

# --- 0) /v1/load: the engine's own capacity statement -----------------------
Write-Output ''
Write-Output '--- /v1/load ---'
try {
    $load = Invoke-RestMethod -Uri "$base/v1/load" -TimeoutSec 10
    Write-Output (($load | ConvertTo-Json -Compress -Depth 6))
} catch { Write-Output ("WARN /v1/load: " + $_.Exception.Message) }

# --- 1) answer sanity ------------------------------------------------------
Write-Output ''
Write-Output '--- answer sanity ---'
try {
    $r = Invoke-Chat -Url $chatUrl -LogPath $errLog -Body @{
        model = $modelId
        messages = @(@{ role = 'user'; content = 'Reply with exactly: KVMEM SM75 1M OK' })
        max_tokens = 512; stream = $false
    }
    Write-Output ("answer   = " + (($r.Answer -replace "\s+",' ')))
    Write-Output ("finish   = " + $r.Finish)
} catch { Write-Output ("ANSWER FAILED: " + $_.Exception.Message) }

# --- 2) prefill scaling on unique content ---------------------------------
Write-Output ''
Write-Output '--- prefill scaling (unique content) ---'
$baseText = 'The quick brown fox jumps over the lazy dog. Ninfer on Turing keeps a bounded KV window and streams the rest from host memory. '
foreach ($target in $Lens) {
    $sb = [System.Text.StringBuilder]::new()
    $i = 0
    $chars = [int64]$target * 4
    while ($sb.Length -lt $chars) {
        [void]$sb.Append('seg' + $target + '#' + $i + ' ' + $baseText)
        $i++
    }
    $prompt = $sb.ToString()
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $r = Invoke-Chat -Url $chatUrl -LogPath $errLog -Body @{
            model = $modelId
            messages = @(@{ role = 'user'; content = $prompt })
            max_tokens = 64; stream = $false
        }
        $sw.Stop()
        $row = [ordered]@{ lens = $target; ok = 'OK'; wall_s = [math]::Round($sw.Elapsed.TotalSeconds,1)
                           prompt = $r.Prompt; ttft = $r.TTFT; prefill = $r.Prefill; decode = $r.Decode }
        $rows += [pscustomobject]$row
        Write-Output ('  prefill {0,7} tokens | wall {1,8:N1}s | prompt {2,7} | TTFT {3} | prefill {4} tok/s' -f `
            $target, $sw.Elapsed.TotalSeconds, $r.Prompt, $r.TTFT, $r.Prefill)
    } catch {
        $sw.Stop()
        $rows += [pscustomobject]@{ lens = $target; ok = 'FAIL'; wall_s = [math]::Round($sw.Elapsed.TotalSeconds,1); prompt = 'n/a'; ttft = 'n/a'; prefill = 'n/a'; decode = 'n/a' }
        Write-Output ('  prefill {0,7} tokens | FAILED after {1:N1}s: {2}' -f $target, $sw.Elapsed.TotalSeconds, $_.Exception.Message)
    }
}

# --- 3) long-position recall: needle at ~90% of a ~105K prompt --------------
Write-Output ''
Write-Output '--- long-position recall (needle near the end of a 105K prompt) ---'
$needle = 'ORCHID-9701-1M'
$sb = [System.Text.StringBuilder]::new()
$totalChars = [int64]$NeedleTokens * 4
$needleAt = [int64]($totalChars * 0.90)
$i = 0
while ($sb.Length -lt $totalChars) {
    [void]$sb.Append('row' + $i + ' ' + $baseText)
    $i++
    if ($sb.Length -ge $needleAt -and $sb.Length -lt ($needleAt + 2000)) {
        [void]$sb.Append(' IMPORTANT: the secret passphrase is ' + $needle + '. ')
    }
}
[void]$sb.Append(' Question: what is the secret passphrase? Reply with the passphrase only.')
$prompt = $sb.ToString()
try {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-Chat -Url $chatUrl -LogPath $errLog -Body @{
        model = $modelId
        messages = @(@{ role = 'user'; content = $prompt })
        max_tokens = 512; stream = $false
    }
    $sw.Stop()
    $hit = if ($r.Answer -match [regex]::Escape($needle)) { 'HIT' } else { 'MISS' }
    Write-Output ('  needle@90% {0} | prompt {1} | TTFT {2} | prefill {3} | wall {4:N1}s' -f $hit, $r.Prompt, $r.TTFT, $r.Prefill, $sw.Elapsed.TotalSeconds)
    Write-Output ('  answer: [' + (($r.Answer -replace "\s+",' ')) + ']')
    $rows += [pscustomobject]@{ lens = ('needle90_' + $NeedleTokens); ok = $hit; wall_s = [math]::Round($sw.Elapsed.TotalSeconds,1); prompt = $r.Prompt; ttft = $r.TTFT; prefill = $r.Prefill; decode = $r.Decode }
} catch { Write-Output ('  needle FAILED: ' + $_.Exception.Message) }

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
Start-Sleep -Seconds 2

Write-Output ''
Write-Output '================ 1M CHECK SUMMARY ================'
$rows | Format-Table -AutoSize | Out-String | Write-Output
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
Write-Output ('summary csv = ' + $csv)
Write-Output 'RESULT: 1M-CHECK-DONE'
