# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - multi-turn (x16) acceptance AND prefix-reuse proof.
#
# WHY THIS EXISTS
#   Every other judgement fires independent single-shot requests. A real agent
#   session is different: each turn re-sends the whole history, so the stack
#   must (a) reuse the shared prefix instead of re-prefilling it every turn,
#   and (b) keep a fact from turn 1 reachable at turn 16.
#   Both are visible in the engine's own "req#N done" line:
#     cache N (P%)   - how many prompt tokens came from the retained history
#     TTFT           - time to first token of this turn
#     prefill tok/s  - the cost of whatever was NOT cached
#     mtp accepted   - speculative decoding acceptance for this turn
#   A turn after the first that reports cache 0 means prefix reuse is NOT
#   working and the session silently re-prefills the whole history.
#
# WHAT IT ASSERTS (verdict from log content and HTTP bodies, never ExitCode)
#   turns_ok   = Turns            (every round returned a 200 with a body)
#   cache turns = Turns-1         (every turn after the first reported cache > 0)
#   recall     = HIT              (the code given in turn 1 comes back at the end)
#   no crash   = process alive after the last turn
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# ============================================================================
param(
    [string]$Model     = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [int]$Port         = 18096,
    [int]$B            = 32768,
    [int]$R            = 5632,
    [int]$Ctx          = 1048576,
    [int]$HostMiB      = 64421,
    [int]$Sessions     = 4,
    [string]$KvDtype   = 'int8',
    # Production recipe draft width. Measured, not copied: see run_kvformat_speed
    # -DraftTokens (2/3/4 sweep) for the value this card actually prefers.
    [int]$DraftTokens  = 4,
    [string]$DiskPath  = (Join-Path $PSScriptRoot '..\build\kvmem_disk_multiturn'),
    [int]$DiskMiB      = 102400,
    [int]$Turns        = 16,
    [int]$MaxTokens    = 512,
    [int]$ReadySec     = 900,
    [int]$GenTimeout   = 1800,
    [string]$Tag       = 'multiturn16',
    [string]$BuildDir  = (Join-Path $PSScriptRoot '..\build')
)

$ErrorActionPreference = 'Stop'
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
$err = Join-Path $BuildDir "$Tag.server.err.log"
$out = Join-Path $BuildDir "$Tag.server.out.log"
$csv = Join-Path $BuildDir ($Tag + '_summary.csv')

foreach ($p in @($exe, $Model)) { if (-not (Test-Path -LiteralPath $p)) { throw "MISSING: $p" } }
if (-not (Test-Path -LiteralPath $DiskPath)) { New-Item -ItemType Directory -Path $DiskPath | Out-Null }
if (Test-Path -LiteralPath $err) { Remove-Item -LiteralPath $err -Force }
if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force }
if (Test-Path -LiteralPath $csv) { Remove-Item -LiteralPath $csv -Force }

Write-Output ("exe sha256 = {0}" -f (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)
Write-Output ("exe mtime  = {0}" -f (Get-Item -LiteralPath $exe).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
Write-Output ("model      = {0}" -f $Model)
Write-Output ("window     = B {0} + R {1} = {2} tokens | ctx {3} | dtype {4} | turns {5}" -f $B, $R, ($B + $R), $Ctx, $KvDtype, $Turns)
Write-Output ''

# Orphans from an earlier arm hold VRAM and change the memory picture.
Get-Process -Name 'ninfer-serve', 'ninfer-perplexity', 'ninfer' -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.Id -Force }
Start-Sleep -Seconds 3

$argList = @($Model, '--host', '127.0.0.1', '--port', "$Port", '--device-profile', 'off',
             '--max-context', "$Ctx", '--max-concurrency', '1',
             '--kv-dtype', "$KvDtype",
             '--kvmem-budget', "$B", '--kvmem-gen-reserve', "$R",
             '--kvmem-host-mib', "$HostMiB", '--kvmem-sessions', "$Sessions",
             '--kvmem-disk-path', "$DiskPath", '--kvmem-disk-mib', "$DiskMiB",
             '--spec', 'mtp', '--draft-tokens', "$DraftTokens", '--log-level', 'info')
Write-Output ("flags      = " + ($argList -join ' '))

$proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError $err -ArgumentList $argList

$base     = "http://127.0.0.1:$Port"
$ready    = $false
$deadline = (Get-Date).AddSeconds($ReadySec)
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) { Write-Output 'server exited before becoming ready'; break }
    try { [void](Invoke-RestMethod -Uri "$base/health" -TimeoutSec 3); $ready = $true; break }
    catch { Start-Sleep -Seconds 5 }
}
if (-not $ready) {
    Write-Output 'RESULT: NOT-READY'
    Write-Output '--- server stderr tail 40 ---'
    Get-Content -LiteralPath $err -Tail 40 -ErrorAction SilentlyContinue
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    exit 1
}
Write-Output 'health     = OK'

$modelId = 'local'
try {
    $m = Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 10
    if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
} catch { }
Write-Output ("model id   = " + $modelId)
$chatUrl = "$base/v1/chat/completions"

function Get-DoneLines {
    param([string]$LogPath)
    return @(Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue |
             Select-String -Pattern 'req#\d+ done')
}

$secret  = 'LOTUS-7391'
$rows    = @()
$okTurns = 0
$messages = @()
$messages += @{ role = 'system'; content = 'You are a careful assistant. Follow each instruction exactly.' }
$messages += @{ role = 'user'; content = ("My access code is {0}. Remember it. Reply with exactly: ACK-1" -f $secret) }

for ($turn = 1; $turn -le $Turns; $turn++) {
    if ($turn -gt 1) {
        if ($turn -eq $Turns) {
            $messages += @{ role = 'user'; content = 'What is my access code? Reply with the code only.' }
        } else {
            $messages += @{ role = 'user'; content = ("Reply with exactly: ACK-{0}" -f $turn) }
        }
    }

    $body = @{ model = $modelId; messages = $messages; max_tokens = $MaxTokens
               stream = $false; temperature = 0.0 } | ConvertTo-Json -Depth 8
    $before = (Get-DoneLines -LogPath $err).Count
    $answer = ''
    $ok = $false
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        # On this PS host Invoke-RestMethod throws a spurious PSCustomObject->Int32
        # conversion while materialising a chat.completion payload (line 136 in the
        # failed runs). Invoke-WebRequest returns the raw body and the hashtable
        # parse sidesteps that conversion entirely.
        $resp = Invoke-WebRequest -Uri $chatUrl -Method Post -ContentType 'application/json' `
                -Body $body -TimeoutSec $GenTimeout
        $h = $resp.Content | ConvertFrom-Json -AsHashtable
        $ch = @($h['choices'])
        if ($ch.Count -gt 0) {
            $msg = $ch[0]['message']
            $answer = if ($null -ne $msg -and $msg.ContainsKey('content')) { [string]$msg['content'] } else { '' }
        } else {
            $answer = ''
        }
        $ok = $true
        $okTurns++
    } catch {
        Write-Output ("turn {0,2} | REQUEST FAILED line={1}: {2}" -f $turn, $_.InvocationInfo.ScriptLineNumber, $_.Exception.Message)
    }
    $sw.Stop()

    # Wait for the engine's own done line for this request (the counters this
    # judgement asserts on live there, not in the HTTP usage object).
    $done = ''
    $dl = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $dl) {
        $lines = Get-DoneLines -LogPath $err
        if ($lines.Count -gt $before) { $done = $lines[-1].Line.Trim(); break }
        Start-Sleep -Milliseconds 500
    }

    $prompt = 0; $output = 0; $cachePct = 0.0; $ttft = ''; $prefill = ''
    $decode = ''; $mtp = ''
    if ($done -match 'prompt ([\d,]+)') { $prompt = [int]($Matches[1] -replace ',', '') }
    if ($done -match '\| output (\d+)') { $output = [int]$Matches[1] }
    if ($done -match 'cache [\d,]+ \(([\d.]+)%') { $cachePct = [double]$Matches[1] }
    if ($done -match 'TTFT ([0-9a-z. ]+?)\s*\|') { $ttft = $Matches[1].Trim() }
    if ($done -match 'prefill ([\d.]+) tok/s') { $prefill = $Matches[1] }
    if ($done -match 'decode ([\d.]+) tok/s') { $decode = $Matches[1] }
    if ($done -match 'mtp accepted (\d+)/(\d+)') { $mtp = $Matches[1] + '/' + $Matches[2] }

    $shown = ($answer -replace "\s+", ' ')
    if ($shown.Length -gt 70) { $shown = $shown.Substring(0, 70) }
    Write-Output ("turn {0,2} | ok={1} | wall={2,6:N1}s | prompt={3,6} | out={4,4} | cache={5,5:N1}% | TTFT={6} | prefill={7,6} | decode={8,5} | mtp={9} | ans=[{10}]" -f `
        $turn, $ok, $sw.Elapsed.TotalSeconds, $prompt, $output, $cachePct, $ttft, $prefill, $decode, $mtp, $shown)

    $rows += [pscustomobject]@{
        turn = $turn; ok = $ok; wall_s = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        prompt = $prompt; output = $output; cache_pct = $cachePct
        ttft = $ttft; prefill = $prefill; decode = $decode; mtp = $mtp; answer = $shown
    }

    if ($ok) {
        if ($answer -eq '') { $answer = '(empty)' }
        $messages += @{ role = 'assistant'; content = $answer }
    }
}

$alive      = -not $proc.HasExited
$cacheTurns = @($rows | Where-Object { $_.turn -gt 1 -and $_.cache_pct -gt 0 }).Count
$lastAnswer = ''
if ($rows.Count -gt 0) { $lastAnswer = [string]$rows[-1].answer }
$recallHit  = ($lastAnswer -match [regex]::Escape($secret))

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
Start-Sleep -Seconds 2

$pass = ($okTurns -eq $Turns) -and ($cacheTurns -eq ($Turns - 1)) -and $recallHit -and $alive

Write-Output ''
Write-Output '================ MULTI-TURN SUMMARY ================'
Write-Output ("turns_ok      = {0}/{1}" -f $okTurns, $Turns)
Write-Output ("cache turns   = {0}/{1}  (every turn after the first must report cache > 0)" -f $cacheTurns, ($Turns - 1))
Write-Output ("recall        = {0}     (last answer must contain {1})" -f $(if ($recallHit) { 'HIT' } else { 'MISS' }), $secret)
Write-Output ("server alive  = {0}" -f $alive)
$rows | Format-Table turn, ok, prompt, output, cache_pct, ttft, prefill, decode, mtp -AutoSize | Out-String | Write-Output
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
Write-Output ('summary csv  = ' + $csv)
if ($pass) { Write-Output 'RESULT: MULTITURN-PASS' } else { Write-Output 'RESULT: MULTITURN-FAIL' }
