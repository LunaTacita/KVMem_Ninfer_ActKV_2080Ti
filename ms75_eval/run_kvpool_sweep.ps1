# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - device-pool (B) sweep, for choosing the KV pool
# from measurements.
#
# WHY THIS EXISTS
#   types.h:522        device_tokens() = selected_tokens(B) + reserve_tokens(R)
#   serve_options.cpp:1852  kv_capacity = explicit_capacity(window * lanes)
#   So --kvmem-budget is BOTH the attention-visible history size AND the number
#   of device KV pages (pages = (B+R)*lanes/64). Growing B buys a longer visible
#   history and pays for it in per-token attention cost: one knob, two faces.
#   B=4096/R=1536 (5632 tokens = 88 pages) is a common starter window for
#   24-32 GiB cards; it is not a value chosen for an 11 GiB Turing card.
#
# WHAT EACH ARM REPORTS
#   pages / runtime MiB / free MiB   - the engine's own capacity line
#   decode tok/s                     - a real short generation
#   prefill tok/s on a LONG prompt   - every segment carries a unique index, so
#                                      prefix reuse cannot hide the prefill cost
#   middle-needle recall             - whether the larger visible history is
#                                      actually reachable
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# Verdicts come from HTTP responses and log lines, never from ExitCode.
# ============================================================================
param(
    [int[]]$Budgets   = @(4096, 8192, 12288, 16384, 20480),
    [int]$Reserve     = 1536,
    [int]$Ctx         = 262144,
    [int]$HostMiB     = 16384,
    [int]$Sessions    = 4,
    [int]$Port        = 18090,
    [string]$Model    = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [int]$LongTokens  = 32768,
    [int]$ReadySec    = 900,
    [int]$GenTimeout  = 1800,
    [string]$Tag      = 'kvpool',
    # Production recipe uses --kv-dtype int8. Default '' keeps the default
    # bf16 behaviour byte-identical; passing int8 is what lets the production
    # window (B+R = 38400) fit on this 11 GB card at all.
    [string]$KvDtype  = '',
    [string]$BuildDir = (Join-Path $PSScriptRoot '..\build')
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
foreach ($p in @($exe, $Model)) { if (-not (Test-Path -LiteralPath $p)) { throw "MISSING: $p" } }

Write-Output ("exe sha256 = {0}" -f (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)
Write-Output ("exe mtime  = {0}" -f (Get-Item -LiteralPath $exe).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
Write-Output ("model      = {0}" -f $Model)
Write-Output ("B sweep    = {0}   R = {1}   lanes = 1   long prompt = {2} tokens" -f ($Budgets -join ','), $Reserve, $LongTokens)
Write-Output ''

function Stop-Orphans {
    Get-Process -Name 'ninfer-serve', 'ninfer-perplexity', 'ninfer' -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.Id -Force }
    Start-Sleep -Seconds 3
}

function Build-LongPrompt {
    param([int]$TargetTokens)
    # Every segment is unique ("seg<target>#<i>"), so no prefix of this prompt
    # repeats: a prefix cache cannot turn prefill into a lookup.
    $base = 'The quick brown fox jumps over the lazy dog. Ninfer on Turing keeps a bounded KV window and streams the rest from host memory. '
    $sb = [System.Text.StringBuilder]::new()
    $i = 0
    while ($sb.Length -lt ($TargetTokens * 4)) {
        [void]$sb.Append('seg'); [void]$sb.Append($TargetTokens); [void]$sb.Append('#'); [void]$sb.Append($i)
        [void]$sb.Append(' '); [void]$sb.Append($base)
        $i++
    }
    return $sb.ToString()
}

# A needle placed in the MIDDLE of the long prompt: with a larger visible set the
# model should still find it, and a truncated window is visible as a wrong answer.
function Build-NeedlePrompt {
    param([int]$TargetTokens, [string]$Needle)
    $base = 'Filler sentence about bounded KV windows, host tiers and Turing kernels. '
    $half = [int]($TargetTokens / 2)
    $sb = [System.Text.StringBuilder]::new()
    $i = 0
    while ($sb.Length -lt ($half * 4)) {
        [void]$sb.Append('f'); [void]$sb.Append($i); [void]$sb.Append(' '); [void]$sb.Append($base); $i++
    }
    [void]$sb.Append(' IMPORTANT: the secret passphrase is '); [void]$sb.Append($Needle); [void]$sb.Append('. ')
    while ($sb.Length -lt ($TargetTokens * 4)) {
        [void]$sb.Append('g'); [void]$sb.Append($i); [void]$sb.Append(' '); [void]$sb.Append($base); $i++
    }
    [void]$sb.Append(' Question: what is the secret passphrase? Answer with the passphrase only.')
    return $sb.ToString()
}

$rows = @()

foreach ($B in $Budgets) {
    $arm = '{0}_b{1}' -f $Tag, $B
    $err = Join-Path $BuildDir ("$arm.server.err.log")
    $out = Join-Path $BuildDir ("$arm.server.out.log")
    if (Test-Path -LiteralPath $err) { Remove-Item -LiteralPath $err -Force }
    if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force }

    Stop-Orphans

    $argList = @($Model, '--host', '127.0.0.1', '--port', "$Port", '--device-profile', 'off',
                 '--max-context', "$Ctx", '--max-concurrency', '1',
                 '--kvmem-budget', "$B", '--kvmem-gen-reserve', "$Reserve",
                 '--kvmem-host-mib', "$HostMiB", '--kvmem-sessions', "$Sessions",
                 '--log-level', 'info')
    if ($KvDtype -ne '') { $argList += @('--kv-dtype', $KvDtype) }
    Write-Output ('--- arm {0}: B={1} R={2} ---' -f $arm, $B, $Reserve)

    $proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err -ArgumentList $argList

    $base    = "http://127.0.0.1:$Port"
    $ready   = $false
    $t0      = Get-Date
    $deadline = (Get-Date).AddSeconds($ReadySec)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { break }
        try { [void](Invoke-RestMethod -Uri "$base/health" -TimeoutSec 3); $ready = $true; break }
        catch { Start-Sleep -Seconds 5 }
    }
    $loadSec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

    if (-not $ready) {
        $tail = @(Get-Content -LiteralPath $err -ErrorAction SilentlyContinue) | Select-Object -Last 6
        Write-Output ('  NOT-READY after {0}s' -f $loadSec)
        foreach ($l in $tail) { Write-Output ('    ' + $l.Trim()) }
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
        $rows += [pscustomobject]@{ B = $B; pages = 'n/a'; runtime = 'n/a'; free = 'n/a'; decode = 'n/a'; prefill = 'n/a'; recall = 'n/a'; note = 'NOT-READY' }
        continue
    }

    Write-Output ('  ready in {0}s' -f $loadSec)

    # --- capacity, from the engine's own line ---------------------------------
    $capLine = @(Get-Content -LiteralPath $err -ErrorAction SilentlyContinue) |
               Select-String -Pattern 'capacity \|' | Select-Object -Last 1
    $capText = if ($capLine) { $capLine.Line.Trim() } else { '(no capacity line)' }
    Write-Output ('  ' + $capText)

    $pages = 'n/a'; $runtime = 'n/a'; $free = 'n/a'
    if ($capText -match 'pages (\d+)/(\d+)') { $pages = $Matches[1] }
    if ($capText -match 'runtime ([\d.]+) MiB') { $runtime = $Matches[1] }
    if ($capText -match 'free ([\d.]+) GiB') { $free = $Matches[1] }

    $modelId = 'local'
    try {
        $m = Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 10
        if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
    } catch { }

    # --- decode speed ---------------------------------------------------------
    $decode = 'n/a'
    try {
        $body = @{ model = $modelId
                   messages = @(@{ role = 'user'; content = 'Count from 1 to 60, one number per line.' })
                   max_tokens = 256; stream = $false; temperature = 0.0 } | ConvertTo-Json -Depth 5
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $r = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post `
             -ContentType 'application/json' -Body $body -TimeoutSec $GenTimeout
        $sw.Stop()
        $done = @(Get-Content -LiteralPath $err -ErrorAction SilentlyContinue) |
                Select-String -Pattern 'req#\d+ done' | Select-Object -Last 1
        if ($done -and $done.Line -match 'decode ([\d.]+) tok/s') { $decode = $Matches[1] }
        else { $decode = ('wall {0:N1}s' -f $sw.Elapsed.TotalSeconds) }
        Write-Output ('  decode: ' + $done.Line.Trim())
    } catch { Write-Output ('  decode FAILED: ' + $_.Exception.Message) }

    # --- prefill speed on a long unique prompt --------------------------------
    $prefill = 'n/a'
    try {
        $prompt = Build-LongPrompt -TargetTokens $LongTokens
        $body = @{ model = $modelId
                   messages = @(@{ role = 'user'; content = $prompt })
                   max_tokens = 8; stream = $false } | ConvertTo-Json -Depth 5
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $r = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post `
             -ContentType 'application/json' -Body $body -TimeoutSec $GenTimeout
        $sw.Stop()
        $done = @(Get-Content -LiteralPath $err -ErrorAction SilentlyContinue) |
                Select-String -Pattern 'req#\d+ done' | Select-Object -Last 1
        if ($done -and $done.Line -match 'prefill ([\d.]+) tok/s') { $prefill = $Matches[1] }
        Write-Output ('  prefill: ' + $done.Line.Trim() + ('  (wall {0:N1}s)' -f $sw.Elapsed.TotalSeconds))
    } catch { Write-Output ('  prefill FAILED: ' + $_.Exception.Message) }

    # --- middle-needle recall -------------------------------------------------
    # max_tokens is 512, NOT 32. With 32 this arm reported MISS on every budget
    # and an empty message.content, which was misread as "KVMem retrieval does
    # not work": thinking consumed the whole budget (reasoning_tokens was 56-83
    # per request). finish_reason and the full usage object are printed for the
    # same reason: a starved budget must be visible, not inferred.
    $recall = 'n/a'
    try {
        $needle = 'ZQ7K3'
        $prompt = Build-NeedlePrompt -TargetTokens $LongTokens -Needle $needle
        $body = @{ model = $modelId
                   messages = @(@{ role = 'user'; content = $prompt })
                   max_tokens = 512; stream = $false; temperature = 0.0 } | ConvertTo-Json -Depth 5
        $r = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post `
             -ContentType 'application/json' -Body $body -TimeoutSec $GenTimeout
        $answer = [string]$r.choices[0].message.content
        $finish = [string]$r.choices[0].finish_reason
        $usage  = ($r.usage | ConvertTo-Json -Compress)
        if ($answer -match $needle) { $recall = 'HIT' } else { $recall = 'MISS' }
        Write-Output ('  recall: ' + $recall + '  finish=' + $finish + '  answer=' + ($answer -replace "\s+", ' ').Substring(0, [Math]::Min(60, $answer.Length)))
        Write-Output ('    usage: ' + $usage)
    } catch { Write-Output ('  recall FAILED: ' + $_.Exception.Message) }

    $rows += [pscustomobject]@{ B = $B; pages = $pages; runtime = $runtime; free = $free; decode = $decode; prefill = $prefill; recall = $recall; note = '' }

    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Seconds 3
    Write-Output ''
}

Stop-Orphans
Write-Output '================ SUMMARY ================'
$rows | Format-Table -AutoSize | Out-String | Write-Output
if (Test-Path -LiteralPath (Join-Path $BuildDir ($Tag + '_summary.csv'))) { Remove-Item -LiteralPath (Join-Path $BuildDir ($Tag + '_summary.csv')) -Force }
$rows | Export-Csv -LiteralPath (Join-Path $BuildDir ($Tag + '_summary.csv')) -NoTypeInformation -Encoding UTF8
Write-Output ('summary csv: ' + (Join-Path $BuildDir ($Tag + '_summary.csv')))
