# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - prefill/decode measurement for chosen (B, R) pairs,
# taken under PRODUCTION host load.
#
# WHY A SEPARATE SCRIPT
#   Prefill numbers move by up to 2x with host load: B=36864 measured 423.7 tok/s
#   and B=32768 measured 210.7 tok/s on the same 34,648-token prompt, i.e. the
#   larger pool looked twice as fast. That is a host-load artifact, not a KVMem
#   property. This script therefore measures the same pairs under a known,
#   recorded host load.
#
# HOST LOAD - NOT SILENT, ON PURPOSE (deliberate production setting)
#   Production conditions are the right conditions. Edge and the production
#   apps stay open, because a judgement that only holds on a quiet host does not
#   describe the machine the user actually runs. The host can move a prefill
#   number by 2x, so each arm records a host-load snapshot and a number is only
#   acted on when
#   a repeat agrees with it. A disagreement is reported, never averaged away.
#
# WHAT IT MEASURES PER (B, R)
#   the engine's own capacity line (pages / runtime / free)
#   decode tok/s on a short generation
#   prefill tok/s on a long prompt whose every segment is unique, so prefix
#   reuse cannot turn prefill into a lookup
#
# RATIONALE FOR THE PAIRS
#   A 16 GiB card can take a retrieval budget of 36864 (36K) or 32768 (32K) with
#   a generation reserve of 16384 / 12288 (its peak measured 460-606 MiB free).
#   On this 11 GB card B+R=38400 already leaves 235 MiB free and B+R=42496
#   collapses (prefill 1.8 tok/s), so B must be traded against R.
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# Verdicts come from HTTP responses and log lines, never from ExitCode.
# ============================================================================
param(
    [string[]]$Pairs   = @('4096:1536', '32768:1536', '32768:5632', '28672:8192'),
    [int]$Ctx         = 262144,
    [int]$HostMiB     = 16384,
    [int]$Sessions    = 4,
    [int]$Sink        = 4096,
    [int]$Recent      = 1024,
    [int]$Port        = 18096,
    [string]$Model    = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [int]$LongTokens  = 32768,
    [int]$ReadySec    = 900,
    [int]$GenTimeout  = 1800,
    [string]$Tag      = 'prefillpairs',
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
Write-Output ("pairs (B:R) = {0}" -f ($Pairs -join ' | '))
Write-Output ''

function Stop-Orphans {
    Get-Process -Name 'ninfer-serve', 'ninfer-perplexity', 'ninfer' -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.Id -Force }
    Start-Sleep -Seconds 3
}

# The measurement is taken under production load, so the load itself is part of
# the evidence: when a repeat disagrees, this snapshot says whether the host
# moved rather than leaving it to guesswork.
function Get-HostSnapshot {
    $gpu  = & nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>$null
    $proc = @(Get-Process).Count
    return ('gpu[' + ([string]$gpu).Trim() + '] procs=' + $proc)
}

function Build-UniquePrompt {
    param([int]$TargetTokens)
    $base = 'The quick brown fox jumps over the lazy dog. Ninfer on Turing keeps a bounded KV window and streams the rest from host memory. '
    $sb = [System.Text.StringBuilder]::new()
    $i = 0
    while ($sb.Length -lt ($TargetTokens * 4)) {
        [void]$sb.Append('u'); [void]$sb.Append($TargetTokens); [void]$sb.Append('#'); [void]$sb.Append($i)
        [void]$sb.Append(' '); [void]$sb.Append($base)
        $i++
    }
    return $sb.ToString()
}

$rows = @()

foreach ($pair in $Pairs) {
    $parts = $pair -split ':'
    $B = [int]$parts[0]; $R = [int]$parts[1]
    $arm = '{0}_b{1}_r{2}' -f $Tag, $B, $R
    $err = Join-Path $BuildDir ("$arm.server.err.log")
    $out = Join-Path $BuildDir ("$arm.server.out.log")
    Remove-Item -LiteralPath $err, $out -ErrorAction SilentlyContinue

    Stop-Orphans

    $argList = @($Model, '--host', '127.0.0.1', '--port', "$Port", '--device-profile', 'off',
                 '--max-context', "$Ctx", '--max-concurrency', '1',
                 '--kvmem-budget', "$B", '--kvmem-gen-reserve', "$R",
                 '--kvmem-host-mib', "$HostMiB", '--kvmem-sessions', "$Sessions",
                 '--kvmem-sink-tokens', "$Sink", '--kvmem-recent-tokens', "$Recent",
                 '--log-level', 'info')
    Write-Output ('--- arm {0} ---' -f $arm)

    $proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err -ArgumentList $argList

    $baseUrl  = "http://127.0.0.1:$Port"
    $ready    = $false
    $deadline = (Get-Date).AddSeconds($ReadySec)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { break }
        try { [void](Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 3); $ready = $true; break }
        catch { Start-Sleep -Seconds 5 }
    }

    if (-not $ready) {
        Write-Output '  NOT-READY'
        Get-Content -LiteralPath $err -ErrorAction SilentlyContinue | Select-String -Pattern 'capacity \||KVMem rejected|error|FATAL' |
            Select-Object -Last 3 | ForEach-Object { Write-Output ('    ' + $_.Line.Trim()) }
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
        $rows += [pscustomobject]@{ B = $B; R = $R; pages = 'n/a'; free = 'n/a'; decode = 'n/a'; prefill = 'n/a'; note = 'NOT-READY' }
        continue
    }

    $cap = @(Get-Content -LiteralPath $err -ErrorAction SilentlyContinue) | Select-String -Pattern 'capacity \|' | Select-Object -Last 1
    $capText = if ($cap) { $cap.Line.Trim() } else { '(no capacity line)' }
    Write-Output ('  ' + $capText)
    Write-Output ('  host load: ' + (Get-HostSnapshot))
    $pages = 'n/a'; $free = 'n/a'
    if ($capText -match 'pages (\d+)/(\d+)') { $pages = $Matches[1] }
    if ($capText -match 'free ([\d.]+) (GiB|MiB|B)') { $free = ($Matches[1] + ' ' + $Matches[2]) }

    $modelId = 'local'
    try {
        $m = Invoke-RestMethod -Uri "$baseUrl/v1/models" -TimeoutSec 10
        if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
    } catch { }

    # --- decode ---------------------------------------------------------------
    $decode = 'n/a'
    try {
        $body = @{ model = $modelId
                   messages = @(@{ role = 'user'; content = 'Count from 1 to 60, one number per line.' })
                   max_tokens = 256; stream = $false; temperature = 0.0 } | ConvertTo-Json -Depth 5
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        # NOT $r: PowerShell variable names are case-insensitive, so a local
        # $r would overwrite the $R parameter that names the generation reserve.
        $resp = Invoke-RestMethod -Uri "$baseUrl/v1/chat/completions" -Method Post `
             -ContentType 'application/json' -Body $body -TimeoutSec $GenTimeout
        $sw.Stop()
        $done = @(Get-Content -LiteralPath $err -ErrorAction SilentlyContinue) |
                Select-String -Pattern 'req#\d+ done' | Select-Object -Last 1
        if ($done -and $done.Line -match 'decode ([\d.]+) tok/s') { $decode = $Matches[1] }
        else { $decode = ('wall {0:N1}s' -f $sw.Elapsed.TotalSeconds) }
        Write-Output ('  decode : ' + $done.Line.Trim())
    } catch { Write-Output ('  decode FAILED: ' + $_.Exception.Message) }

    # --- prefill on a long, unique prompt -------------------------------------
    $prefill = 'n/a'
    try {
        $prompt = Build-UniquePrompt -TargetTokens $LongTokens
        $body = @{ model = $modelId
                   messages = @(@{ role = 'user'; content = $prompt })
                   max_tokens = 8; stream = $false } | ConvertTo-Json -Depth 5
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        # NOT $r: PowerShell variable names are case-insensitive, so a local
        # $r would overwrite the $R parameter that names the generation reserve.
        $resp = Invoke-RestMethod -Uri "$baseUrl/v1/chat/completions" -Method Post `
             -ContentType 'application/json' -Body $body -TimeoutSec $GenTimeout
        $sw.Stop()
        $done = @(Get-Content -LiteralPath $err -ErrorAction SilentlyContinue) |
                Select-String -Pattern 'req#\d+ done' | Select-Object -Last 1
        if ($done -and $done.Line -match 'prefill ([\d.]+) tok/s') { $prefill = $Matches[1] }
        Write-Output ('  prefill: ' + $done.Line.Trim() + ('  (wall {0:N1}s)' -f $sw.Elapsed.TotalSeconds))
    } catch { Write-Output ('  prefill FAILED: ' + $_.Exception.Message) }

    $rows += [pscustomobject]@{ B = $B; R = $R; pages = $pages; free = $free; decode = $decode; prefill = $prefill; note = '' }

    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Seconds 3
    Write-Output ''
}

Stop-Orphans
Write-Output '================ PRODUCTION SUMMARY ================'
$rows | Format-Table -AutoSize | Out-String | Write-Output
$csv = Join-Path $BuildDir ($Tag + '_summary.csv')
Remove-Item -LiteralPath $csv -ErrorAction SilentlyContinue
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
Write-Output ('summary csv: ' + $csv)
