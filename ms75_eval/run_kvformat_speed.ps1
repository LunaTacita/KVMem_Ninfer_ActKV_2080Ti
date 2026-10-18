# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - KV storage format x speed matrix.
#
# WHY THIS EXISTS
#   The device pool is (B+R) x lanes tokens, and the storage format decides how
#   many bytes each token costs: it is what makes or breaks "1M + MTP" on this
#   card. Measured 2026-10-15 with B=32768 / R=5632 / lanes=1 / ctx=1M / MTP on:
#     bf16    reservation 3.18 GiB -> REFUSED (only 2.75 GiB free after weights)
#     int8    reservation 1.79 GiB -> free 845 MiB
#     fp8     reservation 1.76 GiB -> free 847 MiB
#     rk8v4   reservation 1.50 GiB -> free 1.09 GiB
#     k8v4    reservation 1.48 GiB -> free 1.09 GiB
#   Free memory is not the whole decision, so this script measures the part the
#   format also changes: decode tok/s and prefill tok/s, back to back, with only
#   the format varying.
#
# WHAT EVERY ARM REPORTS (from the engine's own words, never ExitCode)
#   ready seconds, the capacity line (pages / runtime / free), the memory ledger
#   (after weights / slack), GPU used+free, decode tok/s, prefill tok/s on a 32K
#   prompt whose every segment is unique (so prefix reuse cannot turn prefill
#   into a lookup), and the full usage object of the recall-free decode.
#
# HOST LOAD - NOT SILENT, ON PURPOSE (deliberate production setting): production
#   conditions are the right conditions. A measurement is accepted only when the
#   repeats AGREE; a disagreeing repeat is reported as SPREAD and never averaged.
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# ============================================================================
param(
    [string]$Model = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [string[]]$KvDtypes = @('bf16', 'int8', 'fp8', 'rk8v4', 'k8v4'),
    [switch]$GdnFp16,
    [string]$Mtp = 'mtp',
    # MTP draft count per step. The production recipe uses 2 (--spec mtp REQUIRES
    # --draft-tokens in [1,15]); this matrix holds it at one value so only the KV
    # format varies.
    [int]$DraftTokens = 3,
    [int]$B = 32768,
    [int]$R = 5632,
    [int]$Ctx = 1048576,
    [int]$HostMiB = 49152,
    [string]$DiskPath = (Join-Path $PSScriptRoot '..\Cache\kvmem'),
    [int]$DiskMiB = 102400,
    [int]$Sessions = 4,
    [int]$Sink = 4096,
    [int]$Recent = 1024,
    [int]$PrefillTokens = 32768,
    [int]$Repeats = 1,
    [int]$BasePort = 18900,
    [int]$ReadySec = 900,
    [int]$GenTimeout = 3600,
    [string]$Tag = 'kvfmt',
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
Write-Output ("window     = B {0} + R {1} = {2} tokens/lane | ctx {3} | mtp {4} | gdn-fp16 {5}" -f `
    $B, $R, ($B + $R), $Ctx, $Mtp, $GdnFp16)
Write-Output ("formats    = {0}  x repeats {1}" -f ($KvDtypes -join ','), $Repeats)
Write-Output ''

function Stop-Orphans {
    Get-Process -Name 'ninfer-serve', 'ninfer-perplexity', 'ninfer' -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.Id -Force }
    Start-Sleep -Seconds 5
}

function Build-UniquePrompt {
    param([int]$TargetTokens, [string]$Prefix)
    $base = 'The quick brown fox jumps over the lazy dog. Ninfer on Turing keeps a bounded KV window and streams the rest from host memory. '
    $sb = [System.Text.StringBuilder]::new()
    $i = 0
    while ($sb.Length -lt ($TargetTokens * 4)) {
        [void]$sb.Append($Prefix); [void]$sb.Append('#'); [void]$sb.Append($i)
        [void]$sb.Append(' '); [void]$sb.Append($base)
        $i++
    }
    return $sb.ToString()
}

function Get-DoneLines {
    param([string]$LogPath)
    return @(Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue |
             Select-String -Pattern 'req#\d+ done')
}

function Get-LastMatching {
    param([string]$LogPath, [string]$Pattern)
    $line = Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue |
            Select-String -Pattern $Pattern | Select-Object -Last 1
    if ($line) { return $line.Line.Trim() }
    return '(none)'
}

function Get-GpuState {
    $v = & nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>$null
    if ($v) { return ([string]$v).Trim() }
    return 'n/a'
}

# Fires one chat completion and waits for the engine to write the matching
# "req#N done" line, so tok/s always comes from the engine's own report.
function Invoke-Chat {
    param([string]$Url, [hashtable]$Body, [string]$LogPath, [int]$TimeoutSec)
    $before = (Get-DoneLines -LogPath $LogPath).Count
    $json   = $Body | ConvertTo-Json -Depth 6
    $resp   = Invoke-RestMethod -Uri $Url -Method Post -ContentType 'application/json' `
                -Body $json -TimeoutSec $TimeoutSec
    $deadline = (Get-Date).AddSeconds(120)
    $lines = @()
    while ((Get-Date) -lt $deadline) {
        $lines = Get-DoneLines -LogPath $LogPath
        if ($lines.Count -gt $before) { break }
        Start-Sleep -Milliseconds 500
    }
    $done = if ($lines.Count -gt 0) { $lines[-1].Line.Trim() } else { '(no done line)' }
    return [pscustomobject]@{ Resp = $resp; Done = $done }
}

$rows = @()
$armIndex = 0
foreach ($dtype in $KvDtypes) {
    for ($rep = 1; $rep -le $Repeats; $rep++) {
        $armIndex++
        $port = $BasePort + $armIndex
        $arm = '{0}_{1}_r{2}' -f $Tag, $dtype, $rep
        $errLog = Join-Path $BuildDir ("$arm.server.err.log")
        $outLog = Join-Path $BuildDir ("$arm.server.out.log")
        Remove-Item -LiteralPath $errLog, $outLog -ErrorAction SilentlyContinue
        Stop-Orphans

        Write-Output ('================ ARM {0} ================' -f $arm)
        $childArgs = @($Model, '--host', '127.0.0.1', '--port', "$port", '--device-profile', 'off',
                       '--max-context', "$Ctx", '--max-concurrency', '1',
                       '--kvmem-budget', "$B", '--kvmem-gen-reserve', "$R",
                       '--kvmem-host-mib', "$HostMiB", '--kvmem-disk-path', "$DiskPath",
                       '--kvmem-disk-mib', "$DiskMiB", '--kvmem-sessions', "$Sessions",
                       '--kvmem-sink-tokens', "$Sink", '--kvmem-recent-tokens', "$Recent",
                       '--log-level', 'debug')
        if ($dtype -ne '' -and $dtype -ne 'bf16') { $childArgs += @('--kv-dtype', $dtype) }
        if ($GdnFp16) { $childArgs += '--gdn-state-fp16' }
        if ($Mtp -eq 'mtp') { $childArgs += @('--spec', 'mtp', '--draft-tokens', "$DraftTokens") }

        $row = [pscustomobject]@{
            arm = $arm; dtype = $dtype; rep = $rep
            ready_s = 'n/a'; capacity = 'n/a'; ledger = 'n/a'; gpu = 'n/a'
            decode = 'n/a'; prefill = 'n/a'; usage = 'n/a'; note = ''
        }

        $t0 = Get-Date
        $proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
            -RedirectStandardOutput $outLog -RedirectStandardError $errLog -ArgumentList $childArgs
        $baseUrl  = "http://127.0.0.1:$port"
        $ready    = $false
        $deadline = (Get-Date).AddSeconds($ReadySec)
        while ((Get-Date) -lt $deadline) {
            if ($proc.HasExited) { break }
            try { [void](Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 3); $ready = $true; break }
            catch { Start-Sleep -Seconds 4 }
        }
        $row.ready_s = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

        if (-not $ready) {
            Write-Output ('  NOT-READY after {0}s' -f $row.ready_s)
            Get-Content -LiteralPath $errLog -ErrorAction SilentlyContinue |
                Select-String -Pattern 'KVMem rejected|requested Engine runtime reservation|it needs|can hold about|FATAL' |
                Select-Object -Last 3 | ForEach-Object { Write-Output ('    ' + $_.Line.Trim()) }
            $row.note = 'NOT-READY'
            if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
            Start-Sleep -Seconds 5
            $rows += $row
            continue
        }

        $row.capacity = Get-LastMatching -LogPath $errLog -Pattern 'capacity \|'
        $row.ledger   = Get-LastMatching -LogPath $errLog -Pattern 'memory ledger'
        $row.gpu      = Get-GpuState
        Write-Output ('  ' + $row.capacity)
        Write-Output ('  ' + $row.ledger)

        $modelId = 'local'
        try {
            $m = Invoke-RestMethod -Uri "$baseUrl/v1/models" -TimeoutSec 10
            if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
        } catch { }
        $chatUrl = "$baseUrl/v1/chat/completions"

        try {
            $res = Invoke-Chat -Url $chatUrl -LogPath $errLog -TimeoutSec $GenTimeout -Body @{
                model = $modelId
                messages = @(@{ role = 'user'; content = 'Count from 1 to 60, one number per line.' })
                max_tokens = 256; stream = $false; temperature = 0.0
            }
            $row.usage = ($res.Resp.usage | ConvertTo-Json -Compress)
            if ($res.Done -match 'decode ([\d.]+) tok/s') { $row.decode = $Matches[1] }
            Write-Output ('  decode: ' + $res.Done)
        } catch { $row.note += 'decodeFail;'; Write-Output ('  decode FAILED: ' + $_.Exception.Message) }

        try {
            $prompt = Build-UniquePrompt -TargetTokens $PrefillTokens -Prefix 'seg'
            $res = Invoke-Chat -Url $chatUrl -LogPath $errLog -TimeoutSec $GenTimeout -Body @{
                model = $modelId
                messages = @(@{ role = 'user'; content = $prompt })
                max_tokens = 8; stream = $false
            }
            if ($res.Done -match 'prefill ([\d.]+) tok/s') { $row.prefill = $Matches[1] }
            Write-Output ('  prefill: ' + $res.Done)
        } catch { $row.note += 'prefillFail;'; Write-Output ('  prefill FAILED: ' + $_.Exception.Message) }

        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
        Start-Sleep -Seconds 5
        $rows += $row
        Write-Output ''
    }
}

Stop-Orphans
Write-Output '================ KV FORMAT SPEED SUMMARY ================'
$rows | Format-Table arm, ready_s, decode, prefill, gpu, note -AutoSize | Out-String | Write-Output
$csv = Join-Path $BuildDir ($Tag + '_summary.csv')
Remove-Item -LiteralPath $csv -ErrorAction SilentlyContinue
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
Write-Output ('summary csv: ' + $csv)
Write-Output 'RESULT: KVFMT-DONE'
