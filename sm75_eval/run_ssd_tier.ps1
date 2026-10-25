# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - SSD/RAM tiering acceptance ("force an eviction").
#
# WHY THIS EXISTS
#   The tier only engages when the accumulated history exceeds --kvmem-host-mib,
#   so the test must create that pressure deliberately; otherwise disk_writes
#   stays 0 and the tier is never exercised.
#
#   Mechanism (program_impl.cpp:967-974):
#     memory.disk_writes = memory_payload_storage.io_stats().writes
#     memory.disk_payload_budget_bytes = kvmem_options.disk_bytes   (--kvmem-disk-mib)
#   Both surface in GET /v1/load via kvmem_status_json.h, which is the verdict.
#
# HOW PRESSURE IS CREATED
#   A BF16 KV page is 64 tokens and about 7.77 MiB on this model, i.e. roughly
#   124 KiB per token. One 32K-token conversation therefore carries ~4 GiB of
#   native KV, far more than a deliberately small --kvmem-host-mib, so the
#   overflow must be written to the tier directory.
#
# WHAT IT COMPARES
#   The same load against two tier locations, so the choice in the start script
#   is made from a measurement rather than from the drive's label:
#     <包根>\Cache   (a fast local disk)
#     the system temp directory (typically NVMe)
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# Verdicts come from HTTP responses and log lines, never from ExitCode.
# ============================================================================
param(
    [string[]]$TierPaths = @(
        (Join-Path $PSScriptRoot '..\Cache\kvmem_tier'),
        (Join-Path $env:TEMP 'disk_Cache')
    ),
    [int]$Budget      = 12288,
    [int]$Reserve     = 1536,
    [int]$Ctx         = 262144,
    [int]$HostMiB     = 1024,     # deliberately small: force the overflow
    [int]$DiskMiB     = 8192,
    [int]$Sessions    = 4,
    [int]$Port        = 18092,
    [int]$LongTokens  = 32768,
    [int]$Requests    = 3,
    [string]$Model    = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [int]$ReadySec    = 900,
    [int]$GenTimeout  = 2400,
    [string]$Tag      = 'ssdtier',
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
Write-Output ("model      = {0}" -f $Model)
Write-Output ("B={0} R={1} hostMiB={2} diskMiB={3} sessions={4} long={5} reqs={6}" -f $Budget, $Reserve, $HostMiB, $DiskMiB, $Sessions, $LongTokens, $Requests)
Write-Output ("tier paths = {0}" -f ($TierPaths -join ' | '))
Write-Output ''

function Stop-Orphans {
    Get-Process -Name 'ninfer-serve', 'ninfer-perplexity', 'ninfer' -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.Id -Force }
    Start-Sleep -Seconds 3
}

function Build-UniquePrompt {
    param([int]$TargetTokens, [int]$Seed)
    $base = 'Bounded KV windows stream the rest from host memory and spill to the tier when it fills. '
    $sb = [System.Text.StringBuilder]::new()
    $i = 0
    while ($sb.Length -lt ($TargetTokens * 4)) {
        [void]$sb.Append('s'); [void]$sb.Append($Seed); [void]$sb.Append('#'); [void]$sb.Append($i)
        [void]$sb.Append(' '); [void]$sb.Append($base)
        $i++
    }
    return $sb.ToString()
}

$rows = @()

foreach ($tier in $TierPaths) {
    Write-Output ('===== tier: ' + $tier + ' =====')
    try { if (-not (Test-Path -LiteralPath $tier)) { New-Item -ItemType Directory -Path $tier -Force | Out-Null } }
    catch { Write-Output ('  cannot create tier dir: ' + $_.Exception.Message); continue }

    Stop-Orphans

    $arm = '{0}_{1}' -f $Tag, (($tier -replace '[:\\]', '_') -replace '\s', '')
    $err = Join-Path $BuildDir ("$arm.server.err.log")
    $out = Join-Path $BuildDir ("$arm.server.out.log")
    Remove-Item -LiteralPath $err, $out -ErrorAction SilentlyContinue

    $argList = @($Model, '--host', '127.0.0.1', '--port', "$Port", '--device-profile', 'off',
                 '--max-context', "$Ctx", '--max-concurrency', '1',
                 '--kvmem-budget', "$Budget", '--kvmem-gen-reserve', "$Reserve",
                 '--kvmem-host-mib', "$HostMiB", '--kvmem-sessions', "$Sessions",
                 '--kvmem-disk-path', $tier, '--kvmem-disk-mib', "$DiskMiB",
                 '--log-level', 'info')

    $proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err -ArgumentList $argList

    $base     = "http://127.0.0.1:$Port"
    $ready    = $false
    $deadline = (Get-Date).AddSeconds($ReadySec)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { break }
        try { [void](Invoke-RestMethod -Uri "$base/health" -TimeoutSec 3); $ready = $true; break }
        catch { Start-Sleep -Seconds 5 }
    }
    if (-not $ready) {
        Write-Output '  NOT-READY'
        Get-Content -LiteralPath $err -ErrorAction SilentlyContinue | Select-Object -Last 8 | ForEach-Object { Write-Output ('    ' + $_) }
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
        $rows += [pscustomobject]@{ tier = $tier; disk_writes = 'n/a'; written_MiB = 'n/a'; payload_MiB = 'n/a'; wall_s = 'n/a'; note = 'NOT-READY' }
        continue
    }

    $modelId = 'local'
    try {
        $m = Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 10
        if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
    } catch { }

    # Push the accumulated history past the Host budget.
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    for ($k = 1; $k -le $Requests; $k++) {
        try {
            $prompt = Build-UniquePrompt -TargetTokens $LongTokens -Seed $k
            $body = @{ model = $modelId
                       messages = @(@{ role = 'user'; content = $prompt })
                       max_tokens = 8; stream = $false } | ConvertTo-Json -Depth 5
            $r = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post `
                 -ContentType 'application/json' -Body $body -TimeoutSec $GenTimeout
            Write-Output ('  req {0}: prompt_tokens={1}' -f $k, $r.usage.prompt_tokens)
        } catch { Write-Output ('  req {0} FAILED: {1}' -f $k, $_.Exception.Message) }
    }
    $sw.Stop()

    $load = $null
    try { $load = Invoke-RestMethod -Uri "$base/v1/load" -TimeoutSec 20 } catch { }

    $dw = 'n/a'; $wb = 'n/a'; $pb = 'n/a'
    if ($load) {
        $j = $load | ConvertTo-Json -Depth 8
        if ($j -match '"disk_writes"\s*:\s*(\d+)') { $dw = $Matches[1] }
        if ($j -match '"disk_written_bytes"\s*:\s*(\d+)') { $wb = [math]::Round([double]$Matches[1] / 1MB, 1) }
        if ($j -match '"disk_payload_bytes"\s*:\s*(\d+)') { $pb = [math]::Round([double]$Matches[1] / 1MB, 1) }
        $kv = $load.kvmem
        if ($kv) {
            Write-Output ('  kvmem: enabled={0} selected={1} reserve={2} host_budget_MiB={3} disk_budget_MiB={4}' -f `
                $kv.enabled, $kv.selected_tokens, $kv.reserve_tokens, `
                [math]::Round([double]$kv.host_payload_budget_bytes / 1MB, 1), `
                [math]::Round([double]$kv.disk_payload_budget_bytes / 1MB, 1))
        }
    }
    Write-Output ("  verdict: disk_writes={0}  disk_written_MiB={1}  disk_payload_MiB={2}  wall={3:N1}s" -f $dw, $wb, $pb, $sw.Elapsed.TotalSeconds)

    $rows += [pscustomobject]@{ tier = $tier; disk_writes = $dw; written_MiB = $wb; payload_MiB = $pb; wall_s = [math]::Round($sw.Elapsed.TotalSeconds, 1); note = '' }

    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Seconds 3
    Write-Output ''
}

Stop-Orphans
Write-Output '================ SUMMARY ================'
$rows | Format-Table -AutoSize | Out-String | Write-Output
$csv = Join-Path $BuildDir ($Tag + '_summary.csv')
Remove-Item -LiteralPath $csv -ErrorAction SilentlyContinue
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
Write-Output ('summary csv: ' + $csv)
