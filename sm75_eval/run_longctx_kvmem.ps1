# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - long-context (256K) acceptance AND prefill-scaling
# judgement, in one run.
#
# WHY THESE FLAGS
#   NO --kv-capacity HERE.  In this stack KVMem owns the device capacity: with
#                           --kvmem-budget set, serve_options.cpp:1852 assigns
#                           explicit_capacity(device_tokens * max_concurrency) itself, and
#                           :1844 rejects ANY explicit --kv-capacity that is not exactly that
#                           value - including "--kv-capacity auto". A 256K recipe that
#                           passes --kv-capacity auto (needed when the default is
#                           explicit_capacity(max_context)) fails here
#                           with "KVMem requires aligned B/R, nonzero H, prefill <= R, and
#                           device capacity B+R <= max-context". The idea carries over; the
#                           spelling does not.
#   --kvmem-budget 4096     selected history B (device window part 1)
#   --kvmem-gen-reserve 1536 append workspace R (device window part 2)
#                           B + R = 5632 device tokens = the 88-page window this
#                           script uses for its 256K runs.
#   --kvmem-host-mib 16384  host native KV payload, same order as a 16 GiB host
#                           tier. This host has 63.9 GB of RAM.
#   --kvmem-sessions 4      also exercises the multi-session host tier.
#   --device-profile off    the calibration sweep is unusable on this card.
#
# WHAT IT MEASURES
#   * a real text answer (max_tokens 512), so "generation works" is shown and not assumed;
#   * prefill tok/s at three prompt sizes with UNIQUE content, so prefix reuse cannot
#     hide the prefill cost. Flat tok/s => attention cost per token is bounded;
#     falling tok/s => the O(cursor^2) prefill behaviour is present here too and
#     chunked prefill must be re-checked.
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# Verdicts come from HTTP responses and log lines, never from ExitCode.
# ============================================================================
param(
    [string]$Model          = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [int]$Port              = 18085,
    [int]$Ctx               = 262144,
    [int]$KvmemBudget       = 4096,
    [int]$KvmemReserve      = 1536,
    [int]$HostMiB           = 16384,
    [int]$Sessions          = 4,
    [int[]]$Lens            = @(8192, 32768, 98304),
    [int]$ReadySec          = 900,
    [int]$GenTimeoutSec     = 1800,
    [string]$Tag            = 'longctx_kvmem',
    # Production recipe uses --kv-dtype int8. Default '' keeps the default
    # bf16 behaviour byte-identical; passing int8 is what lets the production
    # window (B+R = 38400) fit on this 11 GB card at all.
    [string]$KvDtype        = '',
    [string]$BuildDir       = (Join-Path $PSScriptRoot '..\build')
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

foreach ($p in @($exe, $Model)) { if (-not (Test-Path -LiteralPath $p)) { throw "MISSING: $p" } }

Write-Output ("exe sha256 = {0}" -f (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)
Write-Output ("exe mtime  = {0}" -f (Get-Item -LiteralPath $exe).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))

Get-Process -Name 'ninfer-serve', 'ninfer-perplexity', 'ninfer' -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Output ('killing orphan: ' + $_.Name); Stop-Process -Id $_.Id -Force }
Start-Sleep -Seconds 2

$argList = @($Model, '--host', '127.0.0.1', '--port', "$Port", '--device-profile', 'off',
             '--max-context', "$Ctx", '--max-concurrency', '1',
             '--kvmem-budget', "$KvmemBudget", '--kvmem-gen-reserve', "$KvmemReserve",
             '--kvmem-host-mib', "$HostMiB", '--kvmem-sessions', "$Sessions",
             '--log-level', 'info')
if ($KvDtype -ne '') { $argList += @('--kv-dtype', $KvDtype) }
Write-Output ("flags      = " + ($argList -join ' '))

$proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError $err -ArgumentList $argList

$base     = "http://127.0.0.1:$Port"
$ready    = $false
$deadline = (Get-Date).AddSeconds($ReadySec)
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) { Write-Output 'server exited before becoming ready'; break }
    try {
        $h = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 3
        $ready = $true
        Write-Output ("health     = " + ($h | ConvertTo-Json -Compress))
        break
    } catch { Start-Sleep -Seconds 5 }
}

if (-not $ready) {
    Write-Output 'RESULT: NOT-READY'
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Seconds 2
    Write-Output '--- server stderr tail 40 ---'
    Get-Content -LiteralPath $err -Tail 40 -ErrorAction SilentlyContinue
    exit 1
}

$modelId = 'local'
try {
    $m = Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 10
    if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
    Write-Output ("model id   = " + $modelId)
} catch { Write-Output ("WARN /v1/models: " + $_.Exception.Message) }

Write-Output '--- /v1/load (capacity: the long-context judgement) ---'
try {
    $load = Invoke-RestMethod -Uri "$base/v1/load" -TimeoutSec 10
    Write-Output ($load | ConvertTo-Json -Compress -Depth 6)
} catch { Write-Output ("WARN /v1/load: " + $_.Exception.Message) }

function Get-LastDone {
    param([string]$LogPath)
    $line = Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue |
            Select-String -Pattern 'req#\d+ done' | Select-Object -Last 1
    if ($line) { return $line.Line.Trim() }
    return '(no req#N done line yet)'
}

# --- 1) a real answer: proves generation, not just an HTTP 200 -------------------------------
Write-Output ''
Write-Output '--- answer sanity (max_tokens 512) ---'
$body = @{ model = $modelId
           messages = @(@{ role = 'user'; content = 'Reply with exactly: KVMEM SM75 LONG OK' })
           max_tokens = 512; stream = $false } | ConvertTo-Json -Depth 5
try {
    $resp = Invoke-WebRequest -Uri "$base/v1/chat/completions" -Method Post `
        -ContentType 'application/json' -Body $body -TimeoutSec $GenTimeoutSec
    $r = $resp.Content | ConvertFrom-Json -AsHashtable
    Write-Output ("answer     = " + $r['choices'][0]['message']['content'])
    Write-Output ("usage      = " + ($r['usage'] | ConvertTo-Json -Compress))
} catch { Write-Output ("ANSWER REQUEST FAILED: " + $_.Exception.Message) }
Write-Output ("log        = " + (Get-LastDone -LogPath $err))

# --- 2) prefill scaling: unique content per size, so prefix reuse cannot hide the cost -------
$base_text = 'The quick brown fox jumps over the lazy dog. Ninfer on Turing keeps a bounded KV window and streams the rest from host memory. '
Write-Output ''
Write-Output '--- prefill scaling (unique content per size) ---'
foreach ($target in $Lens) {
    $sb = [System.Text.StringBuilder]::new()
    $chars = $target * 4
    $i = 0
    while ($sb.Length -lt $chars) {
        [void]$sb.Append('seg'); [void]$sb.Append($target); [void]$sb.Append('#'); [void]$sb.Append($i)
        [void]$sb.Append(' '); [void]$sb.Append($base_text)
        $i++
    }
    $prompt = $sb.ToString()
    $body = @{ model = $modelId
               messages = @(@{ role = 'user'; content = $prompt })
               max_tokens = 8; stream = $false } | ConvertTo-Json -Depth 5
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-WebRequest -Uri "$base/v1/chat/completions" -Method Post `
            -ContentType 'application/json' -Body $body -TimeoutSec $GenTimeoutSec
        $r = $resp.Content | ConvertFrom-Json -AsHashtable
        $sw.Stop()
        Write-Output ("target {0,7} tokens | prompt_tokens {1,7} | wall {2,7:N1}s | {3}" -f `
            $target, $r['usage']['prompt_tokens'], $sw.Elapsed.TotalSeconds, (Get-LastDone -LogPath $err))
    } catch {
        $sw.Stop()
        Write-Output ("target {0,7} tokens | FAILED after {1:N1}s: {2}" -f $target, $sw.Elapsed.TotalSeconds, $_.Exception.Message)
    }
}

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
Start-Sleep -Seconds 2
Write-Output ''
Write-Output '--- capacity / kvmem / warning lines from stderr ---'
Get-Content -LiteralPath $err -ErrorAction SilentlyContinue |
    Where-Object { $_ -match 'capacity \||kvmem|WARN|ERROR|FATAL|reservation|host KV|context cache' } |
    Select-Object -Last 30
