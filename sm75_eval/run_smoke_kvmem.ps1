# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - serving smoke test.
# Lives in the repo on purpose: judgement scripts must be versioned with the
# code they judge.
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK and then either
# throws "Unexpected token" or silently runs nothing).
#
# Verdicts come from the HTTP responses and the server log, NEVER from
# Start-Process.ExitCode: under PS 5.1 with -NoNewWindow + redirection that
# property can come back empty and misreport a pass as a failure.
#
# Two log channels on purpose: stdout carries the engine's own
# self-report, stderr carries progress/errors; a single redirect loses one.
# ============================================================================
param(
    [string]$Model          = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [int]$Port              = 18081,
    [string]$Extra          = '--device-profile off',
    [int]$ReadySec          = 900,
    [int]$GenTimeoutSec     = 900,
    [string]$Prompt         = 'Reply with exactly: KVMEM SM75 STACK OK',
    [int]$MaxTokens         = 32,
    [string]$Tag            = 'smoke_kvmem',
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

# Anchor the binary by hash and mtime: the "build <id>" string in the log comes
# from a cmake stamp target that does NOT refresh on every rebuild.
Write-Output ("exe sha256 = {0}" -f (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)
Write-Output ("exe mtime  = {0}" -f (Get-Item -LiteralPath $exe).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
Write-Output ("model      = {0}" -f $Model)

Get-Process -Name 'ninfer-serve', 'ninfer-perplexity', 'ninfer' -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Output ('killing orphan: ' + $_.Name); Stop-Process -Id $_.Id -Force }
Start-Sleep -Seconds 2

# NB: not $args -- that is a PowerShell automatic variable.
$argList = @($Model, '--host', '127.0.0.1', '--port', "$Port") +
           @($Extra -split '\s+' | Where-Object { $_ -ne '' })
Write-Output ("flags      = " + ($argList -join ' '))

$proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError $err -ArgumentList $argList

$base     = "http://127.0.0.1:$Port"
$ready    = $false
$deadline = (Get-Date).AddSeconds($ReadySec)
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) { Write-Output "server exited before becoming ready"; break }
    try {
        $h = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 3
        $ready = $true
        Write-Output ("health     = " + ($h | ConvertTo-Json -Compress))
        break
    } catch { Start-Sleep -Seconds 3 }
}

if (-not $ready) {
    Write-Output 'RESULT: NOT-READY'
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Seconds 2
    Write-Output '--- server stderr tail ---'
    Get-Content -LiteralPath $err -Tail 40 -ErrorAction SilentlyContinue
    exit 1
}

# The model id has to match the server's own alias or the request answers 404
# model_not_found. Ask the server rather than guessing.
$modelId = 'local'
try {
    $models = Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 10
    if ($models.data -and @($models.data).Count -gt 0) { $modelId = @($models.data)[0].id }
    Write-Output ("model id   = " + $modelId)
} catch {
    Write-Output ("WARN: /v1/models failed: " + $_.Exception.Message)
}
try {
    $load = Invoke-RestMethod -Uri "$base/v1/load" -TimeoutSec 10
    Write-Output ("load       = " + ($load | ConvertTo-Json -Compress -Depth 6))
} catch {
    Write-Output ("WARN: /v1/load failed: " + $_.Exception.Message)
}

$body = @{
    model      = $modelId
    messages   = @(@{ role = 'user'; content = $Prompt })
    max_tokens = $MaxTokens
    stream     = $false
} | ConvertTo-Json -Depth 5

$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $r = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post `
        -ContentType 'application/json' -Body $body -TimeoutSec $GenTimeoutSec
    $sw.Stop()
    Write-Output ("generated  = {0:N1}s" -f $sw.Elapsed.TotalSeconds)
    Write-Output ("answer     = " + $r.choices[0].message.content)
    Write-Output ("usage      = " + ($r.usage | ConvertTo-Json -Compress))
    Write-Output 'RESULT: OK'
} catch {
    $sw.Stop()
    Write-Output ("REQUEST FAILED after {0:N1}s: {1}" -f $sw.Elapsed.TotalSeconds, $_.Exception.Message)
    Write-Output 'RESULT: REQUEST-FAILED'
}

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
Start-Sleep -Seconds 2
Write-Output '--- server stderr tail (last 40) ---'
Get-Content -LiteralPath $err -Tail 40 -ErrorAction SilentlyContinue
