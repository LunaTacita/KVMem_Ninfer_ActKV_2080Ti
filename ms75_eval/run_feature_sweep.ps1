# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - feature sweep: concurrency, sessions, LRFU,
# transfer verification, SSD tier, MTP and DFlash2.
#
# One arm per server start, because each arm needs different flags. Every arm
# reports what the engine's own log and /v1/load say, never an exit code.
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# The disk path is pure ASCII on purpose: a non-ASCII profile/output path has
# already broken tooling on this host.
# ============================================================================
param(
    [string]$Model      = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [int]$Ctx           = 32768,
    [int]$KvmemBudget   = 4096,
    [int]$KvmemReserve  = 1536,
    [int]$HostMiB       = 8192,
    [int]$DiskMiB       = 4096,
    [string]$DiskPath   = (Join-Path $PSScriptRoot '..\build\kvmem_disk_tier'),
    [int]$ReadySec      = 600,
    [int]$GenTimeoutSec = 600,
    [string]$BuildDir   = (Join-Path $PSScriptRoot '..\build')
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
if (-not (Test-Path -LiteralPath $DiskPath)) { New-Item -ItemType Directory -Path $DiskPath | Out-Null }

$baseKvmem = @('--kvmem-budget', "$KvmemBudget", '--kvmem-gen-reserve', "$KvmemReserve",
               '--kvmem-host-mib', "$HostMiB")

function Get-LastMatching {
    param([string]$LogPath, [string]$Pattern)
    $line = Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue |
            Select-String -Pattern $Pattern | Select-Object -Last 1
    if ($line) { return $line.Line.Trim() }
    return '(none)'
}

function Invoke-Arm {
    param([string]$Name, [string[]]$Extra, [int]$Port)

    Write-Output ''
    Write-Output ('================ ARM ' + $Name + ' ================')
    $err = Join-Path $BuildDir ("feat_$Name.server.err.log")
    $out = Join-Path $BuildDir ("feat_$Name.server.out.log")

    Get-Process -Name 'ninfer-serve', 'ninfer-perplexity', 'ninfer' -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.Id -Force }
    Start-Sleep -Seconds 2

    $argList = @($Model, '--host', '127.0.0.1', '--port', "$Port", '--device-profile', 'off',
                 '--max-context', "$Ctx", '--log-level', 'info') + $baseKvmem + $Extra
    Write-Output ("flags = " + ($argList -join ' '))

    $proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err -ArgumentList $argList

    $base     = "http://127.0.0.1:$Port"
    $ready    = $false
    $deadline = (Get-Date).AddSeconds($ReadySec)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { Write-Output 'EXITED-BEFORE-READY'; break }
        try { $null = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 3; $ready = $true; break }
        catch { Start-Sleep -Seconds 4 }
    }
    if (-not $ready) {
        Write-Output 'RESULT: NOT-READY'
        Write-Output ('first line: ' + (Get-Content -LiteralPath $err -TotalCount 1 -ErrorAction SilentlyContinue))
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
        return
    }

    $modelId = 'local'
    try {
        $m = Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 10
        if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
    } catch { }
    Write-Output ("model id = " + $modelId)

    $body = @{ model = $modelId
               messages = @(@{ role = 'user'; content = 'Reply with exactly: FEATURE OK' })
               max_tokens = 64; stream = $false } | ConvertTo-Json -Depth 5

    # Two concurrent requests (arm A judges concurrency; harmless for the others).
    $jobs = @()
    foreach ($n in 1..2) {
        $jobs += Start-Job -ScriptBlock {
            param($u, $b, $t)
            try {
                $r = Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json' -Body $b -TimeoutSec $t
                return @{ ok = $true; text = $r.choices[0].message.content; usage = ($r.usage | ConvertTo-Json -Compress) }
            } catch { return @{ ok = $false; text = $_.Exception.Message; usage = '' } }
        } -ArgumentList "$base/v1/chat/completions", $body, $GenTimeoutSec
    }
    $res = $jobs | Wait-Job | Receive-Job
    $jobs | Remove-Job -Force
    $idx = 0
    foreach ($r in $res) {
        $idx++
        Write-Output ("  concurrent req {0}: ok={1} | {2}" -f $idx, $r.ok, $r.text)
    }

    # A follow-up request: the session reuse shows up as a cache percentage in the log.
    try {
        $body2 = @{ model = $modelId
                    messages = @(@{ role = 'user'; content = 'Reply with exactly: FEATURE OK' },
                                 @{ role = 'assistant'; content = 'FEATURE OK' },
                                 @{ role = 'user'; content = 'Reply with exactly: SESSION OK' })
                    max_tokens = 64; stream = $false } | ConvertTo-Json -Depth 5
        $r2 = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post `
            -ContentType 'application/json' -Body $body2 -TimeoutSec $GenTimeoutSec
        Write-Output ("  follow-up req  : " + $r2.choices[0].message.content)
    } catch { Write-Output ("  follow-up req FAILED: " + $_.Exception.Message) }

    Write-Output '  --- req done lines ---'
    Get-Content -LiteralPath $err -ErrorAction SilentlyContinue |
        Select-String -Pattern 'req#\d+ done' | ForEach-Object { '  ' + $_.Line.Trim() }
    Write-Output '  --- capacity line ---'
    Write-Output ('  ' + (Get-LastMatching -LogPath $err -Pattern 'capacity \|'))
    Write-Output '  --- kvmem / disk / history counters ---'
    try {
        $load = Invoke-RestMethod -Uri "$base/v1/load" -TimeoutSec 10
        if ($load.kvmem) { Write-Output ('  ' + ($load.kvmem | ConvertTo-Json -Compress)) }
        Write-Output ('  capacity=' + ($load.capacity | ConvertTo-Json -Compress))
    } catch { Write-Output ('  WARN /v1/load: ' + $_.Exception.Message) }
    Write-Output '  --- warnings / errors ---'
    $w = Get-Content -LiteralPath $err -ErrorAction SilentlyContinue |
         Select-String -Pattern 'WARN|ERROR|FATAL|failed'
    if ($w) { $w | Select-Object -Last 8 | ForEach-Object { '  ' + $_.Line.Trim() } } else { Write-Output '  (none)' }

    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Seconds 2
}

Invoke-Arm -Name 'A_concurrency_sessions' -Port 18091 -Extra @('--max-concurrency', '2', '--kvmem-sessions', '4')
Invoke-Arm -Name 'B_lrfu_verify_ssd'      -Port 18092 -Extra @('--kvmem-lrfu', '--kvmem-verify-transfers',
                                                             '--kvmem-disk-path', $DiskPath, '--kvmem-disk-mib', "$DiskMiB")
Invoke-Arm -Name 'C_mtp'                  -Port 18093 -Extra @('--spec', 'mtp', '--draft-tokens', '3')
Invoke-Arm -Name 'D_dflash2'              -Port 18094 -Extra @('--spec', 'dflash2', '--draft-tokens', '3')

Write-Output ''
Write-Output 'RESULT: FEATURE-SWEEP-DONE'
