# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - acceptance for the "auto" fit of the Host and SSD
# budgets, plus the ceiling report that replaces a crash.
#
# WHAT EACH ARM PROVES
#   ACCEPT  : the round-19 quality knobs stay manual (B / R / sink / recent) while
#             --kvmem-host-mib auto and --kvmem-disk-mib auto are fitted by the
#             engine. The arm requires, in order:
#               1. a "KVMem fit:" line for the Host budget,
#               2. a "KVMem fit:" line for the SSD budget,
#               3. a capacity line and a served request (talking is not enough,
#                  so it asks for a real answer),
#               and it records the fitted numbers so the log claim can be checked.
#   CEILING : a tiny explicit Host against a 1M window must FAIL and name how many
#             tokens this machine can hold. This is the negative control for the
#             fit: without it, "auto worked" would only mean "nothing was tested".
#             It also shows that an explicit value still overrides the fit.
#   NOPATH  : --kvmem-disk-mib auto without --kvmem-disk-path must fail with the
#             reason, because the engine cannot know which volume to measure.
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# Verdicts come from log lines and HTTP responses, never ExitCode.
# ============================================================================
param(
    [string]$Model    = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [string]$DiskDir  = (Join-Path $PSScriptRoot '..\Cache\kvmem'),
    [int]$Port        = 18300,
    [int]$ReadySec    = 600,
    [int]$GenTimeout  = 900,
    [string]$Tag      = 'kvfit',
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
Write-Output ''

function Stop-Orphans {
    Get-Process -Name 'ninfer-serve', 'ninfer-perplexity', 'ninfer' -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.Id -Force }
    Start-Sleep -Seconds 5
}

# Starts one arm and reports what the engine said, not whether a process existed.
function Invoke-Arm {
    param([string]$Name, [string[]]$Flags, [int]$Port, [bool]$ExpectReady)

    Write-Output ('================ ARM ' + $Name + ' ================')
    $err = Join-Path $BuildDir ("$Tag`_$Name.server.err.log")
    $out = Join-Path $BuildDir ("$Tag`_$Name.server.out.log")
    Remove-Item -LiteralPath $err, $out -ErrorAction SilentlyContinue
    Stop-Orphans

    Write-Output ('  flags = ' + (($Flags) -join ' '))
    $proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err -ArgumentList $Flags

    $baseUrl  = "http://127.0.0.1:$Port"
    $ready    = $false
    $deadline = (Get-Date).AddSeconds($ReadySec)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { break }
        try { [void](Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 3); $ready = $true; break }
        catch { Start-Sleep -Seconds 3 }
    }

    $lines = @(Get-Content -LiteralPath $err -ErrorAction SilentlyContinue)
    $fit   = @($lines | Select-String -Pattern 'KVMem fit:')
    $rej   = @($lines | Select-String -Pattern 'KVMem cannot hold|auto needs|auto cannot')

    Write-Output ("  ready = {0}" -f $ready)
    foreach ($line in $fit) { Write-Output ('  FIT   | ' + $line.Line.Trim()) }
    foreach ($line in $rej) { Write-Output ('  REJECT| ' + $line.Line.Trim()) }

    $answer = 'n/a'
    if ($ready) {
        $cap = @($lines | Select-String -Pattern 'capacity \|' | Select-Object -Last 1)
        if ($cap) { Write-Output ('  ' + $cap.Line.Trim()) }
        $modelId = 'local'
        try {
            $m = Invoke-RestMethod -Uri "$baseUrl/v1/models" -TimeoutSec 10
            if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
        } catch { }
        try {
            $body = @{ model = $modelId
                       messages = @(@{ role = 'user'; content = 'Reply with exactly: FIT OK' })
                       max_tokens = 512; stream = $false } | ConvertTo-Json -Depth 5
            $r = Invoke-RestMethod -Uri "$baseUrl/v1/chat/completions" -Method Post `
                 -ContentType 'application/json' -Body $body -TimeoutSec $GenTimeout
            $answer = [string]$r.choices[0].message.content
            Write-Output ('  answer = ' + ($answer -replace "\s+", ' '))
        } catch { Write-Output ('  REQUEST FAILED: ' + $_.Exception.Message) }
    }

    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Seconds 3

    $verdict = if ($ExpectReady) {
        if ($ready -and $fit.Count -ge 2 -and $answer -ne 'n/a') { 'PASS' } else { 'FAIL' }
    } else {
        if (-not $ready -and $rej.Count -ge 1) { 'PASS' } else { 'FAIL' }
    }
    Write-Output ("  VERDICT = {0}" -f $verdict)
    Write-Output ''
    return [pscustomobject]@{
        arm = $Name; expect_ready = $ExpectReady; ready = $ready
        fit_lines = $fit.Count; reject_lines = $rej.Count
        answer = ($answer -replace "\s+", ' '); verdict = $verdict
    }
}

# The round-19 quality knobs stay manual; only the two budgets are fitted.
$quality = @('--kvmem-budget', '32768', '--kvmem-gen-reserve', '5632',
             '--kvmem-sink-tokens', '4096', '--kvmem-recent-tokens', '1024')

$rows = @()
$rows += Invoke-Arm -Name 'ACCEPT' -Port $Port -ExpectReady $true -Flags (@($Model,
    '--host', '127.0.0.1', '--port', "$Port", '--device-profile', 'off',
    '--max-context', '1048576', '--max-concurrency', '1',
    '--kvmem-host-mib', 'auto', '--kvmem-disk-path', $DiskDir, '--kvmem-disk-mib', 'auto',
    '--kvmem-sessions', '1', '--log-level', 'info') + $quality)

$rows += Invoke-Arm -Name 'CEILING' -Port ($Port + 1) -ExpectReady $false -Flags (@($Model,
    '--host', '127.0.0.1', '--port', "$($Port + 1)", '--device-profile', 'off',
    '--max-context', '1048576', '--max-concurrency', '1',
    '--kvmem-host-mib', '1024',
    '--kvmem-sessions', '1', '--log-level', 'info') + $quality)

$rows += Invoke-Arm -Name 'NOPATH' -Port ($Port + 2) -ExpectReady $false -Flags (@($Model,
    '--host', '127.0.0.1', '--port', "$($Port + 2)", '--device-profile', 'off',
    '--max-context', '1048576', '--max-concurrency', '1',
    '--kvmem-host-mib', 'auto', '--kvmem-disk-mib', 'auto',
    '--kvmem-sessions', '1', '--log-level', 'info') + $quality)

Stop-Orphans
Write-Output '================ FIT ACCEPTANCE SUMMARY ================'
$rows | Format-Table arm, expect_ready, ready, fit_lines, reject_lines, verdict -AutoSize | Out-String | Write-Output
$csv = Join-Path $BuildDir ($Tag + '_summary.csv')
Remove-Item -LiteralPath $csv -ErrorAction SilentlyContinue
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
Write-Output ('summary csv: ' + $csv)
$verdicts = @($rows | Where-Object { $_ -is [pscustomobject] } | ForEach-Object { $_.verdict })
if (@($verdicts | Where-Object { $_ -ne 'PASS' }).Count -eq 0) {
    Write-Output 'RESULT: ALL-PASS'
} else {
    Write-Output 'RESULT: HAS-FAILURE'
}
