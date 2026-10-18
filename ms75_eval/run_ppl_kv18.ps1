# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - perplexity judgement, LIKE-FOR-LIKE with the
# accepted baseline.
#
# WHY THESE EXACT ARGUMENTS (do not "improve" them):
#   The reference runs used this same artifact and this same argument set and
#   produced:
#     261,167 tokens | 124 windows | PPL 5.62563..5.62685 | [ALIGN] lines 0
#   The engine's own run-to-run spread is about +-0.0006, so the bar for this
#   stack is: 124/124 windows, PPL inside 5.626 +- 0.001, zero [ALIGN] lines.
#   Changing the artifact or the context/stride makes the numbers incomparable.
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# The verdict comes from LOG CONTENT, never from Start-Process.ExitCode: under
# PS 5.1 with -NoNewWindow + redirection that property can come back empty and
# misreport a pass as a failure.
# Two log channels on purpose: stdout carries the device [ALIGN]
# self-report and the result table, stderr the progress and CUDA_CHECK failures.
# --stride must stay < --context. --output must be explicit, or the
# engine writes its report into its default profiles directory.
#
# The device profile is NOT installed by default. The shipped
# sm75-empty-routes.profile.json is schema v1 while this build reads only schema
# v2, so forcing it through NINFER_DEVICE_PROFILES makes the engine ignore the
# file, recalibrate, and REPLACE it inside the source tree. Calibration is safe
# on this card (67 routes in 31 s), so the engine calibrates itself and writes
# nothing into the repository unless -DeviceProfile points at a v2 file the
# caller owns.
# ============================================================================
param([int]$Runs = 8, [string]$Tag = 'ppl_kv18', [string]$KvDtype = '', [switch]$GdnFp16,
      # Judgement artifact. Default stays the accepted path; -Model lets a run
      # read a faster copy (a spinning HDD's file cache is unstable, which stalls
      # repeated loads).
      [string]$Model = (Join-Path $PSScriptRoot '..\model\bonsai-mtponly.ninfer'),
      [string]$DeviceProfile = '')

$ErrorActionPreference = 'Stop'
$R       = (Split-Path -Parent $PSScriptRoot)
$exe     = Join-Path $R 'kvmem_build\apps\ninfer-perplexity.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    $packExe = Join-Path $R 'engine\ninfer-perplexity.exe'
    if (Test-Path -LiteralPath $packExe) { $exe = $packExe }
}
$model   = $Model
$corpus  = Join-Path $R 'kvmem_src\eval\corpora\perplexity-1m\manifest.json'
if (-not (Test-Path -LiteralPath $corpus)) {
    $packCorpus = Join-Path $R 'src\eval\corpora\perplexity-1m\manifest.json'
    if (Test-Path -LiteralPath $packCorpus) { $corpus = $packCorpus }
}

foreach ($p in @($exe, $model, $corpus)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "MISSING INPUT: $p" }
}
if ($DeviceProfile -ne '' -and -not (Test-Path -LiteralPath $DeviceProfile)) {
    throw "MISSING INPUT: $DeviceProfile"
}

# Anchor the binary by hash: the "build <id>" string in the log comes from a
# cmake stamp target that does NOT refresh on every rebuild, so it cannot say
# what actually ran.
Write-Output ("exe sha256 = {0}" -f (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)
Write-Output ("exe mtime  = {0}" -f (Get-Item -LiteralPath $exe).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
Write-Output ("model      = {0}" -f $model)
Write-Output ("profile    = {0}" -f $(if ($DeviceProfile -eq '') { '(none; the engine calibrates itself)' } else { $DeviceProfile }))

# Orphaned workers from a previous arm hold several GB of VRAM and would change
# the memory picture of this arm.
Get-Process -Name 'ninfer-perplexity', 'ninfer-serve', 'ninfer' -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Output ('killing orphan: ' + $_.Name + ' pid=' + $_.Id); Stop-Process -Id $_.Id -Force }
Start-Sleep -Seconds 2

# The hand-authored profile is schema v1 and this build reads only schema v2, so
# forcing it made the engine ignore the file, recalibrate, and REPLACE it inside
# the source tree (+734 lines, measured 2026-10-15). Calibration is safe on this
# card now (67 routes in 31 s), so the profile is opt-in: without it the engine
# calibrates and writes nothing into the repository.
if ($DeviceProfile -ne '') { $env:NINFER_DEVICE_PROFILES = $DeviceProfile }
$env:NINFER_CUDA_SYNC       = 'auto'

# 输出目录：构建机写 kvmem_build\；只拿发布包时写 Cache\judge\
$workDir = Join-Path $R 'kvmem_build'
if (-not (Test-Path -LiteralPath $workDir)) {
    $workDir = Join-Path $R 'Cache\judge'
    New-Item -ItemType Directory -Path $workDir -Force | Out-Null
}
$pass = 0
for ($i = 1; $i -le $Runs; $i++) {
    $name = '{0}_{1}' -f $Tag, $i
    $out  = Join-Path $workDir ($name + '.out.log')
    $err  = Join-Path $workDir ($name + '.err.log')
    $od   = Join-Path $workDir $name
    if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force }
    if (Test-Path -LiteralPath $err) { Remove-Item -LiteralPath $err -Force }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    # -KvDtype / -GdnFp16 are the ONLY variables this matrix adds: the artifact, context,
    # stride, corpus and output stay byte-identical to the accepted baseline, so a PPL
    # difference is a KV-format difference and nothing else.
    $childArgs = @($model, '--corpus', $corpus, '--quick', '--context', '4096',
                   '--stride', '2048', '--output', $od, '--log-level', 'info')
    if ($KvDtype -ne '') { $childArgs += @('--kv-dtype', $KvDtype) }
    if ($GdnFp16) { $childArgs += '--gdn-state-fp16' }
    $proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err `
        -ArgumentList $childArgs
    $proc.WaitForExit()
    $sw.Stop()

    $errLines = @(Get-Content -LiteralPath $err -ErrorAction SilentlyContinue)
    $outLines = @(Get-Content -LiteralPath $out -ErrorAction SilentlyContinue)

    $bad  = $errLines | Select-String -Pattern 'misaligned|CUDA_CHECK.*failed|failed:' | Select-Object -Last 1
    $good = $errLines | Select-String -Pattern 'scoring complete' | Select-Object -Last 1
    $tbl  = @(@($outLines) + @($errLines) | Select-String -Pattern 'PPL|perplexity')

    if ($good) { $ok = 'PASS'; $pass++ } else { $ok = 'FAIL' }

    $align = @(@($outLines) + @($errLines) | Select-String -Pattern '\[ALIGN\]')

    $detail = 'no verdict'
    if ($bad) { $detail = $bad.Line.Trim() } elseif ($good) { $detail = $good.Line.Trim() }
    Write-Output ('{0} | {1} | {2:N0}s | {3}' -f $name, $ok, $sw.Elapsed.TotalSeconds, $detail)
    Write-Output ('    [ALIGN] report lines: {0}' -f $align.Count)
    foreach ($t in ($tbl | Select-Object -Last 3)) { Write-Output ('    ' + $t.Line.Trim()) }

    if ($align.Count -gt 0) {
        $grouped = $align | ForEach-Object {
            if ($_.Line -match '\[ALIGN\]\s+(\S+)\s+need=(\d+)\s+smod=(\d+)\s+src=(\S+)\s+srcmod=(\d+)') {
                '{0} need={1} smod={2} srcmod={3}' -f $Matches[1], $Matches[2], $Matches[3], $Matches[5]
            } else { $_.Line.Trim() }
        } | Group-Object | Sort-Object Count -Descending
        foreach ($g in ($grouped | Select-Object -First 25)) {
            Write-Output ('      {0,8}x  {1}' -f $g.Count, $g.Name)
        }
    }
}

Write-Output ('==== pass {0}/{1} ====' -f $pass, $Runs)
if ($pass -eq $Runs) { Write-Output 'RESULT: ALL-PASS' } else { Write-Output 'RESULT: HAS-FAILURE' }
