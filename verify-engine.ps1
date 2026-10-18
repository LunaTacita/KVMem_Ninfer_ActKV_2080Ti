# Self-check for this engine pack: verifies every file against SHA256SUMS.txt
# and checks the required executables and ten runtime DLLs are present.
# ASCII only. Exit code 0 = PASS.
$here = $PSScriptRoot
$sums = Join-Path $here 'SHA256SUMS.txt'
if (-not (Test-Path -LiteralPath $sums)) { Write-Output 'FAIL: SHA256SUMS.txt missing'; exit 1 }
$bad = 0; $missing = 0
foreach ($line in Get-Content -LiteralPath $sums) {
    if ($line -notmatch '^([0-9A-Fa-f]{64})\s+\*?(.+)$') { continue }
    $hash = $Matches[1].ToUpper(); $rel = $Matches[2].Trim()
    $p = Join-Path $here $rel
    if (-not (Test-Path -LiteralPath $p)) { Write-Output ("MISSING " + $rel); $missing++; continue }
    $c = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash
    if ($c -ne $hash) { Write-Output ("BAD     " + $rel); $bad++ }
}
foreach ($exe in @('ninfer-serve.exe')) {
    $p = Join-Path $here ('engine\' + $exe)
    if (Test-Path -LiteralPath $p) { Write-Output ('present: ' + $exe) } else { Write-Output ('MISSING: ' + $exe); $missing++ }
}
foreach ($dll in @('cublas64_12.dll', 'cublasLt64_12.dll', 'cudart64_12.dll', 'avcodec-63.dll', 'avformat-63.dll', 'avutil-61.dll', 'swresample-7.dll', 'swscale-10.dll', 'libcurl-x64.dll', 'libcurl.dll')) {
    $p = Join-Path $here ('engine\' + $dll)
    if (-not (Test-Path -LiteralPath $p)) { Write-Output ('MISSING: ' + $dll); $missing++ }
}
Write-Output ("manifest: bad=" + $bad + " missing=" + $missing)
if ($bad -eq 0 -and $missing -eq 0) { Write-Output 'RESULT: ENGINE-PACK-OK'; exit 0 }
Write-Output 'RESULT: ENGINE-PACK-FAIL'; exit 1