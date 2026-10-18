# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - recall sweep with diagnostics, and a probe for the
# largest usable B (at B=20480 the engine still reported 1.20 GiB free).
#
# WHY THE BUDGET MATTERS
#   With thinking on, a 32-token max_tokens is consumed entirely by reasoning and
#   message.content comes back empty - it says nothing about recall. So this
#   script:
#     * asks for 512 tokens,
#     * prints finish_reason and the full usage object (reasoning tokens included),
#     * runs a SHORT (4K) needle first as a control: if the short one hits and the
#       long one misses, the miss is a real long-context result; if both miss, the
#       judgement itself is still wrong and the numbers must not be used.
#
# WHY BIGGER B IS STILL ON THE TABLE
#   At B=20480 the engine reported 344/344 pages, runtime 1.67 GiB and free
#   1.20 GiB, so the pool has room left. Decode did not move across the whole
#   sweep; only prefill did, and it flattened above B=16384. Whether a larger
#   visible set actually improves recall is exactly what this script measures.
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# Verdicts come from HTTP responses and log lines, never from ExitCode.
# ============================================================================
param(
    [int[]]$Budgets   = @(4096, 12288, 20480, 28672, 36864),
    [int]$Reserve     = 1536,
    [int]$Ctx         = 262144,
    [int]$HostMiB     = 16384,
    [int]$Sessions    = 4,
    [int]$Port        = 18094,
    [string]$Model    = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [int]$ShortTokens = 4096,
    [int]$LongTokens  = 32768,
    [int]$ReadySec    = 900,
    [int]$GenTimeout  = 2400,
    [string]$Tag      = 'recall',
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
Write-Output ("B sweep    = {0}   R = {1}   needles at {2} and {3} tokens" -f ($Budgets -join ','), $Reserve, $ShortTokens, $LongTokens)
Write-Output ''

function Stop-Orphans {
    Get-Process -Name 'ninfer-serve', 'ninfer-perplexity', 'ninfer' -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.Id -Force }
    Start-Sleep -Seconds 3
}

function Build-Needle {
    param([int]$TargetTokens, [string]$Needle, [int]$Salt)
    $base = 'Filler about bounded KV windows, host tiers and Turing kernels. '
    $half = [int]($TargetTokens / 2)
    $sb = [System.Text.StringBuilder]::new()
    $i = 0
    $tag = 'n' + $Salt
    while ($sb.Length -lt ($half * 4)) {
        [void]$sb.Append($tag); [void]$sb.Append('a'); [void]$sb.Append($i); [void]$sb.Append(' '); [void]$sb.Append($base); $i++
    }
    [void]$sb.Append(' IMPORTANT: the secret passphrase is '); [void]$sb.Append($Needle); [void]$sb.Append('. ')
    while ($sb.Length -lt ($TargetTokens * 4)) {
        [void]$sb.Append($tag); [void]$sb.Append('b'); [void]$sb.Append($i); [void]$sb.Append(' '); [void]$sb.Append($base); $i++
    }
    [void]$sb.Append(' Question: what is the secret passphrase? Reply with the passphrase only.')
    return $sb.ToString()
}

$rows = @()

foreach ($B in $Budgets) {
    $arm = '{0}_b{1}' -f $Tag, $B
    $err = Join-Path $BuildDir ("$arm.server.err.log")
    $out = Join-Path $BuildDir ("$arm.server.out.log")
    Remove-Item -LiteralPath $err, $out -ErrorAction SilentlyContinue

    Stop-Orphans

    $argList = @($Model, '--host', '127.0.0.1', '--port', "$Port", '--device-profile', 'off',
                 '--max-context', "$Ctx", '--max-concurrency', '1',
                 '--kvmem-budget', "$B", '--kvmem-gen-reserve', "$Reserve",
                 '--kvmem-host-mib', "$HostMiB", '--kvmem-sessions', "$Sessions",
                 '--log-level', 'info')
    if ($KvDtype -ne '') { $argList += @('--kv-dtype', $KvDtype) }
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
        $cap = @(Get-Content -LiteralPath $err -ErrorAction SilentlyContinue) | Select-String -Pattern 'capacity \|' | Select-Object -Last 1
        if ($cap) { Write-Output ('  ' + $cap.Line.Trim()) }
        $reason = @(Get-Content -LiteralPath $err -ErrorAction SilentlyContinue) |
                  Select-String -Pattern 'KVMem rejected|error|Error|FATAL|invalid' | Select-Object -Last 2
        foreach ($l in $reason) { Write-Output ('  ' + $l.Line.Trim()) }
        Write-Output '  NOT-READY (this arm is above the device-pool limit)'
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
        $rows += [pscustomobject]@{ B = $B; pages = 'n/a'; free = 'n/a'; short = 'n/a'; long = 'n/a'; note = 'NOT-READY' }
        continue
    }

    $cap = @(Get-Content -LiteralPath $err -ErrorAction SilentlyContinue) | Select-String -Pattern 'capacity \|' | Select-Object -Last 1
    $capText = if ($cap) { $cap.Line.Trim() } else { '(no capacity line)' }
    Write-Output ('  ' + $capText)
    $pages = 'n/a'; $free = 'n/a'
    if ($capText -match 'pages (\d+)/(\d+)') { $pages = $Matches[1] }
    if ($capText -match 'free ([\d.]+) GiB') { $free = $Matches[1] }

    $modelId = 'local'
    try {
        $m = Invoke-RestMethod -Uri "$baseUrl/v1/models" -TimeoutSec 10
        if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
    } catch { }

    $results = @{}
    $salt = 0
    foreach ($len in @($ShortTokens, $LongTokens)) {
        $salt++
        $needle = ('ZQ' + $B + 'X' + $len)
        try {
            $prompt = Build-Needle -TargetTokens $len -Needle $needle -Salt $salt
            $body = @{ model = $modelId
                       messages = @(@{ role = 'user'; content = $prompt })
                       max_tokens = 512; stream = $false; temperature = 0.0 } | ConvertTo-Json -Depth 5
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $r = Invoke-RestMethod -Uri "$baseUrl/v1/chat/completions" -Method Post `
                 -ContentType 'application/json' -Body $body -TimeoutSec $GenTimeout
            $sw.Stop()
            $answer  = [string]$r.choices[0].message.content
            $finish  = [string]$r.choices[0].finish_reason
            $usage   = $r.usage | ConvertTo-Json -Compress
            $hit = if ($answer -match [regex]::Escape($needle)) { 'HIT' } else { 'MISS' }
            $shown = ($answer -replace "\s+", ' ')
            if ($shown.Length -gt 90) { $shown = $shown.Substring(0, 90) }
            Write-Output ('  needle@{0,6} : {1} | finish={2} | wall={3:N1}s' -f $len, $hit, $finish, $sw.Elapsed.TotalSeconds)
            Write-Output ('      usage: ' + $usage)
            Write-Output ('      answer: [' + $shown + ']')
            $results[$len] = $hit
        } catch {
            Write-Output ('  needle@{0,6} : FAILED {1}' -f $len, $_.Exception.Message)
            $results[$len] = 'ERR'
        }
    }

    $rows += [pscustomobject]@{ B = $B; pages = $pages; free = $free; short = $results[$ShortTokens]; long = $results[$LongTokens]; note = '' }

    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Seconds 3
    Write-Output ''
}

Stop-Orphans
Write-Output '================ RECALL SUMMARY ================'
$rows | Format-Table -AutoSize | Out-String | Write-Output
$csv = Join-Path $BuildDir ($Tag + '_summary.csv')
Remove-Item -LiteralPath $csv -ErrorAction SilentlyContinue
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
Write-Output ('summary csv: ' + $csv)
