# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# Parameter matrix for the release.
#
# WHY THIS EXISTS
#   The default start script leaves several switches off: ActKV pillar 2
#   (confidence budget) and pillar 3 (retrieval scorer) are OFF, LRFU is OFF,
#   adaptive MTP is OFF, vision is OFF, and the MTP draft width is a fixed
#   default. Each candidate is therefore measured rather than assumed.
#
# WHAT ONE ARM MEASURES (one server start, one judgement set)
#   short decode tok/s + MTP acceptance   (speed on easy text)
#   a 40,960-token needle prompt          (accuracy: recall past the B=32768
#                                          window, TTFT, prefill, decode, cache)
#   the engine's own capacity/ledger lines (does the variant even fit?)
#
# WHY SERIAL: GPU judgements must never overlap. One arm at a
# time, orphans stopped between arms.
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# ============================================================================
param(
    [string]$Model     = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [int]$B            = 32768,
    [int]$R            = 5632,
    [int]$Ctx          = 1048576,
    [int]$HostMiB      = 64421,
    [int]$Sessions     = 4,
    [string]$DiskPath  = (Join-Path $PSScriptRoot '..\build\kvmem_disk_matrix'),
    [int]$DiskMiB      = 102400,
    [int]$NeedleTokens = 40960,
    [int]$ReadySec     = 900,
    [int]$GenTimeout   = 1800,
    [string]$Tag       = 'param_matrix',
    [string]$BuildDir  = (Join-Path $PSScriptRoot '..\build')
)

$ErrorActionPreference = 'Continue'
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
$out = Join-Path $BuildDir 'matrix_out'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$csv       = Join-Path $out ($Tag + '_summary.csv')
$summaryTx = Join-Path $out ($Tag + '_summary.txt')
if (Test-Path -LiteralPath $csv) { Remove-Item -LiteralPath $csv -Force }
if (Test-Path -LiteralPath $summaryTx) { Remove-Item -LiteralPath $summaryTx -Force }
if (-not (Test-Path -LiteralPath $DiskPath)) { New-Item -ItemType Directory -Path $DiskPath -Force | Out-Null }

Write-Output ("exe sha256 = {0}" -f (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)
Write-Output ("exe mtime  = {0}" -f (Get-Item -LiteralPath $exe).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
Write-Output ("matrix     = {0} arms, needle {1} tokens, window B={2} R={3}" -f 11, $NeedleTokens, $B, $R)
Write-Output ''

# Candidate variants. Every one changes ONLY the named flags; the production
# recipe (int8 + MTP + KVMem long profile + sink/recent) stays identical.
$arms = @(
    @{ name = 'base_d2';        extra = @('--draft-tokens','2') },
    @{ name = 'base_d4';        extra = @('--draft-tokens','4') },
    @{ name = 'adaptive_d4';    extra = @('--draft-tokens','4','--adaptive-mtp') },
    @{ name = 'pertoken';       extra = @('--draft-tokens','2','--kvmem-retrieval-method','per-token') },
    @{ name = 'subblock4';      extra = @('--draft-tokens','2','--kvmem-retrieval-method','sub-block','--kvmem-subblocks','4','--kvmem-subblock-reduce','max') },
    @{ name = 'conf_budget';    extra = @('--draft-tokens','2','--kvmem-conf-budget','--kvmem-conf-init','16384','--kvmem-conf-upper','32768') },
    @{ name = 'lrfu';           extra = @('--draft-tokens','2','--kvmem-lrfu') },
    @{ name = 'ngram_off';      extra = @('--draft-tokens','2','--ngram-draft-tokens','0') },
    @{ name = 'fast_prefill';   extra = @('--draft-tokens','2','--fast-prefill-kernel') },
    @{ name = 'vision_cpu';     extra = @('--draft-tokens','2','--vision','--vision-residency','cpu','--vision-max-merged','256') },
    @{ name = 'gdn_fp16';       extra = @('--draft-tokens','2','--gdn-state-fp16') }
)

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

function Get-DoneLines {
    param([string]$LogPath)
    return @(Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue |
             Select-String -Pattern 'req#\d+ done')
}

function Parse-Done {
    param([string]$line)
    $r = @{ prompt = 0; output = 0; cache = 0.0; ttft = ''; prefill = ''; decode = ''; mtp = '' }
    if (-not $line) { return $r }
    if ($line -match 'prompt ([\d,]+)')                    { $r.prompt = [int]($Matches[1] -replace ',','') }
    if ($line -match '\| output (\d+)')                    { $r.output = [int]$Matches[1] }
    if ($line -match 'cache [\d,]+ \(([\d.]+)%')           { $r.cache = [double]$Matches[1] }
    if ($line -match 'TTFT ([0-9a-z. ]+?)\s*\|')           { $r.ttft = $Matches[1].Trim() }
    if ($line -match 'prefill ([\d.]+) tok/s')             { $r.prefill = $Matches[1] }
    if ($line -match 'decode ([\d.]+) tok/s')              { $r.decode = $Matches[1] }
    if ($line -match 'mtp accepted (\d+)/(\d+) \(([\d.]+)%\)') { $r.mtp = ($Matches[1] + '/' + $Matches[2] + ' ' + $Matches[3] + '%') }
    return $r
}

$rows = @()
foreach ($arm in $arms) {
    $name = $arm.name
    Write-Output ('================ ARM {0} ================' -f $name)
    Stop-Orphans

    $errLog = Join-Path $out ("{0}_{1}.err.log" -f $Tag, $name)
    $outLog = Join-Path $out ("{0}_{1}.out.log" -f $Tag, $name)
    Remove-Item -LiteralPath $errLog, $outLog -ErrorAction SilentlyContinue

    $argList = @($Model, '--host', '127.0.0.1', '--port', '18100', '--device-profile', 'off',
                 '--max-context', "$Ctx", '--max-concurrency', '1',
                 '--kv-dtype', 'int8',
                 '--kvmem-budget', "$B", '--kvmem-gen-reserve', "$R",
                 '--kvmem-host-mib', "$HostMiB", '--kvmem-sessions', "$Sessions",
                 '--kvmem-sink-tokens', '4096', '--kvmem-recent-tokens', '1024',
                 '--kvmem-disk-path', "$DiskPath", '--kvmem-disk-mib', "$DiskMiB",
                 '--spec', 'mtp', '--log-level', 'info')
    $argList += $arm.extra
    Write-Output ("flags = " + ($argList -join ' '))

    $row = [pscustomobject]@{
        arm = $name; ready_s = 'n/a'; capacity = 'n/a'; note = ''
        short_decode = 'n/a'; short_mtp = 'n/a'
        needle_hit = 'n/a'; needle_ttft = 'n/a'; needle_prefill = 'n/a'; needle_decode = 'n/a'
    }

    $t0 = Get-Date
    $proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog -ArgumentList $argList
    $baseUrl = 'http://127.0.0.1:18100'
    $ready = $false
    $deadline = (Get-Date).AddSeconds($ReadySec)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { break }
        try { [void](Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 3); $ready = $true; break }
        catch { Start-Sleep -Seconds 4 }
    }
    $row.ready_s = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    if (-not $ready) {
        $row.note = 'NOT-READY'
        $row.capacity = @(Get-Content -LiteralPath $errLog -ErrorAction SilentlyContinue |
                          Select-String -Pattern 'KVMem rejected|capacity \|' | Select-Object -Last 1).Line
        Write-Output ('  NOT-READY after {0}s : {1}' -f $row.ready_s, $row.capacity)
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
        $rows += $row
        Start-Sleep -Seconds 3
        continue
    }
    $row.capacity = @(Get-Content -LiteralPath $errLog -ErrorAction SilentlyContinue |
                      Select-String -Pattern 'capacity \|' | Select-Object -Last 1).Line
    Write-Output ('  ready {0}s | {1}' -f $row.ready_s, $row.capacity)

    $modelId = 'local'
    try {
        $m = Invoke-RestMethod -Uri "$baseUrl/v1/models" -TimeoutSec 10
        if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
    } catch { }

    # --- 1) short decode (speed on easy text) --------------------------------
    try {
        $before = (Get-DoneLines -LogPath $errLog).Count
        $body = @{ model = $modelId; messages = @(@{ role = 'user'; content = 'Count from 1 to 60, one number per line.' })
                   max_tokens = 256; stream = $false; temperature = 0.0 } | ConvertTo-Json -Depth 6
        [void](Invoke-RestMethod -Uri "$baseUrl/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec $GenTimeout)
        $dl = (Get-Date).AddSeconds(60)
        while ((Get-Date) -lt $dl) { $l = Get-DoneLines -LogPath $errLog; if ($l.Count -gt $before) { break }; Start-Sleep -Milliseconds 400 }
        $d = Parse-Done $l[-1].Line
        $row.short_decode = $d.decode; $row.short_mtp = $d.mtp
        Write-Output ('  short decode {0} tok/s | mtp {1}' -f $d.decode, $d.mtp)
    } catch { $row.note += 'shortFail;'; Write-Output ('  short decode FAILED: ' + $_.Exception.Message) }

    # --- 2) 40K needle: accuracy past the window + long-context speed ---------
    try {
        $before = (Get-DoneLines -LogPath $errLog).Count
        $needle = 'ZQ' + $name.ToUpper() + 'X40960'
        $prompt = Build-Needle -TargetTokens $NeedleTokens -Needle $needle -Salt 7
        $body = @{ model = $modelId; messages = @(@{ role = 'user'; content = $prompt })
                   max_tokens = 256; stream = $false; temperature = 0.0 } | ConvertTo-Json -Depth 6
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        # Invoke-RestMethod throws an internal Int32 conversion on some hosts for
        # the chat.completion payload; read the raw body and parse it as a hashtable.
        $resp = Invoke-WebRequest -Uri "$baseUrl/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec $GenTimeout
        $sw.Stop()
        $h = $resp.Content | ConvertFrom-Json -AsHashtable
        $ch = @($h['choices'])
        $answer = if ($ch.Count -gt 0 -and $null -ne $ch[0]['message']) { [string]$ch[0]['message']['content'] } else { '' }
        $dl = (Get-Date).AddSeconds(60)
        while ((Get-Date) -lt $dl) { $l = Get-DoneLines -LogPath $errLog; if ($l.Count -gt $before) { break }; Start-Sleep -Milliseconds 400 }
        $d = Parse-Done $l[-1].Line
        $hit = if ($answer -match [regex]::Escape($needle)) { 'HIT' } else { 'MISS' }
        $row.needle_hit = $hit; $row.needle_ttft = $d.ttft
        $row.needle_prefill = $d.prefill; $row.needle_decode = $d.decode
        Write-Output ('  40K needle {0} | TTFT {1} | prefill {2} | decode {3} | prompt {4} | cache {5}%' -f `
            $hit, $d.ttft, $d.prefill, $d.decode, $d.prompt, $d.cache)
        Write-Output ('  answer: [' + (($answer -replace "\s+",' ')) + ']')
    } catch { $row.note += 'needleFail;'; Write-Output ('  40K needle FAILED: ' + $_.Exception.Message) }

    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Seconds 3
    $rows += $row
    $rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
    Write-Output ''
}

Stop-Orphans
Write-Output '================ PARAM MATRIX SUMMARY ================'
$rows | Format-Table arm, ready_s, short_decode, short_mtp, needle_hit, needle_ttft, needle_prefill, note -AutoSize | Out-String | Write-Output
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
$rows | Out-String | Set-Content -LiteralPath $summaryTx -Encoding UTF8
Write-Output ('summary csv = ' + $csv)
Write-Output 'RESULT: PARAM-MATRIX-DONE'
