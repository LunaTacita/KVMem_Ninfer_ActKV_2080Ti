# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - DEEP cross-page probe (128K haystack + long decode).
#
# WHY 128K AND NOT 4K - what a short prompt fails to prove
#   The device pool holds B+R = 38400 tokens. Any prompt shorter than that is
#   fully resident, so it never exercises retrieval, eviction, host tiering or
#   the page re-selection path at all. A recall HIT on such a prompt proves
#   only that full attention works; it says nothing about KVMem. A 32K recall
#   prompt (35784 < 38400) is all resident, and a 95-token decode prompt
#   touches no mechanism at all. This probe therefore uses a haystack several
#   times the pool:
#     * 128K tokens with N records scattered through it (default every ~10K),
#       then asks for every record back: each record lives in a block that had
#       to be evicted and re-selected from the host tier, so a lost record is a
#       lost page.
#     * the same haystack, then a LONG answer (default 8192 tokens) under a
#       constraint that must hold at BOTH ends of the output: the first line
#       and the last line must carry the SAME nonce, and the body must cite
#       records from the far end of the haystack. A page swap that loses decode
#       state shows up as a mismatched nonce, a wrong citation, or a loop.
#
# POOL-SIZE CEILING ON max_tokens
#   max_tokens must stay <= the pool token count. A 32000-token request against
#   a small pool was observed to crash a worker with no self-heal. The long
#   answer is therefore clamped to (B+R)/2 per lane.
#
# HOST AND DISK BUDGETS - what THIS stack does, read from the source
#   HostKvStorage allocates with `new uint8_t[]` (host_kv_storage.cpp:371, :316),
#   i.e. ORDINARY heap memory, NOT cudaHostAlloc pinned memory, and its
#   constructor defaults the capacity to UINT64_MAX (host_kv_storage.hpp:114).
#   There is therefore no locked-page ceiling in this stack. Do not carry over
#   "31 GiB / 48.5% of RAM" figures from other host layers: NINFER_HOST_PAGEABLE
#   has zero hits in this tree.
#   Measured on this model: host_payload_bytes 813,694,976 for 12,375 tokens is
#   ~66 KiB per token, so 1M tokens is ~66 GiB of native KV - more than the
#   63.9 GB of RAM. That is exactly why the disk tier must be enabled.
#   SpillFile does NOT preallocate: it creates a sparse FILE_FLAG_DELETE_ON_CLOSE
#   handle (spill_io.cpp:234-249, FSCTL_SET_SPARSE) and charges ranges only as
#   they are used, so a large --kvmem-disk-mib costs no physical space until
#   written and the file disappears when the process exits. A DiskKVBridge-style
#   preallocation step is not needed here.
#   Hence: most of the RAM as the host bound, a large sparse disk quota, and a
#   full 1M logical ceiling; the haystack size is what the probe varies.
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# Verdicts come from HTTP responses and the engine log, never ExitCode.
# ============================================================================
param(
    [string]$Model        = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [string]$Profile      = 'long',
    [string]$Mtp          = 'off',
    [int]$TargetTokens    = 131072,
    [int]$Needles         = 12,
    [int]$AskMax          = 1024,
    [int]$LongMax         = 0,
    [int]$Ctx             = 1048576,
    [int]$HostMiB         = 49152,
    [string]$DiskPath     = (Join-Path $PSScriptRoot '..\Cache\kvmem'),
    [int]$DiskMiB         = 1048576,
    [int]$Sessions        = 1,
    [int]$Sink            = 4096,
    [int]$Recent          = 1024,
    [int]$Port            = 18200,
    [int]$ReadySec        = 900,
    [int]$GenTimeout      = 7200,
    [string]$Tag          = 'deep',
    [string]$BuildDir     = (Join-Path $PSScriptRoot '..\build'),
    # --- KV-format x window variables ---------------------------------------
    # -KvDtype picks the KV storage; -B / -R override the profile's device window
    # so "does a bigger visible set actually recall more, and what does it cost"
    # can be measured without editing the table. -GdnFp16 is the upstream Bonsai
    # recipe's recurrent-state trade.
    [string]$KvDtype      = '',
    [switch]$GdnFp16,
    [int]$B               = 0,
    [int]$R               = 0
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

$TABLE = @{
    'long' = @{ B = 32768; R = 5632; Lanes = 1 }
    'dual' = @{ B = 16384; R = 2816; Lanes = 2 }
    'max8' = @{ B = 3264;  R = 1536; Lanes = 8 }
}
if (-not $TABLE.ContainsKey($Profile)) { throw "unknown profile: $Profile" }
$cfg = $TABLE[$Profile]
# -B / -R override the profile's device window; the lane count stays the profile's.
$poolB = if ($B -gt 0) { $B } else { $cfg.B }
$poolR = if ($R -gt 0) { $R } else { $cfg.R }

# The production rule is max_tokens = max_context / 2. On this stack the device pool
# is the real ceiling for one request (a request longer than the pool has nowhere to append
# into: a 32,000-token request meeting a small pool was observed to crash a worker with no
# self-heal). So the default is the pool's own half, and a judgement must never fail because
# the OUTPUT BUDGET was too small: thinking tokens consume it first, and an empty answer
# then looks like an engine defect.
$poolPerLane = [int](($poolB + $poolR) / $cfg.Lanes)
$longCeiling = [int]($poolPerLane / 2)
if ($LongMax -le 0) { $LongMax = $longCeiling }
if ($LongMax -gt $longCeiling) {
    Write-Output ("long max_tokens clamped {0} -> {1} (per-lane pool {2})" -f $LongMax, $longCeiling, $poolPerLane)
    $LongMax = $longCeiling
}

Write-Output ("exe sha256 = {0}" -f (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)
Write-Output ("exe mtime  = {0}" -f (Get-Item -LiteralPath $exe).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
Write-Output ("profile    = {0}  B={1} R={2} lanes={3}  poolPerLane={4}" -f $Profile, $poolB, $poolR, $cfg.Lanes, $poolPerLane)
Write-Output ("kv-dtype   = {0}   gdn-state-fp16 = {1}" -f $(if ($KvDtype -eq '') { '(default bf16)' } else { $KvDtype }), $GdnFp16)
Write-Output ("haystack   = {0} tokens, {1} records, ask {2}, long decode {3}" -f $TargetTokens, $Needles, $AskMax, $LongMax)
Write-Output ("host= {0} MiB  disk= {1} MiB  ctx= {2}  sessions={3}  mtp={4}" -f $HostMiB, $DiskMiB, $Ctx, $Sessions, $Mtp)
Write-Output ''

function Stop-Orphans {
    Get-Process -Name 'ninfer-serve', 'ninfer-perplexity', 'ninfer' -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.Id -Force }
    Start-Sleep -Seconds 5
}

# Builds a haystack with unique filler (so no prefix cache can turn prefill into
# a lookup) and records scattered at even intervals. Returns the text plus the
# exact codes that must come back.
function Build-Haystack {
    param([int]$TargetTokens, [int]$Count, [string]$Salt)
    $base = 'The quick brown fox jumps over the lazy dog. Ninfer on Turing keeps a bounded KV window and streams the rest from host memory through the tier. '
    $sb = [System.Text.StringBuilder]::new()
    $codes = @()
    $every = [int]($TargetTokens / ($Count + 1))
    $placed = 0
    $i = 0
    while ($sb.Length -lt ($TargetTokens * 4)) {
        $approx = [int]($sb.Length / 4)
        if ($placed -lt $Count -and $approx -ge ($every * ($placed + 1))) {
            $placed++
            $code = ('C' + $placed + $Salt + 'Q' + ($placed * 7 + 3))
            $codes += $code
            [void]$sb.Append(' IMPORTANT RECORD k='); [void]$sb.Append($placed)
            [void]$sb.Append(' value='); [void]$sb.Append($code); [void]$sb.Append('. ')
        }
        [void]$sb.Append('u'); [void]$sb.Append($i); [void]$sb.Append(' '); [void]$sb.Append($base)
        $i++
    }
    return [pscustomobject]@{ Text = $sb.ToString(); Codes = $codes }
}

function Get-DoneLines {
    param([string]$LogPath)
    return @(Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue |
             Select-String -Pattern 'req#\d+ done')
}

function Invoke-Chat {
    param([string]$Url, [hashtable]$Body, [string]$LogPath, [int]$TimeoutSec)
    $before = (Get-DoneLines -LogPath $LogPath).Count
    $json   = $Body | ConvertTo-Json -Depth 6
    $sw     = [System.Diagnostics.Stopwatch]::StartNew()
    $resp   = Invoke-RestMethod -Uri $Url -Method Post -ContentType 'application/json' `
                -Body $json -TimeoutSec $TimeoutSec
    $sw.Stop()
    $deadline = (Get-Date).AddSeconds(120)
    $lines = @()
    while ((Get-Date) -lt $deadline) {
        $lines = Get-DoneLines -LogPath $LogPath
        if ($lines.Count -gt $before) { break }
        Start-Sleep -Milliseconds 500
    }
    $done = if ($lines.Count -gt 0) { $lines[-1].Line.Trim() } else { '(no done line)' }
    return [pscustomobject]@{ Resp = $resp; Wall = $sw.Elapsed.TotalSeconds; Done = $done }
}

function Test-Degenerate {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) { return 'EMPTY' }
    $words = @($Text -split '\s+' | Where-Object { $_ -ne '' })
    if ($words.Count -lt 40) { return 'SHORT' }
    $counts = @{}
    for ($i = 0; $i -le ($words.Count - 8); $i++) {
        $k = ($words[$i..($i + 7)] -join ' ')
        if ($counts.ContainsKey($k)) { $counts[$k] = $counts[$k] + 1 } else { $counts[$k] = 1 }
    }
    $max = ($counts.Values | Measure-Object -Maximum).Maximum
    if ($max -ge 8) { return ('REPEAT-x' + $max) }
    return 'OK'
}

Stop-Orphans

$arm = '{0}_{1}_{2}' -f $Tag, $Profile, $Mtp
$errLog = Join-Path $BuildDir ("$arm.server.err.log")
$outLog = Join-Path $BuildDir ("$arm.server.out.log")
Remove-Item -LiteralPath $errLog, $outLog -ErrorAction SilentlyContinue

$argList = @($Model, '--host', '127.0.0.1', '--port', "$Port", '--device-profile', 'off',
             '--max-context', "$Ctx", '--max-concurrency', "$($cfg.Lanes)",
             '--kvmem-budget', "$poolB", '--kvmem-gen-reserve', "$poolR",
             '--kvmem-host-mib', "$HostMiB", '--kvmem-sessions', "$Sessions",
             '--kvmem-sink-tokens', "$Sink", '--kvmem-recent-tokens', "$Recent",
             '--kvmem-disk-path', "$DiskPath", '--kvmem-disk-mib', "$DiskMiB",
             '--log-level', 'info')
if ($KvDtype -ne '') { $argList += @('--kv-dtype', $KvDtype) }
if ($GdnFp16) { $argList += '--gdn-state-fp16' }
if ($Mtp -eq 'mtp') { $argList += @('--spec', 'mtp', '--draft-tokens', '3') }
Write-Output ('flags = ' + ($argList -join ' '))

$t0 = Get-Date
$proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog -ArgumentList $argList

$baseUrl  = "http://127.0.0.1:$Port"
$ready    = $false
$deadline = (Get-Date).AddSeconds($ReadySec)
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) { break }
    try { [void](Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 3); $ready = $true; break }
    catch { Start-Sleep -Seconds 5 }
}
$readySec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

if (-not $ready) {
    Write-Output ("NOT-READY after {0}s" -f $readySec)
    Get-Content -LiteralPath $errLog -ErrorAction SilentlyContinue |
        Select-String -Pattern 'KVMem rejected|error|Error|FATAL|invalid|out of memory' |
        Select-Object -Last 5 | ForEach-Object { Write-Output ('  ' + $_.Line.Trim()) }
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Write-Output 'RESULT: NOT-READY'
    exit 1
}

$capText = @(Get-Content -LiteralPath $errLog -ErrorAction SilentlyContinue) |
           Select-String -Pattern 'capacity \|' | Select-Object -Last 1
Write-Output ('ready in {0}s' -f $readySec)
if ($capText) { Write-Output ('  ' + $capText.Line.Trim()) }
Write-Output ('  gpu = ' + ((& nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader) -join ' '))

$modelId = 'local'
try {
    $m = Invoke-RestMethod -Uri "$baseUrl/v1/models" -TimeoutSec 10
    if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
} catch { }
$chatUrl = "$baseUrl/v1/chat/completions"

$hay = Build-Haystack -TargetTokens $TargetTokens -Count $Needles -Salt ('S' + $Profile + $Mtp)
Write-Output ("haystack chars = {0}, records = {1}" -f $hay.Text.Length, $hay.Codes.Count)
Write-Output ('  codes: ' + ($hay.Codes -join ' '))

# ---------------------------------------------------------------------------
# 1) cross-page multi-record recall: the haystack is ~3.4x the pool, so every
#    record sits in a block that was evicted and must be re-selected.
# ---------------------------------------------------------------------------
$askPrompt = $hay.Text + " TASK: the document above contains $Needles records of the form 'RECORD k=<index> value=<code>'. List every one as 'k=<index> value=<code>', one per line, ascending index. Output the list only, no preamble."
$recallHits = 0
$recallTotal = $hay.Codes.Count
$recallFinish = 'n/a'
$recallUsage = 'n/a'
try {
    $res = Invoke-Chat -Url $chatUrl -LogPath $errLog -TimeoutSec $GenTimeout -Body @{
        model = $modelId
        messages = @(@{ role = 'user'; content = $askPrompt })
        max_tokens = $AskMax; stream = $false; temperature = 0.0
    }
    $answer = [string]$res.Resp.choices[0].message.content
    $recallFinish = [string]$res.Resp.choices[0].finish_reason
    $recallUsage = ($res.Resp.usage | ConvertTo-Json -Compress)
    foreach ($c in $hay.Codes) { if ($answer -match [regex]::Escape($c)) { $recallHits++ } }
    $shown = ($answer -replace "\s+", ' ')
    if ($shown.Length -gt 300) { $shown = $shown.Substring(0, 300) }
    Write-Output ("RECALL records {0}/{1}  finish={2}  wall={3:N1}s" -f $recallHits, $recallTotal, $recallFinish, $res.Wall)
    Write-Output ('  usage: ' + $recallUsage)
    Write-Output ('  answer: [' + $shown + ']')
    Write-Output ('  done: ' + $res.Done)
} catch { Write-Output ('RECALL FAILED: ' + $_.Exception.Message) }

# ---------------------------------------------------------------------------
# 2) long decode across page swaps, with a constraint that must hold at BOTH
#    ends and citations that must survive from the far end of the haystack.
# ---------------------------------------------------------------------------
$nonce = 'VX' + (Get-Random -Minimum 10000 -Maximum 99999)
$citeTargets = @($hay.Codes[-1], $hay.Codes[[int]($hay.Codes.Count / 2)], $hay.Codes[0])
$longPrompt = $hay.Text + " TASK: write a long, structured technical analysis of how a bounded KV window with host tiering behaves under a 128K-token history. RULES: (1) your very first line must be exactly 'PLAN=$nonce'; (2) your very last line must be exactly 'VERIFY=$nonce'; (3) somewhere in the body, name these three records verbatim: $($citeTargets -join ', '); (4) do not repeat yourself."
$longFinish = 'n/a'; $longComp = 'n/a'; $longReason = 'n/a'
$planOk = 'n/a'; $verifyOk = 'n/a'; $nonceMatch = 'n/a'; $cited = 0; $degen = 'n/a'; $longWall = 'n/a'
try {
    $res = Invoke-Chat -Url $chatUrl -LogPath $errLog -TimeoutSec $GenTimeout -Body @{
        model = $modelId
        messages = @(@{ role = 'user'; content = $longPrompt })
        max_tokens = $LongMax; stream = $false; temperature = 0.0
    }
    $text = [string]$res.Resp.choices[0].message.content
    $longWall = [math]::Round($res.Wall, 1)
    $longFinish = [string]$res.Resp.choices[0].finish_reason
    $longComp = $res.Resp.usage.completion_tokens
    $longReason = $res.Resp.usage.completion_tokens_details.reasoning_tokens
    $degen = Test-Degenerate -Text $text
    $first = (($text -split "`n") | Where-Object { $_.Trim() -ne '' } | Select-Object -First 1)
    $last  = (($text -split "`n") | Where-Object { $_.Trim() -ne '' } | Select-Object -Last 1)
    $planOk   = if ($first -match ('^PLAN=' + [regex]::Escape($nonce))) { 'YES' } else { 'NO' }
    $verifyOk = if ($last  -match ('^VERIFY=' + [regex]::Escape($nonce))) { 'YES' } else { 'NO' }
    # Capture each match separately: a second -match overwrites $Matches, so
    # comparing $Matches[1] to itself is always true and tests nothing.
    $planNonce   = if ($first -match 'PLAN=(\S+)') { $Matches[1] } else { '' }
    $verifyNonce = if ($last  -match 'VERIFY=(\S+)') { $Matches[1] } else { '' }
    $nonceMatch  = if ($planNonce -eq $nonce -and $verifyNonce -eq $nonce) { 'SAME' } else { ('DIFF plan=' + $planNonce + ' verify=' + $verifyNonce) }
    foreach ($c in $citeTargets) { if ($text -match [regex]::Escape($c)) { $cited++ } }
    Write-Output ("LONG wall={0:N1}s finish={1} comp={2} reason={3} degen={4}" -f $res.Wall, $longFinish, $longComp, $longReason, $degen)
    Write-Output ("  PLAN-first={0}  VERIFY-last={1}  nonce={2}  cited {3}/{4}" -f $planOk, $verifyOk, $nonce, $cited, $citeTargets.Count)
    Write-Output ('  first line: [' + $first + ']')
    Write-Output ('  last  line: [' + $last + ']')
    Write-Output ('  done: ' + $res.Done)
} catch { Write-Output ('LONG FAILED: ' + $_.Exception.Message) }

$loadJson = 'n/a'
try {
    $load = Invoke-RestMethod -Uri "$baseUrl/v1/load" -TimeoutSec 30
    $loadJson = ($load.kvmem | ConvertTo-Json -Compress)
    Write-Output ('/v1/load kvmem = ' + $loadJson)
} catch { }

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
Start-Sleep -Seconds 3
Stop-Orphans

$row = [pscustomobject]@{
    arm = $arm; profile = $Profile; mtp = $Mtp
    target = $TargetTokens; needles = $Needles
    ready_s = $readySec; capacity = if ($capText) { $capText.Line.Trim() } else { 'n/a' }
    recall_hits = $recallHits; recall_total = $recallTotal
    recall_finish = $recallFinish; recall_usage = $recallUsage
    long_wall_s = $longWall; long_finish = $longFinish; long_comp = $longComp
    long_reason = $longReason; long_degen = $degen
    plan_first = $planOk; verify_last = $verifyOk; nonce = $nonce
    cited = $cited; cite_total = $citeTargets.Count
    load = $loadJson
}
$csv = Join-Path $BuildDir ($Tag + '_' + $Profile + '_' + $Mtp + '_summary.csv')
$row | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
Write-Output ('summary csv: ' + $csv)
Write-Output 'RESULT: DEEP-DONE'
