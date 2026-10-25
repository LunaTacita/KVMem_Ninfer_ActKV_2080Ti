# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - KV profile matrix: the three shipped profiles with
# MTP off and on, repeated, so every number is like-for-like.
#
# WHY THIS SCRIPT EXISTS
#   The three profiles whose (B+R)*lanes is always 38400 (long / dual / max8)
#   had never been compared back to back under one identical flag set, and a
#   decode figure measured without --spec mtp and --kvmem-sink-tokens /
#   --kvmem-recent-tokens describes a different configuration. This script
#   removes that ambiguity by measuring one matrix with only two variables:
#       profile in {long, dual, max8}  x  mtp in {off, mtp}  x  repeats
#
# HOST LOAD - NOT SILENT, ON PURPOSE (deliberate production setting)
#   Production environment conditions are the right conditions: Edge and the
#   production apps stay open; a judgement that only passes on a quiet host does
#   not describe the machine the user actually runs. The host can move a prefill
#   number by 2x, so a measurement is accepted only when the repeats AGREE. A
#   disagreeing
#   repeat is reported as SPREAD and never averaged away.
#
# WHAT EVERY ARM REPORTS (all from the engine's own words, never ExitCode)
#   ready seconds, capacity line (pages / runtime / free), GPU used/free
#   decode tok/s on a short generation
#   prefill tok/s on a 32K prompt whose every segment is unique (prefix reuse
#     cannot turn prefill into a lookup)
#   middle-needle recall at 4K (control) and 32K, with finish_reason and the
#     full usage object so a thinking-starved budget cannot be misread as a
#     recall failure
#   a production-grade long decode: 1024 max tokens, wall time, decode tok/s,
#     completion/reasoning tokens, finish_reason, and a repetition check
#   mtp accepted X/Y when MTP is on
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK and then throws
# "Unexpected token" or silently runs nothing).
# One port per arm: two servers on one port split requests.
# ============================================================================
param(
    [string]$Model           = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [string[]]$Profiles      = @('long', 'dual', 'max8'),
    [string[]]$MtpModes      = @('off', 'mtp'),
    [int]$Repeats            = 2,
    [int]$Ctx                = 262144,
    [int]$Sink               = 4096,
    [int]$Recent             = 1024,
    [int]$HostMiB            = 16384,
    [int]$Sessions           = 4,
    [int]$BasePort           = 18100,
    [int]$LongTokens         = 32768,
    # 4096, not 1024. Thinking occupies the output budget, so a 1024 cap can be
    # consumed entirely by reasoning and leave message.content empty: the probe
    # arm returned comp 1024 / reasoning 1024 / finish length / text head [].
    # The upstream handbook (04, sections 1.8 OVER_BUDGET, 2 and 3-L1) records
    # this as established and forbids "fixing" long thinking by lowering
    # max_tokens - that only turns "thinking too long" into "zero answer text".
    # Keep max_tokens <= the pool token count (38400 for every profile here).
    [int]$DecodeMaxTokens    = 4096,
    [int]$ReadySec           = 900,
    [int]$GenTimeout         = 3600,
    [string]$Tag             = 'profmtx',
    [string]$BuildDir        = (Join-Path $PSScriptRoot '..\build')
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

# Binary is anchored by hash and mtime: the "build <id>" string in the log comes
# from a cmake stamp target that does not refresh on every rebuild.
Write-Output ("exe sha256 = {0}" -f (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)
Write-Output ("exe mtime  = {0}" -f (Get-Item -LiteralPath $exe).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
Write-Output ("host load  = PRODUCTION (not silent; Edge/apps stay open by design)")
Write-Output ("matrix     = profiles[{0}] x mtp[{1}] x repeats {2}" -f ($Profiles -join ','), ($MtpModes -join ','), $Repeats)
Write-Output ''

# The three shipped profiles. (B+R)*lanes == 38400 for all three, which is the
# device-pool ceiling measured on this 11 GB card (free 236 MiB at that size;
# B+R=42496 collapsed to prefill 1.8 tok/s).
$TABLE = @{
    'long' = @{ B = 32768; R = 5632; Lanes = 1 }
    'dual' = @{ B = 16384; R = 2816; Lanes = 2 }
    'max8' = @{ B = 3264;  R = 1536; Lanes = 8 }
}

function Stop-Orphans {
    Get-Process -Name 'ninfer-serve', 'ninfer-perplexity', 'ninfer' -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.Id -Force }
    Start-Sleep -Seconds 5
}

function Build-UniquePrompt {
    param([int]$TargetTokens, [string]$Prefix)
    $base = 'The quick brown fox jumps over the lazy dog. Ninfer on Turing keeps a bounded KV window and streams the rest from host memory. '
    $sb = [System.Text.StringBuilder]::new()
    $i = 0
    while ($sb.Length -lt ($TargetTokens * 4)) {
        [void]$sb.Append($Prefix); [void]$sb.Append('#'); [void]$sb.Append($i)
        [void]$sb.Append(' '); [void]$sb.Append($base)
        $i++
    }
    return $sb.ToString()
}

function Build-Needle {
    param([int]$TargetTokens, [string]$Needle, [string]$Salt)
    $base = 'Filler about bounded KV windows, host tiers and Turing kernels. '
    $half = [int]($TargetTokens / 2)
    $sb = [System.Text.StringBuilder]::new()
    $i = 0
    while ($sb.Length -lt ($half * 4)) {
        [void]$sb.Append($Salt); [void]$sb.Append('a'); [void]$sb.Append($i)
        [void]$sb.Append(' '); [void]$sb.Append($base)
        $i++
    }
    [void]$sb.Append(' IMPORTANT: the secret passphrase is '); [void]$sb.Append($Needle); [void]$sb.Append('. ')
    while ($sb.Length -lt ($TargetTokens * 4)) {
        [void]$sb.Append($Salt); [void]$sb.Append('b'); [void]$sb.Append($i)
        [void]$sb.Append(' '); [void]$sb.Append($base)
        $i++
    }
    [void]$sb.Append(' Question: what is the secret passphrase? Reply with the passphrase only.')
    return $sb.ToString()
}

function Get-DoneLines {
    param([string]$LogPath)
    return @(Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue |
             Select-String -Pattern 'req#\d+ done')
}

function Get-LastMatching {
    param([string]$LogPath, [string]$Pattern)
    $line = Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue |
            Select-String -Pattern $Pattern | Select-Object -Last 1
    if ($line) { return $line.Line.Trim() }
    return '(none)'
}

function Get-GpuState {
    $v = & nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>$null
    if ($v) { return ([string]$v).Trim() }
    return 'n/a'
}

# Fires one chat completion and waits for the engine to write the matching
# "req#N done" line, so the tok/s always comes from the engine's own report and
# not from a wall-clock guess.
function Invoke-Chat {
    param([string]$Url, [hashtable]$Body, [string]$LogPath, [int]$TimeoutSec)
    $before = (Get-DoneLines -LogPath $LogPath).Count
    $json   = $Body | ConvertTo-Json -Depth 6
    $sw     = [System.Diagnostics.Stopwatch]::StartNew()
    $resp   = Invoke-RestMethod -Uri $Url -Method Post -ContentType 'application/json' `
                -Body $json -TimeoutSec $TimeoutSec
    $sw.Stop()
    $deadline = (Get-Date).AddSeconds(90)
    $lines = @()
    while ((Get-Date) -lt $deadline) {
        $lines = Get-DoneLines -LogPath $LogPath
        if ($lines.Count -gt $before) { break }
        Start-Sleep -Milliseconds 500
    }
    $done = if ($lines.Count -gt 0) { $lines[-1].Line.Trim() } else { '(no done line)' }
    return [pscustomobject]@{ Resp = $resp; Wall = $sw.Elapsed.TotalSeconds; Done = $done }
}

# A production-grade long answer that degenerates into a loop would otherwise
# read as "the long decode worked". 8-gram repetition is the cheapest detector.
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

$rows = @()
$armIndex = 0

foreach ($prof in $Profiles) {
    if (-not $TABLE.ContainsKey($prof)) { throw "unknown profile: $prof" }
    $cfg = $TABLE[$prof]

    foreach ($mtp in $MtpModes) {
        for ($rep = 1; $rep -le $Repeats; $rep++) {

            $armIndex++
            $port = $BasePort + $armIndex
            $arm  = '{0}_{1}_{2}_r{3}' -f $Tag, $prof, $mtp, $rep
            $errLog = Join-Path $BuildDir ("$arm.server.err.log")
            $outLog = Join-Path $BuildDir ("$arm.server.out.log")
            Remove-Item -LiteralPath $errLog, $outLog -ErrorAction SilentlyContinue

            Write-Output ('================ ARM {0} ================' -f $arm)
            Write-Output ('  B={0} R={1} lanes={2} port={3} sink={4} recent={5} mtp={6}' -f `
                $cfg.B, $cfg.R, $cfg.Lanes, $port, $Sink, $Recent, $mtp)

            Stop-Orphans

            $argList = @($Model, '--host', '127.0.0.1', '--port', "$port", '--device-profile', 'off',
                         '--max-context', "$Ctx", '--max-concurrency', "$($cfg.Lanes)",
                         '--kvmem-budget', "$($cfg.B)", '--kvmem-gen-reserve', "$($cfg.R)",
                         '--kvmem-host-mib', "$HostMiB", '--kvmem-disk-path', (Join-Path $PSScriptRoot '..\Cache\kvmem'), '--kvmem-disk-mib', '102400', '--kvmem-sessions', "$Sessions",
                         '--kvmem-sink-tokens', "$Sink", '--kvmem-recent-tokens', "$Recent",
                         '--log-level', 'info')
            if ($mtp -eq 'mtp') { $argList += @('--spec', 'mtp', '--draft-tokens', '3') }
            Write-Output ('  flags = ' + ($argList -join ' '))

            $row = [pscustomobject]@{
                arm = $arm; profile = $prof; mtp = $mtp; rep = $rep
                B = $cfg.B; R = $cfg.R; lanes = $cfg.Lanes
                ready_s = 'n/a'; pages = 'n/a'; runtime_MiB = 'n/a'; pool_free = 'n/a'
                gpu_used_free = 'n/a'; decode_short = 'n/a'; prefill = 'n/a'
                prefill_wall_s = 'n/a'; recall_4k = 'n/a'; recall_32k = 'n/a'
                finish_32k = 'n/a'; usage_32k = 'n/a'
                long_wall_s = 'n/a'; long_decode = 'n/a'; long_finish = 'n/a'
                long_comp_tok = 'n/a'; long_reason_tok = 'n/a'; long_degen = 'n/a'
                mtp_accept = 'n/a'; note = ''
            }

            $t0 = Get-Date
            $proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
                -RedirectStandardOutput $outLog -RedirectStandardError $errLog -ArgumentList $argList

            $baseUrl  = "http://127.0.0.1:$port"
            $ready    = $false
            $deadline = (Get-Date).AddSeconds($ReadySec)
            while ((Get-Date) -lt $deadline) {
                if ($proc.HasExited) { break }
                try { [void](Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 3); $ready = $true; break }
                catch { Start-Sleep -Seconds 5 }
            }
            $row.ready_s = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

            if (-not $ready) {
                Write-Output ('  NOT-READY after {0}s' -f $row.ready_s)
                Get-Content -LiteralPath $errLog -ErrorAction SilentlyContinue |
                    Select-String -Pattern 'KVMem rejected|error|Error|FATAL|invalid|out of memory' |
                    Select-Object -Last 3 | ForEach-Object { Write-Output ('    ' + $_.Line.Trim()) }
                if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
                $row.note = 'NOT-READY'
                $rows += $row
                Start-Sleep -Seconds 3
                continue
            }

            # --- the engine's own capacity line + the real GPU picture ----------
            $capText = Get-LastMatching -LogPath $errLog -Pattern 'capacity \|'
            Write-Output ('  ' + $capText)
            if ($capText -match 'pages (\d+)/(\d+)') { $row.pages = $Matches[1] }
            if ($capText -match 'runtime ([\d.]+) MiB') { $row.runtime_MiB = $Matches[1] }
            if ($capText -match 'free ([\d.]+) (GiB|MiB|B)') { $row.pool_free = ($Matches[1] + ' ' + $Matches[2]) }
            $row.gpu_used_free = Get-GpuState
            Write-Output ('  gpu used/free = ' + $row.gpu_used_free)

            $modelId = 'local'
            try {
                $m = Invoke-RestMethod -Uri "$baseUrl/v1/models" -TimeoutSec 10
                if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
            } catch { }
            Write-Output ('  model id = ' + $modelId)

            $chatUrl = "$baseUrl/v1/chat/completions"

            # --- 1) short decode ----------------------------------------------
            try {
                $res = Invoke-Chat -Url $chatUrl -LogPath $errLog -TimeoutSec $GenTimeout -Body @{
                    model = $modelId
                    messages = @(@{ role = 'user'; content = 'Count from 1 to 60, one number per line.' })
                    max_tokens = 256; stream = $false; temperature = 0.0
                }
                if ($res.Done -match 'decode ([\d.]+) tok/s') { $row.decode_short = $Matches[1] }
                Write-Output ('  decode short: ' + $res.Done)
            } catch { Write-Output ('  decode short FAILED: ' + $_.Exception.Message); $row.note += 'decodeShortFail;' }

            # --- 2) prefill on a long unique prompt ---------------------------
            try {
                $prompt = Build-UniquePrompt -TargetTokens $LongTokens -Prefix 'seg'
                $res = Invoke-Chat -Url $chatUrl -LogPath $errLog -TimeoutSec $GenTimeout -Body @{
                    model = $modelId
                    messages = @(@{ role = 'user'; content = $prompt })
                    max_tokens = 8; stream = $false
                }
                $row.prefill_wall_s = [math]::Round($res.Wall, 1)
                if ($res.Done -match 'prefill ([\d.]+) tok/s') { $row.prefill = $Matches[1] }
                Write-Output ('  prefill: ' + $res.Done + ('  (wall {0:N1}s)' -f $res.Wall))
            } catch { Write-Output ('  prefill FAILED: ' + $_.Exception.Message); $row.note += 'prefillFail;' }

            # --- 3) middle-needle recall, 4K control then 32K ------------------
            foreach ($len in @(4096, $LongTokens)) {
                $needle = ('ZQ' + $cfg.B + 'X' + $len)
                try {
                    $prompt = Build-Needle -TargetTokens $len -Needle $needle -Salt ('n' + $prof + $mtp + $rep)
                    $res = Invoke-Chat -Url $chatUrl -LogPath $errLog -TimeoutSec $GenTimeout -Body @{
                        model = $modelId
                        messages = @(@{ role = 'user'; content = $prompt })
                        max_tokens = 512; stream = $false; temperature = 0.0
                    }
                    $answer = [string]$res.Resp.choices[0].message.content
                    $finish = [string]$res.Resp.choices[0].finish_reason
                    $usage  = ($res.Resp.usage | ConvertTo-Json -Compress)
                    $hit    = if ($answer -match [regex]::Escape($needle)) { 'HIT' } else { 'MISS' }
                    Write-Output ('  needle@{0,6}: {1} | finish={2} | wall={3:N1}s' -f $len, $hit, $finish, $res.Wall)
                    Write-Output ('      usage: ' + $usage)
                    if ($len -eq 4096) { $row.recall_4k = $hit } else { $row.recall_32k = $hit; $row.finish_32k = $finish; $row.usage_32k = $usage }
                } catch { Write-Output ('  needle@{0,6}: FAILED {1}' -f $len, $_.Exception.Message) }
            }

            # --- 4) production-grade long decode ------------------------------
            try {
                $res = Invoke-Chat -Url $chatUrl -LogPath $errLog -TimeoutSec $GenTimeout -Body @{
                    model = $modelId
                    messages = @(@{ role = 'user'; content = 'Write a detailed technical explanation, in at least 700 words, of how paged KV caches work in large language model inference: block tables, page allocation, eviction, and host/disk tiering.' })
                    max_tokens = $DecodeMaxTokens; stream = $false; temperature = 0.0
                }
                $text = [string]$res.Resp.choices[0].message.content
                $row.long_wall_s    = [math]::Round($res.Wall, 1)
                $row.long_finish    = [string]$res.Resp.choices[0].finish_reason
                $row.long_comp_tok  = $res.Resp.usage.completion_tokens
                $row.long_reason_tok = $res.Resp.usage.completion_tokens_details.reasoning_tokens
                $row.long_degen     = Test-Degenerate -Text $text
                if ($res.Done -match 'decode ([\d.]+) tok/s') { $row.long_decode = $Matches[1] }
                Write-Output ('  long decode: wall={0:N1}s finish={1} comp={2} reason={3} degen={4}' -f `
                    $res.Wall, $row.long_finish, $row.long_comp_tok, $row.long_reason_tok, $row.long_degen)
                Write-Output ('    done: ' + $res.Done)
                Write-Output ('    text head: [' + (($text -replace "\s+", ' ')).Substring(0, [Math]::Min(120, ($text -replace "\s+", ' ').Length)) + ']')
            } catch { Write-Output ('  long decode FAILED: ' + $_.Exception.Message); $row.note += 'longDecodeFail;' }

            # --- 5) MTP acceptance --------------------------------------------
            if ($mtp -eq 'mtp') {
                $row.mtp_accept = Get-LastMatching -LogPath $errLog -Pattern 'mtp accepted'
                Write-Output ('  ' + $row.mtp_accept)
            }

            if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
            Start-Sleep -Seconds 3
            $rows += $row
            Write-Output ''
        }
    }
}

Stop-Orphans
Write-Output '================ PROFILE MATRIX SUMMARY ================'
$rows | Format-Table arm, ready_s, pages, pool_free, decode_short, prefill, recall_4k, recall_32k, long_decode, long_degen, mtp_accept -AutoSize |
    Out-String | Write-Output

$csv = Join-Path $BuildDir ($Tag + '_summary.csv')
Remove-Item -LiteralPath $csv -ErrorAction SilentlyContinue
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
Write-Output ('summary csv: ' + $csv)
Write-Output 'RESULT: PROFILE-MATRIX-DONE'
