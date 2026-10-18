# 路径约定（本脚本默认值按“发布包布局”解析）：包根 = 本脚本所在目录的上一级。
#   - 模型：<包根>\model\       - 构建目录：<包根>\build\
#   - 缓存：<包根>\Cache\       - 测试图：<包根>\media\
#   在构建环境中运行请显式传 -Model / -BuildDir 等参数覆盖默认值。
# ============================================================================
# sm75 / KVMem 0.18 stack - release parameter A/B matrix.
#
# WHY THIS EXISTS
#   The bar for this stack is speed, accuracy and completeness. A single-needle
#   judgement answers "does one mid-prompt needle survive"; it does NOT answer
#   "which parameters serve accuracy when the context carries MANY facts".
#   This script measures candidate settings on a multi-fact judgement:
#     * EIGHT vault codes spread over 5%..89% of a 40,960-token prompt, asked
#       for in one request (multi-fact recall);
#     * one follow-up request for a named code (exact single-fact recall; the
#       prefix cache serves it);
#     * a 32K unique-segment prefill (no prefix reuse possible);
#     * short decode tok/s + MTP acceptance;
#     * a 4-turn chat asking turn 1's fact at turn 4 (cross-turn memory).
#
# PRESETS
#   actkv  : ActKV pillar (2) confidence budget + pillar (3) scorer + LRFU.
#            Pillar (1) (action-region query + mandatory retention) is automatic.
#   sink   : --kvmem-sink-tokens / --kvmem-recent-tokens around the production
#            pair (4096 / 1024), and it decides whether
#            the system prompt and task description survive eviction.
#   budget : B/R trade around the production pair (32768 / 5632). B is the
#            visible set; R drives prefill speed. B+R must stay <= 38400 here.
#   p1     : small wins - Q6 output head / Q6 embedding (frees VRAM), CUDA
#            graph allowance (removes the overrun warning), wider n-gram draft.
#   all    : every arm above, one server start each (serial by design: GPU
#            judgements must never overlap).
#
# VERDICTS come from HTTP bodies and the engine's own "req#N done" lines, never
# from ExitCode. Output: <BuildDir>\param_ab_out\*_summary.csv
#
# Encoding: UTF-8 with BOM (a PS 5.1 host reads a non-BOM .ps1 as ANSI/GBK).
# ============================================================================
param(
    [string]$Model      = (Join-Path $PSScriptRoot '..\model\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer'),
    [int]$Port          = 18200,
    [string]$Preset     = 'actkv',
    [int]$NeedleTokens  = 40960,
    [int]$PrefillTokens = 32768,
    [int]$Turns         = 4,
    [int]$ReadySec      = 900,
    [int]$GenTimeout    = 3600,
    [string]$Tag        = 'param_ab',
    [string]$BuildDir   = (Join-Path $PSScriptRoot '..\build')
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
$out = Join-Path $BuildDir 'param_ab_out'
New-Item -ItemType Directory -Force -Path $out | Out-Null
foreach ($p in @($exe, $Model)) { if (-not (Test-Path -LiteralPath $p)) { throw "MISSING: $p" } }

# ---------------------------------------------------------------- arm table
$baseArm = @{ B = 32768; R = 5632; Sink = 4096; Recent = 1024; extra = @() }
$armDefs = @()

if ($Preset -eq 'actkv' -or $Preset -eq 'all') {
    $armDefs += @{ name = 'base_default';  cfg = $baseArm }
    $armDefs += @{ name = 'actkv_full';    cfg = @{ B = 32768; R = 5632; Sink = 4096; Recent = 1024;
                    extra = @('--kvmem-retrieval-method','per-token','--kvmem-conf-budget','--kvmem-conf-init','16384','--kvmem-conf-upper','32768','--kvmem-lrfu') } }
    $armDefs += @{ name = 'actkv_sub';     cfg = @{ B = 32768; R = 5632; Sink = 4096; Recent = 1024;
                    extra = @('--kvmem-retrieval-method','sub-block','--kvmem-subblocks','4','--kvmem-subblock-reduce','max','--kvmem-conf-budget','--kvmem-conf-init','16384','--kvmem-conf-upper','32768','--kvmem-lrfu') } }
}
if ($Preset -eq 'sink' -or $Preset -eq 'all') {
    $armDefs += @{ name = 'sink4096_r1024'; cfg = $baseArm }
    $armDefs += @{ name = 'sink8192_r1024'; cfg = @{ B = 32768; R = 5632; Sink = 8192;  Recent = 1024; extra = @() } }
    $armDefs += @{ name = 'sink12288_r1024'; cfg = @{ B = 32768; R = 5632; Sink = 12288; Recent = 1024; extra = @() } }
    $armDefs += @{ name = 'sink4096_r2048'; cfg = @{ B = 32768; R = 5632; Sink = 4096;  Recent = 2048; extra = @() } }
}
if ($Preset -eq 'budget' -or $Preset -eq 'all') {
    $armDefs += @{ name = 'b32768_r5632';  cfg = $baseArm }
    $armDefs += @{ name = 'b36864_r1536';  cfg = @{ B = 36864; R = 1536; Sink = 4096; Recent = 1024; extra = @() } }
    $armDefs += @{ name = 'b28672_r8192';  cfg = @{ B = 28672; R = 8192; Sink = 4096; Recent = 1024; extra = @() } }
}
if ($Preset -eq 'p1' -or $Preset -eq 'all') {
    $armDefs += @{ name = 'ml_head_q6';    cfg = @{ B = 32768; R = 5632; Sink = 4096; Recent = 1024; extra = @('--lm-head-q6') } }
    $armDefs += @{ name = 'embed_q6';      cfg = @{ B = 32768; R = 5632; Sink = 4096; Recent = 1024; extra = @('--embedding-q6') } }
    $armDefs += @{ name = 'cudagraph128';  cfg = @{ B = 32768; R = 5632; Sink = 4096; Recent = 1024; extra = @('--cuda-graph-allowance-mib','128') } }
    $armDefs += @{ name = 'ngram31';       cfg = @{ B = 32768; R = 5632; Sink = 4096; Recent = 1024; extra = @('--ngram-draft-tokens','31') } }
}
if ($armDefs.Count -eq 0) { throw "Unknown -Preset '$Preset' (actkv|sink|budget|p1|all)" }

$csv = Join-Path $out ($Tag + '_' + $Preset + '_summary.csv')
if (Test-Path -LiteralPath $csv) { Remove-Item -LiteralPath $csv -Force }

Write-Output ("exe sha256 = {0}" -f (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)
Write-Output ("exe mtime  = {0}" -f (Get-Item -LiteralPath $exe).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
Write-Output ("model      = {0}" -f $Model)
Write-Output ("preset     = {0} | arms {1} | multi-needle {2} tokens (8 codes) | prefill {3} | turns {4}" -f `
    $Preset, $armDefs.Count, $NeedleTokens, $PrefillTokens, $Turns)
Write-Output ''

$codes = @('ACTK-0101','ACTK-0202','ACTK-0303','ACTK-0404','ACTK-0505','ACTK-0606','ACTK-0707','ACTK-0808')

function Stop-Orphans {
    Get-Process -Name 'ninfer-serve','ninfer-perplexity','ninfer' -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.Id -Force }
    Start-Sleep -Seconds 3
}

function Build-MultiNeedle {
    param([int]$TargetTokens, [string[]]$Codes)
    $base = 'Filler record about bounded KV windows, host tiers and Turing kernels. '
    $total = $TargetTokens * 4
    $sb = [System.Text.StringBuilder]::new()
    $i = 0
    $k = 0
    $n = $Codes.Count
    while ($sb.Length -lt $total) {
        $frac = [double]$sb.Length / [double]$total
        $nextFrac = if ($k -lt $n) { 0.05 + 0.12 * $k } else { 2.0 }
        if (($k -lt $n) -and ($frac -ge $nextFrac)) {
            [void]$sb.Append(' VAULT-CODE V' + ($k + 1) + ' is ' + $Codes[$k] + '. ')
            $k++
        } else {
            [void]$sb.Append($base)
            [void]$sb.Append('r' + $i + ' ')
            $i++
        }
    }
    while ($k -lt $n) {
        [void]$sb.Append(' VAULT-CODE V' + ($k + 1) + ' is ' + $Codes[$k] + '. ')
        $k++
    }
    [void]$sb.Append(' Question: list all 8 vault codes (V1 to V8) in order, one per line, codes only.')
    return $sb.ToString()
}

function Build-UniquePrompt {
    param([int]$TargetTokens)
    $base = 'The quick brown fox jumps over the lazy dog. Ninfer on Turing keeps a bounded KV window and streams the rest from host memory. '
    $sb = [System.Text.StringBuilder]::new()
    $i = 0
    while ($sb.Length -lt ($TargetTokens * 4)) {
        [void]$sb.Append('segment-' + $i + ' ' + $base)
        $i++
    }
    return $sb.ToString()
}

function Get-DoneLines {
    param([string]$LogPath)
    return @(Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue |
             Select-String -Pattern 'req#\d+ done')
}

# One chat completion. Returns a FLAT hashtable on purpose: an earlier version
# returned a [pscustomobject] whose "Done" member was a hashtable and this PS
# 7.6.6 host threw "Cannot convert ... PSCustomObject ... to System.Int32" at
# the call site. A flat hashtable has no type-conversion surface.
function Invoke-Chat {
    param([string]$Url, [hashtable]$Body, [string]$LogPath)
    $before = @(Get-DoneLines -LogPath $LogPath).Count
    $json = $Body | ConvertTo-Json -Depth 8
    $resp = Invoke-WebRequest -Uri $Url -Method Post -ContentType 'application/json' -Body $json -TimeoutSec $script:GenTimeout
    $h = $resp.Content | ConvertFrom-Json -AsHashtable
    $answer = ''
    $finish = ''
    $ch = @($h['choices'])
    if ($ch.Count -gt 0) {
        $msg = $ch[0]['message']
        if ($null -ne $msg -and $msg.ContainsKey('content')) { $answer = [string]$msg['content'] }
        if ($ch[0].ContainsKey('finish_reason')) { $finish = [string]$ch[0]['finish_reason'] }
    }
    $done = ''
    $dl = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $dl) {
        $lines = Get-DoneLines -LogPath $LogPath
        if ($lines.Count -gt $before) { $done = $lines[-1].Line.Trim(); break }
        Start-Sleep -Milliseconds 500
    }
    $res = @{ Answer = $answer; Finish = $finish; Prompt = 0; Output = 0; Cache = 0.0
              TTFT = ''; Prefill = ''; Decode = ''; Mtp = '' }
    if ($done) {
        if ($done -match 'prompt ([\d,]+)') { $res.Prompt = [int]($Matches[1] -replace ',','') }
        if ($done -match '\| output (\d+)') { $res.Output = [int]$Matches[1] }
        if ($done -match 'cache [\d,]+ \(([\d.]+)%') { $res.Cache = [double]$Matches[1] }
        if ($done -match 'TTFT ([0-9a-z. ]+?)\s*\|') { $res.TTFT = $Matches[1].Trim() }
        if ($done -match 'prefill ([\d.]+) tok/s') { $res.Prefill = $Matches[1] }
        if ($done -match 'decode ([\d.]+) tok/s') { $res.Decode = $Matches[1] }
        if ($done -match 'mtp accepted (\d+)/(\d+) \(([\d.]+)%\)') { $res.Mtp = ($Matches[1] + '/' + $Matches[2] + ' ' + $Matches[3] + '%') }
    }
    return $res
}

$rows = @()
foreach ($arm in $armDefs) {
    $name = $arm.name
    $cfg = $arm.cfg
    Write-Output ('================ ARM {0} ================' -f $name)
    Stop-Orphans

    $errLog = Join-Path $out ("{0}_{1}_{2}.err.log" -f $Tag, $Preset, $name)
    $outLog = Join-Path $out ("{0}_{1}_{2}.out.log" -f $Tag, $Preset, $name)
    Remove-Item -LiteralPath $errLog, $outLog -ErrorAction SilentlyContinue

    $argList = @($Model, '--host', '127.0.0.1', '--port', "$Port", '--device-profile', 'off',
                 '--max-context', '1048576', '--max-concurrency', '1',
                 '--kv-dtype', 'int8',
                 '--kvmem-budget', "$($cfg.B)", '--kvmem-gen-reserve', "$($cfg.R)",
                 '--kvmem-host-mib', '64421', '--kvmem-sessions', '4',
                 '--kvmem-sink-tokens', "$($cfg.Sink)", '--kvmem-recent-tokens', "$($cfg.Recent)",
                 '--kvmem-disk-path', (Join-Path $BuildDir ("kvmem_disk_ab_" + $Preset)),
                 '--kvmem-disk-mib', '102400',
                 '--spec', 'mtp', '--draft-tokens', '2', '--log-level', 'info')
    $argList += $cfg.extra
    Write-Output ("flags = " + ($argList -join ' '))

    $row = [ordered]@{
        arm = $name; B = $cfg.B; R = $cfg.R; sink = $cfg.Sink; recent = $cfg.Recent
        ready_s = 'n/a'; capacity = 'n/a'; note = ''
        short_decode = 'n/a'; short_mtp = 'n/a'
        needle_hits = 'n/a'; needle_hit_list = 'n/a'; needle_ttft = 'n/a'; needle_prefill = 'n/a'; needle_decode = 'n/a'
        single_hit = 'n/a'
        prefill32k = 'n/a'
        turns_ok = 'n/a'; cache_turns = 'n/a'; cross_turn_recall = 'n/a'
    }

    $t0 = Get-Date
    $proc = Start-Process -FilePath $exe -NoNewWindow -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog -ArgumentList $argList
    $baseUrl = "http://127.0.0.1:$Port"
    $ready = $false
    $deadline = (Get-Date).AddSeconds($ReadySec)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { break }
        try { [void](Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 3); $ready = $true; break }
        catch { Start-Sleep -Seconds 3 }
    }
    $row.ready_s = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    if (-not $ready) {
        $row.note = 'NOT-READY'
        $cap = @(Get-Content -LiteralPath $errLog -ErrorAction SilentlyContinue |
                 Select-String -Pattern 'KVMem rejected|capacity \|' | Select-Object -Last 1)
        if ($cap) { $row.capacity = $cap.Line }
        Write-Output ('  NOT-READY after {0}s : {1}' -f $row.ready_s, $row.capacity)
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
        $rows += [pscustomobject]$row
        Start-Sleep -Seconds 3
        continue
    }
    $cap = @(Get-Content -LiteralPath $errLog -ErrorAction SilentlyContinue |
             Select-String -Pattern 'capacity \|' | Select-Object -Last 1)
    if ($cap) { $row.capacity = $cap.Line }
    Write-Output ('  ready {0}s | {1}' -f $row.ready_s, $row.capacity)

    $modelId = 'local'
    try {
        $m = Invoke-RestMethod -Uri "$baseUrl/v1/models" -TimeoutSec 10
        if ($m.data -and @($m.data).Count -gt 0) { $modelId = @($m.data)[0].id }
    } catch { }
    $chatUrl = "$baseUrl/v1/chat/completions"

    # --- 1) short decode (speed on easy text) --------------------------------
    try {
        $r = Invoke-Chat -Url $chatUrl -LogPath $errLog -Body @{
            model = $modelId
            messages = @(@{ role = 'user'; content = 'Count from 1 to 60, one number per line.' })
            max_tokens = 256; stream = $false; temperature = 0.0
        }
        $row.short_decode = $r.Decode; $row.short_mtp = $r.Mtp
        Write-Output ('  short decode {0} tok/s | mtp {1}' -f $r.Decode, $r.Mtp)
    } catch { $row.note += 'shortFail;'; Write-Output ('  short decode FAILED line=' + $_.InvocationInfo.ScriptLineNumber + ': ' + $_.Exception.Message) }

    # --- 2) 8 vault codes over the needle prompt, one request ----------------
    try {
        $prompt = Build-MultiNeedle -TargetTokens $NeedleTokens -Codes $codes
        $r = Invoke-Chat -Url $chatUrl -LogPath $errLog -Body @{
            model = $modelId
            messages = @(@{ role = 'user'; content = $prompt })
            max_tokens = 2048; stream = $false; temperature = 0.0
        }
        $hits = 0
        $hitList = @()
        foreach ($c in $codes) {
            if ($r.Answer -match [regex]::Escape($c)) { $hits++; $hitList += 'H' } else { $hitList += '.' }
        }
        $row.needle_hits = ('{0}/{1}' -f $hits, $codes.Count)
        $row.needle_hit_list = ($hitList -join '')
        $row.needle_ttft = $r.TTFT; $row.needle_prefill = $r.Prefill; $row.needle_decode = $r.Decode
        $shown = ($r.Answer -replace "\s+", ' ')
        if ($shown.Length -gt 140) { $shown = $shown.Substring(0, 140) }
        Write-Output ('  8-needle   {0} [{1}] | TTFT {2} | prefill {3} | decode {4} | finish {5} | cache {6}%' -f `
            $row.needle_hits, $row.needle_hit_list, $r.TTFT, $r.Prefill, $r.Decode, $r.Finish, $r.Cache)
        Write-Output ('  answer: [' + $shown + ']')

        # --- 3) exact single-fact recall (prefix cache serves the 40K prompt) --
        $r2 = Invoke-Chat -Url $chatUrl -LogPath $errLog -Body @{
            model = $modelId
            messages = @(
                @{ role = 'user'; content = $prompt },
                @{ role = 'assistant'; content = $r.Answer },
                @{ role = 'user'; content = 'What is the vault code for V4? Reply with the code only.' }
            )
            max_tokens = 512; stream = $false; temperature = 0.0
        }
        $row.single_hit = if ($r2.Answer -match [regex]::Escape($codes[3])) { 'HIT' } else { 'MISS' }
        $shown2 = ($r2.Answer -replace "\s+", ' ')
        if ($shown2.Length -gt 100) { $shown2 = $shown2.Substring(0, 100) }
        Write-Output ('  single V4  {0} | cache {1}% | TTFT {2} | answer: [{3}]' -f $row.single_hit, $r2.Cache, $r2.TTFT, $shown2)
    } catch { $row.note += 'needleFail;'; Write-Output ('  8-needle FAILED line=' + $_.InvocationInfo.ScriptLineNumber + ': ' + $_.Exception.Message) }

    # --- 4) 32K unique-segment prefill (no prefix reuse possible) ------------
    try {
        $uniq = Build-UniquePrompt -TargetTokens $PrefillTokens
        $r = Invoke-Chat -Url $chatUrl -LogPath $errLog -Body @{
            model = $modelId
            messages = @(@{ role = 'user'; content = $uniq })
            max_tokens = 64; stream = $false
        }
        $row.prefill32k = $r.Prefill
        Write-Output ('  32K prefill {0} tok/s | TTFT {1}' -f $r.Prefill, $r.TTFT)
    } catch { $row.note += 'prefillFail;'; Write-Output ('  32K prefill FAILED: ' + $_.Exception.Message) }

    # --- 5) 4-turn chat, fact from turn 1 asked at turn 4 --------------------
    try {
        $secret = 'LOTUS-7391'
        $messages = @(
            @{ role = 'system'; content = 'You are a careful assistant. Follow each instruction exactly.' },
            @{ role = 'user'; content = ('My access code is {0}. Remember it. Reply with exactly: ACK-1' -f $secret) }
        )
        $okTurns = 0
        $cacheTurns = 0
        $answers = @()
        for ($turn = 1; $turn -le $Turns; $turn++) {
            if ($turn -gt 1) {
                if ($turn -eq $Turns) {
                    $messages += @{ role = 'user'; content = 'What is my access code? Reply with the code only.' }
                } else {
                    $messages += @{ role = 'user'; content = ('Reply with exactly: ACK-{0}' -f $turn) }
                }
            }
            $r = Invoke-Chat -Url $chatUrl -LogPath $errLog -Body @{
                model = $modelId; messages = $messages; max_tokens = 512
                stream = $false; temperature = 0.0
            }
            if ($r.Answer -ne '') { $okTurns++ }
            if ($turn -gt 1 -and $r.Cache -gt 0) { $cacheTurns++ }
            $answers += $r.Answer
            if ($r.Answer -ne '') { $messages += @{ role = 'assistant'; content = $r.Answer } }
        }
        $row.turns_ok = ('{0}/{1}' -f $okTurns, $Turns)
        $row.cache_turns = ('{0}/{1}' -f $cacheTurns, ($Turns - 1))
        $last = [string]$answers[-1]
        $row.cross_turn_recall = if ($last -match [regex]::Escape($secret)) { 'HIT' } else { 'MISS' }
        Write-Output ('  turns {0} | cache {1} | cross-turn recall {2}' -f $row.turns_ok, $row.cache_turns, $row.cross_turn_recall)
    } catch { $row.note += 'turnFail;'; Write-Output ('  turns FAILED: ' + $_.Exception.Message) }

    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Seconds 3
    $rows += [pscustomobject]$row
    $rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
    Write-Output ''
}

Stop-Orphans
Write-Output '================ PARAM A/B SUMMARY ================'
$rows | Format-Table arm, B, R, sink, recent, ready_s, short_decode, short_mtp, needle_hits, single_hit, prefill32k, turns_ok, cache_turns, cross_turn_recall, note -AutoSize | Out-String | Write-Output
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
Write-Output ('summary csv = ' + $csv)
Write-Output 'RESULT: PARAM-AB-DONE'
