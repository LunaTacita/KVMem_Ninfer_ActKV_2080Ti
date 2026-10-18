# =====================================================================
# [发布版] 本脚本随发布包分发，已做脱敏：
#   - 引擎 / 日志 / 缓存路径都是相对本脚本（Start\）的路径
#   - 模型路径留空，需要你填（见下方 $MODEL 一行）
#   - 端口默认 8080（llama.cpp 默认），按需修改
# 首次使用：把 $MODEL 改成你的 .ninfer 模型绝对路径（全 ASCII），保存，
# 然后执行：powershell -ExecutionPolicy Bypass -File .\启动_KVMem_Ninfer_ActKV_2080Ti.ps1
# =====================================================================
# =====================================================================
# KVMem_Ninfer_ActKV_2080Ti v0.1 启动脚本
# =====================================================================
# =====================================================================
# 【30 秒上手速查】
#   1) 只改一处：$MODEL = "你的 .ninfer 模型绝对路径"（全 ASCII；见 §一）
#   2) 常用档位：$KV_PROFILE = "long"（1M 长上下文，默认）/ "dual"（双并发）/ "max8"（8 并发）；
#      三档的 (B+R)×并发 都等于 38400 token（见 §四）
#   3) 端口：$PORT = "8080"；被占用时只改这一行（见 §二）
#   4) 起服：powershell -ExecutionPolicy Bypass -File .\启动_KVMem_Ninfer_ActKV_2080Ti.ps1
#   5) 其余参数保持默认即可 —— 每个参数都按【是什么 / 为什么有它 / 取值与后果 / 实测结论】四段注释
#
# 【章节索引】15 节 + 1 个补充节，按功能分组：
#   §一  基础路径与模型        §二  网络与 API          §三  上下文/并发/批次
#   §四  KVMem 核心 B/R/H      §五  KVMem 检索与打分    §六  KVMem 磁盘分层
#   §七  KV 格式与容量控制     §七·补 ActKV 三支柱      §八  上下文缓存/前缀复用
#   §九  投机解码（MTP/n-gram）§十  多模态（图像/视频） §十一 采样与思考
#   §十二 精度与内核           §十三 提示词移植/API 行为§十四 路由器/模型休眠
#   §十五 日志
#   之后：参数拼装（留空即不传）→ 启动（横幅 + PS 5.1 兼容 + 进程日志）
# =====================================================================
# 【这个脚本是干什么的】
#   一条命令启动 ninfer-serve.exe，在 sm_75（RTX 2080 Ti 11GB）
#   上跑起本引擎，对外提供 OpenAI 兼容 API。
#
# 【它同时打开的三件核心武器】
#   1) KVMem  —— 让 27B 模型在 11GB 卡上用 1M 逻辑上下文：
#                历史放主机内存/SSD，显存里只放"当前要看的窗口"。
#   2) MTP    —— 投机解码：模型自带小头先猜 2 个 token，主模型一次验证；
#                实测 decode 从 44.9 提到 80.3 tok/s。
#   3) int8 KV—— 每 token KV 约 30 KiB（bf16 是 68 KiB），省一半显存且 PPL 只差 0.00072。
#
# 【怎么用】
#   在 PowerShell 里执行：.\启动_KVMem_Ninfer_ActKV_2080Ti.ps1
#   看到 "API : http://..." 后即可请求；Ctrl+C 停止。
#   只想改常用档位 → 只动 $KV_PROFILE（长上下文/双并发/8并发三选一）。
#
# 【硬件参考（构建机实测，不是通用值）】
#   CPU : i9-9900KF (8 物理核 / 16 逻辑核, 有 AVX2 无 AVX-512)
#   GPU : RTX 2080 Ti 11GB (sm_75, 11264 MiB, 驱动 566.36)
#   RAM : 64GB (物理 63.9 GB)
#   盘 : 系统盘 NVMe + 数据盘 HDD（交付目录在 HDD 上；把 Cache 改到 NVMe 会明显更快）
#
# 【API 地址】
#   http://localhost:8080/v1/chat/completions   (OpenAI 兼容)
#   http://localhost:8080/v1/models            (取 model 名，必须是这里的 id)
#   http://localhost:8080/v1/load              (容量/KVMem 状态自证)
#   http://localhost:8080/health
#
# ★★ 三条铁律（违反会直接起不来或静默答错）★★
#   1. KVMem 模式下【绝对不要传 --kv-capacity】（任何形式，含 auto）。
#      引擎自己按 (B+R)×并发 设定容量，传了就报错退出。
#   2. --prefill-chunk 必须是 128 的倍数，且 <= --kvmem-gen-reserve。
#   3. --kvmem-budget 与 --kvmem-gen-reserve 都必须是 64 的倍数；
#      --kvmem-budget >= 128，--kvmem-gen-reserve >= 64。
#
# ★ 参数书写约定：
#   所有参数「留空即完全不传」——包括布尔型开关：变量留空=""则不传该 flag，
#   填任意非空值（如 "1"）则传该 flag。这样默认值完全由引擎决定，脚本不覆盖。
#
# ★ 注释写法约定：
#   每个参数都按【是什么】【为什么有它/什么时候该动】【取值与后果】【实测结论】写，
#   而不是只把参数名翻译一遍。参数名一律以 ninfer-serve.exe --help 实测输出为准。
# =====================================================================

# ============================================================
# 一、基础路径与模型
# ============================================================
$EXE        = Join-Path $PSScriptRoot "..\engine\ninfer-serve.exe"
#   【是什么】引擎可执行文件路径。
#   【什么时候改】几乎不用改。路径错会直接启动失败（报 CreateFileW / 找不到文件）。
$MODEL      = ""   # ★ 必填：你的 .ninfer 模型绝对路径（解压路径必须全 ASCII）
#   【是什么】主模型制品：去审查的 Bonsai（Ternary-Bonsai-2-27B），NInfer v3 容器，自含 text/vision/mtp/dflash2。
#   【为什么是 .ninfer 不是 .gguf】这套引擎只认 NInfer v3 容器；旧格式/gguf 加载不了。
#   【路径要求】解压/模型路径必须全 ASCII（原生 exe 读不了中文路径）。
#   【加载耗时】9.5GB 权重，NVMe 上约 5~9 秒，机械盘冷缓存可能 40 秒（日志会停在 loading，不是死锁）。
$MODEL_ID   = "bonsai2-27b-heretic"
#   【是什么】对外暴露的模型名（API 请求里 model 字段的值）。
#   【为什么需要】客户端必须用这个名字请求；写错会 404（model_not_found）。
#   【建议】客户端第一次先 GET /v1/models 拿准确 id。
$LOG_DIR    = Join-Path $PSScriptRoot "..\logs"
#   【是什么】日志目录。每次启动生成 server_<时间>.out.log / .err.log。
#   【为什么重要】引擎自己的进度、容量行、每个请求的统计（req#N done：TTFT/prefill/decode/缓存命中）
#     都在 err.log 里；排查问题第一现场就是它。
$CACHE_DIR  = Join-Path $PSScriptRoot "..\Cache"
#   【是什么】KV/会话磁盘缓存根目录（被下面 $KV_DISK_PATH 使用）。
#   【效果】E: 是机械盘，改到 NVMe 上会明显加快溢出读写；进程退出时缓存自动清理。

# ============================================================
# 二、网络与 API
# ============================================================
$BIND_HOST            = "0.0.0.0"
#   【是什么】服务监听地址。0.0.0.0 = 局域网可访问；127.0.0.1 = 只本机能连。
#   【什么时候改】只自己用 → 127.0.0.1 更安全；要给局域网其他机器用 → 0.0.0.0（注意无认证风险）。
#   【★ 为什么变量叫 $BIND_HOST 不叫 $HOST】$Host 是 PowerShell 的只读自动变量，
#     写 $HOST = ... 会直接报 "Cannot overwrite variable Host"（已踩过）。
$PORT                 = "8080"
#   【是什么】服务监听端口。被占用会启动失败；1024~65535 之间选。
#   【默认值】8080（llama.cpp 默认端口）；需要时改这一行。
$STATS_PORT           = ""
#   【是什么】额外的健康/统计端口（与主端口分开，方便监控）。
#   【留空=关闭】想让 /health /stats /metrics 走单独端口时才填（如 60009）。
$API_KEY              = ""
#   【是什么】API 认证密钥。填了之后请求必须带 Bearer/x-api-key，否则 401。
#   【什么时候填】监听 0.0.0.0 或要防止别人乱用时；只本机用留空即可。
$CORS                 = ""
#   【是什么】是否发送宽松 CORS 头（允许浏览器网页直连 API）。
#   【留空=不发 / "1"=发】只在浏览器前端要直连时打开。
$NO_WEBUI             = ""
#   【是什么】是否关闭内置 WebUI（GET / 返回的页面）。
#   【留空=提供 / "1"=关闭】不需要网页界面时关掉，少一点暴露面。
$WEBUI_MCP_PROXY      = ""
#   【是什么】WebUI 的 MCP 代理（让网页能代访问外部工具）。
#   【★ 风险】/cors-proxy 可代访问任意 http 目标且不带密钥，只在可信网络开启。
$MAX_REQUEST_MIB      = ""
#   【是什么】请求体大小上限（MiB，默认 384）。
#   【效果】超过在解析前拒绝，防止超大 prompt 打爆内存。
$MAX_PENDING_REQUESTS = ""
#   【是什么】并发满了之后，最多允许几个请求排队（默认 16）。
#   【效果】超过就直接 503（拒绝），而不是无限排队占内存。
$PENDING_TIMEOUT_MS   = ""
#   【是什么】排队+准备阶段的超时（毫秒，默认 600000 = 10 分钟）。
#   【★ 什么时候要调大】长任务（长文档 prefill、思考很长）会被误判超时 503；
#     默认 10 分钟对 1M 上下文可能不够，长档场景建议 1800000（30 分钟）。
$RECOVER_INVARIANT    = ""
#   【是什么】引擎内部不变量损坏时，"只失败当前请求"还是"整个服务退出"。
#   【留空=关引擎（默认）/ "1"=保持服务】长跑服务建议开，避免一个坏请求拖垮整个进程。
$RESP_STORE_RECORDS   = ""
#   【是什么】/v1/responses 接口的有状态记录保留条数（默认 1024）。
#   【效果】超了按 LRU 淘汰旧记录；只影响有状态对话接口，普通 chat 不受影响。
$RESP_STORE_MIB       = ""
#   【是什么】/v1/responses 状态存储的内存预算（MiB，默认 256）。
#   【效果】超了淘汰最久未用的记录；太小会导致状态丢失。

# ============================================================
# 三、上下文、并发与批次
# ============================================================
$MAX_CONTEXT      = "1048576"
#   【这是什么】"这个服务最多允许多长的对话历史"，单位 token。默认设 1M。
#   【最容易误解的一点】这个数字【不占显存】。它只告诉引擎"历史最多记 1M"；
#     显存里永远只放可见窗口 (B+R)。所以 1M 不会 OOM，只是主机内存/SSD 要够。
#   【设大/设小的后果】
#     · 设小了：长文档/长会话到上限被截断（历史被丢弃）。
#     · 设大了：只多占主机内存/SSD，不占显存；请求准备/记账略多一点开销。
#   【怎么选】日常对话 262144（256K）足够；要喂整本书/超长日志 → 保持 1048576。
$MAX_CONCURRENCY  = "1"
#   【这是什么】同时能解码几个请求（几路并发）。
#   【最关键的取舍：并发和窗口是零和的】设备池大小 = (B+R)×并发，而安全上限 (B+R)×并发 <= 38400。
#     并发 1 → 窗口 38400；并发 2 → 每路 19200；并发 4 → 每路 9600。
#     所以"多开几路"不是免费加速：每一路的记忆力都会变小。
#   【什么时候改】只自己一个人用 → 1（默认，窗口最大）；两三个客户端同时连 → 2。
$PREFILL_CHUNK    = ""
#   【这是什么】prefill（读提示词）时每次处理多少 token 一块。
#   【取值约束】必须是 128 的倍数、且 <= R（生成预留）。默认 1024。
#   【效果】越大，prefill 越快；但需要更大的 R 撑住。默认 R=5632 时一般不用动。
#   【什么时候改】极限 prefill 速度可试 2048；改了要同步确认 R 够。
$DEFAULT_MAX_TOK  = ""
#   【这是什么】请求没写 max_tokens 时，默认最多生成多少 token。
#   【★ 最容易踩的坑：thinking 也算这个预算】模型先"想"再答，想的 token 从 max_tokens 里扣。
#     给太小（如 32/64）会出现"回答为空、finish_reason=length"——这不是召回失败，是预算被思考吃光。
#   【怎么选】留空由引擎按可用空间推导；做判据时请求里显式给 512~1024。

# ============================================================
# 四、KVMem 核心：B / R / H / 会话数 / sink / recent
# ------------------------------------------------------------
# ★★ 这四个量决定"显存里能看多少历史、看得有多快"★★
#   B = --kvmem-budget      ：模型每一步能同时"看到"的历史 token 数（可见集/记忆窗口）
#   R = --kvmem-gen-reserve ：给"正在生成/正在读入"预留的空间；它直接决定 prefill 速度
#   H = --kvmem-host-mib    ：主机内存里能存多少历史（显存放不下的部分）
#   设备池大小 = (B+R)×并发 ÷ 64 页；RTX 2080 Ti 11GB 上的安全上限 = 38400 token（600 页）
#
# 【为什么 R 不能太小（同一 34.6K 提示词实测，重复一致）】
#   R=1536  → prefill 210~228 tok/s   （太小，prefill 慢一倍）
#   R=5632  → prefill 444~478
#   R=8192  → prefill 492             （prefill 最快档）
#   R=12288 → prefill 443             （已饱和，再大不再提速）
#   decode 几乎不随 B 变（40~48 tok/s）；放大 B 的代价主要落在 prefill。
#   ★ B=40960（664 页、free 0 B）能起服但 prefill 掉到 1.8 tok/s ⇒ "能起服"不等于"能用"。
#
# 【参照值（16 GiB 卡上的公开推荐配置，本卡 11 GB 放不下）】
#   公开推荐值（16 GiB 卡）：IQ3：B=36864/R=16384；IQ4：B=32768/R=12288；sink 12288/recent 1024。
#   ★ 上述 B+R（45056/53248）在 11 GB 卡放不下；本卡安全上限 B+R <= 38400。
#
# 【三档（三档的 (B+R)×并发 都等于 38400）】
#   long ：B=32768 R=5632 并发1  长上下文（默认）
#   dual ：B=16384 R=2816 并发2  双并发（同池总量，每路减半）
#   max8 ：B=3264  R=1536 并发8  极限8并发（每路只剩 4800，且 R 在 prefill 最差区间，慢）
#   想 prefill 最快（可见集降到 28K）：手动 long 档改 B=28672/R=8192（实测 492 tok/s）。
# ============================================================
$KV_PROFILE       = "long"
#   【这是什么】一档总开关：自动设定 B、R 与并发（三选一）。
#   【怎么选】单会话长文档 → long（默认）；两个客户端同时用 → dual；要 8 个并发小请求 → max8。
#   【注意】改了它就不用再改下面 B/R/并发（switch 会覆盖）。
switch ($KV_PROFILE) {
    "long"  { $KV_BUDGET = "32768"; $KV_GEN_RESERVE = "5632"; $MAX_CONCURRENCY = "1" }
    "dual"  { $KV_BUDGET = "16384"; $KV_GEN_RESERVE = "2816"; $MAX_CONCURRENCY = "2" }
    "max8"  { $KV_BUDGET = "3264";  $KV_GEN_RESERVE = "1536"; $MAX_CONCURRENCY = "8" }
    default { throw "KV_PROFILE must be long | dual | max8" }
}
$KV_HOST_MIB      = "auto"
#   【这是什么】主机内存里给历史 KV 的预算 H。取值：MiB 或 "auto"。
#   【auto 是多少】物理内存 − 1 GiB（64 GB 机器上实测 64421 MiB）。
#   【什么时候改】想让出内存给别的程序 → 显式调小（如 32768）；一般保持 auto。
$KV_SESSIONS      = "4"
#   【这是什么】主机层里同时保留几个会话的历史（多 Agent/多客户端场景）。
#   【效果】多会话能各自恢复前缀；越多越吃主机内存（每会话一份历史）。
#   【怎么选】单人用 → 4 足够；多个长期 Agent 并行 → 8/16。
$KV_SINK_TOKENS   = "4096"
#   【这是什么】"永久保留的开头"：从第 1 个 token 起，前 N 个 token 永不驱逐（计入 B）。
#   【为什么需要它】system 提示、任务描述、角色设定通常在最前面；按滑窗会被挤掉。
#     钉住它们，长对话中模型"永远记得自己是谁、要干什么"。
#   【怎么定 4096】实测一份 1.6K~2.2K token 的系统提示/任务描述能完整容纳；
#     设太大（如 12288）会占掉 B 的三分之一，挤压真正需要检索的历史。
#   【取值】64 的倍数；默认 0=只保 1 页。
$KV_RECENT_TOKENS = "1024"
#   【这是什么】"永久保留的结尾"：最近 N 个 token 永不驱逐（计入 B）。
#   【为什么需要它】避免"刚说的话模型就忘"——最近一轮对话一定在窗口里。
#   【怎么选】一般 1024 足够；长回答场景可加到 2048。

# ============================================================
# 五、KVMem 检索与打分（决定"旧历史里哪些片段值得换回显存"）
# ============================================================
# 【先理解问题】显存里只有 3.84 万 token 的窗口，但 1M 历史都在主机/SSD。
#   每一轮对话引擎都要挑"哪 3.84 万个 token 换进来"。挑法有两层：
#     1) 用当前问题去和每个旧块算相似度（见第七·补节的 ActKV 打分器）
#     2) 记"这个块历史上被用过多少次"（本节的 LRFU）
#   两者叠加：LRFU 不改变"当前问题"的相似度，只给"长期被反复用到"的块加权。
# 【实测】8 针 4 万 token 判据下 LRFU 开/关都是 8/8 全中、速度无差异
#   ⇒ 生产默认关。什么时候值得开：长会话里反复回到同一主题（长期项目助手）。
# ============================================================
$KV_LRFU          = ""
#   【是什么】LRFU = Least Recently/Frequently Used，把"最近用过"和"被用过多少次"合成一个分数。
#   【为什么有它】纯相似度检索会丢掉"虽和当前问题不直接相关、但这个会话一直在用"的块
#     （典型：长期项目的背景文档、反复引用的配置）。开它让这类块不容易被淘汰。
#   【留空=关 / "1"=开】开它不需要别的参数（衰减系数等用默认即可）。
#   【代价与实测】8 针判据下无差异；开启后引擎每轮多做一次衰减+排序（很小）。
$KV_LRFU_LAMBDA   = ""
#   【仅当 LRFU 开着才有效】历史分量的衰减速度：每过一轮，旧的使用记录乘多少。
#   【取值 0~1，默认 0.5】0.9 = 记很久（偏重"历史上常用"）；0.1 = 很快忘（偏重"最近常用"）。
#   【怎么选】长期项目助手 → 0.7~0.9；短期多变任务 → 0.3~0.5（默认即可）。
$KV_LRFU_TOPF     = ""
#   【仅当 LRFU 开着才有效】一个块算"被用过一次"的门槛：相似度要排进前多少比例。
#   【取值 (0,1]，默认 0.9】0.9 = 排进前 90% 就算用过（宽松）；0.5 = 必须前一半（严格）。
#   【为什么有它】防"擦边命中"也积累历史分，导致噪音块越用越顽固。
$KV_SCORE_AGG     = ""
#   【是什么】模型有多个注意力头（GQA），每个头对同一个块给出不同相似度；这里决定怎么合成一个分。
#   【取值 mean（默认）/ max】mean = 所有头平均（稳）；max = 取最相关的那个头（敏感）。
#     直觉：某个头恰好抓到关键词时 max 分更高，但噪音更多；mean 更保守。
#   【什么时候改】基本不用。若发现"偶尔能找回但很不稳定"可试 max；否则保持 mean。
$KV_VERIFY_XFER   = ""
#   【是什么】恢复 KV 时逐字节校验"从主机/SSD 搬回显存的数据没坏"。
#   【为什么平时关】诊断开关，开启后显著变慢（每次换入换出多一次比对）。
#   【什么时候开】怀疑"恢复的数据有问题"（同问题两次答案不一致、长上下文偶发错乱）时临时开。
#   【留空=关 / "1"=开】

# ============================================================
# 六、KVMem 磁盘分层（主机内存也装不下时，把历史溢到 SSD）
# ★ 默认落在解压目录 Cache；放到 NVMe 盘会明显加快换入换出
# ============================================================
$KV_DISK_PATH     = Join-Path $PSScriptRoot "..\Cache\kvmem"
#   【是什么】历史 KV 溢出到磁盘时写哪个目录。必须与 $KV_DISK_MIB 成对给。
#   【什么时候会用到】只有主机内存（$KV_HOST_MIB）也装不下时才写盘；
#     1M int8 历史约 29 GiB、主机预算约 63 GiB ⇒ 单会话通常不溢盘，多会话/超长才用到。
#   【生命周期】进程内有效，进程退出即删（不会长期占盘）。
#   【盘速影响】换入换出速度 = 盘速；放到 NVMe 盘上会明显更快。
$KV_DISK_MIB      = "102400"
#   【是什么】磁盘层最多能用多少 MiB（默认 100 GiB）。防止异常情况下写爆盘。
#   【怎么选】≥ 你预期的最坏溢出量；1M int8 约 29 GiB，100 GiB 足够。
$DISK_KV_PATH     = ""
#   【是什么】"上下文缓存"（另一套机制）的磁盘目录：被驱逐的会话 KV 写这里，可跨重启恢复。
#   【和上面 $KV_DISK_PATH 的区别】$KV_DISK_PATH 是 KVMem 的"临时溢出"，进程退出即删；
#     这个 $DISK_KV_PATH 是"持久缓存"，重启后还能恢复会话前缀。两者不要混。
#   【留空=不启用】需要"重启后少重算"时才填。
$DISK_KV_GIB      = ""
#   【仅当填写 $DISK_KV_PATH 时有意义】持久缓存最多用多少 GiB（默认 64）。
$DISK_KV_RESTORE  = ""
#   【仅当填写 $DISK_KV_PATH 时有意义】启动/请求时从持久缓存恢复匹配前缀（少重算）。
#   【留空=关 / "1"=开】
$DISK_KV_DS       = ""
#   【仅当填写 $DISK_KV_PATH 时有意义】用 DirectStorage 读持久缓存（更快）。
#   【前提】构建时带了 NINFER_DIRECTSTORAGE；否则开了也没用（引擎会忽略）。

# ============================================================
# 七、KV 存储格式与（非 KVMem 才可用的）容量控制
# ============================================================
$KV_DTYPE         = "int8"
#   【这是什么】KV 缓存每个 token 存成什么精度——直接决定"同样显存能装多少历史"。
#   【为什么选 int8】实测（B/R/1M/MTP 同配置）：
#     bf16   68 KiB/token → 设备预留 3.18 GiB → 权重后只剩 2.75 GiB，起不来
#     int8   ~30 KiB/token → 1.79 GiB → 能起，PPL 只 +0.00072（小于引擎自身噪声）
#     fp8    ~30 KiB → 1.76 GiB，PPL +0.00078（sm75 上 fp8 走 SIMT，不划算）
#     rk8v4/k8v4 ~22 KiB → 更省，但 PPL +0.005（明显变差）
#   【结论】int8 是"能起、精度损失在噪声内、速度正常"的唯一选择。
#   【容量上限】int8 下可见窗口实测最大约 67200 token（再大引擎会拒绝并报上限）。
$KV_CAPACITY      = ""
#   ★★ KVMem 模式下必须留空（传任何值含 auto 都会启动报错）★★
#   【为什么】KVMem 自己按 (B+R)×并发 设定设备容量；手动指定会和它算出来的值冲突。
$KV_HEADROOM_MIB  = ""
#   【仅非 KVMem 的 --kv-capacity auto 有意义】给别的程序留多少 MiB 显存；KVMem 下不可用。

# ============================================================
# 七·补、ActKV 三支柱
# ------------------------------------------------------------
# 【先讲清它在解决什么】
#   1M 上下文不等于"显存里能同时看 1M"。显卡一次只能看 3.84 万 token 的窗口，
#   所以每一轮都要决定：历史的哪 3.84 万 token 换进显存？
#   朴素做法有两种，都有病：
#     · 只看最近（滑窗）→ 开头的系统提示、任务描述、关键密码被挤掉，模型"失忆"。
#     · 只看高频（频率）→ 多轮 Agent 里任务指令每轮一字不差，检索退化成只记高频词。
#   ActKV 的三根支柱就是三处改进：
#     ①（自动，无参数）检索 query 改用"模型最新做的动作"，并把 system/任务描述/媒体/最近对话
#        设为永不驱逐 → 解决"每轮检索同一批东西"和"关键内容被挤掉"。
#     ②（默认关，$KV_CONF_*）每步多算一个置信度；模型越不确定，越把可见窗口放大
#        → 解决"该看的没看到、模型在猜"。
#     ③（默认 mean-k，$KV_RETRIEVAL_METHOD）决定"哪些块值得换回来"的打分方式
#        → 解决"块太粗，一个相关小块被整块平均稀释"。
# 【实测结论（为什么三根都保持默认）】
#   判据 = 8 个密码散布在 4 万 token 的 5%~89% 位置、一次请求全部列出。
#   · 默认（①开、②关、③mean-k）：8/8 全中，短 decode 94.1 tok/s
#   · ActKV ②③+LRFU 全开：也是 8/8，但短 decode 93.2（−1%）
#   ⇒ 默认配置已满分、全开净亏 1% ⇒ 生产保持默认；下面参数是"以后遇到更难场景再开"的开关。
# ============================================================

# --- 支柱③：打分器（决定"哪些旧块值得换回显存"）---
$KV_RETRIEVAL_METHOD = ""
#   【默认 mean-k 是什么】把一个 64 token 的块压成 1 个平均向量，用当前问题去和它比相似度。
#     最快、最省；现有判据下已经足够。
#   【什么时候该换】"明显该找回的内容找不回来"（长对话丢早期事实、多针召回丢针），
#     且算力/显存有余时。
#   【四个取值与代价】
#     mean-k   （默认）：每块 1 个代表向量；最快
#     per-token        ：每块保留全部 token 向量；最准，开销最大（实测短 decode −1%）
#     sub-block        ：每块切 4 个子块分别比，取最相关；精度/开销居中
#     deltanet         ：上游标 EXPERIMENTAL，不建议生产用
#   【留空 = mean-k】
$KV_SUBBLOCKS         = ""
#   【仅 $KV_RETRIEVAL_METHOD=sub-block 时有效】每个 64 token 的块再切成几个子块。
#   【为什么切】一个块里可能只有 8 个 token 和问题相关，整块平均会被稀释；
#     切成子块后可以"只命中相关的那一小段"。
#   【取值】2 的幂（1/2/4/8/16/32/64）；4 是常用折中；1 = 不切（退化成 mean-k）。
$KV_SUBBLOCK_REDUCE   = ""
#   【仅 sub-block 时有效】子块分数怎么合成块分数：
#     max（默认）：取最相关的那个子块 → 防止高相关小块被无关子块平均掉
#     sum        ：所有子块相加 → 适合"一个块里有多个相关片段"的场景
$KV_PROTOTYPE         = ""
#   【块内代表向量怎么造】contiguous（默认，按存储顺序取均值）
#     / key-direction-4 / key-direction-adaptive（按 key 方向聚类再分别取代表，
#     适合"一个块混了多个话题"，代价是建索引更慢更吃内存）。基本不用改。

# --- 支柱②：置信度预算（模型不确定时，自动把可见窗口放大）---
# 【它到底在做什么，用一句话】每步算一下"模型有多确定"；连续不确定就在轮边界把窗口放大一点。
# 【详细流程】
#   1) 每生成一个 token，引擎顺手算 top-20 候选的概率（= 置信度）；
#   2) 把最近 64 个 token 的置信度取最小值（滑窗最小池化），得到"最差情况曲线"；
#   3) 只看最近这一段里最低的 10%，拟合"是否还在下降"；
#   4) 若判定"不确定在恶化"，就在【轮边界】把可见窗口乘 1.75（只增不减），冷却 4 轮；
#   5) 上界不超过 B（= 32768）。
# 【什么是"轮边界"】= 一次 chat 请求结束、下一次请求开始的位置。
#   为什么不在生成中途改：窗口一变，KV 布局/缓存都要重排，中途改会抖动、变慢。
# 【代价】每步多算 top-20 logprob ⇒ 实测短 decode 94.1→93.2 tok/s（约 −1%）。
# 【★ 开启时必须设 $KV_CONF_INIT < B，否则起点=B 没有放大空间，等于白开】
$KV_CONF_BUDGET       = ""
#   【总开关】留空=关 / "1"=开。不开时下面所有 $KV_CONF_* 全部无效。
$KV_CONF_INIT         = ""
#   【起点：平时用多少窗口】单位 token，64 的倍数，必须 < B（留空=等于 B ⇒ 白开）。
#   推荐配法：B=32768 时取 16384（平时用一半，不确定时最多放到 32768）。
$KV_CONF_UPPER        = ""
#   【上限：最多放大到多少】<= B（留空=B）；超过 B 引擎拒绝启动。
#   一般就填 B（如 32768）= "不确定时可以看满窗口"。
$KV_CONF_POOL         = ""
#   【趋势怎么算：滑窗长度】最近多少个 token 的置信度收进一个窗口取最小值（默认 64）。
#   越小越敏感（易误判不确定），越大越迟钝。
$KV_CONF_STRIDE       = ""
#   【滑窗多久移动一次】每多少个 token 采一个窗口最小值（默认 8）。越小越灵敏、算得越多。
$KV_CONF_BOTTOM       = ""
#   【只看最差的多少】用最近一段里置信度最低的 b%（默认 0.10 = 10%）拟合"是否下降"。
#   为什么不全看：正常波动很大，只看最差尾部才能抓到"真的在慌"。
$KV_CONF_MIN_POOL     = ""
#   【要几个点才下结论】拟合趋势最少需要几个池化点（默认 8）；点不够这轮不动作。
$KV_CONF_SCALE_UP     = ""
#   【一次放大几倍】判定"不确定"后，把窗口乘这个倍率（默认 1.75）。
#   举例：起点 16384 ×1.75 ≈ 28672，再不够就顶到上限 32768。
#   取值：1.0=不放大（等于关掉动作）；1.5~2.5 合理；越大越激进、越容易吃满 B。
$KV_CONF_COOLDOWN     = ""
#   【两次放大之间至少隔几轮】默认 4。作用：放大后先观察几轮，不要每轮都加。

# ============================================================
# 八、上下文缓存 / 前缀复用（★ 只有"非 KVMem 模式"才有效）
# ============================================================
# 【先说结论】本节参数在 KVMem 模式下基本都留空。
# 【为什么】KVMem 自己接管了"历史复用"（host 层 + 窗口机制）；
#   本节是引擎另一套"兼容前缀缓存"，与 KVMem 是替代关系。
# 【唯一值得知道的一个】$NO_PREFIX_REUSE = "1" 会关闭前缀复用：
#   多轮对话每轮都重算整段历史 → TTFT 从几百毫秒涨到几十秒。只在排查缓存问题时临时开。
# ============================================================
$NO_PREFIX_REUSE  = ""
#   【是什么】关闭"兼容前缀缓存"。留空=开 / "1"=关。
#   【为什么有关它】调试用：怀疑"复用了旧数据导致答案不对"时关掉，让每轮从零重算。
#   【代价】多轮对话每轮重算全部历史：TTFT 从几百毫秒 → 几十秒。
#   【日常】保持留空（=开）；KVMem 模式下这套本来也不生效。
$DEV_STATE_SLOTS  = ""
#   【是什么】设备侧检查点槽数（默认=并发数）。每个槽 = 一份可复用的历史状态快照。
#   【效果】多几个槽 → 更多历史能快速复用；代价是每槽占一份显存。
$HOST_STATE_SLOTS = ""
#   【是什么】主机侧检查点槽数（默认 8）。用内存换复用率。
$HOST_KV_MIB      = ""
#   【是什么】固定主机 KV 容量（MiB，默认 8192）。这是"上下文缓存"的 host 层；与 $HOST_CACHE_MIB 二选一。
$HOST_CACHE_MIB   = ""
#   【是什么】单一 host 内存上限（替代上面两项；默认 8192；0=缓存只留设备）。
#   【效果】引擎自动在"状态槽"与"host KV"之间分配这份预算。
$MAX_PRIV_CONT    = ""
#   【是什么】私有续轮目录深度（默认 2×并发）。多轮会话最多记住多少条"续轮来源"。
$MAX_SHARED_PREF  = ""
#   【是什么】共享稳定前缀目录深度（默认 max(并发,7)）。公共前缀可复用多少份。
$MAX_LONG_ANCHOR  = ""
#   【是什么】每个续轮保留的长锚点数（默认 2）。
#   【效果】用户改写历史时从最近锚点续跑，少重算前面一大段。
$AUTO_LONG_ANCHOR = ""
#   【是什么】按消息边界自动打锚点。留空=关 / "1"=开。长对话编辑场景有用。
$BRANCH_ANCHORS   = ""
#   【是什么】在提示分叉处打锚点。留空=关 / "1"=开。同源对话分叉后能各续各的。
$NO_ENDPOINT_ANC  = ""
#   【是什么】不保留"终点锚点"。留空=保留 / "1"=不保留。
#   【效果】保留时改最后一条消息可续算；设为 1 则改最后消息会重 prefill（一般留空）。
$LONG_ANCHOR_GAP  = ""
#   【是什么】自动锚点最小间距（token，默认 1024；0=每个边界都打）。需配 $AUTO_LONG_ANCHOR。
$MAX_CACHE_MARKS  = ""
#   【是什么】单请求允许的客户端缓存标记上限（默认 4），超过的标记被忽略。
$AUTO_PREFIX_GRID = ""
#   【是什么】按 token 网格自动提供共享候选。留空=关 / "1"=开。客户端不给提示也能共享前缀。
$DERIVE_SESS_KEYS = ""
#   【是什么】为没有 session key 的请求推导会话键。留空=关 / "1"=开。让多轮自动归同一谱系。
$CTX_CACHE_POLICY = ""
#   【是什么】上下文缓存策略。取值 default / rolling。
#   【怎么选】rolling 适合"只增长不回退"的单会话（连续追问）；default 通用。
$REL_DIVERGED     = ""
#   【是什么】压力下优先释放"已分叉"的私有检查点。留空=关 / "1"=开。省内存。
$THOROUGH_ADMIT   = ""
#   【是什么】更彻底的可复用性搜索（最多多搜 250ms）。留空=关 / "1"=开。
#   【代价】每个请求最多多等 250ms；高并发/低延迟场景不建议。
$RECENCY_EVICT    = ""
#   【是什么】压力下优先驱逐最久未用者。留空=关 / "1"=开。与 $VALUE_AWARE 二选一。
$VALUE_AWARE      = ""
#   【是什么】优先驱逐"重建成本最低"的检查点。留空=关 / "1"=开。与上一个二选一。
$KV_LEASE_GROWTH  = ""
#   【是什么】预留 4096 输出窗并按需扩展。留空=关 / "1"=开。
#   【效果】池无法扩展时请求以 finish_reason=length 结束（而不是报错）。
$CONCURRENT_PREF  = ""
#   【是什么】并发 prefill。留空=关 / "1"=开。★ KVMem 独占设备池，二者互斥，KVMem 下开了会报错。
$SLOT_SAVE_PATH   = ""
#   【是什么】会话存档目录（POST /slots/... 保存/恢复会话状态到文件）。
$AUTO_SAVE_EVICT  = ""
#   【是什么】被驱逐前自动写回槽文件。留空=关 / "1"=开。需要 $SLOT_SAVE_PATH。
$ALT_PREFIX_CACHE = ""
#   【是什么】混合前缀缓存。★ 与 KVMem 互斥，KVMem 下不可用。
$ORIG_PREFIX_CACHE= ""
#   【是什么】显式选择默认检查点目录。留空=不传 / "1"=传。★ 与 KVMem 互斥。
$DEV_SNAP_SLOTS   = ""
#   【是什么】混合缓存的设备快照槽（默认 并发+1）。需要 $ALT_PREFIX_CACHE。
$CACHE_TAPS       = ""
#   【是什么】每个新 prefill 打多少快照（默认 8）。需要混合前缀缓存。
$CACHE_TAP_LADDER = ""
#   【是什么】快照阶梯基数（默认 max(4096, 2×prefill)）。同上。
$CACHE_TAP_GAP    = ""
#   【是什么】快照最小间隔（默认 max(1024, prefill)）。同上。
$PREFIX_CACHE_FILE= ""
#   【是什么】Host 层持久化文件路径。重启恢复/退出保存；需要 $ALT_PREFIX_CACHE。

# ============================================================
# 九、投机解码（MTP 草稿 / n-gram 草稿）
# ============================================================
# 【先用一段讲清"投机解码"是什么】
#   decode 是逐 token 生成的，每步都要跑一次完整模型 —— 慢。
#   投机解码的思路：先用"便宜办法"猜接下来几个 token，再让主模型一次验证；
#   猜中就直接跳过那几步（白赚），猜错就回退重算（白算一轮）。
#   本栈有两类"便宜办法"：
#     MTP     ：模型自带的小预测头（真的"学过"预测下一个 token）→ 主力，+70%
#     n-gram  ：从"已经出现过的文本"里复制后续片段（对重复性输出有效）
#   【实测】MTP draft=2 → 80.3 tok/s / 接受率 91.4%（最优）；draft=4 → 66.8 / 52.1%；无 MTP → 44.9。
#   【为什么默认关闭】n-gram 关闭（宽度 31 有对齐崩溃风险、默认路径也偶发）⇒ $NGRAM_DRAFT = "0"。
# ============================================================
$SPEC             = "mtp"
#   【是什么】投机解码后端。留空=关 / mtp / dflash / dflash2。
#   【为什么用 mtp】实测 decode +70%（44.9 → 80.3 tok/s）。
#   【★ dflash/dflash2 与 KVMem 互斥】KVMem 模式下只能用 mtp（本包就是 KVMem 模式）。
$DRAFT_TOKENS     = "2"
#   【是什么】MTP 每轮一次猜几个 token（草稿宽度）。取值 1..15。
#   【★ 必须给】--spec mtp 必须同时给 --draft-tokens，否则引擎直接退出。
#   【实测怎么选的】同一二进制三档各重复 2 次：
#     draft=2 → 80.3 / 79.2 tok/s，接受率 91.4%   ← 最优
#     draft=3 → 75.3 tok/s，接受率 69.5%
#     draft=4 → 66.8 / 67.0 tok/s，接受率 52.1%
#   【为什么不是越大越好】草稿越靠后越容易被拒，被拒的要白算一遍；
#     可预测文本（重复/模板/代码）收益大，发散创作收益小。
$LM_HEAD_DRAFT    = ""
#   【是什么】用优化过的"提案头"起草。留空=关 / "1"=开。草稿质量可能更好（未做专项对比）。
$ADAPTIVE_MTP     = ""
#   【是什么】自适应每轮验证宽度（1..draft-tokens 之间按实测存活率挑）。留空=关 / "1"=开。
#   【前提】需要 $DRAFT_TOKENS >= 3 才有意义。
#   【实测】adaptive(draft4) 81.4 tok/s / 57.4%：比固定 4 略好，但仍不如 draft=2。
$MTP_ATTN_WINDOW  = ""
#   【是什么】MTP 草稿头的注意力窗口。★ KVMem 下只能留空或 0（0=全历史）。
$NGRAM_DRAFT      = "0"
#   【是什么】n-gram 草稿宽度（从历史文本复制草稿），取值 1..63。
#   【★ 默认关闭】显式传 0。
#   【为什么关】宽度 31 在 sm75 上触发 cudaErrorMisalignedAddress（第 63 层 attention）；
#     默认 15 也出现过测试偶发。赶发布不做专项，先关（MTP 已提供主要收益）。
#   【要重新开】留空（=引擎默认 15）或填 1..15；宽度 + draft_tokens 须 <= R。
$NGRAM_MIN_MATCH  = ""
#   【是什么】n-gram 复制所需最短匹配长度（默认 12）。越小越容易命中、越容易猜错。n-gram 关闭时无效。
$NGRAM_ARCHIVE    = ""
#   【是什么】已结束请求的复制源归档（MiB，默认 0=关）。跨请求复用同会话草稿。
$NGRAM_SESSION    = ""
#   【是什么】每会话归档份额（MiB，默认 128）。需要 $NGRAM_ARCHIVE。
$NGRAM_NATIVE     = ""
#   【是什么】识别 Kilo/Codex/Claude 的会话身份（自动归档）。留空=关 / "1"=开。
$LOOKUP_NGRAM     = ""
#   【是什么】旧式上下文查找起草。★ KVMem 禁用此旧参数，用 $NGRAM_DRAFT 代替。

# ============================================================
# 十、多模态（图像/视频输入）
# ============================================================
# 【先说结论】默认"开 vision + 视觉塔跑 CPU"：不占显存、不损文本性能。
#   （引擎 --vision 同时接受图像与视频；本包判据只实测了图像这一路）
#   依据：矩阵实测——int8 + vision CPU 时短 decode 94.0 tok/s、prefill 310.4、
#   40,960 token needle HIT，与不开 vision 完全同档；三张判据图（star/8/7429）全 HIT。
# ============================================================
$ENABLE_VISION    = "1"
#   【是什么】开启图像/视频输入。留空=关 / "1"=开。
#   【为什么默认开】实测不损文本性能（见上）；开了才叫"功能全"。
$VISION_RESIDENCY = "cpu"
#   【是什么】视觉塔（处理图片的那部分网络）放在哪里。
#   【三个取值】resident=占显存；overlay=放主机内存按图借显存；cpu=跑 CPU（最省显存）。
#   【为什么选 cpu】11GB 卡显存紧张；cpu 模式最省显存，实测文本性能不变。
$VISION_CPU       = ""
#   【是什么】等价于 --vision + --vision-residency cpu（一条代替两条）。
#   【留空=关 / "1"=开】与上面两行二选一即可（都填不冲突）。
$VISION_MAX_MERGE = ""
#   【是什么】单媒体合并 token 数上限（默认 16384）。太大图会被下采样细节变少；太小看不清小字。
$MEDIA_CACHE_MIB  = ""
#   【是什么】保留的已预处理媒体（MiB，默认 1024；0=禁用）。重复图片省预处理时间。
$MEDIA_LIVE_MIB   = ""
#   【是什么】活跃媒体负载（MiB，默认 2048）。并发图片请求的内存上限。
$MEDIA_THREADS    = ""
#   【是什么】媒体预处理线程数（0=自动，最多 16）。图片预处理并行度。

# ============================================================
# 十一、采样与思考
# ============================================================
# 【采样是什么】模型每步输出的是"下一个 token 的概率分布"；采样参数决定怎么从分布里挑。
#   temperature 越低越确定、越高越发散；top_p/top_k/min_p 先砍掉低概率候选再抽。
# 【默认策略】
#   采样值由"实际使用时"在请求里配置，脚本全部留空（= 引擎默认）；
#   只有"思考"按用户要求默认开启且最高档（$REASON_EFFORT = "max"）。
# 【★ 一个必须知道的坑】做判据/复现时要在请求里给 temperature=0（或 $GREEDY=1），
#   否则同一问题两次答案可能不同，看起来像"召回不稳定"。
# ============================================================
$TEMPERATURE      = ""
#   【是什么】温度（0..2）。0 = 每次选概率最高的（确定性）；越高越发散。
#   【怎么选】判据/复现 → 请求里 0；日常聊天 → 0.7~0.9（更像人，但不发散）。
#   【留空=用模型/思考模式的默认值】
$TOP_P            = ""
#   【是什么】核采样：只保留累积概率前 p 的候选再抽（0..1）。
#   【效果】越小越保守；0.9~0.95 是常见值。留空=引擎默认。
$TOP_K            = ""
#   【是什么】只保留概率最高的 K 个候选（0=不限）。留空=引擎默认。
$MIN_P            = ""
#   【是什么】min-p：去掉"概率低于最高概率×min_p"的候选。比 top_p 更自适应。留空=默认。
$PRESENCE_PEN     = ""
#   【是什么】存在惩罚（-2..2）：出现过的话题再出现会被压一点，抑制重复绕圈。
$FREQ_PEN         = ""
#   【是什么】频率惩罚（-2..2）：出现次数越多压得越狠，抑制同一词反复出现。
$SEED             = ""
#   【是什么】随机种子。填了 + temperature=0 可复现；不填每次新随机。
$GREEDY           = ""
#   【是什么】强制贪心（等价 temperature=0）。留空=关 / "1"=开。
#   【什么时候开】判据/回归必须开（否则答案不可复现）。
$POST_THINKING    = ""
#   【是什么】"思考结束后切换另一套采样参数"（答案段用更稳的参数）。留空=关 / "1"=开。
$PT_TEMP          = ""
#   【是什么】post-thinking 阶段温度。填了隐含开启 $POST_THINKING。
$PT_TOP_P         = ""
#   【是什么】post-thinking 阶段 top-p。填了隐含开启。
$PT_TOP_K         = ""
#   【是什么】post-thinking 阶段 top-k。填了隐含开启。
$PT_SAMPLER       = ""
#   【是什么】一行写全 post-thinking 采样。格式：temp=F,top_p=F,top_k=N[,min_p=...]。
$NO_THINKING      = ""
#   【是什么】默认关闭"思考"。留空=不关（思考开）/ "1"=关。
#   【★ 思考默认开启 ⇒ 这里留空（不关）。】
#   【代价】思考吃 max_tokens、增加延迟；复杂推理质量更高。
$PRESERVE_THINK   = ""
#   【是什么】多轮里保留"已闭合轮"的推理文本（上下文更完整但更费 token）。留空=关 / "1"=开。
$THINK_BUDGET     = ""
#   【是什么】思考预算上限（token）。超了模型被迫收尾。
#   【★ 默认最高思考 ⇒ 这里留空（不限预算）。】
$THINK_BUDGET_MSG = ""
#   【是什么】思考预算用尽时给模型的提示文案。留空=内置文案。
$REASON_EFFORT    = "max"
#   【是什么】默认推理强度。取值 none/minimal/low/medium/high/xhigh/max。
#   【★ 默认：思考开启且最高档 ⇒ 默认 "max"。】
#   【代价】短问答更慢、更费 token（thinking 计入 max_tokens）；单个请求要快时在请求里显式改低。

# ============================================================
# 十二、精度与内核（★ sm_75 有部分选项不可用，已逐项实测）
# ============================================================
# 【这一段控制什么】用哪种精度/哪个 kernel 算；主要影响速度和显存。
# 【默认原则】默认优先；下面是实测过的几个点：
#   · 快 prefill（$FAST_PREFILL）：矩阵实测无显著差异 → 保持关
#   · GDN 状态 fp16（$GDN_STATE_FP16）：decode 略快但 prefill −13% → 保持关
#   · CUDA graph 预留（$CUDA_GRAPH_ALLOW）：128 与默认无显著差异 → 保持留空
# ============================================================
$FAST_PREFILL     = ""
#   【是什么】快速 prefill 内核（int8/rk* KV 时可用）。留空=关 / "1"=开。
#   【实测】与默认无显著差异（decode 94.1 / prefill 309.8），保持关。
$PREFILL_CUBLAS   = ""
#   【是什么】宽 prefill GEMM 交给 cuBLAS。留空=关 / "1"=开。
#   【代价】PPL 略升、需要更大的 prefill-chunk。未做专项。
$NO_CUBLAS_PROJ   = ""
#   【是什么】cuBLAS 路线下保留注意力/GDN 输入投影。留空=不传 / "1"=传。需配合 $PREFILL_CUBLAS。
$NO_PREFILL_A8    = ""
#   【是什么】prefill 全部走 A16 路线。留空=不传 / "1"=传。★ sm_75 上 A8 本就不可用（永远走不到）。
$MLP_A8_DECODE    = ""
#   【是什么】decode 宽度下 MLP gate_up 用整数激活。★ Ampere+ 专属，sm_75 不可用。
$LM_HEAD_Q4       = ""
#   【是什么】加载时把输出头存成 Q4。留空=不传 / "1"=传。
#   【效果】省显存，代价 PPL 明显上升（约 +0.69%）；只在显存实在不够时用。
$LM_HEAD_Q6       = ""
#   【是什么】加载时把输出头存成 Q6。省显存、PPL 只 +0.01%。
#   【★ 本模型不可用】引擎要求 output_head 是 row-split Q8_G32；本包三元模型不满足
#     （实测起服即报 "--lm-head-q6 requires text/output_head to be stored as row-split Q8_G32"）。
$EMBED_Q4         = ""
#   【是什么】词嵌入存成 Q4。省显存、可能略降质量。本模型格式约束同 Q6（不可用）。
$EMBED_Q6         = ""
#   【是什么】词嵌入存成 Q6。省显存、质量损失小。
#   【★ 本模型不可用】同上（要求 token_embedding 是 row-split Q8_G32）。
$MTP_EXPERTS_Q4   = ""
#   【是什么】MTP 层专家用文本层格式。★ 仅 Qwen3.6-35B-A3B 这类 MoE 有意义；本包单模型无意义。
$GDN_STATE_FP16   = ""
#   【是什么】GDN 循环状态用 FP16。留空=关 / "1"=开。
#   【效果】每份 host 状态镜像减半（省内存），质量影响很小。
#   【实测】decode 略快（97.7 vs 94.3）但 prefill −13%（268.9 vs 310）⇒ 不默认。
$ROPE_YARN        = ""
#   【是什么】超出模型原生窗口时启用 YaRN 外推。留空=关 / "1"=开。
#   【★ 结论（实测）】模型原生窗口 n_ctx_train=262144；287K prompt（超 25K）
#     仍能 prefill + 90% 位置召回 HIT ⇒ 到 287K 不需要手动开 YaRN。1M 全量未验证。
$ROPE_YARN_FACTOR = ""
#   【是什么】固定 YaRN 因子（1..4），对所有位置生效。替代"按 max_context 自动算"。
$ROPE_SCALE       = ""
#   【是什么】线性位置插值因子（1..32，默认 1=关）。另一种外推方式，比 YaRN 简单。
$ROPE_SCALE_ORIG  = ""
#   【是什么】线性插值的"原生窗口阈值"（token 数）。默认用模型自带值。
$CUDA_MEM_POLICY  = ""
#   【是什么】显存不足时的行为（内存池策略）。★ KVMem 下不可用（与 KVMem 互斥）。
$WDDM_EVICTABLE   = ""
#   【是什么】Windows/D3D12 构建按"专用显存"预算（把可驱逐内存算进来）。★ KVMem 下不可用。
$NO_CUDA_GRAPH    = ""
#   【是什么】decode 不用 CUDA Graph。留空=用 / "1"=不用。
#   【效果】省一点驱动侧显存，但每步多一次 launch 开销 → decode 变慢；排错时可用来排除 graph 因素。
$CUDA_GRAPH_ALLOW = ""
#   【是什么】CUDA Graph 的驱动态预留（MiB，默认 0=按 profile 计算）。
#   【什么时候用】并发时若出现 "CUDA graphs used X MiB but KV sizing allowed Y" 警告，调它（如 128）。
#   【实测】128 与默认无显著差异（decode 94.3 / prefill 664.2）⇒ 保持留空。
$DEVICE_PROFILE   = "off"
#   【是什么】每 GPU 的"路线档案"（记录这个卡上各计算的代价，用于自动分配）。取值 auto/off/calibrate。
#   ★ 本卡必须 off：calibrate 会跑标定并撞上 >64512B 动态 smem 的内核；手写 profile 是旧 schema，
#     强制加载会让引擎忽略并重标定、还会改写源码树（已踩坑）。
$DEVICE_PROFILE_P = ""
#   【是什么】指定 profile 文件路径（替代用户缓存）。一般不用。
$CTX_COST_PRESETS = ""
#   【是什么】运行时"上下文代价预设"文件。覆盖编译内置值；一般不用。
$CHAT_TEMPLATE    = ""
#   【是什么】替换制品的对话模板（jinja）。留空=用制品自带；只有模板不兼容时才改。
$DEVICE           = ""
#   【是什么】CUDA 设备号（默认 0）。与 $DEVICES 互斥。
$DEVICES          = ""
#   【是什么】多卡流水线（逗号分隔 CUDA id，2..8 个）。单卡场景不用。
$STAGE_LAYERS     = ""
#   【是什么】各流水级的层数（逗号分隔）。需要 $DEVICES。
$EXPERT_RESIDENCY = ""
#   【是什么】MoE 专家权重的驻留方式：device/host/disk。
#   【意义】让"权重远超显存"的 MoE 模型也能跑（如 Qwen3.8-Flash-Next）；本包 Bonsai 不是 MoE，无效。
$EXPERT_CACHE_MIB = ""
#   【是什么】最常用专家的设备缓存（MiB，默认 auto）。host/disk 专家驻留时使用。
$NGRAM_TABLE      = ""
#   【是什么】n-gram 表制品路径（模型没自带时提供，Flash-Next 用）。本包 Bonsai 自带，不用。
$NGRAM_RAM        = ""
#   【是什么】n-gram 表载入 RAM（不再逐行读文件）。留空=关 / "1"=开。更快但吃内存。
$NO_NGRAM_TABLE   = ""
#   【是什么】不带 n-gram 表运行。★ 实验模式：WikiText-2 PPL 2.66 → 5.01，质量明显下降，仅用于跑通。

# ============================================================
# 十三、提示词移植 / API 行为（一般不用）
# ============================================================
# 【这一段是什么】"把提示词里的某段替换成另一份 KV"（phantom-kv 移植）以及几个 API 兼容开关。
# 【默认】全部留空。下面是"什么场景才需要"的说明。
$GRAFT            = ""
#   【是什么】加载一个 phantom-kv 移植（NAME=PATH，可重复）。请求里带 "graft":"NAME" 时生效。
#   【意义】把一段固定前缀的 KV 直接搬进来（省重算）；一般不用。
$DEFAULT_GRAFT    = ""
#   【是什么】对所有请求默认应用某个移植。请求可用 "graft":"" 退出。
$STRUCTURED_OUT   = ""
#   【是什么】接受 JSON / JSON Schema 结构化输出。留空=关 / "1"=开。开启后才能用 response_format。
$UNCONSTRAINED    = ""
#   【是什么】没有 structured-output 时也不拒绝这类请求（宽松处理）。
$ASSISTANT_PREFILL= ""
#   【是什么】续写最后一条 assistant 消息（Chat 补全的"预填充"）。需要关思考。
$LENIENT_HISTORY  = ""
#   【是什么】接受 reasoning 在 function_call 之后的 Responses 输入（兼容某些客户端）。
$USAGE_CHUNK_CHOICE= ""
#   【是什么】流式 usage 块带零增量 choice（兼容挑剔的流式解析器）。

# ============================================================
# 十四、路由器 / 模型休眠（单模型场景全部留空）
# ============================================================
# 【这一段是什么】多模型路由（按目录/预设加载多个 .ninfer）与闲置休眠。
# 【默认】单模型长期运行，全部留空。$MODEL_SUSPEND 与 KVMem 互斥。
$MODELS_DIR       = ""
#   【是什么】把目录下所有 .ninfer 都挂上（多模型）。与直接给模型路径互斥。
$MODELS_PRESET    = ""
#   【是什么】INI 预设文件（[id] 段里 model= 路径）。多模型时用。
$MODELS_MAX       = ""
#   【是什么】同时加载模型数（默认 1，0=不限）。超了按 LRU 让闲置模型休眠或卸载。
$NO_MODELS_AUTO   = ""
#   【是什么】未加载模型不自动加载。留空=自动 / "1"=不自动。?autoload=1 可覆盖。
$SLEEP_IDLE       = ""
#   【是什么】闲置 N 秒后休眠（需要 $MODEL_SUSPEND）。省显存。
$UNLOAD_IDLE      = ""
#   【是什么】闲置 N 秒后完全卸载。省显存。
$MODEL_SUSPEND    = ""
#   【是什么】启用 /suspend 与 /resume（挂起模型到主机内存）。★ KVMem 下不可用。
$SUSPEND_SNAPSHOT = ""
#   【是什么】挂起状态存放位置：pageable(默认)/pinned。pinned 更快但启动即预留内存。
$SUSPEND_WEIGHTS  = ""
#   【是什么】恢复时权重来源：artifact(默认)/host。host 用一份主机副本换恢复速度。
$NO_AUTO_RESUME   = ""
#   【是什么】挂起期间请求直接 503（不自动恢复）。留空=自动恢复 / "1"=不恢复。

# ============================================================
# 十五、日志
# ============================================================
# 【这一段是什么】控制日志详细程度与落盘。判据脚本一般用 info（能拿到 req#N done 行）。
$LOG_LEVEL        = "info"
#   【是什么】日志级别：trace/debug/info(默认)/warning/error/critical/off。
#   【info 有什么】容量行 + 每个请求的 req#N done（TTFT/prefill/decode/cache/mtp 接受率）。
#   【debug 有什么】每一步动作（排查用，很吵、很占盘）。
$LOG_STATS_MS     = ""
#   【是什么】吞吐统计报告间隔（毫秒，默认 5000；0=关）。
$LOG_COLOURS      = ""
#   【是什么】控制台着色：on/off。重定向到文件时建议 off（否则日志里有颜色码）。
$LOG_STATS_PANEL  = ""
#   【是什么】固定统计面板：on/off（默认 off）。交互终端下方常驻显示吞吐。
$REQ_LOG_JSONL    = ""
#   【是什么】全精度请求日志路径（记录完整请求/响应）。排查/审计用；文件会很大。
$REQ_LOG_MAX_MIB  = ""
#   【是什么】请求日志轮转大小（MiB，默认 0=不轮转）。需要 $REQ_LOG_JSONL。
$REQ_LOG_KEEP     = ""
#   【是什么】轮转保留份数（默认 4）。需要 $REQ_LOG_MAX_MIB。

# =====================================================================
# 组装：留空即完全不传（含布尔）
# =====================================================================
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $EXE))   { throw "缺少引擎: $EXE" }
if (-not (Test-Path -LiteralPath $MODEL)) { throw "缺少模型: $MODEL" }
if (-not (Test-Path -LiteralPath $LOG_DIR))   { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }
if (-not (Test-Path -LiteralPath $CACHE_DIR)) { New-Item -ItemType Directory -Path $CACHE_DIR -Force | Out-Null }

$a = @()
$a += $MODEL

# --- 基础 / 网络 ---
if ($MODEL_ID)            { $a += "--model-id";              $a += $MODEL_ID }
if ($BIND_HOST)           { $a += "--host";                  $a += $BIND_HOST }
if ($PORT)                { $a += "--port";                  $a += $PORT }
if ($STATS_PORT)          { $a += "--stats-port";            $a += $STATS_PORT }
if ($API_KEY)             { $a += "--api-key";               $a += $API_KEY }
if ($CORS)                { $a += "--cors" }
if ($NO_WEBUI)            { $a += "--no-webui" }
if ($WEBUI_MCP_PROXY)     { $a += "--webui-mcp-proxy" }
if ($MAX_REQUEST_MIB)     { $a += "--max-request-mib";       $a += $MAX_REQUEST_MIB }
if ($MAX_PENDING_REQUESTS){ $a += "--max-pending-requests";  $a += $MAX_PENDING_REQUESTS }
if ($PENDING_TIMEOUT_MS)  { $a += "--pending-timeout-ms";    $a += $PENDING_TIMEOUT_MS }
if ($RECOVER_INVARIANT)   { $a += "--recover-invariant-failures" }
if ($RESP_STORE_RECORDS)  { $a += "--response-store-max-records"; $a += $RESP_STORE_RECORDS }
if ($RESP_STORE_MIB)      { $a += "--response-store-max-mib";     $a += $RESP_STORE_MIB }

# --- 上下文 / 并发 ---
if ($MAX_CONTEXT)         { $a += "--max-context";           $a += $MAX_CONTEXT }
if ($MAX_CONCURRENCY)     { $a += "--max-concurrency";       $a += $MAX_CONCURRENCY }
if ($PREFILL_CHUNK)       { $a += "--prefill-chunk";         $a += $PREFILL_CHUNK }
if ($DEFAULT_MAX_TOK)     { $a += "--default-max-tokens";    $a += $DEFAULT_MAX_TOK }

# --- KVMem 核心 ---
if ($KV_BUDGET)           { $a += "--kvmem-budget";          $a += $KV_BUDGET }
if ($KV_GEN_RESERVE)      { $a += "--kvmem-gen-reserve";     $a += $KV_GEN_RESERVE }
if ($KV_HOST_MIB)         { $a += "--kvmem-host-mib";        $a += $KV_HOST_MIB }
if ($KV_SESSIONS)         { $a += "--kvmem-sessions";        $a += $KV_SESSIONS }
if ($KV_SINK_TOKENS)      { $a += "--kvmem-sink-tokens";     $a += $KV_SINK_TOKENS }
if ($KV_RECENT_TOKENS)    { $a += "--kvmem-recent-tokens";   $a += $KV_RECENT_TOKENS }
if ($KV_LRFU)             { $a += "--kvmem-lrfu" }
if ($KV_LRFU_LAMBDA)      { $a += "--kvmem-lrfu-lambda";     $a += $KV_LRFU_LAMBDA }
if ($KV_LRFU_TOPF)        { $a += "--kvmem-lrfu-top-frac";   $a += $KV_LRFU_TOPF }
if ($KV_SCORE_AGG)        { $a += "--kvmem-score-agg";       $a += $KV_SCORE_AGG }
if ($KV_VERIFY_XFER)      { $a += "--kvmem-verify-transfers" }

# --- KVMem 磁盘分层 ---
if ($KV_DISK_PATH)        { $a += "--kvmem-disk-path";       $a += $KV_DISK_PATH }
if ($KV_DISK_MIB)         { $a += "--kvmem-disk-mib";        $a += $KV_DISK_MIB }
if ($DISK_KV_PATH)        { $a += "--disk-kv-path";          $a += $DISK_KV_PATH }
if ($DISK_KV_GIB)         { $a += "--disk-kv-gib";           $a += $DISK_KV_GIB }
if ($DISK_KV_RESTORE)     { $a += "--disk-kv-restore" }
if ($DISK_KV_DS)          { $a += "--disk-kv-directstorage" }

# --- KV 格式 / 容量 ---
if ($KV_DTYPE)            { $a += "--kv-dtype";              $a += $KV_DTYPE }
if ($KV_CAPACITY)         { $a += "--kv-capacity";           $a += $KV_CAPACITY }   # ★ KVMem 下请保持为空
if ($KV_HEADROOM_MIB)     { $a += "--kv-headroom-mib";       $a += $KV_HEADROOM_MIB }

# --- ActKV 三支柱：打分器（支柱③）与置信度预算（支柱②）---
# 支柱① 的「动作区 query」与「system+任务描述强制保留」在源码内自动生效，无开关。
if ($KV_RETRIEVAL_METHOD) { $a += "--kvmem-retrieval-method"; $a += $KV_RETRIEVAL_METHOD }
if ($KV_SUBBLOCKS)        { $a += "--kvmem-subblocks";       $a += $KV_SUBBLOCKS }
if ($KV_SUBBLOCK_REDUCE)  { $a += "--kvmem-subblock-reduce"; $a += $KV_SUBBLOCK_REDUCE }
if ($KV_PROTOTYPE)        { $a += "--kvmem-prototype";       $a += $KV_PROTOTYPE }
if ($KV_CONF_BUDGET)      { $a += "--kvmem-conf-budget" }
if ($KV_CONF_INIT)        { $a += "--kvmem-conf-init";       $a += $KV_CONF_INIT }
if ($KV_CONF_UPPER)       { $a += "--kvmem-conf-upper";      $a += $KV_CONF_UPPER }
if ($KV_CONF_POOL)        { $a += "--kvmem-conf-pool";       $a += $KV_CONF_POOL }
if ($KV_CONF_STRIDE)      { $a += "--kvmem-conf-stride";     $a += $KV_CONF_STRIDE }
if ($KV_CONF_BOTTOM)      { $a += "--kvmem-conf-bottom";     $a += $KV_CONF_BOTTOM }
if ($KV_CONF_MIN_POOL)    { $a += "--kvmem-conf-min-pool";   $a += $KV_CONF_MIN_POOL }
if ($KV_CONF_SCALE_UP)    { $a += "--kvmem-conf-scale-up";   $a += $KV_CONF_SCALE_UP }
if ($KV_CONF_COOLDOWN)    { $a += "--kvmem-conf-cooldown";   $a += $KV_CONF_COOLDOWN }

# --- 上下文缓存 ---
if ($NO_PREFIX_REUSE)     { $a += "--no-prefix-reuse" }
if ($DEV_STATE_SLOTS)     { $a += "--device-state-slots";    $a += $DEV_STATE_SLOTS }
if ($HOST_STATE_SLOTS)    { $a += "--host-state-slots";      $a += $HOST_STATE_SLOTS }
if ($HOST_KV_MIB)         { $a += "--host-kv-mib";           $a += $HOST_KV_MIB }
if ($HOST_CACHE_MIB)      { $a += "--host-cache-mib";        $a += $HOST_CACHE_MIB }
if ($MAX_PRIV_CONT)       { $a += "--max-private-continuations"; $a += $MAX_PRIV_CONT }
if ($MAX_SHARED_PREF)     { $a += "--max-shared-prefixes";   $a += $MAX_SHARED_PREF }
if ($MAX_LONG_ANCHOR)     { $a += "--max-long-anchors-per-continuation"; $a += $MAX_LONG_ANCHOR }
if ($AUTO_LONG_ANCHOR)    { $a += "--auto-long-anchors" }
if ($BRANCH_ANCHORS)      { $a += "--branch-anchors" }
if ($NO_ENDPOINT_ANC)     { $a += "--no-endpoint-anchors" }
if ($LONG_ANCHOR_GAP)     { $a += "--long-anchor-spacing";   $a += $LONG_ANCHOR_GAP }
if ($MAX_CACHE_MARKS)     { $a += "--max-cache-markers-per-request"; $a += $MAX_CACHE_MARKS }
if ($AUTO_PREFIX_GRID)    { $a += "--auto-prefix-grid" }
if ($DERIVE_SESS_KEYS)    { $a += "--derive-session-keys" }
if ($CTX_CACHE_POLICY)    { $a += "--context-cache-policy";  $a += $CTX_CACHE_POLICY }
if ($REL_DIVERGED)        { $a += "--release-diverged-checkpoints" }
if ($THOROUGH_ADMIT)      { $a += "--thorough-admission-search" }
if ($RECENCY_EVICT)       { $a += "--recency-eviction" }
if ($VALUE_AWARE)         { $a += "--value-aware-demote" }
if ($KV_LEASE_GROWTH)     { $a += "--kv-lease-growth" }
if ($CONCURRENT_PREF)     { $a += "--concurrent-prefill" }   # ★ KVMem 下不可用
if ($SLOT_SAVE_PATH)      { $a += "--slot-save-path";        $a += $SLOT_SAVE_PATH }
if ($AUTO_SAVE_EVICT)     { $a += "--auto-save-evicted" }
if ($ALT_PREFIX_CACHE)    { $a += "--use-alt-prefix-caching" }   # ★ KVMem 下不可用
if ($ORIG_PREFIX_CACHE)   { $a += "--use-original-prefix-caching" }
if ($DEV_SNAP_SLOTS)      { $a += "--device-snapshot-slots"; $a += $DEV_SNAP_SLOTS }
if ($CACHE_TAPS)          { $a += "--cache-taps-per-request"; $a += $CACHE_TAPS }
if ($CACHE_TAP_LADDER)    { $a += "--cache-tap-ladder";      $a += $CACHE_TAP_LADDER }
if ($CACHE_TAP_GAP)       { $a += "--cache-tap-min-gap";     $a += $CACHE_TAP_GAP }
if ($PREFIX_CACHE_FILE)   { $a += "--prefix-cache-file";     $a += $PREFIX_CACHE_FILE }

# --- 投机解码 ---
if ($SPEC)                { $a += "--spec";                  $a += $SPEC }
if ($DRAFT_TOKENS)        { $a += "--draft-tokens";          $a += $DRAFT_TOKENS }
if ($LM_HEAD_DRAFT)       { $a += "--lm-head-draft" }
if ($ADAPTIVE_MTP)        { $a += "--adaptive-mtp" }
if ($MTP_ATTN_WINDOW)     { $a += "--mtp-attention-window";  $a += $MTP_ATTN_WINDOW }
if ($NGRAM_DRAFT)         { $a += "--ngram-draft-tokens";    $a += $NGRAM_DRAFT }
if ($NGRAM_MIN_MATCH)     { $a += "--ngram-min-match";       $a += $NGRAM_MIN_MATCH }
if ($NGRAM_ARCHIVE)       { $a += "--ngram-archive-mib";     $a += $NGRAM_ARCHIVE }
if ($NGRAM_SESSION)       { $a += "--ngram-session-mib";     $a += $NGRAM_SESSION }
if ($NGRAM_NATIVE)        { $a += "--ngram-native-sessions" }
if ($LOOKUP_NGRAM)        { $a += "--lookup-ngram";          $a += $LOOKUP_NGRAM }   # ★ KVMem 下不可用

# --- 多模态 ---
if ($ENABLE_VISION)       { $a += "--vision" }
if ($VISION_RESIDENCY)    { $a += "--vision-residency";      $a += $VISION_RESIDENCY }
if ($VISION_CPU)          { $a += "--vision-cpu" }
if ($VISION_MAX_MERGE)    { $a += "--vision-max-merged";     $a += $VISION_MAX_MERGE }
if ($MEDIA_CACHE_MIB)     { $a += "--media-cache-mib";       $a += $MEDIA_CACHE_MIB }
if ($MEDIA_LIVE_MIB)      { $a += "--media-live-mib";        $a += $MEDIA_LIVE_MIB }
if ($MEDIA_THREADS)       { $a += "--media-preprocess-threads"; $a += $MEDIA_THREADS }

# --- 采样 / 思考 ---
if ($TEMPERATURE)         { $a += "--temperature";           $a += $TEMPERATURE }
if ($TOP_P)               { $a += "--top-p";                 $a += $TOP_P }
if ($TOP_K)               { $a += "--top-k";                 $a += $TOP_K }
if ($MIN_P)               { $a += "--min-p";                 $a += $MIN_P }
if ($PRESENCE_PEN)        { $a += "--presence-penalty";      $a += $PRESENCE_PEN }
if ($FREQ_PEN)            { $a += "--frequency-penalty";     $a += $FREQ_PEN }
if ($SEED)                { $a += "--seed";                  $a += $SEED }
if ($GREEDY)              { $a += "--greedy" }
if ($POST_THINKING)       { $a += "--post-thinking" }
if ($PT_TEMP)             { $a += "--post-thinking-temperature"; $a += $PT_TEMP }
if ($PT_TOP_P)            { $a += "--post-thinking-top-p";   $a += $PT_TOP_P }
if ($PT_TOP_K)            { $a += "--post-thinking-top-k";   $a += $PT_TOP_K }
if ($PT_SAMPLER)          { $a += "--post-thinking-sampler"; $a += $PT_SAMPLER }
if ($NO_THINKING)         { $a += "--no-thinking" }
if ($PRESERVE_THINK)      { $a += "--preserve-thinking" }
if ($THINK_BUDGET)        { $a += "--default-thinking-budget"; $a += $THINK_BUDGET }
if ($THINK_BUDGET_MSG)    { $a += "--thinking-budget-message"; $a += $THINK_BUDGET_MSG }
if ($REASON_EFFORT)       { $a += "--default-reasoning-effort"; $a += $REASON_EFFORT }

# --- 精度与内核 ---
if ($FAST_PREFILL)        { $a += "--fast-prefill-kernel" }
if ($PREFILL_CUBLAS)      { $a += "--prefill-cublas" }
if ($NO_CUBLAS_PROJ)      { $a += "--no-prefill-cublas-projections" }
if ($NO_PREFILL_A8)       { $a += "--no-prefill-a8" }
if ($MLP_A8_DECODE)       { $a += "--mlp-a8-decode" }
if ($LM_HEAD_Q4)          { $a += "--lm-head-q4" }
if ($LM_HEAD_Q6)          { $a += "--lm-head-q6" }
if ($EMBED_Q4)            { $a += "--embedding-q4" }
if ($EMBED_Q6)            { $a += "--embedding-q6" }
if ($MTP_EXPERTS_Q4)      { $a += "--mtp-experts-q4" }
if ($GDN_STATE_FP16)      { $a += "--gdn-state-fp16" }
if ($ROPE_YARN)           { $a += "--rope-yarn" }
if ($ROPE_YARN_FACTOR)    { $a += "--rope-yarn-factor";      $a += $ROPE_YARN_FACTOR }
if ($ROPE_SCALE)          { $a += "--rope-scaling-factor";   $a += $ROPE_SCALE }
if ($ROPE_SCALE_ORIG)     { $a += "--rope-scaling-original-context"; $a += $ROPE_SCALE_ORIG }
if ($CUDA_MEM_POLICY)     { $a += "--cuda-memory-policy";    $a += $CUDA_MEM_POLICY }   # ★ KVMem 下不可用
if ($WDDM_EVICTABLE)      { $a += "--wddm-evictable-budget" }
if ($NO_CUDA_GRAPH)       { $a += "--no-cuda-graph" }
if ($CUDA_GRAPH_ALLOW)    { $a += "--cuda-graph-allowance-mib"; $a += $CUDA_GRAPH_ALLOW }
if ($DEVICE_PROFILE)      { $a += "--device-profile";        $a += $DEVICE_PROFILE }
if ($DEVICE_PROFILE_P)    { $a += "--device-profile-path";   $a += $DEVICE_PROFILE_P }
if ($CTX_COST_PRESETS)    { $a += "--context-cost-presets";  $a += $CTX_COST_PRESETS }
if ($CHAT_TEMPLATE)       { $a += "--chat-template";         $a += $CHAT_TEMPLATE }
if ($DEVICE)              { $a += "--device";                $a += $DEVICE }
if ($DEVICES)             { $a += "--devices";               $a += $DEVICES }
if ($STAGE_LAYERS)        { $a += "--stage-layers";          $a += $STAGE_LAYERS }
if ($EXPERT_RESIDENCY)    { $a += "--expert-residency";      $a += $EXPERT_RESIDENCY }
if ($EXPERT_CACHE_MIB)    { $a += "--expert-cache-mib";      $a += $EXPERT_CACHE_MIB }
if ($NGRAM_TABLE)         { $a += "--ngram-table";           $a += $NGRAM_TABLE }
if ($NGRAM_RAM)           { $a += "--ngram-ram" }
if ($NO_NGRAM_TABLE)      { $a += "--no-ngram-table" }

# --- 移植 / API 行为 ---
if ($GRAFT)               { $a += "--graft";                 $a += $GRAFT }
if ($DEFAULT_GRAFT)       { $a += "--default-graft";         $a += $DEFAULT_GRAFT }
if ($STRUCTURED_OUT)      { $a += "--structured-output" }
if ($UNCONSTRAINED)       { $a += "--unconstrained-response-format" }
if ($ASSISTANT_PREFILL)   { $a += "--assistant-prefill" }
if ($LENIENT_HISTORY)     { $a += "--lenient-assistant-history" }
if ($USAGE_CHUNK_CHOICE)  { $a += "--usage-chunk-choice" }

# --- 路由器 / 休眠 ---
if ($MODELS_DIR)          { $a += "--models-dir";            $a += $MODELS_DIR }
if ($MODELS_PRESET)       { $a += "--models-preset";         $a += $MODELS_PRESET }
if ($MODELS_MAX)          { $a += "--models-max";            $a += $MODELS_MAX }
if ($NO_MODELS_AUTO)      { $a += "--no-models-autoload" }
if ($SLEEP_IDLE)          { $a += "--sleep-idle-seconds";    $a += $SLEEP_IDLE }
if ($UNLOAD_IDLE)         { $a += "--unload-idle-seconds";   $a += $UNLOAD_IDLE }
if ($MODEL_SUSPEND)       { $a += "--model-suspend" }        # ★ KVMem 下不可用
if ($SUSPEND_SNAPSHOT)    { $a += "--suspend-snapshot";      $a += $SUSPEND_SNAPSHOT }
if ($SUSPEND_WEIGHTS)     { $a += "--suspend-weights";       $a += $SUSPEND_WEIGHTS }
if ($NO_AUTO_RESUME)      { $a += "--no-auto-resume" }

# --- 日志 ---
if ($LOG_LEVEL)           { $a += "--log-level";             $a += $LOG_LEVEL }
if ($LOG_STATS_MS)        { $a += "--log-stats-interval-ms"; $a += $LOG_STATS_MS }
if ($LOG_COLOURS)         { $a += "--log-colours";           $a += $LOG_COLOURS }
if ($LOG_STATS_PANEL)     { $a += "--log-stats-panel";       $a += $LOG_STATS_PANEL }
if ($REQ_LOG_JSONL)       { $a += "--request-log-jsonl";     $a += $REQ_LOG_JSONL }
if ($REQ_LOG_MAX_MIB)     { $a += "--request-log-max-mib";   $a += $REQ_LOG_MAX_MIB }
if ($REQ_LOG_KEEP)        { $a += "--request-log-keep";      $a += $REQ_LOG_KEEP }

# =====================================================================
# 启动
# =====================================================================
$stamp = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
$outLog = Join-Path $LOG_DIR ("server_$stamp.out.log")
$errLog = Join-Path $LOG_DIR ("server_$stamp.err.log")

Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  KVMem_Ninfer_ActKV_2080Ti v0.1" -ForegroundColor Green
Write-Host ("  引擎 : " + $EXE)
Write-Host ("  模型 : " + $MODEL)
Write-Host ("  参数 : " + ($a -join ' '))
Write-Host ("  API  : http://$BIND_HOST`:$PORT/v1/chat/completions")
Write-Host ("  日志 : " + $errLog)
Write-Host "  按 Ctrl+C 停止服务"
Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray

# PS 5.1 会把引擎写到 stderr 的每一行都当成脚本错误；若 $ErrorActionPreference='Stop'，
# 引擎刚打印第一行状态就会终止脚本。这里临时放宽为 Continue，输出仍全部进日志。
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
try { & cmd /c chcp 65001 > $null } catch { }
$__eap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $EXE @a 2> $errLog | Tee-Object -FilePath $outLog
} finally {
    $ErrorActionPreference = $__eap
}

Write-Host "`n服务已停止。" -ForegroundColor Yellow
