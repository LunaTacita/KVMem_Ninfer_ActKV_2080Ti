# 教程 · 启动与短测（给接收方 agent 与使用者）

> 本教程是**执行手册**，不是背景阅读。目标：**15 分钟内**确认"这个包在你的机器上是真的能跑、数值不是假的"，然后把三个数字报回来。
> 不要一上来就跑长测试矩阵：长测试会让一个能用的部署看起来像失败的部署。
> 运行时限制（含可选的投机解码）见 `指南\已知问题与运行时限制.md`。

---

## §0.0 四步前置（做完再往下）

1. **解压用 7-Zip / WinRAR / `tar -xf`**。Windows 资源管理器自带的解压对中文文件名会乱码（本包文档名是中文）——那是它的老毛病，不是包坏了。
2. **解压路径必须全 ASCII**（例如 `D:\kvmem-sm75`）。原生 exe 读不了中文路径。
3. **先放模型，再起服务**。本包不含模型（权重 9.5 GB，走独立模型包）。下载/复制后按 `模型包\MODELS.md` 的 sha256 校验。
4. **确认 CUDA 12.6 运行时**：`engine\` 目录已自带 `cublas64_12.dll` / `cublasLt64_12.dll` / `cudart64_12.dll`；驱动需支持 CUDA 12.x（下限 ≥ 560.76，推荐 566.36）。

---

## §0.5 纯指令路径（脚本拒绝运行时用这条，不用任何脚本）

```powershell
cd <解压目录>\engine
.\ninfer-serve.exe <你的>\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer `
  --host 127.0.0.1 --port 8080 --device-profile off `
  --max-context 1048576 --max-concurrency 1 --kv-dtype int8 `
  --kvmem-budget 32768 --kvmem-gen-reserve 5632 --kvmem-host-mib auto `
  --kvmem-sink-tokens 4096 --kvmem-recent-tokens 1024 `
  --kvmem-disk-path <解压目录>\Cache\kvmem --kvmem-disk-mib 102400 `
  --log-level info
```

> 默认**不带投机解码**（不传 `--spec` / `--draft-tokens`）。需要实验性开启时见 `指南\已知问题与运行时限制.md` §1。

就绪判据（另开一个窗口）：

```powershell
curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8080/v1/models
# 期望输出：200
curl.exe -s -o NUL -w "%{http_code} %{content_type}" http://127.0.0.1:8080/
# 期望输出：200 text/html   ← 内置 llama.cpp WebUI（构建期嵌入 exe）
```

---

## §1 用启动脚本起服（推荐）

```powershell
powershell -ExecutionPolicy Bypass -File .\Start\启动_KVMem_Ninfer_ActKV_2080Ti.ps1
```

脚本里只需要改一处：`$MODEL`（你的 `.ninfer` 模型绝对路径）。其余默认值就是本包实测的生产配方（1M 上下文 / 单并发 / int8 / 无投机 / 思考 `xhigh`（模板最高档）/ CUDA 图开 / 会话保留 1）。

**长任务等待上限**：内置 WebUI 一次对话默认等待 1 小时（脚本 `$WEBUI_REQUEST_TIMEOUT_MS = "3600000"`，对应 `--webui-request-timeout-ms`；设 `0` = 不设前端超时）；排队 + 准备阶段超时由 `$PENDING_TIMEOUT_MS` 控制（留空 = 引擎默认 600000 ms，长文档场景可按需调大）。

脚本会先打印**配置摘要**与**完整启动命令**，再起服。看到类似下面的输出就是起来了：

```
------------------------------------------------------------
  KVMem_Ninfer_ActKV_2080Ti v0.1.1
  引擎 : <解压目录>\engine\ninfer-serve.exe
  模型 : D:\models\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer

  -- 当前配置 --
  档位     : long   B=32768  R=5632  H=auto  并发=1
  上下文   : 1048576 tokens
  MTP      : 关闭
  CUDA图   : 开启
  会话保留 : 1 个   sink=4096   recent=1024
  超时     : 排队 默认 600000 ms / WebUI 3600000 ms
  思考     : effort=xhigh（本模型模板实际档位：low/medium/xhigh）
  WebUI    : http://127.0.0.1:8080/
  API      : http://127.0.0.1:8080/v1
  日志     : <解压目录>\logs\server_<时间>.out.log  (stdout + stderr)

  -- 完整启动命令 --
  <解压目录>\engine\ninfer-serve.exe "D:\models\...ninfer" --model-id ... 

  按 Ctrl+C 停止服务
------------------------------------------------------------
INFO  capacity | KV 38,400 tokens, int8, explicit | pages 600/600 | runtime 约 1.5–1.8 GiB | free ~1.55 GiB
INFO  listening on 127.0.0.1:8080
INFO  webui http://127.0.0.1:8080/
```

---

## §2 15 分钟短测（只做四件事）

### 2.1 一条"数数"测速请求（1000 token 进 / 1000 token 出）

```powershell
$body = @{
  model = "bonsai2-27b-heretic"
  messages = @(@{ role='user'; content=('Write the numbers 1 to 1000, one per line.') })
  max_tokens = 1200; temperature = 0.0
} | ConvertTo-Json -Depth 6
$r = Invoke-WebRequest -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post `
     -ContentType 'application/json' -Body $body -TimeoutSec 600
($r.Content | ConvertFrom-Json).usage
```

### 2.2 读引擎自己的两行证据（stderr / logs\*.out.log）

```
req#1 done | openai-chat | stop token | prompt 68 | output 1180 | cache 0 (0.0%) |
            TTFT 566 ms | total 15.2s | prefill 118.3 tok/s | decode 55.6 tok/s
```

要报的三个数字：**TTFT（ms）**、**prefill（tok/s）**、**decode（tok/s）**。
（默认配置不带投机解码，所以没有 `mtp accepted` 字段；若你按 `指南\已知问题与运行时限制.md` §1 自行开启了 MTP，才会出现该字段。）

### 2.3 一条跨窗口召回（证明 KVMem 真的在分层）

```powershell
# 用同目录的判据脚本之一（推荐；判据脚本需要 PowerShell 7）：
pwsh -File .\sm75_eval\run_smoke_kvmem.ps1
# 它会：起服 → 短 needle 对照 → 40K 超窗口 needle → 报 HIT/MISS
```

通过标准：**短 needle HIT 且 40K needle HIT**。若短 HIT 长 MISS 才是真长上下文问题；两个都 MISS 说明判据本身还没对上（先查 `max_tokens` 是否给足）。

### 2.4 浏览器打开内置 WebUI

使用任意现代浏览器访问 `http://127.0.0.1:8080/`（**纯 HTTP 回环地址**，不要带 https、不要走代理），应出现 llama.cpp 的对话界面；发一句话能正常回复即通过。
打不开但 API 正常时：先确认浏览器代理已关闭，再确认 `GET /` 返回 200 / `text/html`。

---

## §3 证据行怎么读

| 行 | 含义 | 正常长什么样 |
|---|---|---|
| `capacity \| KV 38,400 tokens, int8, explicit \| pages 600/600 \| runtime 约 1.5–1.8 GiB \| free …` | 设备池大小 / 页数 / 显存占用 | pages 600/600、free 300~1000 MiB |
| `req#N done \| … prompt X \| output Y \| cache Z (P%)` | 本请求的 token 数与前缀复用率 | 第 2 轮起 cache P 应显著 > 0 |
| `kv-history restore/spill/swaps` | 吞吐统计行里的 KV 换页窗口增量（近 5 秒） | 长上下文 prefill 时数值明显增长属正常换页，不是错误 |
| `TTFT n/a (no token)` | 该请求没有产出 token（被取消 / 0 输出） | 不是首 token 时间，不要当成"很慢" |
| `TTFT / prefill / decode` | 首 token 时间 / 预填速度 / 解码速度 | 短上下文 decode 约 45~60 tok/s（无投机，视上下文与档位）、prefill 500~650 tok/s |
| `webui http://…` | 内置 WebUI 已就绪 | 起服日志出现该行 |
| `[ALIGN]` / `misaligned` | 对齐自检失败 | **应该 0 行** |
| `KVMEM_SCORER` / `KVMEM_PROTECT` / `KVMEM_CONF` | ActKV 机制行（需 `--kvmem-verify-transfers`） | 出现即证明对应支柱在跑 |

---

## §4 常见失败对照（按报错文本对号）

| 报错/现象 | 判据 | 处置 |
|---|---|---|
| `CreateFileW: Win32 error 3` | 模型路径不存在/含中文/盘符错 | 换全 ASCII 路径；用绝对路径 |
| 启动即退，无输出 | 缺 DLL | 确认 `engine\` 里 3 个 CUDA DLL 与 FFmpeg DLL 都在，且从 `engine\` 目录启动 |
| `... it needs X bytes and Y are free after weights ... largest visible window ... about Z tokens` | 显存不够放可见窗口 | 按提示降 `--kvmem-budget`（并保持 `B+R ≤ 38400`），或改 `--kv-dtype rk8v4` |
| `KVMem rejected | …` | 容量准入被拒 | 同上；不要传 `--kv-capacity`（KVMem 模式下它必须为空） |
| `--draft-tokens require --spec` 立即退出 | 只给了 `--draft-tokens` 没给 `--spec` | 两者必须成对：要么都给（实验性），要么都留空（默认） |
| 请求 503 / 排队超时 | `--pending-timeout-ms` 太小或并发满 | 单请求场景保持 `--max-concurrency 1`；长任务调大 pending 超时 |
| 长时间无输出、日志停在 loading 27%/82% | 大概率是机械盘冷缓存 | 看任务管理器的磁盘读取是否还在增长；把模型复制到 NVMe |
| 回答为空但 `finish_reason=length` | `max_tokens` 被 thinking 吃光 | 加大 `max_tokens`（thinking 也计入预算） |
| WebUI 打不开但 API 正常 | 浏览器代理 / https | 用 `http://127.0.0.1:8080/`，关代理 |
| 日志出现 `ninfer-serve: FATAL: …` 且进程退出 | 不可恢复的 CUDA / 内部错误；本版打印 FATAL 后以退出码 1 退出 | 记录 FATAL 详情与日志末段 → 重启服务；长上下文偶发 GPU 错误见 `指南\已知问题与运行时限制.md` §12 |
| WebUI 提示请求超时 / Request timed out | 前端等待上限默认 1 小时；长任务可能超过，或浏览器 / 代理侧超时 | 调大 `$WEBUI_REQUEST_TIMEOUT_MS` 或设为 0 后重启；确认浏览器代理已关 |
| 用 `Start-Process` 隐藏窗口起启动脚本时，日志文件 0 字节 | PowerShell 在隐藏窗口 / 重定向下 Tee 输出可能为空 | 前台窗口运行脚本，或用 `-RedirectStandardOutput` / `-RedirectStandardError` 捕获控制台输出 |

---

## §5 边界（别把本教程当万能）

- 本教程的判据是「**部署是否可用**」，不是性能排行榜：短测数字会因机器、驱动、盘、后台负载波动（同一机器同一配置重复测可差 5~10%）。
- 性能判据要在**宿主静默**下测，同配置重复一致才采纳；与其他作业并行时的读数不要用。
- 不要一上来跑长测试矩阵：先跑 §2 四件事；通过了再按需跑 `sm75_eval` 里的专项判据。
- 本包不含模型；模型的许可与来源见 `模型包\MODELS.md`。
- 运行时已知限制（可选的投机解码、n-gram 路径等）见 `指南\已知问题与运行时限制.md`。
