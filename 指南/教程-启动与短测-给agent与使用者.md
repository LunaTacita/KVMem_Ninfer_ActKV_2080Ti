# 教程 · 启动与短测（给接收方 agent 与使用者）

> 本教程是**执行手册**，不是背景阅读。目标：**15 分钟内**确认"这个包在你的机器上是真的能跑、数值不是假的"，然后把三个数字报回来。
> 不要一上来就跑长测试矩阵：长测试会让一个能用的部署看起来像失败的部署。

---

## §0.0 四步前置（做完再往下）

1. **解压用 7-Zip / WinRAR / `tar -xf`**。Windows 资源管理器自带的解压对中文文件名会乱码（本包文档名是中文）——那是它的老毛病，不是包坏了。
2. **解压路径必须全 ASCII**（例如 `D:\kvmem-sm75`）。原生 exe 读不了中文路径。
3. **先放模型，再起服务**。本包不含模型（权重 9.5 GB，走独立模型包）。下载/复制后按 `模型包\MODELS.txt` 的 sha256 校验。
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
  --spec mtp --draft-tokens 2 --log-level info
```

就绪判据（另开一个窗口）：

```powershell
curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8080/v1/models
# 期望输出：200
```

---

## §1 用启动脚本起服（推荐）

```powershell
powershell -ExecutionPolicy Bypass -File .\Start\启动_KVMem_Ninfer_ActKV_2080Ti.ps1
```

脚本里只需要改一处：`$MODEL`（你的 `.ninfer` 模型绝对路径）。其余默认值就是本包实测的生产配方（1M 上下文 / 单并发 / int8 / MTP draft=2 / 思考最高档）。

看到类似下面的行就是起来了（stderr 里）：

```
INFO  capacity | KV 38,400 tokens, int8, explicit | pages 600/600 | runtime 1.79 GiB | free ~996 MiB
INFO  listening on 127.0.0.1:8080
```

---

## §2 15 分钟短测（只做三件事）

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

### 2.2 读引擎自己的两行证据（stderr / logs\*.err.log）

```
req#1 done | openai-chat | stop token | prompt 68 | output 1180 | cache 0 (0.0%) |
            TTFT 566 ms | total 15.2s | prefill 118.3 tok/s | decode 79.6 tok/s | mtp accepted 139/152 (91.4%)
```

要报的三个数字：**TTFT（ms）**、**prefill（tok/s）**、**decode（tok/s）**，外加 **mtp accepted 百分比**。

### 2.3 一条跨窗口召回（证明 KVMem 真的在分层）

```powershell
# 用同目录的判据脚本之一（推荐；判据脚本需要 PowerShell 7）：
pwsh -File .\ms75_eval\run_smoke_kvmem.ps1
# 它会：起服 → 短 needle 对照 → 40K 超窗口 needle → 报 HIT/MISS
```

通过标准：**短 needle HIT 且 40K needle HIT**。若短 HIT 长 MISS 才是真长上下文问题；两个都 MISS 说明判据本身还没对上（先查 `max_tokens` 是否给足）。

---

## §3 证据行怎么读

| 行 | 含义 | 正常长什么样 |
|---|---|---|
| `capacity \| KV 38,400 tokens, int8, explicit \| pages 600/600 \| runtime 1.79 GiB \| free …` | 设备池大小 / 页数 / 显存占用 | pages 600/600、free 300~1000 MiB |
| `req#N done \| … prompt X \| output Y \| cache Z (P%)` | 本请求的 token 数与前缀复用率 | 第 2 轮起 cache P 应显著 > 0 |
| `TTFT / prefill / decode / mtp accepted` | 首 token 时间 / 预填速度 / 解码速度 / 投机接受率 | 短上下文 decode 75~95 tok/s、MTP 90%±5 |
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
| `--spec mtp` 立即退出 | 漏了 `--draft-tokens` | 成对给：`--spec mtp --draft-tokens 2` |
| 请求 503 / 排队超时 | `--pending-timeout-ms` 太小或并发满 | 单请求场景保持 `--max-concurrency 1`；长任务调大 pending 超时 |
| 长时间无输出、日志停在 loading 27%/82% | 大概率是机械盘冷缓存（不是死锁） | 看任务管理器的磁盘读取是否还在增长；把模型复制到 NVMe |
| 回答为空但 `finish_reason=length` | `max_tokens` 被 thinking 吃光 | 加大 `max_tokens`（thinking 也计入预算） |

---

## §5 边界（别把本教程当万能）

- 本教程的判据是「**部署是否可用**」，不是性能排行榜：短测数字会因机器、驱动、盘、后台负载波动（同一机器同一配置重复测可差 5~10%）。
- 性能判据要在**宿主静默**下测，同配置重复一致才采纳；与其他作业并行时的读数不要用。
- 不要一上来跑长测试矩阵：先跑 §2 三件事；通过了再按需跑 `ms75_eval` 里的专项判据。
- 本包不含模型；模型的许可与来源见 `模型包\MODELS.txt`。
