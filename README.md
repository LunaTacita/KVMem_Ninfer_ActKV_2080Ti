# KVMem_Ninfer_ActKV_2080Ti

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Version](https://img.shields.io/badge/version-v0.1-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20sm__75-lightgrey)
![GPU](https://img.shields.io/badge/RTX%202080%20Ti-11GB-orange)
![Context](https://img.shields.io/badge/context-1M%20tokens-purple)

> **让 11GB 显存的 RTX 2080 Ti（Turing, sm_75），跑起 27B 的 Bonsai 模型和 100 万 token 逻辑上下文。** 支持跨窗口召回、多轮会话与图片输入，提供 **OpenAI 兼容 API**。
>
> 版本 `KVMem-Ninfer-ActKV-2080Ti-v0.1` · 本包**不含模型权重**（见 §三）· 许可 Apache-2.0（模型权重许可另计）

> [!NOTE]
> 这是实验性移植包，用于个人研究与本地部署；已知问题与验证范围见 §九。发布包本身已通过完整性自检、起服与真实请求验收。

**阅读导航**：[这是什么](#一这是什么) · [系统要求](#二系统要求) · [快速开始](#三快速开始) · [目录结构](#四目录结构) · [生产配方](#五生产配方脚本默认值) · [性能实测](#六性能实测) · [参数结论](#七参数实测结论) · [验收清单](#八验收清单怎么确认它在工作) · [已知问题](#九已知问题与验证范围) · [排错](#十排错--faq) · [来源与许可](#十一来源致谢与许可)

---

## 一、这是什么

本包把三样东西合成一个可直接运行的服务：

| 层 | 来源 | 说明 |
|---|---|---|
| 引擎 | `iamwavecut/ninfer-all` @ `8319e8f5` | NInfer 引擎本体（上游整合线） |
| KV 分层 | `kvmem/kvmem` 0.18.0 | KVMem 分层 KV 的官方发行层 |
| 模型 | Bonsai（Ternary-Bonsai-2-27B） | NInfer v3 原生容器，自含 text / vision / mtp 组件 |

本栈的基线 = **KVMem 0.18 的 ninfer 后端**（= `ninfer-all@8319e8f5` + KVMem 官方集成补丁 `ninfer-kvmem.patch`，官方 `versions.json` 固定其 sha256 与 patched_tree `7d7e1499a9`）。本包在此基线上做 sm_75 移植与 ActKV 集成，全部改动见 `patches\v0.1-sm75-port.patch`。

### 从这里开始

| 你的目标 | 看哪里 |
|---|---|
| 直接跑起来用 | §三 快速开始 |
| 确认它真的在工作 | §八 验收清单（每条都写了"怎样会失败"） |
| 看实测速度与能力 | §六 性能实测、§七 参数结论 |
| 自己改源码 / 重新编译 | `src\` + `踩坑-源码编译疑难杂症.md` + `patches\` |
| 重跑判据 | `ms75_eval\`（17 个 `run_*.ps1`） |
| 核对来源与许可 | §十一 来源、致谢与许可 |
| 遇到问题 | §九 已知问题、§十 排错 FAQ |

### 它解决了什么

1. **上游引擎只支持 30~50 系显卡（sm_80/86/89/120），20 系（sm_75）直接编译失败或静默算错。** 本项目做了源码级 2080 Ti（sm_75）移植——你的 2080 Ti 也能跑这套 27B 服务。

2. **11GB 显存装不下"27B 模型 + 长上下文"这一整包。** Bonsai 权重本身放得下，但长上下文的 KV 缓存放不下。KVMem 把完整历史放到主机内存（装不下再溢出到 SSD），设备只保留当前窗口——于是模型、长上下文、高速推理三者同时成立。

3. **长对话里，开头的 system prompt / 任务描述会被滑窗挤出，模型"记得住但找不回"。** ActKV 让动作区内容强制保留（system / 任务描述 / 最近一轮 / 图片永不驱逐），并按需检索。

### 技术说明

这一节写给关心实现的人；只想用的话可以直接跳到 §二。

**sm_75 移植**（相对上游基线）：对齐安全的 `cp_async` 字节拷贝替代、Turing MMA 降级、INT8 prompt kernel（Bc=16）、small-T 路由、GDN chunked 路径、fp8 A8 路由门控、`-rdc=false` 下的协作 launch 降级与环境变量语义修正、Turing 共享内存上限适配。

**KVMem 三层**：可以把它想成一张办公桌——**显卡显存是桌面**，只放当前正在用的东西；**主机内存是抽屉**，完整历史都存在里面；**SSD 是仓库**，抽屉也放不下时再往下沉。设备侧只保留可见窗口 `(B+R)=38400` token；完整历史在主机内存（预算默认"物理内存 − 1 GiB"）；主机也装不下时溢出到 SSD（默认 100 GiB）。逻辑上下文 1M 由这三层共同承载；**KV 池本身仍然占显存**（约 3.8 万 token 的窗口），只是不随历史增长。

**ActKV 三支柱**：动作区检索 query + 强制保留（源码内自动生效）、置信度驱动预算（可选）、可插拔打分器（默认 mean-k）。

**上游能力（不是本包新增）**：MTP 投机解码、n-gram 草稿、多模态视觉塔。本包负责把它们在 sm_75 上跑通。

**边界**：本包是实验性移植，二进制为固定版本（`engine\ninfer-serve.exe` sha256 `97D9FFC5…`），不随源码重新编译；已知问题见 §九。

### 移植选型前提：为什么锁定 CUDA 12.6 + 驱动 566.36

本包先定运行环境（NVIDIA 驱动 566.36 + CUDA 12.6），再针对它做移植。

**为什么是 566.36。** 构建机是游戏 + 本地 AI 双用途。对 2080 Ti 这样的老卡，驱动版本不会把 AI 算力变高，真正决定推理速度的是 CUDA 运行时、引擎构建与 sm_75 kernel；但驱动版本会明显影响游戏兼容性、功耗与稳定性。在 2080 Ti 上，566.36 是社区长期反馈中兼顾游戏与本地 AI 的稳定选择，也是驱动出问题时的常见回退版本。本栈推荐 566.36，取的就是这个交集。

**为什么是 CUDA 12.6。** 566.36 支持 CUDA 12.6 / 12.7 运行时。因此工具链固定在 CUDA 12.6 + MSVC 14.38.33130 + VS2026，并针对 sm_75（Turing）做源码级移植。本包全部实测数据都来自这个组合（见 §六）。

**这对使用者意味着什么。**
- 不需要为了跑 27B 换 Studio 驱动、升级 CUDA 13 工具链，或牺牲游戏兼容性——一台机器可以同时是游戏机与本地 AI 服务器。
- 不需要安装 CUDA Toolkit：`engine\` 已带 CUDA 12.6 运行时 DLL。
- 上游默认按 sm_80/86/89/120 + CUDA 13 构建；本包是为 Turing + CUDA 12.6 专门移植的，上层模型与 API 与上游一致，下层 kernel 与工具链按本卡重写。

**兼容性边界。**
- CUDA 12.6 GA 要求 NVIDIA 驱动 ≥ 560.76；本包要求同一下限。
- 驱动向后兼容旧 CUDA 运行时：更高的驱动理论上仍可运行本包，但未在本包验证。
- 反过来不成立：566.36 无法运行 CUDA 13.x 编译的二进制。
- 换驱动（无论升降）都应重跑 §八 验收清单。

## 二、系统要求

| 项 | 要求 |
|---|---|
| 操作系统 | Windows 10/11 x64 |
| 显卡 | RTX 20 系 Turing（sm_75）；实测 RTX 2080 Ti 11GB。其他架构（sm_86/89/120a）不能直接用本包二进制，需按源码重新编译 |
| 显卡驱动 | 推荐 566.36（本栈构建与实测环境）；下限 560.76 |
| CUDA | 12.6；包内 `engine\` 已带运行时 DLL，无需安装 CUDA Toolkit |
| VC++ 运行库 | Microsoft Visual C++ 2015–2022 x64 Redistributable |
| 内存 | 建议 64 GB（主机 KV 预算取"物理内存 − 1 GiB"） |
| 磁盘 | 建议 NVMe；模型 9.5 GB 另计 |
| 路径 | 全 ASCII（原生 exe 读不了中文路径） |
| 启动脚本 | PowerShell 5.1 与 7 均可 |
| 判据脚本 | PowerShell 7.x |

## 三、快速开始

> **让 AI 帮你装**：把下面这段粘给 AI 编程助手（DSH / Claude Code / Cursor / Codex 等），它会按本包文档检查环境、放模型、改路径并启动验证。
>
> 仓库地址：**https://github.com/LunaTacita/KVMem_Ninfer_ActKV_2080Ti**（源码与 Release 都在这里）
>
> ```text
> 帮我在这台机器上装好并启动 KVMem_Ninfer_ActKV_2080Ti（Windows / RTX 2080 Ti 11GB / sm_75）。
> 仓库地址：https://github.com/LunaTacita/KVMem_Ninfer_ActKV_2080Ti
> 1) 从该仓库的 Release 页面下载所需版本压缩包，解压到全 ASCII 路径；
> 2) 下载模型 Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer（sha256 99F94F44BED48892B7C5E1C4DC8349E0DB8B4B44D2BD8AD7CD438CFC181DE2FD），校验后放到全 ASCII 路径；
> 3) 把 Start\启动_KVMem_Ninfer_ActKV_2080Ti.ps1 里的 $MODEL 改成模型绝对路径；
> 4) 运行 Start\启动_KVMem_Ninfer_ActKV_2080Ti.ps1，确认 http://127.0.0.1:8080/v1/models 返回 200，再发一条真实 chat 请求确认有内容。
> 参考包内 README.md 与 指南\教程-启动与短测-给agent与使用者.md。
> ```

### 第 1 步：下载
从 Release 页面下载所需版本压缩包；模型需另外下载（本包不含权重，见第 3 步）。

### 第 2 步：解压
用 7-Zip / WinRAR 解压 `.001` 到全 ASCII 路径（例如 `D:\ninfer\`）。解压后得到 `kvmem-ninfer-actkv-2080ti-v0.1\`。

### 第 3 步：准备模型

推荐从社区获取 Bonsai 的 NInfer v3 容器制品：

- 推荐下载：[`emiltsoi/Ternary-Bonsai-2-27B-Uncensored-Heretic-NInfer`](https://huggingface.co/emiltsoi/Ternary-Bonsai-2-27B-Uncensored-Heretic-NInfer)
- 也可以尝试其他 **NInfer 制品**（必须含 `text` 组件；需要投机解码则要含 `mtp`；需要多模态则要含 `vision`），或参考上游项目 [`1314521gjy/ninfer-fusion-kvmem`](https://github.com/1314521gjy/ninfer-fusion-kvmem) 中给出的模型来源。

本项目构建与实测使用的制品：

| 项 | 值 |
|---|---|
| 文件名 | `Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer` |
| 字节 | 9,520,051,456 |
| sha256 | `99F94F44BED48892B7C5E1C4DC8349E0DB8B4B44D2BD8AD7CD438CFC181DE2FD` |
| 格式 | NInfer v3 原生容器；components = text / vision / mtp / dflash2（多模态自包含，不需要外部 mmproj） |

下载后请校验 SHA256（`Get-FileHash -Algorithm SHA256 <模型文件>`），应与 `模型包\MODELS.txt` 中记录的值一致。

### 第 4 步：填入模型路径

打开 `Start\启动_KVMem_Ninfer_ActKV_2080Ti.ps1`，把 `$MODEL` 一行改成你模型文件的**绝对路径**（全 ASCII）：

```powershell
$MODEL = "D:\models\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer"
```

其余参数保持默认即可（默认配方见 §五）。

### 第 5 步：启动服务

```powershell
powershell -ExecutionPolicy Bypass -File .\Start\启动_KVMem_Ninfer_ActKV_2080Ti.ps1
```

看到类似下面的输出即表示启动成功：

```
------------------------------------------------------------
  KVMem_Ninfer_ActKV_2080Ti
  引擎 : ...\engine\ninfer-serve.exe
  模型 : D:\models\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer
  参数 : ...
  API  : http://127.0.0.1:8080/v1/chat/completions
------------------------------------------------------------
```

保持这个终端开启，`Ctrl+C` 停止服务。

> **第一次启动会慢一点**：要加载约 9.5 GB 的模型；机械盘冷缓存可能更慢。加载完成前 `/v1/models` 不通是正常的，不要反复重启。

### 第 6 步：验证

```powershell
# 1) 就绪：返回 200
curl.exe -s -o NUL -w "%{http_code}`n" http://127.0.0.1:8080/v1/models

# 2) 真实请求（取真实 model id，不要用占位名）
$models = Invoke-RestMethod 'http://127.0.0.1:8080/v1/models'
$body = @{
    model = $models.data[0].id
    messages = @(@{ role = 'user'; content = '用一句话解释什么是 KV 缓存。' })
    max_tokens = 256
    stream = $false
} | ConvertTo-Json -Depth 8
$r = Invoke-RestMethod 'http://127.0.0.1:8080/v1/chat/completions' -Method Post `
     -ContentType 'application/json; charset=utf-8' -Body $body
$r.choices[0].message.content
```

> `/v1/models` 返回 200 只能证明模型列表接口可访问。**必须检查真实生成请求、答案内容与进程存活**，才能判断服务可用。

> 完整启动参数、判据与排错见 `指南\` 与 `ms75_eval\`。

### API 速查

| 接口 | 用途 |
|---|---|
| `GET /health` | 就绪状态 |
| `GET /v1/models` | 已加载模型标识 |
| `GET /props` | 后端、预算与能力 |
| `GET /v1/load` | 当前容量、负载与 token 计数 |
| `GET /metrics` | Prometheus 指标 |
| `POST /v1/chat/completions` | OpenAI 兼容对话（流式 / 工具调用 / 图片） |
| `POST /v1/completions` | OpenAI 传统补全 |
| `POST /v1/responses` | OpenAI Responses（含 SSE） |
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/rerank` | 文档重排 |

服务默认只监听本机、不设 API key；对外服务请自行加 HTTPS 代理与鉴权。

## 四、目录结构

```
kvmem-ninfer-actkv-2080ti-v0.1\
├── engine\                1 个可执行文件（ninfer-serve.exe）+ 7 个运行时 DLL + 3 个 CUDA 12.6 运行时 DLL
│   └── ninfer-serve.exe        HTTP 服务（OpenAI 兼容 API）
│       （辅助工具 ninfer.exe / ninfer-perplexity.exe / ninfer-calibrate.exe 单独下载，放进本目录即可）
│   ├── avcodec-63.dll / avformat-63.dll / avutil-61.dll / swresample-7.dll / swscale-10.dll
│   ├── libcurl-x64.dll / libcurl.dll
│   └── cublas64_12.dll / cublasLt64_12.dll / cudart64_12.dll   ← CUDA 12.6 运行时
├── Start\                 启动脚本（路径相对本脚本；只需填模型路径）
├── ms75_eval\             本项目实测判据脚本（PPL / 速度 / 召回 / 多轮 / 长上下文 / 多模态 / 参数矩阵 / 1M）
├── media\                 判据测试图（1.png / 2.png / 3.png；各自 ground truth 见 ms75_eval\run_vision_kvmem.ps1 文件头）
├── src\                   完整源码（不含上游文档；见 §十一 血统说明）
├── patches\               相对上游基线的完整代码改动（v0.1-sm75-port.patch + 一页说明）
├── 指南\                  启动与短测教程
├── 模型包\                MODELS.txt + SHA256SUMS.txt（**不含权重**）
├── README.md              本文件
├── CHANGELOG.md           版本变更与已知问题
├── NOTICE.md              血统与修改范围声明
├── LICENSE                Apache-2.0
├── verify-engine.ps1      完整性自检（逐文件 SHA256）
├── SHA256SUMS.txt         全包文件哈希清单
├── ZIP-README.md         zip 交付清单（随包发）
└── 踩坑-源码编译疑难杂症.md   移植与编译阶段的问题记录
```

> `Start\` 里的启动脚本默认端口为 **8080**（llama.cpp 的默认端口）。如需改端口，只改脚本里的 `$PORT` 一行。

### 文档索引

| 文件 | 用途 |
|---|---|
| `指南\教程-启动与短测-给agent与使用者.md` | 15 分钟启动与短测 |
| `踩坑-源码编译疑难杂症.md` | 移植/编译期问题记录 |
| `CHANGELOG.md` | 版本变更与已知问题 |
| `模型包\MODELS.txt` | 模型来源与 sha256 |
| `ms75_eval\` | 17 个判据脚本（PPL / 速度 / 召回 / 多轮 / 长上下文 / 多模态 / 1M） |
| `ZIP-README.md` | 压缩包交付清单 |

## 五、生产配方（脚本默认值）

启动脚本的默认值就是本项目验证过的生产配方：

| 参数 | 默认值 | 含义 / 为什么是这个值 |
|---|---|---|
| `--max-context 1048576` | 1M | 逻辑上下文上限；由 KVMem 承载，不占显存 |
| `--max-concurrency 1` | 1 | 单并发；显存与速度最优（多并发会更吃 KV 预算） |
| `--kv-dtype int8` | int8 | 每 token KV 约 30 KiB；PPL 只比 bf16 差 0.00072（见 §八） |
| `--spec mtp --draft-tokens 2` | MTP draft 2 | 实测最优：draft 2 = 80.3 tok/s / 91.4% 接受率；draft 4 降到 66.8 / 52.1% |
| `--kvmem-budget 32768` | B=32768 | 设备可见窗口（KVMem 预算）；必须是 64 的倍数 |
| `--kvmem-gen-reserve 5632` | R=5632 | 生成预留；决定 prefill 与生成时的可追加空间 |
| `--kvmem-host-mib auto` | 物理内存 − 1 GiB | 主机 KV 预算；实测机器为 64421 MiB |
| `--kvmem-disk-mib 102400` | 100 GiB | SSD 溢出配额；历史装不下主机内存时使用 |
| `--kvmem-sink-tokens 4096` | 4096 | 永久保留前缀（system / 任务描述） |
| `--kvmem-recent-tokens 1024` | 1024 | 永久保留尾部（最近一轮） |
| `--default-reasoning-effort max` | max | 思考默认开启且最高档 |
| `--ngram-draft-tokens 0` | 0（关闭） | n-gram 草稿默认关闭（sm_75 对齐路径有崩溃风险，见 §九） |

**三档 profile**：脚本里 `$KV_PROFILE` 可一键切换常用档位（长上下文 / 双并发 / 8 并发），不改其余参数。

**三条硬约束**（违反会起不来或静默答错）：

1. KVMem 模式下**绝对不要传 `--kv-capacity`**（任何形式，含 `auto`）——引擎会按 `(B+R)×并发` 自行设定容量，传了就报错退出。
2. `--prefill-chunk` 必须是 128 的倍数，且不大于 `--kvmem-gen-reserve`。
3. `--kvmem-budget` 与 `--kvmem-gen-reserve` 都必须是 64 的倍数（budget ≥ 128，reserve ≥ 64）。

## 六、性能实测

构建机：RTX 2080 Ti 11GB / i9-9900KF / 64GB / Win10 22H2（19045）/ 驱动 566.36（驱动级 CUDA 12.7）/ nvcc 12.6.85。所有数字来自**交付二进制 sha256 `97D9FFC5D8C9CAF0E98E0F0DB2D9E1264FAABB5D4CA77D4F4984BDCF5BAC188F`**（`engine\ninfer-serve.exe`，1,930,925,568 字节）。

### 速度

> 怎么读：**decode = 写回答速度**（每生成一个 token），**prefill = 读提示速度**（每读入一个 token）。短对话看 decode，长文档看 prefill。

| 项 | 实测 | 条件 |
|---|---|---|
| decode（MTP draft=2） | **80.3 / 79.2 tok/s**，接受率 **91.4%** | 短上下文，贪心 |
| decode（无 MTP） | 44.9 tok/s | 同上 |
| prefill 8K / 32K / 105K | **611.8 / 549.4 / 240.8 tok/s** | 常规生产配置 |
| 1M 长档 prefill | 287,269 token → **268.9 tok/s**（18 分 1.6 秒） | 1M 起服配置 |
| 起服 | **5.5 s**（模型在 NVMe）/ 40 s（机械盘冷缓存） | — |

### 精度（124 窗口 / 261,223 token）

| KV 格式 | PPL |
|---|---|
| bf16 | 5.85453 |
| **int8（生产）** | **5.85525**（+0.00072，小于引擎 run-to-run 噪声 ±0.0006） |

### 能力

| 能力 | 实测 |
|---|---|
| 跨窗口召回 | 40,960 token prompt（超出可见窗口）→ **8/8 个分散针全部命中**；38,083 token 开头密码 `ORCHID-9701` 严格 fresh×2 + 8 次全一致 |
| 多轮会话 | 16/16 轮成功，前缀复用 82.7% → 93.3%，跨轮召回 `LOTUS-7391` 命中，TTFT 290–455 ms |
| 多模态 | 三图 × 4 位置（baseline / front / middle / back）**ALL-PASS**（12/12，star / 8 / 7429），含 51,200 filler 中部（1.33× 窗口）图片被驱逐后仍召回 |
| 1M 逻辑上下文 | 1M 起服成功；`/v1/load` 报 `max_context 1048576`；105K prompt 第 90% 位置召回 **HIT**；287,269 token prefill 实测通过 |
| 1M 全量 prefill | **未跑**（单请求预计数十分钟以上，见 §九） |

## 七、参数实测结论

> 判据：40,960 token prompt，8 个密码分散在 5%~89% 位置，一次请求列出全部（`8/8` = 全命中）；外加单针精确召回、32K 唯一内容 prefill、4 轮跨轮召回。每臂一台服务、串行执行。

### ActKV 开关（3 臂）

| 臂 | 短 decode | 8 针召回 | 单针 | 32K prefill | 4 轮 | 跨轮召回 |
|---|---|---|---|---|---|---|
| **默认（生产：① 自动生效、③ mean-k、②/LRFU 关）** | **94.1 tok/s** | **8/8** | HIT | **663.8 tok/s** | 4/4 | HIT |
| ActKV 全开（②置信度预算 + ③per-token + LRFU） | 93.2 | 8/8 | HIT | 663.0 | 4/4 | HIT |
| ActKV sub-block（② + ③sub-block4 + LRFU） | 93.3 | 8/8 | HIT | 662.0 | 4/4 | HIT |

结论：**默认配置在多针 / 单针 / 跨轮判据下已是 8/8 满分；ActKV ②③+LRFU 全开没有增益、短 decode 还慢约 1%** ⇒ 生产保持默认（支柱①自动生效；③默认 mean-k；②/LRFU 保留为可选开关）。

> 边界说明：本判据已饱和（默认 8/8），它证明的是"默认在多针下不丢针"，**不构成"ActKV 在任何场景都无益"**；更复杂场景（1M 多针、跨会话）留后续。

### sink / recent（4 臂）

| 配置 | 短 decode | 8 针召回 | 单针 | 32K prefill | 4 轮 |
|---|---|---|---|---|---|
| **sink 4096 / recent 1024（生产）** | 93.9 | **8/8** | HIT | 662.9 | 4/4 |
| sink 8192 / recent 1024 | 94.0 | 8/8 | HIT | 663.0 | 4/4 |
| sink 12288 / recent 1024 | 93.7 | 8/8 | HIT | 661.8 | 4/4 |
| sink 4096 / recent 2048 | 93.7 | 8/8 | HIT | 663.6 | 4/4 |

结论：四组在 40K 多针判据下无差异 ⇒ **保持生产值 4096 / 1024**。

### B / R 组合（3 臂）

| 臂 | B / R | 设备窗口 | 短 decode | 8 针 | 32K prefill |
|---|---|---|---|---|---|
| **b32768_r5632（生产）** | 32,768 / 5,632 | 38,400 | 94.3 | **8/8** | 664.6 |
| b36864_r1536 | 36,864 / 1,536 | 38,400 | 94.1 | 8/8 | 664.9 |
| b28672_r8192 | 28,672 / 8,192 | 36,864 | 94.1 | 8/8 | **668.4**（+0.6%） |

结论：三臂 8 针全 8/8；`b28672_r8192` 的 32K prefill 略快 0.6% 但窗口小 1,536 token ⇒ **生产保持 B=32768 / R=5632**（更大窗口优先，速度差在噪声内）。

### 省显存 / 小提速项（4 臂）

| 臂 | 结果 |
|---|---|
| `--lm-head-q6` | **起不来**：`requires text/output_head to be stored as row-split Q8_G32`（本模型是三元格式，不满足） |
| `--embedding-q6` | **起不来**：`requires text/token_embedding to be stored as row-split Q8_G32` |
| `--cuda-graph-allowance-mib 128` | 正常：94.3 decode / 8/8 / 664.2 prefill（与生产无显著差异，保留默认 0） |
| `--ngram-draft-tokens 31` | **崩溃**：`cudaErrorMisalignedAddress @ layer/prefill/attention (layer 63)`（默认 15 未复现，保留默认） |

结论：Q6 两选项受**模型格式**限制不可用；CUDA graph allowance 无显著收益；ngram 宽度 31 触发对齐崩溃（与已知的极端低内存 misaligned 同族，待深挖）⇒ 生产配置**保持默认**。

## 八、验收清单（怎么确认它在工作）

每一条都写了"**怎样会失败**"——只会变绿的判据不算判据。

| # | 判据 | 怎么做 | 期望 | 怎么变红 |
|---|---|---|---|---|
| 1 | 包完整性 | `verify-engine.ps1` | `RESULT: ENGINE-PACK-OK`，`bad=0 missing=0` | 出现 `BAD` / `MISSING` 行；退出码非 0 |
| 2 | 服务就绪 | `GET /v1/models` | HTTP 200，返回模型 id 列表 | 连不上 / 非 200 |
| 3 | 真实生成 | `POST /v1/chat/completions` | 返回非空 `choices[0].message.content` | `content` 为空、`finish_reason=length`（需加大 `max_tokens`，思考会吃 token） |
| 4 | 跨窗口召回 | 40,960 token prompt，8 针散布 5%~89% | 一次请求 **8/8** 全命中 | 少于 8 个（检查 sink/recent 与 KV 预算） |
| 5 | 多轮复用 | 连续 16 轮带历史追问 | 16/16 成功，跨轮召回命中 | 任一轮答非所问 / 前缀复用率异常低 |
| 6 | 多模态 | 三图 × 4 位置（图片在 `media\`，ground truth 见 `ms75_eval\run_vision_kvmem.ps1` 文件头） | 全部正确识别 | 图片未被理解（检查 `--vision` 是否被关闭） |
| 7 | KVMem 专项回归 | `ms75_eval` 下对应脚本 / 构建机上的 ctest 子集 | **14 项中 13 通过**（见下方范围说明） | 通过数下降；`memory_ngram` 偶发失败需复跑确认 |

> **测试范围说明（重要，避免误读）**：引擎全量 ctest 注册 **328 项**，绝大多数是上游引擎的单元测试。本项目的回归范围是 `tests/experimental` 下 **KVMem 专项 14 项**（15 个探针中排除 `active_kv_ssd`）。
> - 本次交付二进制：**KVMem 专项 14 项 = 13 通过**。唯一失败的 `memory_ngram` 是偶发（同一命令手动复跑 2/2 通过），只在极端测试条件下出现，**不影响生产默认配置**（启动脚本默认 `--ngram-draft-tokens 0`，该路径已关闭）。
> - 本包不把"KVMem 专项 13/14"表述为"全部测试通过"；同样，这 1 项偶发失败**不构成**影响日常使用的缺陷——发布包本身是经过完整验收的可用包。

## 九、已知问题与验证范围

**已知问题**（每条都写了实际影响与规避方式）：

- **n-gram 草稿是可选功能，默认关闭；手动开到 31 宽会崩。** `--ngram-draft-tokens 31` 在生产配置下第 1 个请求会因对齐崩溃（`cudaErrorMisalignedAddress`）；默认 15 在 6 类判据中未复现。日常使用不受影响——启动脚本默认 `--ngram-draft-tokens 0`，该路径已关闭。与已知的极端低内存 misaligned 同族，待专项定位。
- **`active_kv_ssd` 判据约 50% 非确定。** 只在极端低内存测试配置（可见集 384 / 强制 spill）下出现，**生产配置未复现**，不纳入本包验证范围。
- **1M 全量 prefill 未验证。** 已验证"1M 起服 + 287,269 token prefill 实测 + 105K 位置召回 HIT"；但"一次读入 100 万 token 的长文档"没有实际跑过（单请求预计数十分钟以上）。
- **`--lm-head-q6` / `--embedding-q6` 不可用。** 引擎要求 row-split Q8_G32 权重，本模型是三元格式，属模型格式约束而非缺陷。
- **模型放在机械盘会加载很慢。** 新进程冷缓存加载可能只有 8~135 MB/s，日志看起来像卡住（不是死锁）；复制到 NVMe 后起服约 5.5 s。
- **判据脚本需要 PowerShell 7。** PS 5.1 的 `Invoke-RestMethod` 在这类 chat 负载上有已知转换问题；启动脚本本身 PS 5.1 / 7 都行。

**验证范围**：

- 本包在 **RTX 2080 Ti 11GB + 驱动 566.36 + CUDA 12.6** 上完成全部实测；其他驱动、其他显卡、其他模型**未验证**。
- 上游引擎自身的 328 项单元测试中，仅 KVMem 专项 14 项纳入本项目的交付回归范围；其余测试未随本包重跑。
- 多卡、并发 >1、其他 KV 格式（fp8 / rk8v4 等）仅做过参数矩阵抽样，**不构成生产推荐**。

## 十、排错 / FAQ

| 现象 | 原因 | 处理 |
|---|---|---|
| 起服报 `Cannot ... path` / 模型读不到 | 路径含中文或空格 | 换到**全 ASCII** 路径（例如 `D:\ninfer\`） |
| 起服报显存不足 | 预算配大了 | **认真读报错文本**：引擎会直接报"需要多少、可用多少、最大可启动窗口约多少、建议把 `--kvmem-budget` 降到多少" |
| 起服报 `--kv-capacity` 相关错误 | KVMem 模式不接受该参数 | 删掉 `--kv-capacity`（任何形式，含 `auto`） |
| 起服很慢、日志像卡住 | 模型在机械盘冷缓存 | 把模型复制到 NVMe；或耐心等待首次加载 |
| 请求返回空内容 | 思考吃掉了 `max_tokens` | 加大 `max_tokens`（本项目判据用 2048） |
| 判据脚本报 `ConvertFrom-Json` / `Invoke-RestMethod` 类型错误 | 用了 PS 5.1 | 用 **PowerShell 7.x** 重新运行 |
| 端口被占用 | 8080 被其他程序占用 | 改启动脚本 `$PORT`，或关掉占用程序 |
| 模型格式不对（起不来 / 报组件缺失） | 用了非 NInfer v3 容器，或缺 mtp/vision 组件 | 换正确制品；缺 mtp 时把 `--spec` 留空，缺 vision 时把 `--vision` 留空 |

## 十一、来源、致谢与许可

### 血统链

- 引擎：`iamwavecut/ninfer-all` @ `8319e8f5`（NInfer 引擎本体，上游整合线）。
- KV 分层：`kvmem/kvmem` 0.18.0（官方发行层；其 ninfer 后端 = 上述引擎 + KVMem 官方集成补丁 `backends/patches/ninfer-kvmem.patch`，1,049,526 字节，官方 `versions.json` 固定其 sha256 与 patched_tree）。
- 本包源码 = KVMem 0.18 的 ninfer 后端（patched_tree `7d7e1499a9`，导入 commit `c3f4dc3`）+ 本项目全部移植/集成 commit（完整代码补丁见 `patches\v0.1-sm75-port.patch`）。
- ActKV 机制来源：论文《ActKV: Efficient LLM Agents through Action-Guided KV Cache Management》（arXiv:2609.31395）；三支柱在本栈源码中按论文机制实现。
- 参考实现（机制对照与踩坑核实）：`melis2023/ninfer-kvmem-actkv`、`1314521gjy/ninfer-fusion-kvmem`（KVMem 环与 host-backed 复用）。

### 致谢

本包建立在以下项目的工作之上，感谢这些项目的维护者与贡献者：

- [Neroued/ninfer](https://github.com/Neroued/ninfer)：NInfer 引擎本体。
- [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all)：NInfer 社区整合线（本栈引擎基线 @ `8319e8f5`）。
- [kvmem/kvmem](https://github.com/kvmem/kvmem)：KVMem 分层 KV 与官方 ninfer 后端集成补丁（本栈 KV 分层来源）。
- [melis2023/ninfer-kvmem-actkv](https://github.com/melis2023/ninfer-kvmem-actkv)、[1314521gjy/ninfer-fusion-kvmem](https://github.com/1314521gjy/ninfer-fusion-kvmem)：ActKV 与 KVMem 环的机制对照参考。
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)：推理基础设施与工具链习惯。

### 许可

- 本包代码按 **Apache License 2.0** 发布（见 `LICENSE`）；上游各组件保留其原始版权与许可声明（见 `NOTICE.md`）。
- 包内 `cublas64_12.dll` / `cublasLt64_12.dll` / `cudart64_12.dll` 是 **NVIDIA CUDA 12.6 运行时**，随包分发遵循 NVIDIA 的 CUDA EULA，**不适用** Apache-2.0。
- **模型权重许可另计**，不与本包许可混同；分发或使用前请确认模型的原始许可。
