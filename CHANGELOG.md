# CHANGELOG · KVMem_Ninfer_ActKV_2080Ti

版本：**KVMem-Ninfer-ActKV-2080Ti-v0.1.1**
本包 = 引擎基座（KVMem 的 ninfer 后端）+ 本项目全部 sm_75 移植与集成改动（完整代码补丁见 `patches\v0.1.1-sm75-port.patch`）；上游署名与精确版本见 `NOTICE.md`。

本包**不含模型权重**，也**不含上游项目的 README 与说明文档**——`src\` 只保留源码与法律文件（`LICENSE` / `NOTICE` / 第三方许可证），上游文档请到各自仓库查阅。

## v0.1.1

### 新增
- **内置 llama.cpp WebUI**：llama.cpp 官方 WebUI（b11796，70 文件）在构建期经 `-DNINFER_WEBUI_DIR` 静态嵌入 `ninfer-serve.exe`；起服后 `GET /` 直接返回网页（HTTP 200 / `text/html`），浏览器打开 `http://127.0.0.1:8080/` 即可对话。**不需要 Node.js、不需要前端资源目录**。
- **流式会话断线续传**：服务端把流式生成与 HTTP 连接解耦——客户端刷新页面、切换标签页或短暂断网不再中断生成；页面重开后按会话身份（`X-Conversation-Id`）从已收到的位置继续推送，**不重复、不丢失已生成内容**。新增 `GET /v1/stream` / `POST /v1/streams/lookup` / `DELETE /v1/stream` 三条路由（协议细节与缓冲上限见 README 与 `指南\已知问题与运行时限制.md`）。
- **启动脚本增强**：起服前打印配置摘要（档位 / B / R / 上下文 / MTP / CUDA 图 / 会话保留 / 超时（排队 + WebUI）/ 思考档位 / sink / recent / WebUI / API / 日志）与完整启动命令；控制台编码切换放在所有输出之前（配置摘要不会被清屏冲掉）；`--log-stats-interval-ms 5000` 每 5 秒打印吞吐统计；每请求输出 `req#N done | prompt | output | cache | TTFT | prefill | decode`。
- **内置 WebUI 请求超时可配置**：新增启动参数 `--webui-request-timeout-ms N`（默认 `3600000` = 1 小时；`0` = 不设前端超时），并通过 `/props` 的 `webui.request_timeout_ms` 下发给 WebUI；长 prompt 的 KVMem prefill 常需 15–30 分钟，避免浏览器侧误报超时。启动脚本里对应 `$WEBUI_REQUEST_TIMEOUT_MS`（默认 3600000）。
- 生产配方默认 `--kvmem-sessions 1`：只保留 1 个会话历史，避免「长上下文之后新开的对话复用旧会话信息」导致输出多样性下降（详见 `指南\已知问题与运行时限制.md` §11）；需要多会话并行复用时可调大。

### 修复
- **attention envelope 映射补齐**：长 prompt 跨窗口时 envelope 尾部页未映射的问题（`decode.cpp`）。
- **restore 回绕修复**：KV 恢复时 `compact_end` 改用 int64 并加 clamp，消除回绕风险（`kvmem_window.cpp`）。
- **small-T 参数 staging**：`t2_small_t_i8_kernel` 的 `parents` 改为逐字段从参数缓冲 staging 到共享内存，避免整体赋值的跨代/复用问题。
- **日志可读性改进**：吞吐统计行新增 `prefill-units` 与 `host` / `wait` 细分、`kv-history restore/spill/swaps` 窗口增量；`output 0` 的请求 TTFT 显示 `n/a (no token)`，避免把取消请求的退化值误读为首 token 时间。
- **致命错误改为可诊断退出**：遇到不可恢复的 CUDA / 内部错误时，进程打印 `ninfer-serve: FATAL: <详情>` 与重启提示后以退出码 1 退出，便于日志定位与服务管理；不会自动重启（是否重启由使用者决定）。

### 默认配置
- 默认使用常规（非投机）解码路径；**MTP 投机解码与 n-gram 草稿为可选实验功能，默认关闭**（见 `指南\已知问题与运行时限制.md`）。
- `--device-profile` 固定 `off`（启动脚本默认）。
- 默认思考档 `--default-reasoning-effort xhigh`：模型内置的 Qwen3.8 模板只接受 `low` / `medium` / `xhigh`（默认 `xhigh`）；`high` / `max` 会被引擎映射到最近档 `xhigh`、`minimal` → `low`。其他模型的模板若接受更多档可改回 `max`。

### 交付二进制
- `engine\ninfer-serve.exe` sha256 = `1DF23CA3B0851BC96D3E6C175B8F003A885F17C83BC05DC566521A6B84A4B6CB`，1,941,118,976 字节（内置 llama.cpp WebUI b11796；构建标识 `v0.1.1`）。
- 辅助工具（单独 Release 资产，可选）：`ninfer-perplexity.exe` / `ninfer-calibrate.exe` / `ninfer.exe`，与主 exe 同批构建。

### 验收
- 包完整性 `verify-engine.ps1` = `ENGINE-PACK-OK`；起服 + `/v1/models` 200 + 真实 chat 生成非空；`/props` 返回 `webui.request_timeout_ms`；内置 WebUI `GET /` = 200 / `text/html`。
- 断线续传实测：流式中途断开后 `lookup` 仍能找回会话且服务端继续生成（缓冲 44,677 → 58,908 B），`GET /v1/stream?from=25,005` 续传 33,903 B（25,005 + 33,903 = 58,908，字节完全对账，无重复无丢失）；从 0 全量重放含 `[DONE]`；`DELETE /v1/stream` 后 `lookup` 不再命中、`GET /v1/stream` 返回 404；浏览器 F5 刷新自动续上（真实页面实测）。
- KVMem 专项回归 14 项 = 13 通过（`memory_ngram` 偶发，复跑通过；默认配置已关闭该路径）。开发期早期构建上全量 ctest 258 项 100% 通过；当前构建注册 314 项，交付回归范围为专项 14 项。

## v0.1

### 引擎 / sm_75 移植
- 对齐安全的 `cp_async` 字节拷贝替代（Turing 无 cp.async 硬件指令；三个私有副本统一）
- Turing MMA 降级（ldmatrix 原生可用；fp8 / nvfp4 MMA 在 sm75 隔离）
- INT8 prompt kernel（Bc=16）、small-T 路由、fp8 A8 路由门控、GDN chunked 路径
- 构建：CMake 架构白名单 +75、CUDA 下限 12.6、`NINFER_SM75`、工具链 pin（MSVC 14.38.33130 + VS2026 + CUDA 12.6）、MSVC 依赖前缀字节级修复
- 运行环境：驱动 566.36 + CUDA 12.6（全部实测数据来自该组合）

### KVMem 0.18 集成
- 三层 KV：设备可见窗口 / 主机原生负载 / SSD 溢出；1M 逻辑上下文不占显存
- 自动 Host/SSD 预算（auto fit）；容量不足时报出"最大可启动窗口 + 建议降哪个参数"
- 设备池 `B+R = 38,400` token（int8，600 页）

### ActKV 三支柱
- 支柱①：动作区检索 query + 强制保留（system / 任务描述 / 媒体 / 最近）——自动生效
- 支柱②：置信度驱动预算（`--kvmem-conf-*`）——默认关，可选
- 支柱③：可插拔打分器（mean-k 默认 / per-token / sub-block / deltanet）

### 多模态
- 三图 × 4 位置（baseline / front / middle / back）判据 ALL-PASS（star / 8 / 7429），含 51,200 filler 中部驱逐后召回
- vision 塔默认 CPU 驻留（实测不损文本性能）

### 判据 / 交付
- `sm75_eval`：17 个判据脚本 + 1 个设备 profile JSON，覆盖 PPL / 速度 / 召回 / 多轮 / 长上下文 / 多模态 / 参数矩阵 / 1M
- 启动脚本：逐个参数四段式中文注释（是什么 / 为什么有它 / 取值与后果 / 实测结论）；兼容 PowerShell 5.1 与 7
- 回归范围：`tests/experimental` 下 KVMem 专项 14 项（本交付二进制 13/14，`memory_ngram` 偶发、复跑通过）

## 版本号约定

- 对外发布版本 = `KVMem-Ninfer-ActKV-2080Ti-v0.1.1`（源码分支 / tag 同为 `v0.1.1`）
- 包内 `src\VERSION` 是上游 ninfer 引擎的版本标识（`0.12.0-rtx3090`），**不是**本发布版本

## 运行时限制

见 `指南\已知问题与运行时限制.md`（可选投机路径、1M 全量 prefill 范围、判据脚本环境等）。
