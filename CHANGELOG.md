# CHANGELOG · KVMem_Ninfer_ActKV_2080Ti

版本：**KVMem-Ninfer-ActKV-2080Ti-v0.1**（首个本地发布）
基线：`iamwavecut/ninfer-all@8319e8f5` + `kvmem/kvmem` 0.18 ninfer 集成补丁（导入 commit `c3f4dc3`）
本包源码 = 基线 + 本项目全部移植/集成 commit（完整代码补丁见 `patches\v0.1-sm75-port.patch`，即相对上游基线 `c3f4dc3` 的单一全量 diff）。

本包**不含模型权重**，也**不含上游项目的 README 与说明文档**——`src\` 只保留源码与法律文件（`LICENSE` / `NOTICE` / 第三方许可证），上游文档请到各自仓库查阅。

## v0.1

### 引擎 / sm_75 移植

- 对齐安全的 `cp_async` 字节拷贝替代（Turing 无 cp.async 硬件指令；三个私有副本统一）
- Turing MMA 降级（ldmatrix 原生可用；fp8 / nvfp4 MMA 在 sm75 trap，调用者已隔离）
- INT8 prompt kernel（Bc=16）、small-T 路由、fp8 A8 路由门控、GDN chunked 路径
- 构建：CMake 架构白名单 +75、CUDA 下限 12.6、`NINFER_SM75`、工具链 pin（MSVC 14.38.33130 + VS2026 + CUDA 12.6）、MSVC 依赖前缀（`rules.ninja`）字节级修复
- **移植选型前提**：运行环境锁定 **驱动 566.36 + CUDA 12.6**（2080 Ti 游戏与本地 AI 双用途下的最稳交集）；全部实测数据来自该组合

### KVMem 0.18 集成

- 三层 KV：设备可见窗口 / 主机原生负载 / SSD 溢出；1M 逻辑上下文不占显存
- 自动 Host/SSD 预算（auto fit）；容量不足时报出"最大可启动窗口 + 建议降哪个参数"
- 设备池 `B+R = 38,400` token（int8，600 页）

### ActKV 三支柱

- 支柱①：动作区检索 query + 强制保留（system / 任务描述 / 媒体 / 最近）——自动生效
- 支柱②：置信度驱动预算（`--kvmem-conf-*`）——默认关，可选
- 支柱③：可插拔打分器（mean-k 默认 / per-token / sub-block / deltanet）

### 投机解码

- MTP：实测 draft=2 最优（decode 80.3 tok/s、接受率 91.4%）；draft=4 只有 66.8 / 52.1%
- n-gram 草稿：v0.1 默认关闭（sm75 对齐路径有崩溃风险；见「已知问题」）

### 多模态

- 三图 × 4 位置（baseline / front / middle / back，共 12 组）判据 ALL-PASS（star / 8 / 7429），含 51,200 filler 中部驱逐后召回
- vision 塔默认 CPU 驻留（实测不损文本性能）

### 判据 / 交付

- `ms75_eval`：**17 个判据脚本** + 1 个设备 profile JSON；多模态判据的三张测试图随包放在 `media\`（`1.png` / `2.png` / `3.png`，各自 ground truth 见 `run_vision_kvmem.ps1` 文件头）；覆盖 PPL / 速度 / 召回 / 多轮 / 长上下文 / 多模态 / 参数矩阵 / 1M
- 启动脚本：173 个参数逐个四段式中文注释（是什么 / 为什么有它 / 取值与后果 / 实测结论）
- 启动脚本兼容 **PowerShell 5.1 与 7**（引擎写到 stderr 的状态行不会被当成脚本错误终止）
- 交付二进制 sha256：`97D9FFC5D8C9CAF0E98E0F0DB2D9E1264FAABB5D4CA77D4F4984BDCF5BAC188F`
- 回归范围：`tests/experimental` 下 **KVMem 专项 14 项**（15 个探针中排除 `active_kv_ssd`）。
  - 本交付二进制：KVMem 专项 **13/14 通过**，唯一失败 `memory_ngram` 为偶发（同命令手动复跑 2/2 通过）。
  - 引擎全量 ctest 注册 328 项，其中非 KVMem 的单元测试本次未重跑——本 CHANGELOG 不把"13/14"表述为"全部测试通过"。

### 已知问题

- `active_kv_ssd` 极端低内存判据约 50% 非确定（可见集 384 / 强制 spill）；**生产配置未复现**，不纳入本包验证范围。
- n-gram 路径对齐问题（宽度 31 必崩、默认 15 偶发；v0.1 默认关闭）。
- 1M 全量 prefill 未验证（发布范围：1M 起服 + 287,269 token 实测 + 105K 90% 位置召回）。
- 判据脚本需 PowerShell 7（PS 5.1 的 `Invoke-RestMethod` 在 chat 负载上有已知转换问题）。

## 版本号约定

- 发布版本 = `KVMem-Ninfer-ActKV-2080Ti-v0.1`
- 包内 `src\VERSION` 是上游 ninfer 仓库的版本标识（`0.12.0-rtx3090`），**不是**本发布版本
