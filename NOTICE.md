# NOTICE · 血统与修改范围

本发布包是 **KVMem / NInfer 栈在 sm_75（Turing，RTX 20 系）上的源码级移植与集成**。

## 上游血统

- 引擎上游：`iamwavecut/ninfer-all` @ `8319e8f5`（NInfer 引擎本体）。
- 技术基线：`kvmem/kvmem` 0.18（KVMem 分层 KV；其 ninfer 后端集成补丁）。
- ActKV 机制来源：论文《ActKV: Efficient LLM Agents through Action-Guided KV Cache Management》（arXiv:2609.31395）；三支柱在本栈源码中按论文机制实现。
- 参考实现（机制对照与踩坑核实）：`melis2023/ninfer-kvmem-actkv`、`1314521gjy/ninfer-fusion-kvmem`（KVMem 环与 host-backed 复用）。

## 本包的修改范围

1. **sm_75 移植**：对齐安全的 `cp_async` 字节拷贝替代、Turing MMA 降级、INT8 prompt kernel（Bc=16）、small-T 路由、fp8 A8 路由门控、GDN chunked 路径、`-rdc=false` 下的设备变量语义修正。
2. **KVMem 0.18 集成**：三层 KV（设备窗口 / 主机 / SSD）、1M 逻辑上下文、自动 Host/SSD 预算、容量不足时给出可操作建议。
3. **ActKV 三支柱**：动作区检索 query + 强制保留（源码内自动生效）、置信度驱动预算（可选）、可插拔打分器（mean-k 默认）。
4. **判据资产**：`ms75_eval\` 下的本项目实测判据脚本（PPL / 速度 / 召回 / 多轮 / 长上下文 / 多模态 / 参数矩阵 / 1M）。

## 关于上游文档

本包**不附带**上游项目的 README 与说明文档——它们不属于本栈，请到各自上游仓库查阅。
`src\` 内只保留源码与法律文件（`LICENSE` / `NOTICE` / 第三方许可证）。

## 许可

Apache 2.0（见 LICENSE）。上游各组件保留其原始版权与许可声明。
模型权重许可另计，不与本包许可混同。
CUDA 运行时 DLL 随包分发遵循 NVIDIA CUDA EULA，不适用 Apache-2.0。