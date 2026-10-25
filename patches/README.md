# patches\ — 相对上游基线的完整代码改动

本目录有两个文件：

- `v0.1.1-sm75-port.patch` —— 本发布包源码相对**技术基线**的完整代码 diff（纯源码改动）。
- 本说明页。

## 基线

- **技术基线**：`kvmem/kvmem` 0.18 的 **ninfer 后端**（= `iamwavecut/ninfer-all@8319e8f5` + KVMem 官方集成补丁 `ninfer-kvmem.patch`，patched_tree `7d7e1499a9`）。

本补丁 = 相对上述技术基线的单一全量 diff，共 **135 个文件、+6897 / −347 行**（564,112 B）。

## 范围

- 只含**源码改动**。`sm75_eval\`（判据脚本）与随包文档不在 diff 内 —— 它们以文件形式随包提供。
- `src\` 是应用完本补丁后的完整源码；本补丁用于审计与在基线上复现。

## 用法

在 `kvmem/kvmem` 0.18 的 ninfer 后端（patched_tree `7d7e1499a9`）工作树上：

```powershell
git apply --check <解压目录>\patches\v0.1.1-sm75-port.patch   # 先验证可应用
git apply <解压目录>\patches\v0.1.1-sm75-port.patch
```
