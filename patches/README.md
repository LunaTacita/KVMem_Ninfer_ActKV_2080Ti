# patches\ — 相对上游基线的完整代码改动

本目录有两个文件：

- `v0.1-sm75-port.patch` —— 本发布包源码相对上游基线的**完整代码 diff**（纯源码改动）。
- 本说明页。

## 基线

- 引擎上游：`iamwavecut/ninfer-all@8319e8f5`
- 技术基线：`kvmem/kvmem` 0.18 的 ninfer 后端集成补丁（导入 commit `c3f4dc3`）

## 范围

- 只含**源码改动**。`ms75_eval\`（判据与构建工具）与随包文档不在 diff 内 —— 它们以文件形式随包提供。
- `src\` 是应用完本补丁后的完整源码；本补丁用于审计与在基线上复现。

## 用法

```powershell
git clone <上游基线仓库> ninfer-all
cd ninfer-all
git checkout 8319e8f5
git apply --check <解压目录>\patches\v0.1-sm75-port.patch
git apply        <解压目录>\patches\v0.1-sm75-port.patch
```

> 补丁与 `src\` 同源；两者不一致时以 `src\` 为准。