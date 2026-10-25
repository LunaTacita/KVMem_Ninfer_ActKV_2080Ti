# 7z 交付清单 · 版本 KVMem-Ninfer-ActKV-2080Ti-v0.1.1（本页在压缩包内）

> **本包不含模型权重**（9.5 GB，单独下载，见 `模型包\MODELS.md`）。
> 先读 `README.md`；15 分钟快速验证见 `指南\教程-启动与短测-给agent与使用者.md`；运行时限制见 `指南\已知问题与运行时限制.md`。

## 一、这个压缩包里是什么

| 项 | 值 |
|---|---|
| 包名 | `KVMem-Ninfer-ActKV-2080Ti-v0.1.1.7z` |
| 给谁 | RTX 20 系（sm_75，Turing）单卡、想在本机跑 27B + 1M 逻辑上下文的人 |
| 内容 | `engine\`（`ninfer-serve.exe` + 10 个 DLL；perplexity / calibrate / CLI 三个辅助 exe 另作为独立资产）+ `Start\` 启动脚本 + `sm75_eval\` 判据 + `src\` 源码 + `patches\` + 文档 |
| 解压后 | **2.637 GiB / 2,583 文件**（含本清单 `SHA256SUMS.txt`，共 2,582 条哈希） |
| 解压后 sha256 | 见随包 `SHA256SUMS.txt` |
| 7z 大小 | 发布时回填（见 Release 资产大小） |
| 主 exe 指纹 | `ninfer-serve.exe` sha256 = `1DF23CA3B0851BC96D3E6C175B8F003A885F17C83BC05DC566521A6B84A4B6CB`（1,941,118,976 B，内置 WebUI b11796） |

## 二、四条交付提醒（请一并转给接收方）

1. **解压用 7-Zip / WinRAR / `tar -xf`**。Windows 资源管理器自带的解压对中文文件名会乱码——那是它的老毛病，不是包坏了。解压路径**必须全 ASCII**（原生 exe 读不了中文路径）。
2. **先放模型，再起服务**：`模型包\MODELS.md` 写了下载、放哪、怎么校验。放好后把 `Start\启动_KVMem_Ninfer_ActKV_2080Ti.ps1` 的 `$MODEL` 改成模型绝对路径，起服务；就绪判据 `curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8080/v1/models` ⇒ `200`；内置 WebUI 在 `http://127.0.0.1:8080/`。
3. **长任务等待上限**：内置 WebUI 一次对话默认等待 1 小时（脚本 `$WEBUI_REQUEST_TIMEOUT_MS`，`0` = 不设前端超时）；长文档 prefill 可能持续 15–30 分钟，属正常，不要反复重启。
4. **断线续传**：内置 WebUI 生成过程中刷新页面会**自动续上同一次生成**（服务端保留响应缓冲，不重复、不丢失已生成内容）；服务重启后缓冲丢失，需要重新发送。

## 三、验收读数（构建机 RTX 2080 Ti 11GB / i9-9900KF / 64GB，交付二进制 sha256 1DF23CA3…）

| 档位 | prefill | decode | 说明 |
|---|---|---|---|
| 生产配方（int8，默认无投机） | 566.0 – 571.3 tok/s（短上下文） | **51.4 – 52.2 tok/s** | 1M 逻辑上下文，设备窗口 38,400 |
| 1M 长档 | 287,269 token → **268.9 tok/s** | — | 1M 起服 + 长档 prefill 实测 |

精度：int8 PPL = 5.85525（124 窗口 / 261,223 token；bf16 = 5.85453）。

能力判据：40,960 token 跨窗口召回 8/8；多轮 16/16；多模态三图 12/12；1M 起服 + 105K 长上下文在 90% 深度处召回命中；内置 WebUI `GET /` = 200 / `text/html`。

## 四、模型

| 模型 | 字节 | 说明 |
|---|---|---|
| `Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer` | 9,520,051,456 | Bonsai（Ternary-Bonsai-2-27B），NInfer v3 容器，自含 text/vision/mtp/dflash2 |

权重不在压缩包里；下载地址与校验值见 `模型包\MODELS.md`。
