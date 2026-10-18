# zip 交付清单 · 版本 KVMem-Ninfer-ActKV-2080Ti-v0.1（本页在 zip 内）

> **本包不含模型权重**（9.5 GB，单独下载，见 `模型包\MODELS.txt`）。
> 先读 `README.md`；15 分钟快速验证见 `指南\教程-启动与短测-给agent与使用者.md`。

## 一、这个 zip 里是什么

| 项 | 值 |
|---|---|
| 包名 | `KVMem-Ninfer-ActKV-2080Ti-v0.1.zip` |
| 给谁 | RTX 20 系（sm_75，Turing）单卡、想在本机跑 27B + 1M 逻辑上下文的人 |
| 内容 | `engine\`（`ninfer-serve.exe` + 10 个 DLL；perplexity / calibrate / CLI 三个辅助 exe 另作为独立资产）+ `Start\` 启动脚本 + `ms75_eval\` 判据 + `src\` 源码 + `patches\` + 文档 |
| 解压后 | 2.63 GiB / 2632 文件 |
| 解压后 sha256 | 见随包 `SHA256SUMS.txt` |
| zip 大小 | 压缩后填写 |

## 二、两条交付提醒（请一并转给接收方）

1. **解压用 7-Zip / WinRAR / `tar -xf`**。Windows 资源管理器自带的解压对中文文件名会乱码——那是它的老毛病，不是包坏了。解压路径**必须全 ASCII**（原生 exe 读不了中文路径）。
2. **先放模型，再起服务**：`模型包\MODELS.txt` 写了下载、放哪、怎么校验。放好后把 `Start\启动_KVMem_Ninfer_ActKV_2080Ti.ps1` 的 `$MODEL` 改成模型绝对路径，起服务；就绪判据 `curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8080/v1/models` ⇒ `200`。

## 三、验收读数（构建机 RTX 2080 Ti 11GB / i9-9900KF / 64GB，交付二进制 sha256 97D9FFC5…）

| 档位 | prefill | decode | 说明 |
|---|---|---|---|
| 生产配方（int8 + MTP draft=2） | 611.8 tok/s（8K）/ 549.4（32K）/ 240.8（105K） | **80.3 tok/s**（MTP 接受率 91.4%） | 1M 逻辑上下文，设备窗口 38,400 |
| 1M 长档 | 287,269 token → **268.9 tok/s** | — | 1M 起服 + 长档 prefill 实测 |

## 四、模型

| 模型 | 字节 | 说明 |
|---|---|---|
| `Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer` | 9,520,051,456 | Bonsai（Ternary-Bonsai-2-27B），NInfer v3 容器，自含 text/vision/mtp/dflash2 |

权重不在 zip 里；下载地址与校验值见 `模型包\MODELS.txt`。