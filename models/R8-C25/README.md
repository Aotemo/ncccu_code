# R8-C25

ConvNeXt Tiny，无年龄分支，唯一保留 EMA@2844；完整官方 AUROC 0.9890383629536272。原 `job.json` 中年龄合同字段仍存在，但 `enabled=false`，不能因字段存在而启用年龄分支。

| 内容 | 原件 |
| --- | --- |
| 精确权重与 variant | [MODEL.json](MODEL.json) |
| 模型、age、normalization 等参数 | [convnext-tiny-job.json](convnext-tiny-job.json) |
| 完整配置 | [config.json](r8-6c-c25-image-ema-config.json) |
| 公开来源、EMA、schedule 和 identity | [contract.json](r8-6c-c25-image-ema-contract.json) |
| 源码逐文件 SHA | [source-files.json](r8-6c-c25-image-ema-source-files.json) |

这些是服务器既有参数原件的只读传回。本次未重新从 checkpoint 提取或加载验证；原环境已消失。验证范围与待办见 [models/README.md](../README.md)。历史路径仅作来源身份。
