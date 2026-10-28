# N06：研究专用

FLAIR ResNet50，唯一保留 EMA@2133；完整官方 AUROC 0.9952862924941565。禁止作为后续比赛的初始化、teacher、BN 或训练参数来源。

| 内容 | 原件 |
| --- | --- |
| 精确权重与 variant | [MODEL.json](MODEL.json) |
| 模型、age、normalization 等参数 | [flair-resnet50-job.json](flair-resnet50-job.json) |
| 完整配置 | [config.json](n06-4c-flairr50-s20261027-b64-p8-fp32-fixedbn-ema-config.json) |
| 公开来源、EMA、schedule 和 identity | [contract.json](n06-4c-flairr50-s20261027-b64-p8-fp32-fixedbn-ema-contract.json) |
| 源码逐文件 SHA | [source-files.json](n06-4c-flairr50-s20261027-b64-p8-fp32-fixedbn-ema-source-files.json) |

这些是服务器既有参数原件的只读传回。本次未重新从 checkpoint 提取或加载验证；原环境已消失。验证范围与待办见 [models/README.md](../README.md)。历史路径仅作来源身份。
