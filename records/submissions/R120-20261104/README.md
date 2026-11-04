# R120 test 提交

2026-11-04 用户要求的 R120 提交已生成、下载并回传 Pod 校验 SHA，未上传比赛平台。唯一执行者为 4e；分析会话 5d3 已读取本机文件并负责交付。实际产物在 `exp-v3/submissions/r120-20261104`，本机可下载文件为 [submit.csv](../../../outputs/submissions/R120/submit.csv)，原推理 metadata 为 [submit.metadata.json](../../../outputs/submissions/R120/submit.metadata.json)。CSV、逐例预测、模型权重和生产日志均不进入 Git。

真实使用 R120 原冻结 bundle 的 `retrain.inference.run_prediction`，唯一 EMA c6@4800，保留原 age、512 输入、ImageNet normalization、FP32；未重训、替换模型，未做内部 dev 或官方 valid5849 前向。先在 CPU 实际载入并核对 EMA，随后一次 GPU test 生产，生产进程正常退出。未找到独立赛事模板，按项目现有预测入口三列 `Image_name,Probability,Pred_Label` 生成。

权威 [CPU 验证](FINAL-CPU-VALIDATION.json)通过：5855 行、test.csv 原顺序、唯一覆盖、有限 `[0,1]` 概率，标签为 `int(Probability >= 0.5)`；阳性 278。test.csv 仅有 Image_name/Age，未读取 test 标签。[下载往返验证](DOWNLOAD-CPU-VALIDATION.json)将实际下载副本回传 Pod，再比较原文件 SHA，一致。两次 CPU guard 分别为 [6u0d1g1c](FINAL-CPU-RECEIPT.json) 和 [3nq_ck2t](DOWNLOAD-CPU-RECEIPT.json)，均 exit0、0–23 核、24 核准备池；没有额外模型前向。

| 文件或身份 | SHA256 |
| --- | --- |
| submit.csv，232672 bytes | `e7a6f1c9cd91c21527bb3845b8c9a426c1ab36261593404f6d6ea74ee0142621` |
| submit.metadata.json | `414a1b6172a38d26e4175d23b1782ef53a85a4e90342f9f88a64395466fde3b5` |
| checkpoint blob | `25229efdea3a91baeb0b5e5142a470c2a68a15bab9470f92e9de76b1b616cac6` |
| EMA tensor | `f21bffae88acbf7c216c4141c0ef0b9d990a624e3e92a1baf750a501c415fc85` |
| test.csv | `3eba62c4e23201f85ce2a25e2290d195221159cf78b7fe8c49ecfb0f30a17c54` |
| FINAL-CPU-VALIDATION.json | `7c2a7614d9d9518c754ec2653534b4fbbe943733a8915934f3d99925cb29042f` |
| FINAL-CPU-RECEIPT.json | `7f07a6c5c3b9a835793e1f90401a7ec2e8fb047a52808e80bb6e63bcba275674` |
| DOWNLOAD-CPU-RECEIPT.json | `8e027ad40e89e9a5e86a7dd9be0936429e98248897a8341afd2cbcf28374d9db` |

R120 生产驱动源码发布于 `59436ec313153b9fabf994ec48e1745196960260`；模型 identity site 为原 R120 的 `26cae5aefe899351d153dff9c1dd6d685c129dcc0cb578bc73a2f03af409ce43`。硬件准入复用仍有效的通用 owner 资源额度和实例锁，没有执行旧 R236R2 训练入口或重置其 clock/token。R236R2 已自然闭轮，结果见 [R236R2](../../R236R2/README.md)；本次提交不代表长期实验指标已达到。
