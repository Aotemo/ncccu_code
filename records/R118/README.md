# R118：DINOv1 ResNet50 来源替换

固定 9:1 内部验证的唯一选点为 **online、cycle 5、成功更新 4000**：整体 AUROC `0.970734180141627`、AP `0.7233388551998549`。训练在 7200/7200 次尝试与成功更新、0 次跳过、9 个内部评估周期后依 patience 4 正常早停。结果低于直接来源对照 R111 `0.9754398301727533`，也低于预冻实用门槛 T29（R107）`0.9884405731067525`；该固定 DINOv1 来源加 R111 下游合同的分支结束，不进行 seed27 或 A1 替补。此判断只使用完整内部结果。

内部选点低于 `0.996`，因此完成冻结后官方 5849 评估按政策记为 `not_run_by_policy / threshold_not_met`，无官方指标。2026-10-31 05:01:11 北京时间只读核到 round `completed`、原 owner 同 PID/start 的退出态、8 rank 均已退出、原锁无持有、设备用户为空且 8 卡 busy 0，实际资源 `safe=true`。

冻结源码 Git `9e5e95ed94a38ab0ca9bacda203678aebb40ec99`。公开 DINOv1 原权重 SHA、严格转换后的 backbone SHA、provenance SHA、唯一模型张量 SHA 和无 token 训练摘要见 [R118-RESULT.json](R118-RESULT.json)。该摘要的 Pod SHA256 为 `ecec0e024da020dd869eeb4d92e7ab37cf6adc3b2fb81f82f3c46510c070701e`；原始 freeze、TRAIN、ROUND、done 各自 SHA 分列于摘要，不将脱敏摘要视为原件。含运行 token 的完整原件保留在 exp-v3 的该轮目录及工作区 raw 导回目录，不进入 Git。
