# R127：训练旋转 identity 消融

R127 从 R107 的 ConvNeXt Small、seed20261029、pairwise logistic 合同重新公开初始化，仅在训练输入中抽取原旋转随机数后跳过实际旋转；其他光度与几何增强及年龄支路保持原合同。首次训练批次直接记录 `training_rotation=identity-v1`、`training_photometric=null`、每 rank 8 个主样本、四元组与年龄启用；8 rank 初态张量 SHA-256 为 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，当时 Adam 状态为空。这些是初态和首批次证据，不能代表终态参数或训练全程。

训练成功/尝试更新 **9600/9600**，跳过 0 次，内部评估 12 轮，连续 4 轮无提升而早停。唯一选中 EMA cycle 8、update 6400：内部 AUROC `0.9886320261146719`、AP `0.825473259026753`、`error_twice=18882`，张量 SHA-256 `e6bd15b353706b039734ef648d2a0229e352c4a66abdcfca01822f6b30f4ba76`，checkpoint blob SHA-256 `18b7b35603da912da404daa8e14df8e12c9b362adcfd9ad21befaae95ea192b0`。相对预冻 R107 `0.9884405731067525` 严格更高，是有限的同 seed 正向观察；未超过 R120 单次最佳 `0.9894773092062407`，本阶段结束，不启动 seed20261027，也不将其升为方法基线。R127 与 R107 的训练停止步数及选点步数不同，不是等更新量比较；结果不证明稳定复现、显著性或像素瓶颈。

选点低于内部 AUROC 0.996 门槛，官方 5849 按政策 `not_run_by_policy / threshold_not_met`，没有官方指标。北京时间 2026-10-31 20:02:37.455 的只读观察记录 TRAIN exit0、ROUND/done completed、原 owner 为 Z 态、8 rank 已退出、锁和设备使用者为空、8 卡空闲、safe=true；Z 不表示进程已回收。

[R127-RESULT.json](R127-RESULT.json) 是有效 Pod 准备池从现有原件生成的去令牌摘要，Pod SHA-256 `2ad5f8c71d362d9082400237f0275f049c79c9850373d7973b3535fa294eadc1`。原 freeze、TRAIN、ROUND、done、progress 与 CPU 回执的 SHA-256 分列于 `original_sha256`。原 checkpoint、clock、运行令牌及完整回执留在 exp-v3，未纳入 Git 或改写。
