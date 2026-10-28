# R100：binary focal gamma2

冻结Git `b8126fdc77fb81a615ea5f4b638275b148bd7345`、tree `78a3e87c32fadafaa019e45e968c72d369606df8`。直接对照R95，仅BCE→binary_focal gamma2/alphaNone/float32/mean/modulation_detachedfalse/version binary-focal-v1；其余R95 dropout0.2、backbone1e-4/head1e-3/age1e-3、seed20261027及数据/更新时钟/EMA保持。旧模型权重不作为初始化，原native fresh整体tensor仍eba53fee；13项微型数值/梯度/配置与新Git正式CPU初态证据已归档，真实8rank初态通过后启动。

实际successful=attempted6400、skip0、8完整周期。c4刷新后c5–c8连续4轮内部max(online,EMA) AUROC无新高，bad4、正常early_stopping；完整16内部唯一c4 EMA@3200，tensor `5a39fbcbc8c9fd37c17924c5986f17a716ee7d9f56273a5fe7073fa856d80745`。

| 来源 | 同一唯一c4 EMA | AUROC | AP |
| --- | --- | --- | --- |
| 内部dev5140 | @3200 | 0.9851617898327616 | 0.8337256490854664 |
| 官方5849，原最终一次 | @3200 | 0.9879769922215498 | 0.8463110838863377 |

内部error_twice24646，AUROC未超过R95的0.9866645153288838，AP高于R95的0.8193841805300731，不称全指标下降或机制已定位。单seed固定dev多轮探索不足以证明稳定收益。按照读取官方值前已预注册的规则，不扫gamma/alpha；下一轮回R95 BCE，仅固定训练seed20261029做一次复核，不选最好seed、不改变splitseed20261024。官方不参与训练/早停/选点/科学分支，raw AUROC未达0.999。

06:52观测原一次official仍活跃；06:58:26随后既有产物只读确认ROUND/done completed/complete、train/freeze/official全部exit0，8rank及组件inactive，设备idle无使用者、原实例锁释放。owner60931保留Z，不称reaped；未发信号、重置clock、取锁、重试official或做额外模型前向。终态 `exp-v3/technical-prep/r100-frozen-b8126fdc77fb/TERMINAL-OBSERVATION-2.json` Pod SHA `1a9d7f898693da0c09898b147446b5903779aea4b2e99d77d361b016b5ed4b78`。

运行根 `/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r100-tiny22k-focal2-24k-p4-9010-20261029`。本目录原件均普通字节导回：`SINGLE-MODEL-FREEZE.json`（完整16及selected，Pod SHA `e4a0af3f0a17cdc7b20a7332cf2b78d386fb1a600583bbbe0b8120aab3916158`）、`TRAIN_RESULT.json`（`c9818cf93f13ed07036ddbfe8d5fff0b7c11955ac86af08bd5716762b7b118f7`）和 `ROUND_RESULT.json`（Pod SHA见终态source_snapshots）。权重、checkpoint和样本预测留exp-v3，本机不解析或重哈希。

本次R101新seed必要CPU初态控制与10项配置窄验、R100只读终态共用验证池任务，guard-ebokl1if Pod SHA `def0583be62441108e77a97b3cfd130a621898802bd5e3867ba309942ea40b67`，见 `records/preparation/r101-seed20261029/CPU-RECEIPT.json`。未因归档增加科学试验或过程Git。
