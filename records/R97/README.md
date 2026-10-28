# R97：末端head LR5e-4

冻结Git `33e35d68d8f9e013c3e60e0c3470076850295bf6`、tree `413dbdcfa9b7cd0764ec14c31c151167b857ae53`。直接对照R95，仅末端原生Linear的head LR由1e-3降至5e-4；backbone1e-4、独立age1e-3。冻结SCIENCE、CPU参考参数组和真实fresh参数组均一致，外来快照把age并写为5e-4是报告误标，没有代码修复或重跑。

最大24000成功更新、800一轮、内部max(online,EMA) AUROC连续4轮无提升早停及绝对warmup/EMA warmcopy1600均保留。实际6400成功/attempted6400/skip0，8完整周期；c4刷新后c5–c8无新高，bad4，合法early_stopping。完整16条内部选唯一c4 EMA@3200，tensor `09db9a1d77d5a58ffb60ece36e97ea22eb021d9d6460c1467d0d00747ae0cb88`。

| 来源 | 唯一checkpoint | AUROC | AP |
| --- | --- | --- | --- |
| 内部dev5140 | c4 EMA@3200 | 0.9854519796120608 | 0.8277913868944404 |
| 官方5849，仅原最终一次 | 同一冻结EMA | 0.9885877199313764 | 0.8405994179668198 |

内部error_twice24164；相对R95所选内部AUROC较低、AP较高，不能概括为所有指标退步或以单seed诊断机制。c3 EMA的error24168至c4的24164虽刷新很小，仍严格按min_delta0计数。官方不参与早停、选点和后续方向，raw AUROC未达0.999。R98的单轴head2e-3方向在此次官方终态读取之前已依据完整内部证据冻结。

04:03观测官方仍活跃；04:11:04后续已有产物只读核实train/freeze/official均exit0、ROUND/done completed/complete，8rank及组件inactive、设备idle无使用者且原锁已释放。owner35966保留Z，不称reaped；没有发送信号、取锁、重置clock或再前向。终态观测原件 `exp-v3/technical-prep/r97-frozen-33e35d68d8f9/TERMINAL-OBSERVATION-2.json`，Pod SHA `6ff0ac24ef389185f365f6ec23caaa4d3602ac70329ce6701f27a95f7c7fa3fd`。

权威运行根 `/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r97-tiny22k-head5e4-24k-p4-9010-20261029`。本目录仅普通字节导回的必要小原件：完整16条 `SINGLE-MODEL-FREEZE.json`（Pod SHA `b70f23889fb7a361d7e93c6f4af66ea749e544269ac54927c775c18b42a1e795`）、`TRAIN_RESULT.json`（`b238ef14c0b65969f6fb46483c9bfbd1c60d704330b0a00b81ce6509642dd135`）与含原官方结果的 `ROUND_RESULT.json`（精确Pod SHA见终态观测source_snapshots）。不带权重、checkpoint或样本级预测。

这次终态只读与R98配置窄验共享验证池任务guard-n3bn166q，原Pod SHA `1c1df2765771f1c6e3b39defa1575e20724c60f3e472f27dd931d5d3f9fefeb5`，归档 `records/preparation/r98-head2e3/CPU-RECEIPT.json`。所有SHA来自Pod原件，本机未解析或重哈希。
