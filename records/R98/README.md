# R98：末端head LR2e-3，内部训练终态

冻结Git `481e5903bd569de3784f5d16b45121551cae9a9c`、tree `0826c8846d39c2ca910b5afe7db7569e2c11220d`。直接对照R95，仅末端原生Linear的head LR由1e-3提高到2e-3；backbone1e-4、独立age1e-3。已导回的真实参数组和SCIENCE明确这些值，外来快照“head/age均2e-3”是报告误标，没有新增参数验证或修复。

实际successful=attempted5600、skip0、7完整周期。内部max(online,EMA) AUROC首轮baseline，每800步评估，c3刷新后c4–c7连续4轮无新高，合法early_stopping；保留24000上限、patience4/min_delta0/min_cycles1、绝对warmup/EMA warmcopy1600。完整14条内部结果选唯一c3 EMA@2400：AUROC `0.9853363853431284`、AP `0.8194004616388242`、error_twice24356，tensor `2e472e71890c7281ebac406caa72548c55e2a2714666ef5c7f902f08947e3c7f`。

相比R95内部AUROC较低，AP非常接近；这是单seed固定划分观察，不证明统计显著性或机制。官方不参与早停、选点或R99科学方向，raw AUROC目标0.999尚未达成。

本次04:53:05只读观测时，train/freeze均exit0，原一次official进程51642仍活跃，owner44334S、8rank inactive，ROUND/done及official结果尚未出现。因此这里只归档训练内部终态，不称整轮完成、安全交接或官方通过；原流程自然推进，后续GPU仍须实核安全退场。没有新forward/评估/模型load、信号、launch、clock修改或官方重试。R99基于内部14条的决定独立于官方结果。

原运行根 `/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r98-tiny22k-head2e3-24k-p4-9010-20261029`。本目录原件均普通字节导回：

- `SINGLE-MODEL-FREEZE.json`（完整14内部及selected），Pod SHA `a545f16f4b9444c2a7598317635e1944cdc27bea60466381d064b7a399c96841`。
- `TRAIN_RESULT.json`，Pod SHA `2b738cfdb939ea630e156091d48838e99b6123b855f92bf71df9d2fcf9fa4c32`。

观测原件为 `exp-v3/technical-prep/r98-frozen-481e5903bd56/TERMINAL-OBSERVATION-1.json`，Pod SHA `fc6a6db7040fe9ed440a69907de0007a1037ba9ffd27deaf8068ce2735438a33`；guard-w278_u5y SHA `967194ac728623042d6933b4e8d8e9419ac21a125ad6343bffc91642143a63f9`，exit0。原小回执留工作区 `work/exp-v3-8card-review-20261028/r98-update/r98-terminal-export`。本机未解析或重哈希；权重、checkpoint、样本级预测均留exp-v3。后续完整整轮结果随必要结果归档，不为过程状态另造提交。
