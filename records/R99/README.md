# R99：末端Dropout0.4

冻结Git `cf556e954e6ef491b019fe7ad32fca5be2fd7271`、tree `d8991545c4bbf0785d3dfa57fe8e6bc18048a4db`。直接对照R95，仅末端LN/Flatten后、原生Linear前的Dropout由0.2改为0.4；backbone1e-4/head1e-3/age1e-3、其余R95合同保持。模型实际non-tensor p由原生CPU/真实8rank初态核对，fresh tensor仍eba53fee、空Adam，未恢复旧模型。

实际successful=attempted5600、skip0、7周期；c3刷新后c4–c7连续4轮无新高，正常early_stopping。完整14内部选择唯一c3 EMA@2400：AUROC `0.9843622628059786`、AP `0.8160422876816051`、error_twice25974，tensor `5e2b47e6850988ca2790b193713bda4972bfa6e0b7ac0e74c285cad34b2cdc55`。固定24k上限/800每轮/内部maxonlineEMA AUROC/patience4/min_delta0/min_cycles1/绝对warmupEMA1600不变。

| 来源 | 同一唯一c3 EMA | AUROC | AP |
| --- | --- | --- | --- |
| 内部dev5140 | @2400 | 0.9843622628059786 | 0.8160422876816051 |
| 官方5849，原最终一次 | @2400 | 0.9879344263508649 | 0.8423919163221405 |

内部所选AUROC/AP未胜R95，不据单seed固定划分诊断机制或显著性。官方只作独立分报，不参与早停/选点/新的R100科学方向；raw AUROC尚未达到0.999。R100目标改变在读取本次官方终态之前已冻结，未继续扩大LR/dropout扫点。

05:43观测原一次official仍活跃；05:59:32随后既有产物只读核实ROUND/done completed/complete、train/freeze/official均exit0，8rank及组件inactive、设备idle无使用者且原锁释放。owner52625保留Z，不称reaped。未新评估/模型、发信号、重置clock、取锁或重试official。TERMINAL-OBSERVATION-2原件 `exp-v3/technical-prep/r99-frozen-cf556e954e6e/TERMINAL-OBSERVATION-2.json`，Pod SHA `c1aa9927d347f25198649d45512d5e06d55080724e932708e81cabceed24b137`。

运行根 `/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r99-tiny22k-drop04-24k-p4-9010-20261029`。本目录只普通字节导回的14条 `SINGLE-MODEL-FREEZE.json`（Pod SHA `8c895cd193fbd95c34e4b304b78c2253e8b9c4d12489c2f81bb1ba68fcee5881`）、`TRAIN_RESULT.json`（`c0aa9b4010b149e8abd7d436ecbfa67a6eac09e26b16db913b0865e336bd3a62`）与原 `ROUND_RESULT.json`（Pod SHA见终态source_snapshots）。权重、checkpoint及样本级预测留exp-v3，本机未解析或重哈希。

同一验证池任务完成R100必要微型CPU测试及上述只读终态，guard-xn57oqjh原件SHA `fd316f4a2b8311a70e93653afa96bfe7fb9c43eb0eb6ad1ca1890d0431511ccf`，见 `records/preparation/r100-focal2/CPU-RECEIPT.json`。不是重新评估或额外科学试验。
