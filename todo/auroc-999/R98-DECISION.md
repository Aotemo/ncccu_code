# R98：对照R95，仅将末端head LR提高到2e-3

2026-10-29科学决定已直接交A实施。直接对照为R95：从`configs/R95-SCIENCE.json`创建`configs/R98-SCIENCE.json`，唯一科学delta为`job.head_lr: 0.001 → 0.002`；backbone仍为`0.0001`，age独立仍为`0.001`，`head_lr_scope=last_linear`。R97为同一head轴的下侧取值对照，不以官方指标决定本次方向。

精确profile=`single-tiny22k-bc48a87f-fixed-9010-24k-earlystop-head2e3-v1`；candidate=`r98-convnext-tiny-imagenet22k-to-1k-24k-p4-head2e3-9010`；order8。此页收口时R98尚未运行，A已接单准备；无文档批准前置。

## R95与R97完整内部事实

规划仅文本读取原生内部freeze、TRAIN_RESULT及实际参数组，不读取官方指标选参，不增加解析脚本、模型评估或本机计算。R97实际五个参数组的`initial_lr`为backbone两组1e-4、head两组5e-4、age一组1e-3，CPU参数组原件同值，真实8rank初态completed且pin实际参数组文件；外来“head/age均5e-4”为报告误标，不需要代码修复或重跑。

| 唯一内部选择 | R95：head1e-3 | R97：head5e-4 |
| --- | --- | --- |
| 实际成功更新／完整周期 | 5600／7 | 6400／8 |
| 选择位置 | c3 EMA@2400 | c4 EMA@3200 |
| AUROC | 0.9866645153288838 | 0.9854519796120608 |
| AP | 0.8193841805300731 | 0.8277913868944404 |
| error_twice（越低越好） | 22150 | 24164 |

各周期内部AUROC展示到6位小数；选点和早停使用原精度。R95第8周期未运行，不补值。

| 周期 | R95 online | R95 EMA | R97 online | R97 EMA |
| --- | --- | --- | --- | --- |
| 1 | 0.973097 | 0.973097 | 0.970029 | 0.970029 |
| 2 | 0.985323 | 0.985323 | 0.981870 | 0.981870 |
| 3 | 0.982684 | 0.986665 | 0.983701 | 0.985450 |
| 4 | 0.977746 | 0.985536 | 0.977569 | 0.985452 |
| 5 | 0.977276 | 0.983334 | 0.973677 | 0.984530 |
| 6 | 0.970660 | 0.981346 | 0.968628 | 0.982060 |
| 7 | 0.968865 | 0.979318 | 0.969259 | 0.979575 |
| 8 | 未运行 | 未运行 | 0.970168 | 0.977997 |

R97的c3 EMA为AUROC 0.9854495713981247/error_twice24168；c4以24164严格刷新，随后c5–c8连续4周期无新高，6400/skip0正常早停。这个很小的刷新仍按既定min_delta0生效，不据此改变早停合同。c5 EMA的AP 0.8320960600146975高于所选c4，但error_twice25696较差，不能为AP改选点优先级。

本次head减半未胜R95所选AUROC/error_twice，AP则较高；部分后段EMA及c7读数也较R95同期高，不能概括为所有指标或所有周期退步。两轮最大预算和早停政策相同，实际停止步数不同，不能称等计算量比较。这是单seed、单固定划分的观察，不足以证明统计显著性、一般最优值或“head过拟合”等机制。

## 单一假设及保留合同

R98检验R95末端head LR的上侧2倍取值，与R97的下侧减半构成有限的同轴比较。下侧未见主指标收益不证明上侧会改善；2e-3是待检验取值，不是已知最优。保留R95的backbone步长，不同时改变age、WD、EMA、增强、模型或来源。

现有优化器把末端Linear权重/bias归head，final LN仍归backbone，age两参数另组LR1e-3；既有SCIENCE消费者可表达本实验，无需新接口。提高head LR同时影响其AdamW权重收缩及后续反向传播轨迹，不能把“backbone LR不变”解释为backbone参数更新轨迹相同。

除head LR及新候选身份外完整沿用R95：固定fit46256/dev5140、同公开Tiny22K→1K revision/权重、seed20261027、fresh native binary+零age、初态eba53fee控制值、空Adam、online=EMA；512/FP32、G64/P8、8rank各8/accum1、原采样/增强/WD/clip/BCE。最大24000成功更新、每800一周期，每完整周期max(online,EMA)内部dev AUROC严格刷新才清零，首轮baseline，patience4/min_delta0/min_cycles1；warmup和EMA warmcopy绝对1600、首blend1601、cosine horizon24000/尾倍率0.01、EMA decay0.999。不得从R95/R97或其他训练checkpoint初始化。

在实际完成周期的全部online/EMA结果中沿用error_twice/AP/update/EMA规则选唯一checkpoint，直接与R95比较；实际停止步数单独报告。官方5849只做一次最终独立比较，不参与早停、选点或后续调参；raw AUROC未达0.999不称总体完成。

## 执行与证据

A仅做新SCIENCE、candidate/profile/order接线及改动必需的现有窄验证，复用已有准备、Git同步、实际tree/SHA冻结部署及一次fresh启动链；不新增平台、套件或独立GPU预演，不重来源、迁移、native或旧检查。现有墙钟及集中资源policy沿用，准备20%与训练80%互斥并行，实际份额/TTL由既有入口实时核验。04:03的R97原一次官方进程仍活跃，不能把训练组件completed当整轮已安全交接；A独占Git/GPU并在真实安全退场后启动R98，不改旧冻结包、allocation、clock或token。

R97冻结Git为`33e35d68d8f9e013c3e60e0c3470076850295bf6`。原件已普通导回工作区`work/exp-v3-8card-review-20261028/r97-update/r97-terminal-export/`；以下SHA均来自Pod原件，本机未重哈希。

- R97 `SINGLE-MODEL-FREEZE.json`：`b70f23889fb7a361d7e93c6f4af66ea749e544269ac54927c775c18b42a1e795`；`evidence/TRAIN_RESULT.json`：`b238ef14c0b65969f6fb46483c9bfbd1c60d704330b0a00b81ce6509642dd135`。
- R97 `parameter-groups.json`：`c6d7832de5da6358dacd9cd10573d9c70b8758bb232192de70057f4196c24ca9`；`evidence/ACTUAL_FRESH_INITIAL.json`：`fc99871f067c6d131a2e2bf3c2708c5a4e63d56caf50a2c0a4dd0533a40c86a9`。
- R95对应freeze：`13aa989198ca9b4e64ed5e021f6da9b8733eda748878bd5750d19e2ebccc217b`；TRAIN_RESULT：`dd253b3aa0fdd33bb6a539f01ba596d7f5520dbea1547d543787fdae69f306c1`。基线配置为`configs/R95-SCIENCE.json`，既有合同见[R95计划](R95-PLAN.md)。

本次规划只交付本文件，由A与必要实现/结果一次集成；不为过程状态另造提交或等待。
