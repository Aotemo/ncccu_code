# R97：对照R95，仅将末端head LR减半

2026-10-29科学决定已直接交A实施。**直接对照是R95，不是紧邻的R96**：从`configs/R95-SCIENCE.json`创建`configs/R97-SCIENCE.json`，唯一科学delta为`job.head_lr: 0.001 → 0.0005`。backbone保持R95的`0.0001`，age独立保持`0.001`。相对R96还包含backbone返回R95基线，不能把R96→R97称为单变量比较。

精确profile=`single-tiny22k-bc48a87f-fixed-9010-24k-earlystop-head5e4-v1`；candidate=`r97-convnext-tiny-imagenet22k-to-1k-24k-p4-head5e4-9010`；order7。名称head5e4表示5e-4，以版本化SCIENCE数值及实际SHA为准。此页收口时R97尚未运行；不等待文档批准或官方分数来决定科学值。

## R95→R96完整内部事实

规划仅文本读R95的14条和R96的16条原生`dev_entries`、唯一`selected`及TRAIN_RESULT，未读官方指标、执行本机解析计算或增加模型评估。R96训练completed、组件exit0、8周期/6400成功更新；内部reference为0.983341180097075，c5–c8均未超过c4峰值，`bad_cycles=4, stopped=true`，属于正常早停。R96的03:04终态观测时官方原一次进程仍活跃，整轮GPU安全交接尚未完成；这不影响CPU准备，但不授权新GPU并发。

| 唯一内部选择 | R95：backbone1e-4 | R96：backbone5e-5 |
| --- | --- | --- |
| 实际完成成功更新 | 5600 | 6400 |
| 选择位置 | c3 EMA@2400 | c4 EMA@3200 |
| AUROC | 0.9866645153288838 | 0.983341180097075 |
| AP | 0.8193841805300731 | 0.825803312915609 |
| error_twice（越低越好） | 22150 | 27670 |

各周期内部AUROC展示到6位小数，选择及早停仍使用原精度；R95没有第8轮结果，不补值或外推。

| 周期 | R95 online | R95 EMA | R96 online | R96 EMA |
| --- | --- | --- | --- | --- |
| 1 | 0.973097 | 0.973097 | 0.955901 | 0.955901 |
| 2 | 0.985323 | 0.985323 | 0.977180 | 0.977180 |
| 3 | 0.982684 | 0.986665 | 0.982349 | 0.981823 |
| 4 | 0.977746 | 0.985536 | 0.981760 | 0.983341 |
| 5 | 0.977276 | 0.983334 | 0.973158 | 0.982804 |
| 6 | 0.970660 | 0.981346 | 0.976908 | 0.980592 |
| 7 | 0.968865 | 0.979318 | 0.970714 | 0.979366 |
| 8 | 未运行 | 未运行 | 0.979378 | 0.979342 |

本次减半backbone LR使最佳位置后移，但所选AUROC/error_twice较差；所选AP较高，不能说所有指标均退步，也不能为AP改变既定选择优先级。R96部分后段online读数较R95同期高，仍没有超过R95所选峰值。两轮具有相同最大预算与早停政策，实际停止步数不同；这是单seed、单固定划分的结果，不能推断一般最优LR、统计显著性或训练机制。

## 为什么选择head这一轴

R96没有为继续降低backbone LR提供本次内部主指标收益，因此回到同24k/p4政策下表现更好的R95基线。下一项仅检验较慢的fresh末端分类器适配，保留已有backbone步长；5e-4是对既有head值的简单减半试验，不是已证明的最优值。当前汇总没有定位“head过拟合”或其他具体病因，也不保证这个改动有效；不同时调整WD、EMA、增强、模型或预训练来源。

已文本核现有`engine._optimizer`：`head_lr_scope=last_linear`只把末端Linear的权重/bias归head，final LN仍属于backbone，age两参数另组固定LR1e-3；`experiment91.validate_job`和`check_candidate`均从绑定配置读取head LR，已有接口可表达本实验。降低head LR也会改变其AdamW权重收缩及后续反向传播轨迹，不能解释为backbone参数更新轨迹保持不变；只是配置层面仅改变一个科学变量。

除上述head LR与新身份外完整沿用R95：固定fit46256/dev5140、同公开Tiny22K→1K revision/权重、seed20261027、原生fresh binary+零age、初态eba53fee控制值、空Adam、online=EMA；512/FP32、G64/P8、8rank各8/accum1、原采样/增强与WD/clip/BCE；最大24000成功更新、每800一周期、每周期max(online,EMA)内部AUROC严格刷新才清零、首轮baseline、patience4/min_delta0/min_cycles1；warmup与EMA warmcopy绝对1600、首blend1601、cosine horizon24000/尾倍率0.01、EMA decay0.999。不得从R95/R96训练checkpoint初始化。

完成后从实际完整周期的全部online/EMA结果按既定error_twice/AP/update/EMA规则选唯一checkpoint，直接与R95比较；R96仅为已完成旁证。停止更新数可不同，不把多跑周期本身当收益。官方5849只做一次最终独立分报，不参与选点、早停或后续调参；raw AUROC未达0.999不称总体完成。

A仅做新SCIENCE与candidate/profile/order接线及改动必需窄验证，复用现有配置注入/打包/同步/实际Git树部署/一次fresh启动链；不散写新常量、不新增接口/平台/独立GPU预演、不重来源/迁移/model/native或旧套件。原墙钟及集中资源policy沿用。allocation3到期后只为未执行CPU准备阶段按实时资源建立新显式份额，不改R96绑定；准备仍总20%，训练80%，互斥并行。A独占Git/GPU，须核R96原一次官方与整轮安全退场后才启动R97；不能因训练组件已completed提前接管GPU，不重启旧运行或重置clock/token。

## 来源身份

R96冻结Git`35ad0cb11854d57511e07685d514316c09725abd`。权威根`/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r96-tiny22k-lr5e5-24k-p4-9010-20261029`；已普通导回工作区`work/exp-v3-8card-review-20261028/r96-update/r96-terminal-export/`，本机未重哈希。

- R96 `SINGLE-MODEL-FREEZE.json` Pod SHA `4ecfa8cd95dfa438a6ae6cdfb73fd9ae1998179aff5b768892b5c85eefb84e8b`；`evidence/TRAIN_RESULT.json` Pod SHA `967c225f596fb3109cab091c04b139b8c85df7e9c9533debf66e47374adb45da`。选中tensor SHA `dcce91a2737f50d59e2d8b028d8fcf40dbb76e773d5bee420d357c487ea2d997`。
- R95对应freeze Pod SHA `13aa989198ca9b4e64ed5e021f6da9b8733eda748878bd5750d19e2ebccc217b`；TRAIN_RESULT Pod SHA `dd253b3aa0fdd33bb6a539f01ba596d7f5520dbea1547d543787fdae69f306c1`；已定科学配置及事实见[R95计划](R95-PLAN.md)、[R96决策](R96-DECISION.md)。精简结论由A与必要实现一次集成，不另造过程提交或等待。
