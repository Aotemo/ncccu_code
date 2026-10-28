# R100：对照R95，仅将BCE改为无alpha的focal loss

2026-10-29协调在两份只读复盘后确定次序：先检验不同损失函数，停止继续扩展相近LR/dropout点。直接对照为R95，唯一科学变化为训练损失BCE→binary focal；其余全部保持R95。此页收口时R100尚未运行，不把建议或配置登记写成验收通过。

精确配置=`configs/R100-SCIENCE.json`；profile=`single-tiny22k-bc48a87f-fixed-9010-24k-earlystop-focal2-v1`；candidate=`r100-convnext-tiny-imagenet22k-to-1k-24k-p4-focal2-9010`；order10。身份与字段已直接交A对齐，不设文档批准门。

## 已有事实及次序分歧

规划已文本核R99原生完整14条内部dev_entries、selected及TRAIN_RESULT：5600成功更新、7周期、skip0，c3 EMA@2400唯一选中，AUROC 0.9843622628059786/AP 0.8160422876816051/error_twice25974；c4–c7无新高，bad_cycles4，训练组件exit0，正常早停。R99 c1内部AUROC高于R95同期，所选峰值却较差，不能把完整轨迹概括为每一周期均退步。

| 相对R95的单项改变 | 唯一所选内部AUROC |
| --- | --- |
| R95基线 | 0.9866645153288838 |
| R96 backbone LR减半 | 0.983341180097075 |
| R97 head LR减半 | 0.9854519796120608 |
| R98 head LR加倍 | 0.9853363853431284 |
| R99 dropout提高到0.4 | 0.9843622628059786 |

这些结果只支持“当前seed、固定划分下，这些已测改动未胜R95的所选主指标”。相同初态和seed有利于条件对照，但没有估计seed变异；反复依同一dev挑方案也存在选择偏差。不能据此证明过拟合、易负例主导、标签噪声、模型容量或分辨率上限，不能宣称显著性或R95全局最优。各轮未走满24000，不能把早停结果解释为24000预算必然无用。

次序并非两位一致：规划首选先试focal，B首选先以固定训练seed20261029复核R95基线。B并未否定focal可实施，双方均未证明focal更优。协调选择先检验不同学习目标，保留seed复核作为明确后续分支；这是一项实验次序判断，不是已经得到的收益结论。

## 唯一科学变化

损失字段由版本化SCIENCE注入：`kind=binary_focal`、`version=binary-focal-v1`、`gamma=2.0`、`alpha=null`、`dtype=float32`、`reduction=mean`、`modulation_detached=false`。对当前硬二值标签，逐样本稳定BCE乘以`(1-p_t)^2`后求均值，调制项参与梯度；不加class alpha、pos_weight、额外归一化或损失缩放，不同时改变G64/P8采样、LR、clip或任何第二科学轴。两个训练入口使用同一实现；保持既有8rank等样本数下的全局均值语义。

当前BCE对采样样本等权，P8是类配额而非同类内按预测难度分配贡献。focal检验改变相对样本梯度是否有益，区别于再移动统一步长或dropout值；这不证明易样本是当前瓶颈，也不等同于直接优化AUROC。它同时改变梯度尺度及可能的clip触发，可能放大错标影响；结果只能归于该完整损失配方，不得单独归因为“难例机制”。不同损失的训练loss数值不直接比较。

以R95为起点保留：backbone1e-4/head1e-3/age独立1e-3、dropout0.2、SD0.1、原WD与clip；seed20261027、固定fit46256/dev5140和split seed20261024、同公开Tiny22K→1K revision/权重、512/FP32、G64/P8、8rank各8/accum1、同采样/增强、LN1e-6、native fresh binary+零age、空Adam、online=EMA。损失不改变模型参数定义，初态tensor预计仍为既有eba53fee控制值，以现有正式包和实际初态证据为准。相对R99还包含dropout回到0.2，因此R99不是单变量直接对照。

最大24000成功更新、每800一周期；每完整周期max(online,EMA)内部dev AUROC严格刷新才清零，首轮baseline，patience4/min_delta0/min_cycles1；warmup及EMA warmcopy绝对1600、首blend1601、cosine horizon24000/尾倍率0.01、EMA decay0.999全保留。实际完整周期中按error_twice/AP/update/EMA规则选唯一checkpoint，分报AP及实际停止更新数，不以focal训练loss替代选择指标。官方5849仅一次最终独立比较，不能用于早停、选点或后续选择；不承诺0.999或总体完成。

## 明确的后续分支

若R100工程有效，但唯一所选内部AUROC不严格超过R95的0.9866645153288838，则不继续扫gamma/alpha。后续转为R95原BCE基线，仅将训练seed预先固定为20261029，split seed及固定清单不变；不再使用先前规划提议的20261028，也不看结果后挑最好seed。改变训练seed同时改变fresh head、采样顺序、增强及模型随机性，不能称为只检head；须生成新的真实初态身份，不能复制旧tensor或训练权重。

该复核用于检查随机性是否足以影响目前的局部排序，两个seed也不足以确证统计显著性。即使R100超过R95，也只先记录本次固定条件下的改善，不直接宣布可泛化收益。工程无效的运行不触发科学成败结论。此分支只是预定方向，不在本次提前生成或启动下一配置。

## 最小实现、资产和分工

当前两个训练循环固定BCE，配置入口尚无loss字段，不能声称仅改JSON就能运行。B仅独占新增`runtime/bundle/retrain/classification_loss.py`和`tests/test_classification_loss.py`的文本编码；A负责其余SCIENCE/身份、`config.py`接纳并验证loss、`experiment91.py`消费绑定配置、`engine.py`与`distributed_engine.py`接同一损失实现。现有`job_contract`会保留新字段，不建设新身份框架。A统一做改动必需的Pod CPU数值/梯度与调用验收，复用现有正式包和启动链，不扩全套、模型诊断或独立GPU预演。

现有exp-v3数据、512缓存、固定9:1元数据和公开22K→1K权重足够，无新资产或下载。R17仅借用其保存的loss定义；其旧1K来源、LN1e-12、80:20划分及其他训练设置与本轮不同，不能用历史成绩证明本轮收益，也不加载R17或R8-C25权重。N06完全不使用。

准备20%与训练80%按集中配置和实时有效CPU/TTL互斥并行。A独占Git写和GPU：必要验收、一次同步、实际tree/SHA冻结部署后，一次fresh启动；仍须确认R99原一次官方和整轮自然安全退场。05:43原观测official仍活跃，不能将train已completed当整轮已安全。旧运行、allocation、clock/token不热改或重启。

## 来源身份

R99冻结Git=`cf556e954e6ef491b019fe7ad32fca5be2fd7271`。已普通导回工作区`work/exp-v3-8card-review-20261028/r99-update/r99-terminal-export/`，以下为既有Pod SHA，本机未重哈希：

- R99 freeze：`8c895cd193fbd95c34e4b304b78c2253e8b9c4d12489c2f81bb1ba68fcee5881`；TRAIN_RESULT：`c0aa9b4010b149e8abd7d436ecbfa67a6eac09e26b16db913b0865e336bd3a62`。
- R95 freeze：`13aa989198ca9b4e64ed5e021f6da9b8733eda748878bd5750d19e2ebccc217b`；TRAIN_RESULT：`dd253b3aa0fdd33bb6a539f01ba596d7f5520dbea1547d543787fdae69f306c1`；科学基线为`configs/R95-SCIENCE.json`。
- 历史配方只读参考：`models/R17/convnext-tiny-job.json`及`models/R17/r17-4c-tiny-a25-b64-focal2-fp32-ema-config.json`，不使用历史官方指标。

规划本次仅新增本文件，交A与必要实现/结果统一集成；不更新过程状态，不另设批准或归档等待。
