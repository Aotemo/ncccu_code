# R102：ConvNeXt Small 公开表征对照

2026-10-29协调合并两份只读建议后，选择Small公开表征对照；P4本轮不执行。规划原首选P4以复用资产、检验数据暴露，B首选Small以保持G64/P8及学习时钟。本决定保留这项分歧，不表述为双方证明Small更优或已经发现容量瓶颈。科学方向在读取R101官方指标之前确定。

直接基线为预先登记的[R95](../../configs/R95-SCIENCE.json)，训练seed保持20261027；不是根据R95与R101两个seed的结果重新挑最高seed。唯一科学轴是**架构与对应公开checkpoint成对变化**：ConvNeXt Tiny ImageNet22K→1K改为ConvNeXt Small ImageNet22K→1K。公开权重、结构深度、fresh head及构造时消耗随机数的轨迹自然不同，不能把结果归因于纯参数量，也不要求新模型沿用Tiny整体初态tensor。

## 精确身份与固定合同

- 配置：`configs/R102-SCIENCE.json`。
- profile：`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-v1`。
- candidate：`r102-convnext-small-imagenet22k-to-1k-24k-p4-9010`；order12。
- `job.model=convnext_small`，运行名称、模型身份和来源期望从绑定SCIENCE消费，不能遗留Tiny名称或计数作为Small已验依据。
- 公开repo：`timm/convnext_small.fb_in22k_ft_in1k`；固定revision：`fe6b358aa409414cdb522ae3d43a2c9922f7cc1b`。只选一种原始权重格式下载，源与转换产物SHA由Pod实际字节产生，不能以远端广告SHA代替。

保持R95全部其余训练设置：BCE（缺失loss字段沿原BCE路径，不继承R100 focal）、seed20261027、backbone LR1e-4、head LR1e-3、age独立LR1e-3、dropout0.2、**显式stochastic_depth0.1**、LN eps1e-6、原WD/clip、原增强与ImageNet normalization；不能采用Small工厂默认SD等值而引入额外科学变化。保留native avgpool、fresh binary head、零age、无BN、空Adam、online=EMA。只从该公开源fresh初始化，不加载R95/R101或任何历史训练checkpoint；N06及三保留EMA均不作初始化或参数来源。

数据保持原fit46256/dev5140、固定9:1清单与splitseed20261024、fit-only age统计，512输入及现有prepared512缓存。保持FP32、G64/P8、reference physical32、8rank各8、accum1；不改采样配额、排序、增强算法或分布式均值语义。

最多24000成功更新，每800成功更新完成一周期；每完整周期以max(online,EMA)内部dev AUROC严格刷新历史最好值才清零，首轮baseline、patience4/min_delta0/min_cycles1。warmup与EMA warmcopy绝对1600，首blend1601，cosine horizon24000/尾倍率0.01及EMA decay0.999保持。早停满足原合同就是正常完成，不要求跑满24000。按实际完成周期的error_twice/AP/update/EMA规则选唯一checkpoint，完整官方5849仅一次最终独立分报，不参与早停、选点或后续科学决定。

## 实际依据和解释边界

R101原生freeze完整16条与TRAIN_RESULT已文本核对：6400成功更新、8周期、skip0，训练组件exit0，连续4周期未刷新后正常早停。唯一选择c4 EMA@3200，内部AUROC0.9843911613732117/AP0.8278718323176861/error_twice25926。R95唯一c3 EMA@2400为AUROC0.9866645153288838/AP0.8193841805300731/error_twice22150，5600更新正常早停。两次差异说明这两次训练结果对随机性敏感，不足以估计可靠方差、证明显著性或选择最佳seed。

相对R95、同seed20261027的R96 backbone减半、R97 head减半、R98 head加倍、R99 dropout0.4、R100 focal gamma2均未超过R95所选内部AUROC；其中部分AP更高，不能称所有指标都恶化。它们不能证明过拟合、样本噪声、采样缺陷或Tiny容量瓶颈，也不能否定整个LR、正则或损失设计。固定dev多轮选择造成的偏差仍在。

本次有限假设是：在采样、预处理和成功更新时钟不变时，不同的同家族公开表征组合可能改善排序表现。较大模型也可能没有收益，并增加训练耗时或显存成本。结果按同规则分别报告R95与R102的唯一选择、完整内部轨迹、AP和实际停止更新数；不把不同早停步数称为等计算量比较，不承诺0.999或从单次改善推断稳定收益。后续不自动扫Small的LR、seed或P值，待该轮内部与工程证据再决定。

## 资产、实现与资源接续

[timm官方模型卡](https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k)列出该Small 22K→1K模型并标注Apache-2.0；[原作者代码许可](https://github.com/facebookresearch/ConvNeXt/blob/main/LICENSE)为MIT。卡片信息不是本地来源验收，也不新增比赛批准或底层数据权利断言。本页冻结时，固定revision的来源原件、原始权重本地bytes、转换严格载入与Small整体初态尚待A/B实证；不得写成已下载、已匹配或已准入。

B负责该新公开来源的单格式下载、固定revision原件与许可记录、必要CPU严格转换/加载和非张量结构对应证据，数据/权重/缓存/产物全部放exp-v3独立prep。A负责生产工厂、SCIENCE和Tiny专属身份消费处的必要接入、Small实际参数组与全模型fresh初态控制、统一Pod集成及唯一Git/GPU。公开全头strict加载、保留的backbone参数、实际head/age组、空Adam及新初态属于新模型必需证据；旧Tiny来源/迁移/已过套件不重跑，不新增平台、身份桥、benchmark或独立GPU预演。新源哈希、转换哈希和初态控制值如实填入SCIENCE后同步，不能复制Tiny的eba或R101的18313控制值。

准备20%与训练80%按当前有效CPU及互斥池运行。只为未执行R102通过现有集中CLOCK/GPUgrant配置设置保守工程预算：train最多21600秒（6h）、total最多28800秒（8h），GPU grant至少32400秒（9h），具体实际配置与创建/到期身份由A记录。这是覆盖更大模型的工程上限，不是测得吞吐或租期保证，不改变24000/800/1600和早停科学合同；不申请新资源、延长云实例或修改旧运行时钟。实际显存和耗时由真实训练报告，超限不能伪称正常完成。

必要新CPU证据齐后，A统一一次Git同步，核实际commit/tree/文件与包SHA冻结部署；R101整轮和原一次官方进程真实安全退场后，沿既有链一次fresh activation/launch。规划读取内部证据时最近07:53观测仍为原official活跃、ROUND/done未出，不能将训练完成等同于资源安全。后续由A核实际退出，不等待官方数值决定本方案，不重发官方，不增加文档批准或归档门。

## 内部证据定位

R101冻结Git：`a215f6fb76fc25b24e1108274cb81f266c500e7d`。已普通导回工作区`work/exp-v3-8card-review-20261028/r101-update/r101-terminal-export/`；以下SHA引用既有Pod回执，本机未重哈希：

- `SINGLE-MODEL-FREEZE.json`：`6783af788175b731b96c018c010f4a45aa60e7bc28c8bea9207bb424f54ca604`。
- `evidence/TRAIN_RESULT.json`：`f198c07f18cbf684a341c2ebba6a7cedceccca4092c05586d1cc9ed45e603225`。
- R95 freeze：`13aa989198ca9b4e64ed5e021f6da9b8733eda748878bd5750d19e2ebccc217b`；TRAIN_RESULT：`dd253b3aa0fdd33bb6a539f01ba596d7f5520dbea1547d543787fdae69f306c1`。

规划本次仅新增本页，科学身份冻结交A/B；这不代表来源/CPU/GPU验收已通过。后续实际证据由A与实现统一集成，不为过程状态另作提交。
