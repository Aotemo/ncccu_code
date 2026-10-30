# R115：仅用 fit 图像的 VICReg 式续训，再按 R107 监督训练

2026-10-30，协调已选择本合同，承接用户持续训练直至 raw AUROC≥0.999 的要求，不保证必达。规划只写本页和R114终态短结论；A可并行实施，本页不是启动审批门。A `e62c5cb0-3516-4798-8b37-ed8da2d958ad` 唯一集成、Git写入与GPU执行；B `2d827e8e-667c-427d-b002-5867e305edb7` 仅下述两件损失文件。规划交接只发协调，不直接打断A。

## Goal、直接对照与取舍

检验：从同一合规公开Small出发，仅在当前sealed fit图像上进行固定1600成功步的无标签表征续训，再恢复原监督训练合同，能否提升完整内部dev所选AUROC。直接预登记锚点是 **R107、seed20261029，c4 EMA@3200，AUROC `0.9884405731067525`、AP `0.8274326888154083`**。这增加了一个预训练阶段，包含额外图像曝光和计算；不是等FLOP比较，也不能把任何差异纯归因于损失函数。

R114R1完整18条内部已文本核：7200/9cycles/skip0，c5 ONLINE@4000 AUROC `0.9875399010946536`、AP `0.8091074656953037`，未胜R107。其stride1分支结束；R115恢复原stride2及全部参数可训，不以R114R1作为单变量基线。R106的双随机全图平均BCE曾未胜匹配基线；本轮采用临时投影空间的无标签方差/协方差目标，仍不能据方法不同预断会获益。

N06的标准ResNet包装、固定公开BN、普通R50、一次BN替换和冻结早层均不等于取得其医学预训练表征。本轮只检验**官方fit图像是否能提供有用的域内表征适配**，不使用N06/FLAIR权重、teacher、BN、统计量或输出，不声称复现N06。

## 身份与唯一生成规则

| 字段 | 固定值 |
| --- | --- |
| SCIENCE（A生成） | `configs/R115-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-fitssl1600-v1` |
| candidate | `r115-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-fitssl1600-9010` |
| order | `25` |
| round | `r115-small22k-seed20261029-scale085-pairlog-fitssl1600-24k-p4-9010-20261030` |
| SSL规则版本 | `r115-fit-vicreg-1600-v1`，绑定为本轮唯一 `ssl_recipe` |
| 阶段 | `public_fresh → fit_only_ssl → derived_commit → supervised → freeze → official_once` |

监督job完整取R107：Small/seed20261029、512、`training_scale=[0.85,1.0]`、原单视图、SD0.1/dropout0.2、全部参数可训、原pairwise六字段和P8。不继承R114的`final_downsample_stride=1`、R113冻结字段、R106监督双视图或R50专用head_seed/BN字段。SSL规则放独立阶段配置，不把监督job改为views2或无P8，再靠运行中改回。

公开源保持 `timm/convnext_small.fb_in22k_ft_in1k`，revision `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b`；converted SHA `7aaea5ce6fe6db953874d9792d3f1711a5cc0193a10d740e023108043918c610`、provenance SHA `8ea3e957e2381a40e2f8fc5feae5df31d0d8b33faf3bdd3fa5749e64ae98368b`。沿既有公开full1000 strict来源验证，再构造原binary+age；预SSL public整体控制仍为 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，由正式原生流程实核。**该50c仅证明public起点，不是监督阶段derived初态的期望值。**

10/23规则第1条及10/27补充第4部分允许合规非医学公开权重在官方训练数据上自监督续训；本轮进一步限制为当前固定9:1的fit，排除dev/official/test。外部医学主来源仍禁止，Apache许可或重叠unknown不构成例外。已提取文本见 `work/technical-prep/official-rules-body-continue-20261030-b/RULES-EXCERPTS.md`；[10/27官方补充](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261027/b23d6a1f19602da2a9bd4048108175d0.pdf)不授权本轮使用FLAIR。

## SSL输入、抽样与网络

仅消费原sealed manifest的`fit_indices`及对应图像；原9:1划分、清单、原age fit统计保持，SSL不读取标签或年龄作为样本返回值、分层条件、损失、权重、教师或目标。现成ImageDataset可显式返回无label/age的views与主index。原manifest具有标签不意味着SSL使用它；抽样器只接fit索引序列。

- 每次全局64个主样本，8rank各8；均匀打乱fit完整排列，每个排列尾部不足64个样本丢弃，下一排列重新打乱。无P8配额、无类别平衡、无重复补齐、无hard mining。监督阶段仍原G64/P8。
- 每主样本两视图，沿原512 prepared图像、ImageNet归一化、原温和增强和scale `[0.85,1.0]`；同一未增强图读取一次，再按独立sample-key生成两个视图。没有原图局部、额外分辨率、裁块缓存或新增医学数据。
- 每rank `[8,2,3,512,512]` sample-major展平为16图，**一次concat模型forward**后还原为`[8,2,256]`。两view配对按同一主index保留；不用两次DDP forward再一起backward。全局64主图/128视图，一次SSL optimizer更新。
- Encoder为原 `network.features → avgpool → classifier[0]原finalLN → flatten`，输出768；全部encoder参数训练、原SD0.1保持train模式，原GAP/finalLN eps1e-6保持。分类dropout、binary linear和age路径全部绕过。
- 临时projector为 `Linear(768,1024,bias=True) → GELU(approximate='none') → Linear(1024,256,bias=True)`；两Linear使用原生Torch初始化，无BN、额外LN、dropout或归一化层。SSL wrapper只注册encoder与projector，不能把未使用head/age交给DDP并依赖隐式unused处理。
- 无teacher、动量encoder、负样本队列、显式负对、stop-gradient、标签或年龄目标。不同图之间只通过全局统计耦合，不能称其为已知阴性样本。

原binary head/age从public fresh工厂封存，SSL不更新、不衰减、不放入其optimizer；阶段交接实际核封存值。临时projector在SSL末步后丢弃，不进入监督模型、最终checkpoint或推理输入。

## 精确损失与分布式归一化

对两个本地embedding分别调用B共享 `gather_embeddings(local, rank=None, world_size=None)`，使用默认**全局WORLD**，保留二维embedding及梯度，得到配对的 `X,Y∈R^(64×256)`。生产省略可选rank/world参数，取真实WORLD；不能用`ctx.group`半组、detach gather或把现有会flatten标量logit的pairwise gather直接套用。每rank计算同一全局标量 `vicreg_loss(X,Y)`。

设N=64、D=256，`var`沿样本维使用无偏估计（分母N−1），`offdiag`不含对角线：

\[
I=\frac1{ND}\sum_{n,d}(X_{nd}-Y_{nd})^2,
\quad V=\frac12\left[\frac1D\sum_d\max(0,1-\sqrt{\operatorname{var}(X_d)+10^{-4}})+\frac1D\sum_d\max(0,1-\sqrt{\operatorname{var}(Y_d)+10^{-4}})\right].
\]
\[
C_X=\frac{(X-\bar X)^T(X-\bar X)}{N-1},\quad
C_Y=\frac{(Y-\bar Y)^T(Y-\bar Y)}{N-1},\quad
C=\frac{\sum_{i\ne j}(C_X)_{ij}^2}{D}+\frac{\sum_{i\ne j}(C_Y)_{ij}^2}{D},\quad
L=25I+25V+C.
\]

所有输入与loss/梯度FP32。方差项两view取平均，协方差项两view各自`sum/D`后相加，不能再除2或D；对角线不惩罚。不增BCE/pairwise混合项，不调系数/epsilon/投影维度。可微gather backward跨rank SUM，与DDP参数梯度平均抵消；每rank同算全局loss时**不再乘除world_size**。B提供数学与实际CPU两进程梯度窄验，不能把Gloo通过写成实际八卡通信已通过；后者由本次真实SSL更新体现。

[VICReg原论文](https://arxiv.org/abs/2105.04906)及[作者实现](https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py)仅提供一般目标依据。本轮的256维无BN投影器、G64、AdamW、1600步与原文大batch/投影器/优化器/时长不同，称“VICReg式有限recipe”，不声称复现论文或已证明眼底收益。

## 固定优化时钟与独立RNG

SSL固定1600次**成功**更新，FP32/noTF32，AdamW默认betas `(0.9,0.999)`、eps `1e-8`；encoder峰值LR1e-4、projector1e-3、WD1e-4，norm/bias不衰减，梯度clip norm1。所有encoder/投影器参数各归唯一组，head/age不入组。每步保留loss、梯度以及更新后参数/Adam finite检查和全rank一致退出；不能用非有限步凑足1600。

SSL成功步t从1计，LR因子为`t/160`（t≤160），否则 `0.01+0.99*(1+cos(pi*(t-160)/1440))/2`；t=160为峰值，t=1600为峰值1%。固定取第1600步encoder，无SSL dev评估、best选择、EMA、早停或根据训练loss提前选点。监督原LR函数及warmup1600完全不变，不能将SSL的160步暖启用于监督。

所有派生用现有 `data._seed(*parts)`：UTF-8冒号拼接、SHA256前8bytes按big-endian转整数，记为H；域字符串固定`D='r115-fit-vicreg-1600-v1'`，基础s=`job.seed=20261029`。不是搜索新seed，实际整数由Pod计算并记录：

| 独立流 | 预定派生 |
| --- | --- |
| projector初始化（各rank同值） | `H(s,D,'projector')` |
| SSL模型随机流（含SD） | `H(s,D,'model',rank)` |
| SSL全局抽样流 | `q=H(s,D,'sampler')` |
| 第e次fit排列（e从0计） | `random.Random(H(q,'epoch',e)).shuffle(list(manifest.fit_indices))` |
| 第a次全局batch/slot j/主index i的view1 | `H(q,'sample',a,j,i)`；a从0计，j=0..63 |
| 同主图view2 | 原接口 `H(view1_seed,'training-view',1)` |
| SSL DataLoader独立generator | `H(s,D,'loader',rank)` |

Python、Torch CPU与本rank设备seed按现有`single._seed`语义设置，NumPy只取seed modulo2^32；不对其他rank设备额外播种。projector构造在独立RNG上下文，随后明确设置SSL模型流；PIL增强、抽样与loader不消耗模型随机流。每rank取全局batch连续8个slot，两view全局gather保持rank-major主图配对。固定drop_last，排列边界与1600成功步时钟分开，不在800步人为重置SSL排列。

**监督RNG与head/age交接以原R107真实初始化顺序为准：** 原监督路径按 `single._seed(s+rank)` 构造public fresh模型；各rank的临时随机head在WORLD rank0广播前可能不同，因此封存统一head/age和比较必须使用广播后的同一public值，不能在广播前把rank>0 head与rank0强比。实现可在SSL前封存共同seed s的public head/age，在监督入口按原路径重建，WORLD广播后逐键核完全一致；不得将其替换为SSL随机head。

监督入口重放原`seed=s+rank → 原factory构造`顺序，捕获每rank正常factory后的Python/NumPy/Torch CPU/本rank设备RNG状态；载入derived encoder不采样、不改head/age。完成监督WORLD广播、空Adam、新EMA及既有保存/恢复RNG的reducer priming后，恢复该正常factory后的每rank状态，再从原监督sampler attempt0和原loader独立generator起步。若实现选择提前保存这一等价状态，须同样来自原初始化顺序。**不能使用SSL末步RNG、将所有rank重设为同seed、只凭同seed声称随机起点相同，或在广播前做统一head断言。** 此恢复隔离SSL随机消耗；encoder值已经改变，训练梯度/轨迹当然不会与R107相同。

## Runtime derived身份与监督恢复

冻结SCIENCE/源码只登记 `ssl_recipe`、上述派生规则版本、public来源、fit manifest身份、固定末步与输出位置。未来derived SHA未产生，不把50c或空hash当作已经核过的监督初态；**不在SSL之后修改SCIENCE、Git冻结副本、DEPLOYMENT、clock或token来补hash**。现有public初态字段如保留50c，仅用于pre-SSL门；监督门显式改读本attempt运行生成的derived绑定。

在同一round的 `ssl-stage/derived-encoder.pt` 仅保存encoder的 `network.features.*` 与 `network.classifier.0.*`，并原子提交 `ssl-stage/SSL-RESULT.json`。运行receipt至少绑定：本round/attempt、Git/SCIENCE及recipe摘要、源资产与public tensor身份、实际fit manifest摘要、SSL成功/尝试/skip与1600末步、真实WORLD/rank、每rankderived encoder digest、asset字节SHA、封存head/age前后逐键一致证明和RNG派生/还原事实。无样本级值入Git。实际SHA由Pod产生、各rank消费同一已提交asset与receipt；不从其他attempt目录择优读取，不以recipe登记代替实际生成。

严格核完实际derived资产后，监督模型沿原factory恢复Small结构、原public head/age，精确替换encoder名单；训练checkpoint的`initialize=False`加载直接使用训练state，不重新套public或derived覆盖它。当前和最终推理均无需projector、SSL optimizer、教师或多模型。

丢弃projector与SSL Adam及其所有状态，监督重新建立原Adam（空state）、原EMA（从**derived完整模型**精确拷贝，游标0），原age统计不重估。真实8rank监督零步同时记录derived whole tensor、encoder、保留head/age、结构stride2/全可训/0BN、参数组和空Adam。derived whole必须实测，不能沿用50c；公开strict proof、SSL末步proof和监督初态proof属于三个不同阶段，不互相冒替。

监督科学回到R107：64主图/64单视图、8P56N448对，原六字段 `binary-pairwise-logistic-v1 / binary_pairwise_logistic / temperature1 / mean / float32 / global_batch`；backbone1e-4、head/age1e-3、原WD/clip/SD/dropout/LN/age，原完整dev单视图、每800成功步评估、上限24000、warmup及EMA绝对1600、max(online,EMA) AUROC strict/min_delta0/patience4。SSL步数不计入监督成功步、采样cursor、EMA或dev周期，但计入联合总成本。

## 一个联合attempt的预算、恢复与真实代价

本阶段先**1个联合fresh attempt**，一个activation、一个owner/锁、同一GPUgrant；固定1600 SSL成功步加最多24000监督成功步。共享从同一T0起算的 **train21600秒 / round28800秒 / GPUgrant32400秒**，不是每阶段各一套。SSL、阶段交接、监督及原训练组件内dev/保存的wall全部占用同一训练截止时间；freeze/official也受同一整轮截止时间。阶段计数归零不代表clock/token/资源期限归零。不加新实例、延长云资源、额外GPU试跑或独立预演。

**恢复范围单列：** 监督阶段现有300秒checkpoint、原子提交和恢复语义保持，不降频或改异步；新增SSL阶段本轮只支持fresh，不建设跨进程中途恢复。SSL记录phase/成功与尝试cursor、抽样位置和RNG身份，进度日志不冒充包含optimizer/RNG全状态的可恢复checkpoint，旧监督checkpoint也不能用于跳过SSL。本合同只授权一次fresh派发，不自动重启恢复、重新校准或重做1600步。工程异常/超时保留实际阶段和成本，未完整走通不当科学阴性，不用部分encoder替代固定末步；中途故障可能损失本次SSL工作，这是明确工程代价。监督原恢复/earlystop/finite事务不因加入SSL而弱化。

SSL每步128视图，监督每步64视图；同主batch不是同图像计算。既有R106双视图训练项约0.736秒/成功步仅作数量级参考，1600步约20分钟，加原R107最多24000步训练项约2.8小时，说明总限额内存在可执行空间；这不是本轮SSL benchmark或总wall承诺，未含全部IO/dev/保存/通信差异。实际吞吐、峰值显存与联合剩余预算由正常运行日志分阶段报告，不称精确2倍wall或免费预训练。超时/OOM不得私改batch、维度、SSL步数、精度或时限救场。

最终推理恢复原stride2的单Small/单输出，投影器不部署；没有新增五路编码。但指定硬件端到端100ms及≤500MB仍须按比赛实际口径判断，当前设备训练吞吐不是部署证明；两份规则PDF未给出具体T4/16GB型号，不再将该型号当已核事实。

## 文件边界与最短实施计划

- **B独占** `runtime/bundle/retrain/vicreg.py`、`tests/test_vicreg.py`：只提供 `gather_embeddings(local, rank=None, world_size=None)` 和 `vicreg_loss(z1,z2)` 及约3项必要小fixture。值/梯度对独立数学参考、无偏方差/协方差归一化、实际CPU两进程WORLD DDP参数梯度对完整batch参考覆盖本次新增数学；不重pairwise/Gloo旧套件或构造真实Small。
- **A独占其余必要接线**：新SCIENCE、`config.py`/`experiment91.py`、新有限 `ssl_stage.py`、`data.py`无标签返回的可选分支、`parallel_fresh_train.py`/`distributed_engine.py`联合阶段调用、`fresh.py`及正式初态/`single91.py`的derived身份、现有prepare/run_round配置与共用deadline消费点、原candidate fixture。只扩本轮两阶段路径，不建任意SSL平台；B两文件不共写。源`7e8e2eb`已过CPU的三处通信工程优化可复用，原科学、逐步finite和300秒恢复合同保持。

- [ ] B完成3项数学/可微gather文本，A在实时有效20%份额统一运行；A必要配置/阶段交接小fixture覆盖fit-only均匀分片与两view顺序、原head/RNG与空Adam/EMA、derived receipt及同一绝对deadline。合并真正相关断言，不扩拒绝矩阵、旧全套或control-only模型预演。
- [ ] A同次集成精确Git并按实际清单部署；正式原生CPU仍核public来源/50c/原stride2与配置，不能在CPU尚未SSL时伪造derived hash。生产阶段分支消费实际运行receipt，不跳过真实8rank初态。
- [ ] A当前动态核身份/资源/安全后一次activation，真实SSL首次更新报告64主图128视图、GLOBAL64×256及finite loss/grad；1600末步提交derived，沿同owner/deadline转监督，报告新whole/原head/空Adam/EMA及首64主图448对成功更新。只利用正常训练证据，不增加独立GPU预演或额外forward证明阶段。
- [ ] 完整监督内部freeze后一次official；内部与官方分报。统一必要结果归档，运行期间不追逐状态写过程commit，不碰旧冻结包。

## 可证伪判据与下一问题责任

工程有效且完整内部按原selector选出的AUROC必须**严格大于R107 `0.9884405731067525`**，才另行冻结同一recipe的既定seed20261027，对照R108 `0.9872328538178018`；本页不预建第二轮身份。AP独立分报，原固定难组若已有辅助分析只作描述，不能救主目标、重选checkpoint或用官方/test决定参数。原固定dev已多次使用，单seed或两seed同向也不能消除自适应选择偏差或证明显著性。

主要反例：G64下256维样本协方差秩最多63，统计估计不等大batch；无标签不变性可能压掉稀少病征，温和视图也可能学到取景/色彩捷径；1600步可能不足以获得有用适配，也可能损害原公开表征。额外曝光、投影器和优化阶段共同变化，因此正负结果仅评价这一预登记组合，不定位N06差距的根因。

首跑不胜则关闭**这一1600步VICReg式recipe**，不自动扫步数/投影维度/损失系数/温度/seed，AP不挽救。若工程未完成，先按真实故障区分工程与科学，不能把partial当失败。任何终态后规划均主动依据完整内部及实际成本提出下一独立问题并交协调/A，不再等待用户再次说继续；也不在本页预批准另一方法、第三跑或扩大数据来源。raw AUROC未达0.999不称总体完成。
