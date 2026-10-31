# R130R1运行期间：固定第二seed与一个后续方向的有界准备

**接续：R130R1完整两门false/actualsafe；唯一R131/order42固定pointwise-r8已获真实首成功。** R1最终11200/eval14/bad4，唯一online c10@8000 AUROC `.9850016436060114`，固定layerwise .8关闭/no seed27，见[完整结果](../../records/R130R1/README.md)。R131身份和步骤转入[R131决定页](R131-DECISION.md)：runGit7aa3232，身份1/1、normalformal一次21/21、692源码exact及唯一activation/正确launch0；05:07:33.619实查running225/0skip/eval0，真实八rank初态与自然step1的147实际Adam[1]/WORLDfinite/冻基座exact已证。新public whole9a0665、原345共同50c/489总/342冻/147训/4组，空Adam只指初态。新SCI显式`.996`，尚无完整科学结果；小件见[启动归档](../../records/R131/README.md)，不重toy或新增GPU预演。本文仍由A唯一写入。

下文保留准备阶段的两条条件路径与证据边界：A路径因R1两门未过而关闭；B方向现已作为唯一R131采纳，不与R1叠加。历史“未采纳/无身份”仅指当时阶段，新身份/新时钟/新grant由R131独立承接，旧轮与保护资产保持。

## A. 两门通过后的固定seed27

资格仍需R1工程完整、唯一selector冻结、实际安全退出，selected internal AUROC严格大于R107 `.9884405731067525`且严格大于R120 singlebest `.9894773092062407`。任一不满足不能派这个seed；partial/AP/官方不救。合格后最新持续推进授权覆盖固定第二seed，无额外候选菜单。

| 科学字段 | 固定差异与不变项 |
| --- | --- |
| job.seed | 仅`20261029 → 20261027`；split seed仍`20261024`，原9:1名单及AGE合同保持 |
| 新fresh初态 | 从相同公开Small fe6b358a重建，空Adam；seed改变自然改变head及训练随机流。沿[R108 SCIENCE](../../configs/R108-SCIENCE.json)既有seed27预期whole `91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67`，须由新formal实际证明，不套seed29的50c，也不加载R108/R1 checkpoint |
| layerwise | 原固定0.8/12层映射、全部层从step1更新、原30组、scaled initial_lr与全局成功游标schedule不变 |
| 其余科学值 | 完整沿R1：同源/512/单视图/G64P8/FP32、pairlog、age、旋转光度、LR/WD/clip/SD/dropout、24k/800/p4/绝对1600 warmup和EMA、原selector及`.996`官方门 |
| 运行身份 | order/profile/candidate/config/round及新运行绑定待资格证实后由根冻结；名称变化不是新科学变量，不复用R1 clock/once marker |
| 主比较 | 第二seed完整selected严格胜匹配R108 `.9872328538178018`才支持两seed同向；不选最好seed、不改用较弱基线或以AP挽救。两seed正差仍不等于显著性或`.999`保证 |

A可先准备一个配置差异断言及现有`experiment91`/formal双profile接线的文本补丁：去掉运行身份/note后，只允许job.seed及其自然fresh initial控制值改变，layerwise policy全等。新身份未定时不杜撰SHA/profile，不再跑R1已过的算法fixture；真实新初态、空Adam及命名组由未来正常formal与真实8rank首步承接。

## B. 历史依据及仅两个替代的取舍

本次复用[全历史复盘](EXPERIMENT-REVIEW-20261030.md)、[历史结论](../../records/storage/EXPERIMENT-CONCLUSIONS.md)、[当前终态入口](../README.md)及相关决定/结果页，不重算指标。旧8:2、当前9:1、官方和平台分列，旧实验只能提供非等价先验。Tiny/SAM/L2-SP/EMA教师在入口与交接中已明确评议未采用；迁移工作区未见这些评议的详细原件，本页不虚构原论证、不把它们包装成未试方向，也不重新评议。

| 已有证据 | 本次采用的限制 |
| --- | --- |
| R91–R108及旧R/N/V汇总 | source/容量/seed的影响并不单调；scale .85与pairlog有配对支持，但不外推`.999`可达。旧融合/N06医学参数不是候选；旧V16 head-only、V48参数锚定、V51冻结读出未胜各自旧基线，见[R128历史说明](R128-DECISION.md) |
| R96/R97/R98/R99/R100/R109/R110 | LR、head LR、dropout、focal、P4、SD等固定尝试已有负先验；本页不从这些近邻扫描中挑下一轮 |
| [R113](../../records/R113/README.md) | 永久冻结早期features[:4]仅`.9838264352051979`，明显未胜R107；保留公开特征并非普遍有益，不能因为冻结会省Adam就推断泛化改善 |
| [R128](R128-DECISION.md) | 头/age先训1600再全量仅`.9882093845688875`；本次不扫冻结时长、深度、解冻ramp或LR |
| R114R1/R115/R116/R117/R118/R129 | stride1、固定SSL1600、Base/Swin/DINO和22K-only各自固定合同已闭；不换近邻checkpoint/源epoch，不重开这些组合 |
| R120/R121 | sigmoid一胜一负，R120 `.9894773092062407`只作singlebest；不再选近邻有界surrogate、温度或混合目标 |
| R122/R123/R124R1/R125/R126/R127 | 去光度、去age、content pool、stage3读出、roleclip未胜；R127 `.9886320261146719`有限正向但未刷新best，不能写双门false，也不扫旋转/几何 |
| R119及FIT/cache/C0 | R119是12CPU读取筛查失败，非局部科学阴性或实际96CPU超时；FIT256 unknown/首cache missing/C0均已收口，不修后补考、扩样或建全局部cache |

此次只比较两个实质方向：

| 方向 | 信息价值与代价 | 结论 |
| --- | --- | --- |
| 全36block的pointwise rank8增量，原骨干冻结 | 改变可学习更新集合：每个block仍可通过非零增量适应，原权重不动；同公开来源、单次图像编码，不新数据/cache。新增因子、冻结及其优化/EMA语义需精确绑定 | **唯一保留给根评估的建议**；是受限适配整体策略，不保证收益 |
| 归一化/余弦分类头 | 改变样本特征范数及读出几何，算力增量小；但须另选logit尺度、与age残差的相对量级，且R124R1/R125已给读出修改负先验 | 本次不选，不形成备用自动队列，也不在R1阴性后转而扫尺度 |

归一化头的原作者实现`distLinear`明确包含特征归一化、可学习类权重范数及额外scale，不能称“去掉一个范数但完全不改优化尺度”。这里只支持上述取舍，不套用其few-shot结论为当前任务收益。[作者源码](https://raw.githubusercontent.com/wyharveychen/CloserLookFewShot/master/backbone.py)

## C. 唯一建议的机制、预设和负先验

LoRA原文给出冻结`W0`并训练`BA`的参数化，原实验主要是语言模型；它没有证明当前Small或本医学图像任务的更新低秩，也没有证明本轮收益。固定`r=8, alpha=8`是本次规划为单次可证伪实验选定的小容量，使`alpha/r=1`，不是用户指定、论文给出的本任务最优值或搜索结果。失败即关闭这个精确组合，不改rank/alpha、覆盖层、LR或seed补考。[原论文§4.1–4.2](https://arxiv.org/html/2106.09685v2)

直接对照回到R107 seed29，不继承R1 layerwise倍率；候选`layerwise_lr_policy=None`，原均一backbone LR1e-4用于新增A/B，head/age仍1e-3，其余R107科学值不变。主干只能从原公开fe6b358a fresh加载，禁止旧训练checkpoint、N06/FLAIR、外部医学teacher或统计。

精确作用点为`network.features.s.j.block.k`，`s∈{1,3,5,7}`分别含`3/3/27/3`个block、通道`C=96/192/384/768`，每个block的`k=3`为Linear C→4C、`k=5`为Linear 4C→C，共72个Linear。stem、downsample、depthwise卷积、所有LN/层scale及原Linear weight/bias全冻结；只训练72对A/B、原binary head和age。保留原模块train mode、SD.1/dropout.2及单次DDP forward，不用`eval()`或`no_grad()`切断跨层输入梯度。该位置来自官方TorchVision0.25源码，并须未来与Pod实际模块名称/类型核对，不升级依赖。[CNBlock及Small配置](https://docs.pytorch.org/vision/0.25/_modules/torchvision/models/convnext.html)

每层运算固定为`linear(x,W0,b0) + linear(linear(x,A),B)`；无新增dropout、bias、非线性或可学习scale。A形状`[8,in]`，B形状`[out,8]`；采用微软Linear实现的A Kaiming uniform（a=√5）、B全零，而非把源码误说成论文Gaussian初始化。[微软原始Linear实现](https://raw.githubusercontent.com/microsoft/LoRA/main/loralib/layers.py) 不新增loralib依赖；若复制其代码则保留[MIT许可](https://raw.githubusercontent.com/microsoft/LoRA/main/LICENSE.md)，不下载任何该库训练权重。

参数规模最初由上述形状与仓库原Small计数作静态推导，随后本次Pod CPU原生结构核对得到相同计数；技术模型不加载public权重、无图像forward，不代替未来formal。推导为：`Σ blocks C = 3*96+3*192+27*384+3*768 = 13536`；每block两Linear共`10*r*C`个新增参数，所以A/B共1,082,880元素/144tensor。原骨干49,454,688元素/342tensor全冻；head769元素/2tensor、age2元素/1tensor仍训。合计可训1,083,651元素/147tensor；全模型50,538,339元素/489tensor。新增A/B保持技术模型原345个张量的值/名称；公开fresh的新whole仍待正式采用后实核，不能仍填50c。

这不是R113冻结深度补考：R113的早期block不能学习，本方案每个block的两个pointwise都有可学习增量；也不是R128阶段长度变化，没有1600后解冻；不是L2-SP惩罚或R1层LR倍率，优化变量已变成矩阵因子。但共同的保留公开基座假设仍受R113/R128/旧V16/V48/V51负结果约束，不能因机制不同就忽略它们。固定低秩可能限制必要域适应，且depthwise/normalization完全冻结比R113更严格；这是一项有信息价值但证据较弱的候选，不声称纯低秩因果、特征失真已证或收益可保证。

## D. 不能省略的数值与状态语义

1. **初始化与RNG。** 先走原native1000 public strict加载、原binary head/age构造，再在CPU隔离RNG上下文中附加A/B。保留原`...weight/bias`路径和张量值，不嵌套成改名base.weight。隔离seed固定为SHA256文本`job.seed:convnext-small-pointwise-lora8-v1:module_name`的前8字节big-endian模`2**63`，仅本地CPU generator/fork_rng消费；退出后原Python/NumPy/Torch CPU及设备RNG均不得被附加构造改动。B=0只给数学零增量，不证明整state或RNG等价；验证原345共同张量/head/age、RNG后态、实际零增量及新whole SHA分别报告。`initialize=False`只重建相同结构后strict加载训练state，不回灌public覆盖参数。
2. **优化器及WD。** 冻结原骨干不入Adam，无grad、moment或WD；A/B按现有二维参数判据走backbone LR1e-4及AdamW WD1e-4，head/age沿原角色分组，所有可训参数合做原globalclip1。因子WD不等于对有效`W0+BA`做WD，也不等于只约束rank；冻结+低秩+因子WD是整体策略。B=0时A的数据梯度初始可为0，仍有合法梯度张量/WD/state；不得要求所有A/B首步非零位移或延迟其Adam年龄。
3. **EMA。** 保持现有成功游标、前1600copy/之后原decay；冻结基座每次copy，A/B及head/age按现有参数EMA。明确`EMA(B) @ EMA(A) != EMA(B @ A)`一般成立；本方案选择的是**因子参数EMA**，不临时改成有效权重EMA，不称与R107有效权重平均等价。online与EMA仍各是一个完整单模型variant，沿原selector择一。
4. **禁止自动merge。** train/dev/freeze/official统一使用上述未合并两支FP32运算。微软库默认eval会merge、train会undo，故不能直接照搬其默认切换；合并权重再单次linear与未合并路径的舍入可不同。本次不在观测、冻结、官方或交付时偷偷merge，不凭原论文“无推理开销”声称本实现100ms通过。[作者merge说明](https://raw.githubusercontent.com/microsoft/LoRA/main/README.md)
5. **保存与严格加载。** 保存完整base+A/B+head+age及optimizer/EMA/RNG，不只保存adapter差分。给可训名单、冻结名单、完整名称/顺序、shape、LR/WD、saved param-ID位置绑定；未来resume fixture只验状态语义，不授权实际续训旧轮。不得把当前R1的345个Adam state/30组要求照搬为候选期望147可训tensor；实际数量由候选正常formal证明。冻结base的online/EMA必须逐张量exact，原source身份独立保存。

## E. 资源可行性与最小缺口

仍一个Small、单图512、G64/P8、FP32、8卡，不新增编码视图、原图读取或缓存，不扩大6h训练/8h整轮/9hgrant/Small12GiB与free20GiB/floor16GiB/300s合同。省去多数Adam矩有静态储存优势，但全层adapter仍需传播激活梯度，72处额外两小矩阵乘也非免费，不能推断GPU更快或显存一定降低。

按当前payload中一份online、一份EMA及可训参数的两份FP32 Adam矩，主要浮点张量字节推导为`4*[2*50,538,339 + 2*1,083,651]`，约0.413GB/非initial checkpoint；这不含步计数、RNG、metadata、原子写临时件/所有保护副本及文件系统余量，不是实际序列化回执。未来A只需在正常候选formal中核实际全state/Adam规模与健康9C+I及当前free；不能为不足删保护或扩资源。满24k耗时和端到端100ms仍未知，不增加独立GPU预演，不靠早停保证预算。

两项小CPU正向证据及结果无关接线已完成，见下文。剩余缺口是条件采纳后的正式身份、真实公开fresh whole、新正常formal和GPU实际行为；不是需要新临床/标签/全样本审计。若根采用，沿现有正常formal一次及真实训练承接，不重跑已过toy或另建GPU预演。若发现需重写DDP/保存平台、全数据cache或超既有资源合同，则报告具体不兼容并撤回此候选，不扩成调参网格。

## F. 给A的具体文件及CPU准备边界

根已授权A按下面范围实现并验证CPU技术准备，**不是根已经采用LoRA或要求A现在派发训练**。新的独立目录为`exp-v3/technical-prep/lowrank-cpu-20261101-7c17`，CPU allocation SHA `f5c19340844c353b257725717b6ec195215b17bd7ac776d3ea08c0e9ff4cf5b0`，创建epoch`1793475322.2924988`、有效至`1793486122.2924988`。本次实核effective120，A12核0–11、根保留12–23未使用，与活跃训练96核24–119互斥；旧allocation/grant保持。nice19、隐藏CUDA/HIP/ROCR，不取训练锁、不初始化GPU；每次入口仍核UID/Ready/TTL。无新增Agent/物理资源/续租。

| 文件/接口 | 最小职责及边界 |
| --- | --- |
| 新`runtime/bundle/retrain/low_rank.py` | 固定policy、`attach_low_rank(model, job)`和`low_rank_proof(model, job, optimizer)`；仅明确72个Linear/上述r8策略，None严格no-op；保持原weight/bias路径、隔离A/B RNG、固定冻结名单与命名组身份，不做通用PEFT平台或自动merge |
| `models.py` | public strict及native head/age完成后附加；普通fresh/initialize=False同结构，现有其他模型与None路径不变；不跳过原来源验收 |
| `config.py`、`experiment91.py`、未来SCIENCE | 新显式`low_rank_policy`进入job_contract/science/candidate/describe_model/validate_checkpoint；精确源、层名/形状、frozen/trainable与无buffer新增身份。正式order/profile/config路径待根采纳后冻结；不松绑旧profile，不改R1 SCIENCE |
| `engine.py`的`_optimizer`及`distributed_engine.py`保存/恢复 | 原optimizer只消费requires_grad参数，数值更新不重写；仅在新policy下附命名组/param-ID绑定并在load前核；将low_rank_proof沿既有parameter_trainability/initial-ranks/checkpoint路径传递，None不变。沿原一次WORLD finite、clip、sampler/gather，不改loss或通信 |
| `ema.py`、`fresh.py`、`single91.py` | 已有冻结参数copy与参数EMA可复用；核既有job/variant/strict加载是否自动传播新policy，仅补缺失身份，不改EMA算法、selector排序或official once。冻结/官方都经initialize=False重建未合并模型；原图像/age推理接口不变 |
| `tools/validation/check_candidate.py`、`runtime/r92_initial_gate.py` | 新候选formal证明共同原345张量/新whole、精确147可训/342冻结/489全tensor、实际组和空Adam；现有rank入口消费对应CPU期望，不重建GPU控制模型、不留硬编码旧30组/345 Adam断言 |
| 新`tests/test_low_rank.py`与`tests/test_low_rank_binding.py` | 仅下面两项有实际风险针对性的正向证据；不重跑R1三项算法fixture、全Gloo/source/cache/EMA测试或拒绝矩阵 |

两项必要正向证据：

- `test_zero_delta_rng_and_trainable_membership`：小Linear合成输入，在原weight/bias/head保持、独立A非零/B零时验证零增量与RNG退出后不变；对非零A/B比较显式两支参考的输出/梯度；确认原基座冻结、optimizer只含A/B+head/age且唯一全覆盖，首步A零数据梯度不误判失败。验证train/eval切换不改weight/state、没有merge副作用。
- `test_named_state_roundtrip_factor_ema_and_variant`：两个小block、原AdamW/clip和现有EMA，以warmcopy边界附近连续参考对保存/strict load后的下一步，核base保持、命名组/Adam step/moments、A/B/head/age、RNG和因子EMA一致；用小fixture明确EMA variant按自己的因子重建，不混online A或B。新SCIENCE身份采用后将该fixture接现有freeze/load正向入口，不增加真实图像/模型前向探针。

实际GREEN2为两项2/2、0error，耗时6.885秒，11个修改Python源在Pod compile通过。原件[CPU-CHECKS](../../records/preparation/lowrank-cpu-20261101/CPU-CHECKS.json) SHA `a7597deb5a598b7e9984ffc233e1d98e0b63bd618aa9bb8653a1dbeb0319959d`；[CPU-GUARD](../../records/preparation/lowrank-cpu-20261101/CPU-GUARD.json) SHA `1301656c74a019daba28c92d2030be38162a526817f5953493b00671396dd868`，guard0、GPU未初始化。native Small技术构造实测345共同张量exact、72目标、489总/342冻结/147可训及原生空Adam；两小Linear模型实际7个可训Adam state的step475→476与保存游标相符，矩、RNG、冻结base、因子EMA及variant严格重建相同。toy使用现有warmcopy474规则覆盖copy/blend边界，不冒作未来科学合同warmcopy1600或native147个Adam的实际训练。

GREEN2之后只改`experiment91.describe_model`一处可选分支：向结构描述补传`resolved_age_spec()`，因为原SCIENCE.job不包含运行时age。本次[FINAL-SOURCE-PINS](../../records/preparation/lowrank-cpu-20261101/FINAL-SOURCE-PINS.json) SHA `9167a535c6863a496a3d1c921585432d26000f1db6913d979d6c8944ca867a00`记录该变更后11源compile、其余10源与GREEN2字节pin全同；这是一行修正的compile/SHA证据，不是最终全部源码重新2/2或descriptor实际调用通过。该任务guard0，原Pod回执`tasks/...-r4ei1elx/CPU-RECEIPT.json` SHA `b731cf85b3aa95c6637d3f571fc94d56a01cab9c320e394bec50036a0ba6b428`。

实现复用现有EMA、fresh job_contract、single91 selector与严格加载，不改EMA算法、loss、通信或调度。可选policy才产生factor/optimizer/proof字段；None不附参数、不消费RNG、旧job_contract不多写None。保存及载入后的Adam证明逐state实读step/矩/覆盖并与成功游标比较；冻结base另算实际digest与初态reference比较。EMA readout全部requires_grad=False，因此结构角色与online实际mask分开描述。新实际首步证明、normalformal和八rank门仅接线，当前没有执行或伪造通过。

独立只读stage2审查未发现本次CPU准备范围的Critical/Important，非测试或GPU实证。审查确认上述age单行适配合理，现阶段无需另跑wrapper检查；正式采用后的既定normalformal直接承接。已知通用standalone `export_model`/脱离冻结包predict仍未支持或验证该正式候选：原materials限制ImageNet-1K，export源码也未闭合experiment91及绑定依赖。本阶段仅承诺现有single91 freeze+完整checkpoint路径的接线，不把新增三个export文件名当作可搬运artifact通过，不为本次扩建导出平台。descriptor及保存证明会在保存边界实读/哈希冻基座，实际GPU保存时延、吞吐与显存仍unknown，不据CPU通过承诺预算内时耗。

上述toy不证明公开fresh Small初态、GPU吞吐、数据或收益。根采纳后A再做新候选正常formal、必要Git push/readback及精确部署、唯一新activation/launch；首自然成功核实际新可训集合和frozen base。合格seed27分支优先时，LoRA准备保持候选状态，不抢GPU或叠加参数改动。当前SCIENCE/order/profile/round和新whole仍未冻结；本次future-prep源码归档版本由本页Git历史定位，与活跃R1 runGit `bc5d788b51bb2d1be3a9f76f6088e7c3cc3aabdd`分列，活跃冻结包不动。

## G. 结果分支与最终口径

若LoRA最终被采纳，首轮仅seed20261029、直接R107，沿完整工程有效selector两门strict>R107且>R120 singlebest；不合格关闭该精确rank8策略，不能回头扫rank/alpha/位置/WD/LR/seed，也不自动改做本页已放弃的归一化头。合格才固定seed27对照R108，具体身份由根届时冻结。R1当前结果不能预写为该分支触发。

两条路径均保留完整freeze selected>=.996才官方5849一次；官方不训练/选点/调参。内部dev、官方及平台分列，内部`.999`不冒官方/平台达标，raw`.999`总体目标未证前不称完成。公开自然图像源、单模型、原固定9:1及全部资产保护保持；不生成未经授权的平台test CSV。根唯一heartbeat继续报告和有界完成检查，A不新增timer，规划不自行轮询。

本次一手检索限定LoRA论文/微软代码及许可、TorchVision CNBlock、归一化头作者代码；CVPR PDF返回403、OpenReview PDF遇验证页，未用博客或新闻补证，归一化头只引用实际读到的作者源码。源码main链接仅作机制资料，不作为新冻结部署版本或参数资产；没有下载/运行其代码或权重。文档完成交根评估，不把推荐写成已采用。
