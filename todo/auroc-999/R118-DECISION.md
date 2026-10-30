# R118决定：DINOv1 ResNet50公开来源替换

2026-10-31。根已采用并分配唯一R118/order28，沿用户持续训练授权执行；本页复用[用户指定的来源方案](N06-FRAMEWORK-SOURCE-PLAN-20261031.md)，不重写其研究或再搜索来源。R117完整内部未胜，条件seed27不执行，该分支已收口。**DINO公开来源CPU验收已完成；生产新初态、正式包及实际启动仍待A回执。** 本页不是文档审批门或启动成功证明。

## 唯一身份与科学合同

| 项目 | 固定值 |
| --- | --- |
| 配置（仅A生成） | `configs/R118-SCIENCE.json` |
| order | `28` |
| profile | `single-resnet50-dino-imagenet-fixedbn-9010-24k-earlystop-seed20261029-v1` |
| candidate | `r118-resnet50-dino-imagenet-fixedbn-24k-p4-seed20261029-9010` |
| round | `r118-resnet50-dino-seed20261029-fixedbn-24k-p4-9010-20261031` |
| 唯一科学变化 | 相对R111有效下游合同，ImageNetV2 backbone来源→官方DINOv1 ImageNet backbone-only来源，包含新源BN统计 |

直接来源对照为R111，不重跑该对照。标准TorchVision R50→原生GAP2048→Dropout.2→Linear(2048,1)，Normal(0,.01)/bias0，**head_seed=20261027、job.seed=20261029**；同固定9:1 fit-only age统计、两个零初始化年龄残差。53层BN保持eval，全部159个公开源buffer固定，卷积及BN affine继续训练；EMA沿R111规则复制固定buffer，无重新校准、早层冻结或新teacher。

输入/优化沿R111：prepared/input512、单视图、缺省scale[.94,1]及其余原增强、ImageNet归一化、G64/P8、8rank各8、参考micro32、FP32/noTF32；BCE（loss缺省）、backbone LR2e-5、head/age1e-3、WD1e-4及原分组例外、clip1、原AdamW。SD0且标准R50无SD模块；不继承R117的pairlog、scale.85、Transformer或LN配置含义。每800成功步完整内部dev、最多24000、warmup/EMA绝对1600、原max(online,EMA) selector/min_delta0/patience4保持。

相对N06并非严格复现，旧划分/归一化/时钟均不移植；N06/FLAIR的任何权重、teacher、BN或年龄统计继续隔离。新源改变卷积表征与BN一起，收益不能拆成某一个的纯因果。历史DINOv2-S14阴性不等于本同构DINOv1来源已试，也不提供本轮收益保证。

## 已有来源实证与生产边界

本页已文本读取 `work/dino-source-20261031-e62a/receipts/HANDOFF-2.json`：passed=true，代码repo `facebookresearch/dino`、revision `7c446df5b9f45747937fb0d72314eb9f7b66930a`。现成转换资产为 `exp-v3/assets/dino-resnet50-v1-relay-1/converted/dino-resnet50-backbone.pt`，SHA **`7a228bfac4d2eb2332c91f93d5b0daaffd93bb2df4a1b8ad2918b2acc156cc68`**；同目录provenance SHA **`fb01cc841bf1beccfc9a2dfacd49390b51e345d6dc1cfa5c637dec54d57c516b`**。原公开文件由协调来源回执给定94345885字节、SHA `156f8c4166a23dc2951ae811e39d76a06269c565932edf647c0187e65cd7aa7c`，与代码revision、转换SHA分别记录，不混成一个pin。

HANDOFF已实证318个backbone state＝159参数tensor+159buffer、23508032公开参数、53BN/eps1e-5、GAP2048、fc=Identity；53个num_batches_tracked均来自源，派生counter=0。协调已报告metadata及3个tiny来源项GREEN。本轮**不重做来源验收**，不声称资产仍未验；许可/非医学ImageNet出处及边界沿原方案和已交provenance。

生产源分支必须严格加载这318项；DINO没有公开1000类头，不制造fc张量冒充来源、不用strict=False吞关键缺项。之后以原固定head seed创建binary头并零age。全新whole/参数/BN digest必须由实际生产工厂及正式流程取得，不能复制R111初态或Small的50c；初始化、EMA及checkpoint/inference均绑定同一DINO源和固定BN身份，`initialize=False`加载训练state时不重新覆盖公开参数。

## 分支启动前固定的比较门槛

| 比较用途 | 已固定的基线/门槛 |
| --- | --- |
| 来源对照 | R111 EMA c5@4000：AUROC `0.9754398301727533`，AP `0.757855599765702` |
| 首seed实用门槛T29 | R107 EMA c4@3200：AUROC **`0.9884405731067525`**，AP `0.8274326888154083`；高于完整R117 seed29的`0.9850076641408516` |
| 条件seed27实用门槛T27 | R108 EMA c4@3200：AUROC **`0.9872328538178018`**，AP `0.8354670944168833`；R117 seed27不存在，不补跑 |

阶段只一个新来源，先1次fresh seed29。工程完整、原selector唯一selected AUROC须**同时严格胜R111和T29**才另行固定同来源seed27复核；胜R111但未胜T29只记录来源改善线索，仍停止本阶段。条件第二轮必须严格胜T27才推进主线，两轮都报告、不挑最好seed。当前缺R111同合同seed27对照，因此不能称两seed来源因果已复现，也不增加第三跑。所有门槛启动前固定，启动后不移动；AP/难组不救主目标。

主源只有在正式训练前工程不可用时才由根另决策A1；本次已有来源验收，不能借一般不确定性自动切源。DINO完整科学阴性后不跑替补，不扫source/LR/BN/seed。工程失败记未完成，不冒充科学阴性或自动重启原attempt。每个完整结果后规划主动交下一问题决定，关闭该来源不等于总体目标暂停。

## 评估、工程预算与最短接线

完整训练结束并冻结唯一checkpoint后，selected内部5140 dev AUROC≥0.996才做完整official5849一次；否则`not_run_by_policy / threshold_not_met / metrics=null`，整轮可正常completed，safe仍需实际退出证据。partial触阈不执行官方。**0.996官方门与上述T29/T27条件复核独立**，未达官方门不能取消已满足条件的seed复核；不授权平台test CSV/提交，官方不用于选择。总体raw AUROC≥.999目标不变。

沿原train21600/total28800/grant32400秒，采用R50的**默认12GiB新round软件上限**，不延伸Swin专属16GiB；启动free20GiB、运行floor16GiB及全部保护/300秒保存/atomic/prune/fail-fast不变。A按实际参数、磁盘与运行绑定自然核验，不新资源/续租。既有R50实现支持该预算安排，不把早停或未测吞吐当全额完成保证；最终唯一模型500MB及指定硬件100ms仍以真实交付证据为准。

- A沿已派范围负责R118 SCIENCE及现有R50加载器的DINO backbone-only窄分支、来源/固定BN/新初态与正式候选身份、checkpoint/inference消费；复用训练/data/EMA/优化算法，只改真正受影响的来源断言及绑定，不继承旧运行token/clock/optimizer。
- 复用已完成source/HANDOFF及3tiny，不重来源、旧BN校准、loss/Gloo/像素套件。仅必要生产配置、backbone无fc严格消费、同head/age初值、固定BN与实际组/空Adam的受影响窄验；正式新包覆盖保存加载/推理。无额外control-only或独立GPU预演。
- A统一必要Git同步、实际清单冻结、实时环境与20%/80%互斥绑定、正式原生包、新初态；在前驱实际safe后唯一activation/fresh，由真实8rank证明启动。页面不作等待门，规划不联系A/B或派GPU；本页完成后停写，来源已验不等于生产已启动。
