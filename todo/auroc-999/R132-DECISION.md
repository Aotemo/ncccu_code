# R132：固定 raw-logit 平方0.1联合目标

用户已授权持续推进，根在R131完整结果与actualsafe后采用本轮。R131为11200success=attempt/0skip/eval14/bad4正常早停，唯一online c10@8000 AUROC `.9809751099048635`，未胜R107或R120 singlebest，关闭精确r8/no seed27。原R1301503中断、R1/R131完整产物及R120/KEEP全部保留，不加载旧checkpoint或initial、不拼接曲线或成本。

## 冻结身份与唯一科学变化

| 字段 | 固定值 |
| --- | --- |
| order/config | 43 / `configs/R132-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-logitsquare01-v1` |
| candidate | `r132-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-logitsquare01-9010` |
| round | `r132-small22k-seed20261029-scale085-pairlog-logitsquare01-24k-p4-9010-20261101` |
| newprep | `exp-v3/technical-prep/r132-20261101-7c17` |
| direct comparison | R107/seed20261029，公开fe6b358a fresh/step0/空Adam、原345全部可训；无LoRA、layerwise或冻结 |

仅新增精确`OUTPUT_REGULARIZATION_RULE`，同时绑定candidate与job；原pairlog六字段保持。目标为原64个最终image+age raw logits上的`mean_448 softplus(z_negative-z_positive) + 0.1*mean_64(z_i²)`，FP32/global_batch/mean，从step1起常数，不中心化、不detach、不截断、不退火、不classweight。沿原同一次forward和可微WORLD gather，无额外collective、WORLD倍率或图像forward；新增logit直接梯度`0.2z/N`，公共bias和age链都参与。dev/freeze/official继续原raw预测。

完整保持R107的固定9:1 fit46256/dev5140/splitseed20261024、Small公开ImageNet22K→1K/native1000 strict及原生head/age、prepared512/scale[.85,1]/原旋转光度、单视图/G64P8/八rank/FP32/noTF32、SD.1/dropout.2、backbone1e-4/head-age1e-3/原WD/一次globalclip1、24k成功步/每800完整online+EMA dev/连续4轮无提升早停、warmup1600与EMA copy1600/1601 blend、唯一selector。预期whole沿R107 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`；本轮正常formal真实重建证明，不另做公开fresh测量。原_optimizer分桶不改，全参数通常5组；由实际唯一成员/role/LR/WD与原保存身份验收，不造4/5组门。

论文的CE/线性化网络以输出L2替代参数L2分析，不能直接外推本pairlog+AdamW；本轮继续原AdamW，是不同联合策略。0.1来源是DomainBed多类CE默认值，不是本任务最优或收益证据，不声称纯特征效应或`.999`保证。出处及focal/Mixup/其他固定正则、R120/R121、R131的非等价负先验见[NEXT](NEXT-PREPARATION-20261101.md)。Random Erasing明确不选，不建立备用队列。

## 既有准备与有限执行

future-prep `e0f3d618c7efcea48f0318b3b1930f4f71a091a7`已有两项数学/Gloo/None/保存身份CPU检查2/2，6源compile及703源码Git blob/mode exact/GREEN pin一致；它不是R131 runGit，也不代替本轮正式验收。本次不重这两项、旧fixture/sourceprobe或独立GPU预演，只增加一项旧R13142/新R13243的真实身份正向窄验。生产只新增精确experiment91注册/.996/双policy绑定，所有旧profile仍None；不改模型、优化器、EMA、selector、保存算法或通信。

执行顺序按现有prepare实际依赖：身份窄验通过后实际commit/nonforce push/readback，精确远端源码→当前UID/Ready及有效本人CPU份额/新80%训练grant→正常formal一次→deployment freeze→新鲜薄链一次activation/正确live launch。准备时本人A12核0–11/nice19/隐藏GPU，根12–23保留未用；实际仍effective120时训练96核24–119。每阶段核TTL/互斥mask，f5到06:35:22；不足只新建future软件绑定，不改旧entry或训练grant。GPU段显式8卡可见，SITE SHA在A12核出并消费pin，使用正确`kubectl exec -i`；任一真marker存在即停，不重入或重置clock/token/grant。原6h训练/8h整轮/9h grant、Small12GiB/free20/floor16/300s存储保护保持，不新资源、续租、关实例或新timer。

真实八rank初态接管345全训/实际组/空Adam/50c/设备和80%整树。此策略没有LoRA/layerwise的step1专用事件，保持原log_every25；启动观察只读取小件及最多2MiB metrics首部，以真实首条成功update的实际游标/finite loss、8rank输入/64P8/448pairs及初态门确认，不为step1加collective/forward/probe或改日志频率。未逐项观察step1 Adam就明确未观察。观测成功后current-small显式用本轮SITE+真实first receipt；R131 completion只作历史。

唯一新身份窄验已实际1/1通过、0error，SCI SHA `8d48dd163240ec2df99ef2435445bbb088a771e055f8d65c9a895a09697790a4`；`NARROW-GREEN.json` SHA `bc9f97172686f1a85d0381b70e4a4e9d83f3f12089fb3cf6f308801ac71a3fb6`、guard SHA `6747ff66a0cc55e60f0f54c2533982b7a62b73e3d4835491d0056dc490e3d3c1`，GPU未初始化。此前RED为新profile未接入的1FAIL/0error，不是训练失败。规划对SCI/test/最小ex映射一次只读审查未见Critical/Important；旧两项数学/Gloo未重测。原件在本轮prep及`work/r132-20261101-a/NARROW-GREEN.txt`。当前尚在Git冻结前准备，不冒写normalformal、activation或训练成功。

## 完整结果分支

完整工程有效selector必须同时strict>R107 `.9884405731067525`且strict>R120 singlebest `.9894773092062407`才固定seed27，对照R108 `.9872328538178018`；任一未胜即关闭精确0.1目标，不扫系数/中心化/classweight/WD/LR/seed，不转Erasing。技术中断或不完整结果不冒科学阴性、不reclock补跑。完整freeze selected>=`.996`才官方5849一次，不足才正常`not_run_by_policy/threshold_not_met`。官方不参与训练、年龄、选点、早停或调参；内部/官方/平台分列，raw`.999`未证不称总体完成。

根唯一heartbeat `9c70b2bc`继续5分钟轻检及`:00/:30`汇报，准备阶段不重复旧R131 completion/safe；A不另建timer/服务或派生Agent，实际完整终态与actualsafe后才串行推进。
