# R131：公开Small的固定pointwise rank8适配

根依据用户持续推进授权，在R130R1完整收口后正式采用本轮。R1为完整11200成功更新/14轮/bad4早停，唯一online c10@8000 AUROC `.9850016436060114`，未胜R107及R120 singlebest；固定layerwise .8关闭/no seed27，不扫邻近factor/LR/group/seed。R130原1503中断仍单独保留，不拼曲线或继承其initial/Adam。

**R131已真实运行，05:07:33.619216北京时间观察为225success=attempt/0skip/eval0。** runGit `7aa3232b7a2e47dedf68839e19deb29ce4ec6751`；身份窄验1/1、normalformal一次21/21及692源码exact均已通过，唯一activation0/正确live launch0。真实8rank初态489/147/4组/Adam0与自然step1的147个实际Adamstate/observed[1]/WORLDfinite/frozen342基座exact已证；owner342435与8rank342447–54的start/argv/实际96核mask/设备均匹配。完整小件和阶段pins见[启动归档](../../records/R131/README.md)。首个05:06 pending是正常等待原controller初态门，未重复入口/重clock。

## 唯一身份与比较

| 字段 | 冻结值 |
| --- | --- |
| order / config | 42 / `configs/R131-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-pointwise-lora8-v1` |
| candidate | `r131-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-pointwise-lora8-9010` |
| round | `r131-small22k-seed20261029-scale085-pairlog-pointwise-lora8-24k-p4-9010-20261101` |
| newprep | `exp-v3/technical-prep/r131-20261101-7c17` |
| direct comparison | R107，seed20261029；`layerwise_lr_policy=None`，不继承R1倍率 |

固定9:1 fit46256/dev5140/splitseed20261024、Small公开fe6b358a/ImageNet22K→1K/native1000 strict后原生binary head+age fresh、prepared512/scale[.85,1]/原旋转光度、单视图/G64P8/八rank/FP32/noTF32、原pairwise logistic、SD.1/dropout.2均保持。不能用旧R1/R130 checkpoint或initial初始化。

## 固定整体策略

使用`LOW_RANK_RULE`：全部36个block的两个pointwise Linear，共72个精确位置，r8/alpha8；每层为`linear(x,W0,b0)+linear(linear(x,A),B)`。A用按seed/version/module-name派生的独立CPU Generator做Kaiming uniform，B为零；原weight/bias路径和原345张量保持，不消费原模型RNG。原骨干342张量全部冻结，只训练144个因子、binary head2、age1，共147可训、489总张量。stem/downsample/depthwise/LN/layerscale及原pointwise权重均冻结，保留原train mode与跨层梯度。

因子/backbone均一initial LR1e-4，head/age1e-3；原二维/非bias WD规则、一次globalclip1不变。因子WD并不等于有效全权重WD；这是冻结公开基座+低秩更新+因子WD+因子EMA的整体策略，不宣称纯低秩因果效应。B0时A的数据梯度可为零，但存在梯度张量及合法Adam state，不能要求其首步非零位移。

最多24000次成功更新，每800步完整online/EMA配对dev，连续4轮内部AUROC无提升早停；原1600 warmup和EMA前1600 copy/1601起blend。现有EMA每次copy冻结基座，平均可训因子；`EMA(B)@EMA(A)`不等于`EMA(B@A)`。训练/dev/freeze/official统一使用未merge FP32路径，完整state保存base+因子+head+age/Adam/EMA/RNG，唯一selector及并列规则不变。

## 已有CPU证据与本轮公开初态

算法/状态两项toy已在[CPU准备](NEXT-PREPARATION-20261101.md)完成；6f31002是future-prep代码，不是R1 runGit，不重复这两项、原fixture/Gloo/cache测试或独立GPU预演。原最后age适配只做compile/SHA，正式外层describe/online-EMA/strict load由本轮既定normalformal承接。

本轮已在当次有效A12/nice19/隐藏GPU入口，使用精确6f源码做一次正常公开fresh CPU初态测量，不加载训练checkpoint、不forward图像。实际native public1000 strict 344张量，完整公头[1000,768]，无missing/unexpected，全部exact/FP32/finite；原345共同tensor SHA仍`50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`。新489 whole实际为`9a0665aac156939f13bb6dffbb10077d8c583d193f1eb291526c187834680fc4`，总489/冻342/训147、实际4个命名组/空Adam。

[PUBLIC-FRESH-INITIAL](../../records/preparation/r131-public-fresh/PUBLIC-FRESH-INITIAL.json) SHA `b0520794830c7a6ed94e275efc501394f6dc8fbe156baa6cdd1adfc54e842f8c`；任务CPU SHA `b47b65ef4c2f2b0406180a16c842733799f5c0f05d1a877a25b2bc29f5881418`。该测量阶段clone R107后产生SCIENCE SHA`ac66335ad5c54e77f776c5556c26c60b1531bde459708402ee0a34cc7b5ff54e`，当时尚无显式官方gate；最终SCIENCE另补`OFFICIAL_GATE_0996`并由新窄验/normalformal实核最终pin，不重建公开模型、不把阶段SHA冒作最终SCIENCE。

新身份唯一正向窄验已实际1/1通过，最终SCIENCE SHA `45995c2fe6783199c8df561accd5da623c6726c2276eef491c2d429ba0b7a321`。Pod `IDENTITY-CHECK.json` SHA `0570a2570d0609f6ac43bc1abedbbfa16b6912b7e4263a21962138ec69b29c02`，CPU任务`...-pri45ger` SHA `bea393c1435a81ef0196b9431e79ee9182a2244af21aa7778938d0d95e134264`、guard0/GPU未初始化。实际检查旧41/新42接受、双LOW_RANK_RULE、无layerwise/initial、正式`.996`门、new whole/共同345绑定，移除允许差异后整件科学值与R107全等。该项不替代本轮normalformal。

## 最小接线与执行顺序

- A唯一Git/Pod/GPU owner，不派生。精确注册profile/order42、固定policy与candidate/job一致约束及显式`.996`官方门；旧R130/R1身份和None路径保留。一项`test_low_rank_identity.py`核新42/旧41接受、直接R107科学值及实际public whole pin，不新增拒绝矩阵。
- 实际commit/push/readback后部署精确Git文件，按prepare对实际commit的依赖安排新技术包、新80% grant、normalformal一次、deployment freeze；本页不替代实际回执，未执行项不写通过。
- 当次重新核UID/Ready、Python/依赖、CPU份额TTL、有效CPU两互斥池、独占设备/锁、当前free与健康9C+I。现有policy按实时floor80%训练整树、floor20%准备；仍effective120时训练96核24–119，A准备12核0–11、根12–23保留不使用。新阶段独立绑定，旧allocation/grant/clock/token不改。
- 6h训练/8h整轮/9h软件grant、Small12GiB/free20GiB/floor16GiB/300s保护保持；不新增物理资源、不续租、不关闭实例。全部R130 source/manifest/cache/checkpoint/protected、R120及KEEP-MODELS保留。
- 新鲜薄链只在R131新根一次activation+一次正确live launch，真实8rank初态核489/147/空Adam/4组与frozen base；首自然成功核WORLDfinite、147个实际Adam state/step1及冻结基座exact，不追加forward/GPU预演。新saved-ID与后续step从实际state读，不用期望游标造证明。

normalformal、一次启动和真实首步已由本页首小件证实；尚无完整科学结果。活动冻结包不改，后续归档commit不冒runGit。通用standalone export/predict仍未支持或验证22K→1K材料与experiment91依赖闭包，本轮使用现有single91完整checkpoint冻结路径，不扩建导出平台。额外因子与保存边界冻基座hash的实际长期GPU吞吐、显存、保存时延、满24k时耗、500MB交付或100ms均未证；若需要新平台/缓存或扩大资源，报告具体不兼容并撤回该实现路径。

当前轻查仅复用既有工具，在本轮prep的`CHECK-RUNNING-SMALL.py`显式传当前source-git observer、本轮live `SITE_BINDING.json`、`START-OBSERVATION-R131-2.json`及独立新输出路径；核round/Git/site SHA与首次实证一致后新读done/status及进程PID-start。R1completion不作R131 locator，旧helper和原件保留，不为适配额外观察。当前A软件entry为f5c193、有效至epoch1793486122.2924988；过期只另建future prep绑定，不热改本轮冻结grant。

## 完整结果分支

完整工程有效且唯一selector冻结后，selected internal AUROC必须同时strict>R107 `.9884405731067525`、strict>R120 singlebest `.9894773092062407`，才取得固定seed20261027资格，对照R108 `.9872328538178018`。任一门未过就关闭这个精确rank8整体策略，不扫rank/alpha/位置/WD/LR/seed、不自动转归一化头，AP/partial/官方不救主判。工程中断或资源失败没有完整selector时不判科学阴性、不重置clock补跑。

最终完整freeze selected>=`.996`才官方5849一次；不足才`not_run_by_policy/threshold_not_met`。官方不训练、选点、早停或调参。内部dev、官方、平台分列，raw`.999`未证不称总体完成。根唯一heartbeat `9c70b2bc`继续每5分钟轻检及每`:00/:30`汇报，完成后先核全链终态与actualsafe再推进；A不另建timer。
