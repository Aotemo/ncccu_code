# R128：预热1600成功步仅更新head/age，再全量更新

R128已完整工程有效并于2026-10-31 22:38:23.630北京时间实际安全退出。唯一EMA c6@4800内部AUROC `.9882093845688875`，低于R107科学门及R120实用门；本固定1600 head+age策略关闭/no seed27，不长度/LR/ramp/近邻解冻或seed补考。下文保留原预登记科学合同，实际分阶段证据与有限结论见末段；不是所有LP-FT/冻结方法均无效，也不是总体暂停。raw AUROC尚未达到`.999`。

## 唯一身份与科学问题

| 字段 | 固定值 |
| --- | --- |
| order / config | 38 / `configs/R128-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-headwarmup1600-v1` |
| candidate | `r128-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-headwarmup1600-9010` |
| round | `r128-small22k-seed20261029-scale085-pairlog-headwarmup1600-24k-p4-9010-20261031` |
| readout | `native_avgpool_final_LN_dropout_linear_plus_age` |
| newprep | `technical-prep/r128-20261031-e62a` |
| activation授权reference | 根对唯一A的执行派工 `9701c811-a5b3-4b76-9644-087749868c2b` |

可证伪问题是：保持原总预算，在原预热内先让新head/age适应，再允许全部骨干更新，能否改善R107同seed的完整内部选点。固定1600借用既有warmup终点，不额外增加LP阶段、样本扫描、训练步或缓存，不按dev表现提前或延后转段。

[LP-FT原文§4.1–4.2](https://arxiv.org/pdf/2202.10054)提供更新次序的机制先验，但原实验是固定特征上的L2 logistic regression再FT。本方案保留随机增强、SD/dropout、AdamW、pairlog、age和完整反向，是LP-FT-inspired固定变体，非严格复现；未证当前特征失真、ID/OOD关系、噪声或原warmup无效。最强非等价反例包括R113永久部分冻结`.9838264352051979`未胜R107、R96低骨干LR未胜其R95对照；旧V16完整5400步head-only两臂`.9791857418`/`.9792135975`均低于自身父`.9792535643`，旧V48参数锚定/V51冻结读出亦未胜旧基线。它们不是本次公开fresh全骨干暂缓后解冻，不跨合同相减，也不由定向检索未见等价完成而保证收益。

用户已明确True_Label按单张图可见征象；这是用户说明，非独立标注验证，不推无噪声或`.999`可达，不新建临床、规则或样本审计门。R127旋转identity的有限正向不作本轮收益证据。

## 精确policy与更新顺序

唯一科学变化是`job.head_warmup_policy`和`candidate.head_warmup_policy`均为下列完整`HEAD_WARMUP_RULE`，不得散写另一份可调参数；缺省None严格沿旧数值路径，不扩AMP或旧单卡age支持。

```python
{
    'version': 'head-age-only1600-then-full-v1',
    'kind': 'optimizer_role_head_warmup',
    'head_only_successful_updates': 1600,
    'roles': ['backbone', 'head', 'age'],
    'warmup_active_roles': ['head', 'age'],
    'backbone_action': 'grad_none_after_world_finite',
    'computed_gradient_check': 'all_roles_before_mask',
    'active_clipping': 'combined_global_l2',
    'max_norm': 1.0, 'norm_type': 2.0, 'epsilon': 1e-6,
    'nonfinite': 'world_consensus_before_mutation',
    'parameter_membership': 'unique_optimizer_parameters',
    'optimizer_groups': 'unchanged',
    'backbone_lr_schedule': 'original_global_successful_cursor',
    'ema_schedule': 'original_absolute_successful_cursor',
    'precision': 'fp32'
}
```

按已committed `successful+1`判下一步：1..1600仅head+age active，1601起全体active。原DDP注册、全参数requires_grad=true、train mode、SD/dropout、完整forward/backward/gather与RNG流程保持；不eval/no_grad冻结特征、不重建optimizer/DDP/paramgroup、不用find_unused遮图。骨干包括finalLN，原head/age角色成员和衰减分组不变。

原unscale后，在任何mask/clip/step前只读全部computed gradients，包括稍后丢弃的骨干梯度；所有role norm及实际active总global norm均须finite，合成norm也检查溢出。合并进入原一次WORLD finite判断，全部通过才将骨干`grad=None`，对active head+age合做一次global L2 clip1；1601后对全体合做一次globalclip1。不R126独立roleclip、不调用其apply、不追加第二globalclip。系数为device FP32 0d的`min(1, 1/(active_global_norm+1e-6))`，不每步逐role item/float或增加collective。FP32非finite仍fail-fast，拒绝前不部分修改grad、参数、Adam或successful阶段；原参数/Adam finite、ctx.any_flags、LR/skip/cursor失败语义保持，不能用mask隐藏骨干NaN/Inf。

`grad=None`不能换成零梯度或LR0：前1600骨干参数无更新、无WD且Adam state真正不存在，head/age沿原Adam更新。成功s=0三role state均absent；1≤s≤1600骨干expected step为None（无state），head/age各为s；s>1600骨干为s−1600、head/age为s。不补零moments或边界reset。第1601骨干第一次Adam更新直接使用原全局schedule峰值LR，不增骨干第二warmup/ramp。移除原骨干早期适应、改变head/age所受clip约束和Adam年龄都是该策略的内在影响，可能导致适应不足或切换冲击，不能称纯特征保护因果。

阶段仅由saved successful与绑定policy推导，无独立阶段counter。保存/严格恢复保留部分Adam映射、原参数组及各role计数、EMA/RNG；新policy的fresh status明确role预期/absent范围，不能仍用统一optimizer_expected_step=s误报，None旧显示保留。旧stage.py续训路径不扩改。EMA按原绝对successful：≤1600 warmcopy，1601起blend；online全requires_grad=true，不沿R113永久冻结copy名单。恢复窄验不授权任何已绑定轮的实际relaunch或旧checkpoint续训。

## R107保留合同

直接seed20261029、合规公开Small revision `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b` fresh、原GAP/finalLN/dropout/linear+age enabledtrue及原AGEcontract、fulltrain/stride2；原345state预期whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`及空Adam须本轮actual核，预期值不是实证。512prepared单图、scale[.85,1]、G64P8/每rank8主图与8视图/全局64view/448pairs、原binary-pairwise-logistic六字段T1/mean/float32/global_batch及WORLD可微gather保持。

SD.1/dropout.2、FP32/noTF32、backbone LR1e-4/head与age各1e-3、原WD/AdamW/globalclip1/EMA、固定9:1、原selector、每800评估/max24000/patience4、绝对warmup和EMA1600均不变；采样、coverage和earlystop从头执行，不为head阶段延期。`training_rotation=None`恢复原旋转，不继承R127identity；`training_photometric=None`、`gradient_clip_policy=None`、`attention_pool=None`、`stage3_residual=None`、training_views原缺省。不R120sigmoid、不医学源/N06/FLAIR或旧训练权重，不增加模型、原图扫描、缓存或辅助loss。

## 最短接线与三项窄证据

- **B仅两新文件**：`runtime/bundle/retrain/head_warmup.py`、`tests/test_head_warmup.py`。接口为`HEAD_WARMUP_RULE`、`build_update_groups`、`mask_and_clip_gradients_`、`expected_optimizer_steps`，只文本实现，不执行。两toy分别覆盖：①真实AdamW/独立手算active合并globalclip、前段无WD无state和1600/1601角色计数；②被mask骨干NaN/Inf、合成norm非finite和WORLD拒绝均发生在mutation前，无部分修改。
- **A独占其余**production/config/experiment91/两个engine/SCIENCE/正式绑定及`tests/test_head_warmup_binding.py`；第三项覆盖1599/1600/1601保存恢复对连续参考的参数、Adam/role步、EMA/RNG一致及SCIENCE/None兼容绑定。规划仅本页与todo入口，不同人不共写。
- 本轮三项窄验及正式20/20均已由A实际完成，详见末段，不重source/loss/Gloo/cache/EMA大套件、拒绝矩阵、控制模型或独立GPU预演；50c/空Adam与非tensor policy消费分别实证，不互相代替。
- 实际首成功及自然1600/1601阶段证据已由原回执消费，核骨干状态、head/age更新与各role Adam计数；未增加forward/model/gradprobe/query/poller。不把这三个时点推广为全程梯度主导或科学收益。

## 成本、执行与判据

每步仍原单编码完整前反向、pairwise和输入IO，readonly norm/mask有成本、早段少Adam不保证加速；满24000步+30eval约3–3.5h仅历史Small估计，非本轮benchmark。后段全量Adam存储峰值仍按原Small合同；最终500MB交付/指定硬件100ms尚未实证。原6h训练/8h整轮/9hgrant、12GiB round cap、启动free20GiB/运行floor16GiB、健康9C+I/reserve/并行prep容量及全部warm、best online/EMA、last/previous、initial和R120真实selected/source保护、300s atomic/prune/monitor/failfast保持。不靠earlystop、降batch/精度/尺寸/步数、扩时或删保护救限。

A唯一Git/GPU，B与规划仅文本；沿现有三角色，不新Agent/子Agent/云资源/租约/监控/轮询器/定时器。Mac仅文本/Git/rawtransfer；所有Python/配置解析/hash/model/tests仅A工具discover原Pod并核UID/Ready/Python/依赖/effectiveCPU/本人entryTTL后guard执行，互斥prep20%与训练准入owner/rank/loader/control/helper整树80%，不默认核数。CPU22 allocation SHA `80a4b755a5c16fdee26ead3f107e7fd0639eebc8003e32c0a5c5a6cc1395bb77`仅历史，TTL未报不猜；过期只同Pod必要future下一号软件份额，不改旧allocation/grant/clock、不续租。R127已safe归档不再查询，R128仍须实时独占设备/锁/存储/身份准入。新Git实际同步精确部署、长准备完成后沿原薄链两处120s live preflight→唯一activation/launch，不伪造观测或重试旧attempt；必要证据齐A直接执行，无文档批准门。

完整工程有效原selector selected内部5140 AUROC **strict>.9884405731067525（R107）**才同seed科学正向，同时 **strict>.9894773092062407（R120 singlebest）**才由根另派固定seed20261027对 **R108 .9872328538178018**。只第一门为有限正向endstage/no27；未胜关闭本固定1600策略，不阶段长度/LR/ramp/近邻解冻/seed补考。本次仅首seed，无自动27；AP/partial/难组/official/platform不救主判。R120不升级方法基线，R107/R108仍科学锚点。

完整训练结束并冻结唯一checkpoint后，selected内部 **>=.996**才原official5849完整一次；未达正常completed并记`not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，safe另实核。partial不触发；工程未启动无selected不能写threshold_not_met。无平台testCSV/提交，官方不选参；总体raw`.999`未达持续协调，不保证收益、显著性或可达，不称总体暂停/完成。此页到此freeze，不生成第二候选或下一评议。

## 实际终态与有限结论

根转交A实际2026-10-31 **22:38:23.630北京时间**终态：completed/early_stopping，success=attempt **8000** / max24000 / skip0 / eval10 / earlybad4，TRAIN completed/exit0/hard_deadline_met=true/signals[]，ROUND/done complete。规划仅一次文本核本机freeze的selected/顶层blob及TRAIN限定字段，未整件打印含token的TRAIN，未读预测或官方。唯一 **EMA c6@4800**、internal5140 AUROC **.9882093845688875**、AP **.8282870957132695**、error_twice **19584**；其他attempt/官方/safe等沿A经根转交的实证。

science_gate strict>R107 **.9884405731067525**为false，practical_best_gate strict>R120 **.9894773092062407**为false。关闭本固定1600策略/no27；不长度/LR/ramp/近邻解冻或seed补考，AP不救。R128训练8000/selected4800相对R107训练6400/selected3200并非等更新或等计算；结果不证明所有LP-FT/冻结普遍无效、预训练特征优劣或临床原因。R107原全程更新路径继续比较，R120仅singlebest、R107/R108锚点和总体`.999`目标保持。

原新三项窄验、formal20/20、native345/whole50c/空Adam、首25及实际成功步1/1600/1601的8rank policy消费、唯一activation/launch均分阶段保留，不重验。1600骨干342参数无Adam state、head2/age1 step1600；1601骨干step1、head/age1601，写expected_step前已逐参数核step和矩，不仅配置声明。原精确树首次缺R127归档三件的构建失败保留，remote补齐后的正式成功不覆盖失败。runGit为 `9caaf3e060da6b5d4d4291456094463e8cddf1be`、tree `277119598438b7eeb888afd4bb3690c138a93a6f`；当前actual main `f0d6902ef2b8e8cd424cfbbbad33db4990bda376`为独立observer提交，不冒充运行Git或本轮结果归档，archive Git待A actual。原source/部署/clock/token与原件不改。

selected不足`.996`，official正常`not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，不是执行失败。同次actualsafe=true：owner310573/start348080922现Z，只称Z不称reaped；原8rank310585–310592 absent，locks[]/deviceusers[]/denied0/8busy0，非agent idle推断。资源与保存合同保持，无活跃训练，不查询旧safe轮或自动新fresh。

本机raw根为 `work/r128-20261031-a/progress-2235/`，内部文件按exp-v3相对路径位于 `rounds/r128-small22k-seed20261029-scale085-pairlog-headwarmup1600-24k-p4-9010-20261031/SINGLE-MODEL-FREEZE.json` 和同round的 `evidence/TRAIN_RESULT.json`，不是旧平铺目录。以下pins仅转引原回执，不本机重算；模型tensor、checkpoint blob与文件SHA分别记录：

- selected tensor：`0edb6ee48f74935e662fde8151255053e86ab5c1319c1554a8222865719b8be1`；checkpoint blob：`7068d3f955640a326fc6dc64cb97fe36724d92b675e128c36e699f40dca7404b`。
- progress文件：`cfcfc8384246419de1452b7642f3895e35f0e0b859b89e8cdb116e013cd83246`；CPU回执：`a1c881e60583da1608dfedd1e0ec515247686aee836853b9293f4dc64841cbed`。
- freeze文件：`1e27800cb27b08b932e45bd23c071b278b937bdd23c19a114613da8153454f3d`；TRAIN文件：`0d16d3d61de755560329b33bfc3f3b48c9bc189e82167d02450c67b758cae481`。
- ROUND文件：`7cbc343f8102be313e4acae9f45e8ed588e6811f79fdc4039cedb4c5dc2dadd5`；done文件：`b118950a67d3601d6b2ae00beebd97e21dbc780f41694eff88edfa932f58f17f`。官方和安全事实仅引回执，不用官方选科学。
