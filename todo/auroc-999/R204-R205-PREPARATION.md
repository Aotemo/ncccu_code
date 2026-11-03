# R204/R205 固定困难负例四分之一混合 Implementation Plan

11-03 06:11独立R205已在R203训练期间完成[normal18/18和精确部署](../../records/preparation/hard-negative-20261103/NORMAL-R205.json)，guardlkcjtc1w/72f2680e exit0，actual SITE10afb903、原7ead/SCI7aaa/whole50c6/345全训/5空组；原6目标5compile按SHA复用。无clock/activation/GPU。完整R203双门及actualsafe前不启动下一GPU；条件R204仍仅原technical包，当前R203/573保持。

04:23已完成Git7ead152实际commit/tree回读与[两technical包](../../records/preparation/hard-negative-20261103/TECHNICAL-PACKAGES.json)，guard均0，R204实际run_name104/16GiB、R205实际100/default12GiB；无normal/GPU allocation/clock/GPU。原13b helper、现役R201/8635和先行R203/573保持。下文“尚待Git/技术包”为之前CPU完成时点。

11-03 04:18已完成[六必要目标、五源码编译及两actual public fresh](../../records/preparation/hard-negative-20261103/README.md)，CPU汇总1b7cd941/guardtvbimxcy exit0。R204 whole6a261e48/165全训/5空组、R205 whole50c6/345全训/5空组，0图像forward；SCI分别f079cdd6/7aaa297d。原RED及两次私有汇总脚本错误均保留。当前尚待Git冻结/实际回读与两technical包，没有normal/clock/GPU；R201/8635与后续R203/573不混入新源码。

> 使用 superpowers:brainstorming / writing-plans / executing-plans / TDD，由32e3本人在现有20%池实施；用户已明确最小修改无需审批，不恢复/派生Agent。

**Goal:** 为R203之后准备条件seed27复核及一条独立、批内困难负例梯度假设，不阻塞当前R199或既定R201/R203。

**Architecture:** 复用原classification_loss、WORLD gather/DDP、AdamW及保存/评估/启动链。新损失只在当前完整8正×56负softplus矩阵沿正例平均，保留0.5全体均值，加0.5最高14个负例均值；WORLD负例原顺序稳定破同分。没有额外参数、状态、forward、历史分数或名单。

**Tech Stack:** 当前已核Pod/Python3.12.7/PyTorch2.10+DTK；无新依赖。全部Python/配置解析/测试/哈希在实核Pod nice19/隐藏GPU下执行，本机只文本/Git/传输。

**Spec:** 本页是封存设计与实施计划。候选做法为继续改公开模型、重试困难正例、或只改变批内负例权重。前者已有R203，原R156正例top2固定配方完整阴性；本次选第三项。14/56与R156的2/8同为四分之一，0.5混合也保持，便于区分梯度侧重方向；不扫描k/系数/seed或按dev样本加权。R197封存20正例局部净改善未胜完整R120，不据它证明负例是根因。

[When AUC meets DRO](https://proceedings.mlr.press/v162/zhu22g.html)及[LibAUC原实现](https://docs.libauc.org/_modules/libauc/losses/auc.html)提供关注困难负例/低FPR区域的概念参考。这里是自己固定的有限batch原全AUC-surrogate与负例尾部混合，不是论文/SOPA/LibAUC逐字复现、总体无偏估计或pAUC替代目标；不安装LibAUC、不引入其逐样本状态/dual optimizer。选点仍完整内部AUROC，不选pAUC；可能放大含噪负例，不能证明临床机制或AUROC收益。

## Global Constraints

单模型/固定9:1/合规公开fresh/官方隔离；最多24000成功更新、每800评估、p4、min_cycles1、EMA1600与唯一内部AUROC选点保持。完整selected>=.996才官方一次，raw .999未达。当前R199/82、R201/8635及R202/390、R203/573冻结不改；准备ec5/root0–11、peer12–23保留，与训练24–119互斥。绑定续期只改未启动stage，旧clock/token/grant/失败原件保持。

R204/order117只在完整有效R203严格胜R107 `.9884405731067525`及R120 `.9894773092062407`并actualsafe后，同native ViT-B/publicc6a/CLS/512/原logistic/backbone1e-5及其余R203合同、seed20261027独立fresh复核，否则仅技术准备。精确新round为`r204-vitb-dinov3-seed27-lr1e5-cls-storage16-24k-p4-9010-20261103`，仅加其16GiB工程余量，旧override/default12/free16不变。

R205/order118独立direct R107原Small/fe6/seed20261029/512/FP32/ImageNet/logistic/G64P8/448对/原AdamW/LR/WD/24k余弦，仅loss改固定negative-top14-mix05。`L=.5*mean(softplus(n-p))+.5*mean(stable_descending_top14(mean_over_8_positives(softplus(n-p))))`；矩阵维度8×56，沿dim0得到56负例项，stable=True按WORLD负例顺序取14。九字段独立version/kind加固定top_negative_count14、hard_negative_weight.5、tie_break WORLD_negative_order_stable，拒绝变值/多缺字段。全部448对仍有原项梯度，112个选中对多获梯度，不增加独立标签样本。不叠WD.05、wide2、gamma、低LR、困难正例或旧checkpoint。round `r205-small22k-seed29-pairlog-neg14mix05-24k-p4-9010-20261103`默认12GiB。

历史R107没有当前显式official_evaluation字段，R205沿用现R120的完整内部selected>=.996才一次官方门，原R107文件不改。未封存草稿缺该字段已在首次候选GREEN前发现并保留；补齐后SCI为7aaa297d，不能称官方字段逐字来自R107。

## Review Focus

1. 局部无正例rank仍通过既有可微WORLD gather拿完整8×56；真实global/local logit及DDP参数梯度对独立解析值，不另乘除world。
2. 全同分与第14名边界同分必须稳定；用分类器零空间不同特征使实际参数梯度可辨选择，不把expected index写成实际观测。
3. None/logistic/sigmoid/wide旧值与梯度保持；新目标FP32、平移不变/共享bias梯度0、有限极端值及严格合同。
4. SCI/candidate/job_contract/保存身份拒绝错loss/LR/seed/来源或叠加；模型/EMA/cold路径源码未变按SHA复用。
5. 原生ViT非持久RoPE、两actual fresh、5空组及native组LR/WD；新短run_name经真实technical构造验证<=128，存储只加R204精确键。

## Task 1：独立损失与真实WORLD窄验

Files：`tests/test_pairwise_hard_negative_loss.py`、`runtime/bundle/retrain/classification_loss.py`。消费原classification_loss/gather接口，新增固定PAIR_HARD_NEGATIVE_LOSS及kind，其他函数不改。

- [x] 先写独立Python标量/解析8×56梯度、全同分/边界同分、legacy bitwise/严格schema和真实2rank Gloo目标；实际RED后最小实现，GREEN。WORLD的distinct/alltie/boundary三case各每rank一次Toy Linear，真实保留global/local logits.grad及非零dW，含本地0正例；无公开模型/实图forward。

## Task 2：两身份及存储最小登记

Files：`configs/R204-SCIENCE.json`、`R205-SCIENCE.json`、`ROUND-STORAGE-POLICY.json`、`runtime/bundle/retrain/experiment91.py`、`tests/test_candidate_binding.py`、`tests/test_round_storage_policy.py`。config/data/engine/distributed_engine/model/EMA/保存/推理/源加载不改。

- [x] Pod内从R203/R107生成SCI；候选/存储目标先RED，再仅登记对应profile/来源/固定loss及LR/seed/互斥，GREEN。R204除seed、R205除loss之外job逐项相等；原9:1/结构/日程/早停/source保持，R205官方门明确采用现R120。存储原guard做size/free fixture，旧16/32/default保持，超cap/free15拒；不实际写大文件。

## Task 3：实际fresh与源码闭合

- [x] 只编译受影响源码；两actual public fresh核R204原165全训/5空组/原seed27 tensor身份和非持久RoPE、R205原345/5空组/whole50c6，原LR/default betas/eps/WD保持，0图像forward。原模型/P8/EMA/cold及gather/helper函数按相同SHA复用，汇总明确新测与复用，不重跑全套。

## Task 4：Git冻结与既有串行接续

- [x] 精简真实回执入records，Git7ead152同步/实际API回读commit/tree后两technical包，真实生成名/配置验证。
- [ ] 正常CPU/精确部署在前轮可行收尾前完成；实际safe后唯一owner沿原入口一次activation/direct launch，以真实八rank初态及首自然成功更新确认GPU链。不追加GPU预演，不把准备状态当训练或收益。
