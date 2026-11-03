# R206/R207 固定逐配对平方 hinge Implementation Plan

> 使用已启用的 brainstorming / writing-plans / executing-plans / TDD，由32e3本人实施。用户授权最小修改直接执行、唯一Git/GPU且不恢复或派生Agent；复用独立technical-prep源码快照隔离验证，不新增审批、平台或GPU预演。

**Goal:** 在R204/R205之后准备条件seed27复核及独立的逐配对有限间隔目标，持续使用现有DCU。

**Architecture:** 直接复用classification_loss的完整WORLD 8×56差值矩阵，新增固定`mean(relu(1+n-p)^2)/4`。无参数、状态、额外forward、历史分数或dev名单；原gather/DDP、AdamW、保存/评估/推理不改。

**Tech Stack:** 当前已实核Pod/Python3.12.7/PyTorch2.10+DTK；ec5准备root0–11/nice19/隐藏GPU，与训练24–119互斥，无新依赖。

**Spec:** 本页同时封存设计。备选为新增公开架构、改变优化器或逐pair目标；EfficientNet当前BN/EMA消费者需多处适配，暂不采纳；优化器已有Lion/AdamW beta2实验。本次选择逐pair平方hinge，保留原Small/公开fe6/seed29/RGB512/FP32/ImageNet/G64P8/448对/原AdamW/LR/WD/24k余弦/EMA。R160类内方差+均值间隔目标即使所有pair过间隔仍有方差梯度；本目标每个`p-n>=1`的pair损失与梯度均为0，不是R160重跑。固定1/4使d=n-p=0处导数为0.5，与原softplus同点相等；不声称梯度范围、clip、优化轨迹相同，不扫描margin/系数。

[Mao et al. ICML2023附录L.1.5](https://proceedings.mlr.press/v202/mao23a/mao23a.pdf)给出pairwise squared-hinge形式；这里仅以其函数为参考，1/4是本固定归一化，不把线性假设类理论转述为本神经模型的AUROC或临床保证。错序尾部梯度线性增大，可能放大离群/含噪pair；没有错标、过拟合或临床机制证明。现configs/todo/历史汇总未找到同逐pair配方，不宣称所有历史任务从未试过。

## Global Constraints

单模型/固定9:1/合规公开fresh/官方隔离；最多24000成功更新、每800内部评估、p4/min_cycles1/EMA warmcopy1600、完整内部AUROC唯一选点；selected>=.996才官方5849一次。raw .999未达。当前R203/573及既定R204/R205/7ead冻结不改。

R206/order119仅完整有效R205严格胜R107 `.9884405731067525`和R120 `.9894773092062407`并actualsafe后，以同负例top14/mix05配方seed20261027独立公开fresh复核，否则仅技术准备。round `r206-small22k-seed27-pairlog-neg14mix05-24k-p4-9010-20261103`。

R207/order120直接R107原科学job，仅loss为六字段独立`binary-pairwise-squared-hinge-margin1-quarter-v1`/kind `binary_pairwise_squared_hinge`，temperature1/reduction mean/dtype float32/scope global_batch。原R107没有当前显式官方门，本轮明确沿现R120的>=.996门，历史文件不改。round `r207-small22k-seed29-pairsqhinge-m1q025-24k-p4-9010-20261103`。不叠hard-negative、wide2、margin-primal、WD.05、gamma或低LR，默认12GiB/free16GiB，无新存储例外。

## Review Focus

1. 原8×56全部pair，逐pair取正部后平方；不能写成先均值再hinge或R160方差形式。
2. 每pair过间隔/恰边界时梯度0，混合正负间隔及极端FP32有限值；不将无效binary/missing-class输入当通过。
3. 新logit梯度对独立Python双循环解析值；真实2rank Gloo本地0正例、global/local梯度与DDP参数梯度，不另乘除world。
4. 旧None/logistic/sigmoid与既有分支源码保持；SCI/candidate/job_contract/保存loss身份拒绝错值、来源、seed、原LR或非法叠加。
5. 两actual public fresh/345全训/5空Adam及短run_name；原模型/P8/EMA/cold链按不变SHA复用，CPU通过不当GPU吞吐/峰值证明。

## Task 1：数学目标及WORLD

Files：新增`tests/test_pairwise_squared_hinge_loss.py`；最小修改`runtime/bundle/retrain/classification_loss.py`固定版本/kind/单分支。

- [x] 独立8×56双循环value/逐logit梯度、shift/shared-bias、混合/全过/边界/全同分及有限极端，旧None/logistic/sigmoid兼容；实际RED lwpn5gco/5818a920后最小GREEN。
- [x] 真实2rank Gloo逐global/local logit和DDP参数梯度，包含本地0正例，不新模型或实图forward。3目标guard1pi0tzpl/1e28588d exit0，8.297秒；编译尚待下一步。

## Task 2：候选最小登记

Files：`configs/R206-SCIENCE.json`、`configs/R207-SCIENCE.json`、`runtime/bundle/retrain/experiment91.py`、`tests/test_candidate_binding.py`。其余runtime与存储策略不改。

- [x] Pod由R205/R107生成两SCI草案，候选实际RED blh632qd/aff2df33后最小登记/身份互斥GREEN；逐job/split/structure/schedule/earlystop及原source相等，官方字段差异明确。新候选及受影响旧R204/R205共2目标guard16llskka/abb5957f exit0，10.474秒。
- [x] 两actual public fresh与5空组，0图像forward；4源码compile通过。CPU汇总47996632、最终guardkksy1dne/b8b655da exit0，R206 whole91507a65/SCI11e6f001、R207 whole50c6/SCIa146cfe2，均345全训/5空Adam。原model/WORLD/EMA/cold/optimizer按不变SHA复用，未重跑旧目标或GPU预演。

## Task 3：同步与原链接续

- [x] 精简真实回执入records，Git78e80aa实际push/commit-tree回读，两个technical包guard均0，实际run_name均100/default12GiB；normal尚待前轮可行收尾完成。源/SCI与5目标4compile/两fresh证据逐项一致。
- [ ] 完整双门及actualsafe决定唯一分支；原入口一次activation/direct launch，真实八rank初态及自然首25证实可运行。不重复旧safe/官方或改现役，不为文档空等。

## Ledger

- 05:32 UTC设计已固定，尚未实现/测试/配置身份/训练。Ruling：暂不引入EfficientNet BN适配，选择现有loss矩阵内单分支；这是本次范围决定，不是该架构的科学阴性。
- 05:35 UTC已5目标GREEN，尚待compile/两actual fresh/最终归档/Git/technical包。首次新测试传输早于源码目录完成而失败，确认后顺序补传；无源码或训练后果。R203第9次观测请求早于16llskka guard完全退场，被既有owner flock拒绝，未执行观察；该次为unknown，准备无并发越池。后续等待guard结束再派发，未加流程或改互斥实现。
- 05:40 UTC受影响CPU准备已完成，真实原件归入[records](../../records/preparation/squared-hinge-20261103/README.md)，无formal/clock/GPU。私有既有prepare helper仅四处身份/证据增补并compile，SHA89857348；原8a172及storage段保持。当前R203实观3050/0skip/eval3/bad0，暂EMA `.9771051101095617`，八rank活跃；后续R204/R205仍原7ead两technical包。
- 05:43 UTC两technical包已实际完成、归档guardk9rxoipf/99b98596 exit0，未新normal/clock/GPU。05:42 Paseo只读实核仅32e3 running、peer idle/activeTurn null，无消息/恢复/派生。当前R203最近3525/0skip/eval4/bad0，暂EMA `.9798155548946346`，仍沿原队列接续。
- 07:00 UTC R205实观4850/0skip/eval6/bad3、八rank活跃，后备[R207 normal18/18及精确部署](../../records/preparation/squared-hinge-20261103/NORMAL-R207.json)已在训练期间完成；guard6to2lglz/e3cb8c8d exit0，actual SITE1718c7f3、345全训/5空Adam/whole50c6、原5目标4compile按精确SHA复用。未新clock/activation/GPU，R206仍仅technical；完整双门/actualsafe后一次原入口接续。06:52实际只读核peer idle/activeTurn null、32e3唯一Git/GPU，无消息/恢复/派生。
