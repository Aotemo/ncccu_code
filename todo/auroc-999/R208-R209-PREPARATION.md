# R208/R209 原生 AMSGrad Implementation Plan

> 已启用 brainstorming / writing-plans / executing-plans / TDD，由32e3本人在20%准备池实施。用户授权最小修改无需审批；唯一Git/GPU，不恢复或派生Agent，独立technical-prep快照验证，不新建运行链。

**Goal:** 为R206/R207之后准备条件seed27复核及一条独立的原生AdamW AMSGrad假设。

**Architecture:** R209直接R107 Small/publicfe6/seed29/RGB512/原logistic，原五组仅通过`torch.optim.AdamW(groups,amsgrad=True)`开启原生AMSGrad。beta/eps、组序/参数/LR/WD、loss、clip、采样、模型/forward、24k/800/p4/EMA保持。无自制优化器、额外loss状态或训练阶段。

**Tech Stack:** 当前已核Pod/Python3.12.7/PyTorch2.10+DTK，ec5/root0–11/nice19/隐藏GPU，与训练24–119互斥，无新依赖。

**Spec:** 本页同时封存设计。备选新架构需BN/来源适配，继续调loss已有R205/R207；本次只选原生二阶矩历史最大值策略，区别于R168固定beta2=.95，不扫描beta/LR/eps或叠加。当前configs/任务/历史汇总未找到amsgrad=True，不声称所有历史工作从未使用。

[PyTorch2.10 AdamW原公式](https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html)对**未偏差校正的**v取逐元素历史max，随后除`1-beta2^t`。保留raw v的历史峰值，不宣称校正后的分母或有效LR必单调，不把收敛理论转述为本非凸网络/AUROC保证。OpenReview原论文页面与PDF均遇到公开验证页，本次算法消费以已读官方2.10原生文档和实际Pod实现为准。可能使罕见大梯度后的更新过小；没有梯度失稳、临床机制或改善证明。

## Global Constraints

单模型/固定9:1/合规公开fresh，最多24000成功更新、每800内部评估、p4/min_cycles1/EMA1600，完整内部AUROC唯一选点；selected>=.996才完整官方5849一次，raw .999未达。当前R203/573、既定R204/R205/7ead及R206/R207/78冻结不改。

R208/order121仅完整有效R207严格胜R107 `.9884405731067525`和R120 `.9894773092062407`并actualsafe后，同逐pair平方hinge/margin1/quarter、seed20261027独立公开fresh复核，否则仅技术准备。round `r208-small22k-seed27-pairsqhinge-m1q025-24k-p4-9010-20261103`。

R209/order122独立直接R107，仅新增字符串`adamw_amsgrad_policy='adamw-amsgrad-v1'`，同时写入SCI/candidate/job/保存身份，None或省略均不启用且从job_contract省略。原R107无当前显式官方字段，本轮沿R120 >=.996门，历史文件不改。round `r209-small22k-seed29-pairlog-amsgrad-24k-p4-9010-20261103`。不叠beta095、Lion、hinge、hard-negative、强WD/低LR等。默认12GiB/free16GiB不改；新max_exp_avg_sq增加一份同参数大小FP32状态，实际参数计数与9C+I情景须CPU计算，不能把它当磁盘/设备峰值上界。

## Review Focus

1. 原生五组均amsgrad=True，原age组仍LR1e-3/WD1e-4；不误走Lion年龄/组倍率或自制更新。
2. 独立FP64 oracle明确先raw max再bias correction，首大梯度后两小/零梯度可区分AMSGrad与普通AdamW；检查m/v/vmax/参数/步数。
3. 原生state保存冷载后的下一步exact，包括max_exp_avg_sq及amsgrad组配置；Toy回执不冒生产八rank断点恢复实测。
4. None/省略原参数/RNG/组/逐步更新和身份保持，新policy非法值/叠加拒绝；旧beta095与两先行候选兼容。
5. 两actual public fresh仍345全训/5空组/原whole；参数状态情景在既有12GiB内才继续。原模型/P8/EMA/冷推理/保存/gather按未变SHA复用，不追加GPU预演。

## Task 1：原生优化器及配置

Files：`tests/test_adamw_amsgrad_policy.py`、`runtime/bundle/retrain/config.py`、`runtime/bundle/retrain/engine.py`。

- [x] 五组独立oracle及native save/load下一步、None/省略/native默认和RNG/身份检查，实际2 RED后2 GREEN，guardxvs5o8k0/8d8be30b exit0，1.573秒。
- [x] policy由load_config保留/省略None、job_contract绑定；固定Small/logistic/G64P8，拒绝beta095/Lion等叠加。6源码compile、原beta095/默认两目标通过，guardyl7q0bq8/f5e20287 exit0。

## Task 2：候选与既有消费者

Files：`configs/R208-SCIENCE.json`、`configs/R209-SCIENCE.json`、`runtime/bundle/retrain/experiment91.py`、`tools/validation/check_candidate.py`、`tests/test_candidate_binding.py`。

- [x] Pod由R207/R107生成SCI；候选实际RED后新/旧两目标GREEN，guardmo0qncoi/861912a5 exit0，9.736秒。R208仅seed、R209仅新policy及明确官方字段，保存job绑定和实际五组amsgrad消费检查。
- [x] 两actual public fresh/345全训/5空组/0图像forward及状态情景通过；[CPU汇总](../../records/preparation/amsgrad-20261103/CPU-CHECKS.json)862ad766，最终guards8jhvna9/fb7f5ebf exit0，原件/SHA已核。R209主要float 9C+I+原子新C为10,286,735,472B，非实际峰值/序列化上界；默认12GiB保持。当前/先行冻结不改。

## Task 3：同步与原链接续

- [x] 精简真实回执入records，Gitd6fe92f实际push/commit-tree回读后原入口两technical包，实际run_name100/97、默认12GiB，两个guard和归档均0。normal尚待前轮可行收尾完成，无clock/GPU。
- [x] R207完整6400/c8/bad4、唯一EMA3200内部.9865633703435679，两门false/官方未跑/同次actualsafe，R208不eligible。[独立R209](../../records/R209/README.md)一次activation/direct launch0，真实八rank/345全训/各rank空state及自然首25有限已证，08:09实观325/0skip/eval0。AMSGrad逐组kwargs证据来自正式CPU/source绑定，不冒充八rank逐组遥测；旧safe/clock/token/grant保持，无GPU预演。

## Ledger

- 05:52 UTC仅封存设计，尚未实现/验证/新SCI或训练。当前R203最近4600/0skip/eval5/bad0，暂EMA `.9812761366468752`，八rank活跃；R204/R205和R206/R207均已各自冻结技术准备，不因本假设延后接续。
- 06:02 UTC必要CPU准备已完成，6目标/6compile/两actual public fresh与final guard均0，源/SHA/原始OUTPUT已核；原件见[records](../../records/preparation/amsgrad-20261103/README.md)。私有原prepare只四处身份/证据增补，bc141803/guardhzbnpv8x0，原8985及storage消费段保持。尚待Git实际同步/回读和两technical包，无normal/clock/GPU。当前R203最近6075/0skip/eval7/bad1，暂EMA `.9816433892721294`，原573与先行7ead/78不改。
- 06:07 UTC两个technical包及归档实际完成，6源/两SCI/证据保持，归档swxjqf69/8cac0996 exit0；无normal/clock/GPU。R203最近6800/0skip/eval8/bad2、八rank活跃，原573不改；接下来优先R203完整收尾及7ead既定分支接续，不让本准备成为等待门。
- 07:56 UTC R207实观5650/0skip/eval7/bad3且八rank活跃，后备[R209 normal19/19及精确部署](../../records/preparation/amsgrad-20261103/NORMAL-R209.json)已在训练期间完成；guard_kps1z7h/fb08f308 exit0、actual SITE37c08d4e、345全训/5空Adam/whole50c6及正式CPU逐组AMSGrad配置相符，原6目标6compile按SHA复用。无clock/activation/GPU，R208仍仅technical；完整双门/actualsafe后一次原入口接续。07:49实际只读核peer idle/activeTurn null、32e3唯一Git/GPU，无消息/恢复/派生；非Lion八rank观测只读取空state/自然更新，逐组AMSGrad kwargs不冒充八rank遥测。
