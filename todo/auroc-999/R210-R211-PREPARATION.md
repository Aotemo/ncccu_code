# R210/R211 原生 NAdam 解耦衰减 Implementation Plan

> 已启用 brainstorming / writing-plans / executing-plans / TDD，由32e3本人实施。用户已明确最小修改无需审批、唯一Git/GPU、不恢复或派生Agent；复用独立technical-prep快照及现有运行链，不追加审批、全套重复验证或GPU预演。

**Goal:** 为R209之后准备固定seed27条件复核，以及一个独立原生NAdam对照，持续使用现有DCU。

**Architecture:** 直接复用`engine._optimizer`原五组，仅新增`torch.optim.NAdam(groups, decoupled_weight_decay=True)`分支。无模型、数据、loss/WORLD、EMA、保存或推理算法改动；新可选`nadam_policy`进入配置及保存job身份，None/省略保持原身份和AdamW行为。

**Tech Stack:** 当前已核Pod/Python3.12.7/Torch2.10+DTK；ec5准备root0–11/nice19/隐藏GPU，与训练24–119互斥，无新依赖或计算资源。

**Spec:** 本页同时封存设计。备选为新公开架构、再次调整读出/配额/SSL、原生优化器。已有Swin/DeiT/DINO/CAFormer及固定读出/配额/SSL历史不重开；新BN架构需更宽适配，暂缓。本次选择原生NAdam以控制改动范围，不把R209正在运行的AMSGrad先判阴性，也不宣称Nesterov会解决临床机制或达到.999。仓库configs/todo/runtime/tests文字检索未见NAdam/Nesterov候选，非所有历史资产的穷尽证明。

[PyTorch2.10 NAdam](https://docs.pytorch.org/docs/2.10/generated/torch.optim.NAdam.html)支持`decoupled_weight_decay=True`，其余原生默认betas(.9,.999)、eps1e-8、momentum_decay.004、foreach=None、maximize/capturable/differentiable=False。全部五组保留原显式LR/initial_lr/WD及参数顺序，age仍lr1e-3/wd1e-4；NAdam构造器默认lr不替代组级LR。只固定这一组合，不扫momentum_decay/beta/LR/WD/seed。

## Global Constraints

单模型、原固定9:1、合规自然图像public fresh、官方隔离，最多24000成功更新/每800内部dev评估/patience4/min_cycles1/EMA1600，以完整内部AUROC选择唯一checkpoint。完整selected>=.996才官方5849一次；raw .999未达。当前R209/d6冻结不改，后续只在完整终态/actualsafe后串行原入口。

R210/order123：仅完整有效R209所选严格胜R107 `.9884405731067525`及R120 `.9894773092062407`并actualsafe后，原AMSGrad配方独立seed20261027复核，否则仅技术准备。round `r210-small22k-seed27-pairlog-amsgrad-24k-p4-9010-20261103`。

R211/order124：直接R107 Small/fe6/public fresh/seed29/RGB512/FP32/ImageNet/G64P8/448pairs/原logistic/原LR-WD-增强-age-时钟，仅`nadam_policy='nadam-decoupled-native-v1'`。不叠AMSGrad/beta095/Lion/hinge/ROI等policy；原R107没有当前显式官方门，本轮沿现R120的.996门，历史不改。round `r211-small22k-seed29-pairlog-nadam-24k-p4-9010-20261103`，默认12GiB/free16GiB，无新存储例外。

## Review Focus

1. 五组均为原生NAdam且decoupled=True，保持原组成员/LR/WD；不能漏改age或落回AdamW/耦合衰减。
2. 独立FP64公式核Nesterov的mu_t/mu_next、mu累乘、m/v和参数更新；零梯度首步保持m/v为0，能识别误用耦合WD。生产仍FP32，容差明确。
3. 原生Toy状态保存/恢复后的下一步exact，检查step/mu_product/exp_avg/exp_avg_sq，不冒充生产八rank断点恢复。
4. None/省略配置和job_contract完全兼容；非法policy及AMSGrad/beta095/Lion/新loss叠加拒绝，SCI/candidate/validate_job逐项绑定。
5. 两actual public fresh/345全训/5空组、源/SCI/SHA与原模型链不变证据复用；CPU结果不当作DCU峰值/吞吐或AUROC收益。

## Task 1：最小优化器分支

Files：新增`tests/test_nadam_policy.py`；修改`runtime/bundle/retrain/engine.py`与`config.py`。

- [x] 写实际五组/独立公式三步/零梯度/原生状态下一步及None身份测试，在当前源码观察预期RED。
- [x] 仅新增固定policy、配置消费和原生构造分支，GREEN后运行受影响旧AMSGrad/默认目标；不重跑无关模型或全套。

## Task 2：候选与正式消费者

Files：`configs/R210-SCIENCE.json`、`configs/R211-SCIENCE.json`、`runtime/bundle/retrain/experiment91.py`、`tools/validation/check_candidate.py`、`tests/test_candidate_binding.py`。

- [x] Pod由R209/R107生成两SCI，新增候选正向与非法身份测试并观察RED，再做精确两身份/互斥登记。
- [x] 正式消费者已加入5组NAdam参数及空state检查；两actual public fresh、必要编译与最终源码/回执身份核验完成，正式normal尚待。原模型/WORLD/EMA/保存链按不变SHA复用。

## Task 3：同步与原链接续

- [x] 真实精简回执入records；代码/config同步GitHub并实际回读后精确部署两个technical包。
- [ ] 本轮训练期间完成可行后备normal，完整结果决定唯一分支；actualsafe后一次activation/direct launch，以真实八rank初态及自然首25证明能跑。

## Ledger

- 08:13 UTC设计固定，尚未实现/测试/生成SCI/训练。Ruling：沿用户明确授权由本人实施、不派生或追加审批；验证限实际受影响路径，旧已通过且源身份相同证据复用。独立technical-prep源快照用于CPU隔离，主仓唯一Git写入者保持。最终由本人另作完整diff自审，不把它称为独立Agent审查。
- 08:30后实际[8目标/6compile/两public fresh及最终guard](../../records/preparation/nadam-20261103/README.md)已通过；完整diff本人自审未见阻断，旧参数分组/调度/保存为不变通用实现，不追加GPU预演。原source上传过早的目录不存在运输错误与预期RED均如实保留。R209仍八rank训练，无新normal/clock/GPU，待Git/technical。
- 08:34 source158bfb0/tree2e993be9已push并GitHub实际回读，[两technical包](../../records/preparation/nadam-20261103/TECHNICAL-PACKAGES.json)guard0/原storage保持；无新normal/clock/GPU。source archive首次taskset参数错误已记录，正确重传exit0后才绑定/打包。R209完整结果仍未知，不先判条件R210。
