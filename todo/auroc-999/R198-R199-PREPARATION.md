# R197 之后固定 AdamW 参数收缩对照计划

11-03接续已实际完成：[R197完整selected](../../records/R197/R197-RESULT.json)严格R107门true/R120门false，R198不eligible，仅技术包。R199正常CPU18/18、精确部署在R197训练期间一次通过，actualsafe后一次activation/direct launch；[真实八rank与首自然25](../../records/R199/README.md)已证345全训/5空Adam/whole50c6、有限loss/WORLD8/G64P8/448及0skip。当前82冻结不改，无完整R199指标或官方结果；本段覆盖下文早期尚未normal/GPU状态。

使用 brainstorming / writing-plans / TDD / executing-plans，32e3本人继续既有20%准备池；用户已授权最小修改，无审批或新Agent。当前R195/693与先行R196/R197/436保持冻结，本组仅更后续准备。

目标：复用原生AdamW及既有训练/保存/评估/启动链，检验较强参数收缩这一固定假设。备选是继续改公开架构、延伸日程或改变参数收缩；前两类已有独立实验，选第三项且仅改原有weight_decay字段。现有R*-SCIENCE未查到.05/.01/.1/.0005值，不将其视为任意历史任务均未试过的证明。作者[main.py](https://github.com/facebookresearch/ConvNeXt/blob/main/main.py)通用默认weight_decay=.05；[TRAINING.md](https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md)ImageNet微调示例明确1e-8。本实验独立固定.05，其他超参数仍本任务R120原合同；不是作者微调配方复现，也没有过拟合或临床机制证明，不扫描衰减值。

R198/order111仅在完整有效R197唯一内部selected严格胜R107 `.9884405731067525`与R120 `.9894773092062407`且actualsafe后，同gamma-reciprocal075-v1、seed20261027独立public fresh复核，否则仅技术准备。R199/order112独立直接R120原Small/public fe6/seed20261029/512/FP32/ImageNet/sigmoid/G64P8/default AdamW、backbone1e-4/head-age1e-3与原24000余弦，只把job.weight_decay从1e-4改为.05，并在candidate身份明确记录.05。原两个backbone/head decay组消费.05，原两个bias/norm no-decay组仍0，独立age组仍原age合同1e-4；参数/分组/顺序/LR/betas/eps/空状态/RNG均不改。不叠gamma、8k衰减、半LR、其他policy或旧checkpoint。

最小永久范围：`configs/R198-SCIENCE.json`/`R199-SCIENCE.json`、`runtime/bundle/retrain/experiment91.py`登记两身份/来源/互斥，且其validate_job原非Lion硬编码WD1e-4仅按新candidate显式WD取值（旧无该字段仍1e-4）；`tools/validation/check_candidate.py`原native组检查以role区分age1e-4、实际decay组job.weight_decay、no-decay0，保留Lion原支路。`tests/test_candidate_binding.py`两新目标及最近旧R196/R197兼容目标。原config、data、engine、model/loss/EMA/WORLD/保存/推理不改，无新依赖/字段/平台/forward或GPU预演，两轮默认12GiB/free16GiB保持。

- [x] Pod内由既有R197/R120生成两SCI；新身份目标先实际RED（尚未登记），已有WD消费与旧身份目标先通过，未改原优化器。原SSH255后读取并逐字节核原任务回执，未重跑测试。身份目标逐项比对新job仅seed或WD变化、公源/原9:1/日程/早停/官方门保持、saved job及非法叠加拒绝。
- [x] 用真实_optimizer与显式Toy参数对象映射核5组、.05/0/age1e-4作用域、原LR/betas/eps/空状态/RNG；所有参数给零梯度，单次原生step与独立`old*(1-lr*expected_wd)`exact。不是生产八rank恢复证明，不复测已验证的AdamW通用算法。
- [x] 最小登记/必要checker条件修复后，两新目标与旧R196/R197身份GREEN，编译3份受影响源码。两actual public fresh核345全训/5空组/真实初态与SCI；原Gamma像素、model/P8/EMA/cold/loss/WORLD按精确未变SHA复用。新整段checker仅编译/静态复核，完整producer在normal一次执行前不称已通过。[实际CPU证据](../../records/preparation/weight-decay-20261103/README.md)含原传输恢复与汇总组顺序纠正，生产实现未因汇总失败改变。
- [x] 真回执与精简CPU汇总已Git `82add932e59e61f8eb3e8a7f0a4b4cfe1dfd454d` / tree `89c22313619ddadd1e24139ef6d3878fa8b871e5`同步/实际远端回读/[两actual technical包](../../records/preparation/weight-decay-20261103/TECHNICAL-PACKAGES.json)，均guard0，无normal/新clock/GPU。首次汇总组顺序预期错误原件保留，只修归档断言。仅在既定R197之后按完整结果与actualsafe选择唯一分支；后备正常CPU/精确部署于前轮可行收尾前完成，接续临界段连续处理，缩短终态发现/准入间隔，不改旧clock/token/grant或跳过必要检查。

准备仍53ab/root0–11、peer12–23保留，与训练24–119互斥；所有Python/配置解析/测试/哈希在逐次核UID/Ready的既有Pod内nice19/隐藏GPU。无官方或逐例标签/权重修改，24k上限/每800评估/p4/EMA1600与唯一内部AUROC选点保持；完整内部selected≥.996才官方一次，raw .999未达。
