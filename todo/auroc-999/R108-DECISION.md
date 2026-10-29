# R108：固定 seed27 的 BCE／Pairwise 配对复核与短实施计划

**状态：科学合同已冻结。** 已文本核 R107 实际内部 freeze16／TRAIN 原件，训练正常完成与 selected 指标一致；这不等于原一次官方已结束或资源已安全退场，后者由 A 实核。本页不设文档批准门。

**Goal:** 补齐两个既定训练 seed 的 BCE／pairwise logistic 配对，检验 R107 的主指标方向能否在 seed20261027 对照中重现，不寻找最好 seed。

**Architecture:** 完整复用 R107 单视图、全局 WORLD 可微 gather 与固定 pairwise loss；相对 R107 只改训练 seed 及自然 fresh 初态控制和身份。直接科学对照是同 seed 的 R105，仅 BCE→pairwise。

**Tech Stack / Spec:** 现有 Small／PyTorch DDP／prepared512／版本化 SCIENCE 与直接训练链；依据本次 R108 授权、[R107 合同](R107-DECISION.md)、[R105 决定](R105-DECISION.md)及现有 `configs/R105-SCIENCE.json`、`configs/R107-SCIENCE.json`。writing-plans 短计划仅在本页；不另建计划、测试套件、Agent 或批准步骤。

## 身份与唯一变化

- 配置：`configs/R108-SCIENCE.json`。
- profile：`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairlog-v1`。
- candidate：`r108-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairlog-9010`；order18。
- round：`r108-small22k-seed20261027-scale085-pairlog-24k-p4-9010-20261029`。
- `job.seed=20261027`，`training_scale=[0.85,1.0]`；`training_views` 保持缺省单视图，不复制 R106 的2。
- 对 R105 唯一科学变化为以下固定六字段 `job.loss`；对 R107 只换 seed 及派生初态/身份，不改 loss。

```json
{
  "version": "binary-pairwise-logistic-v1",
  "kind": "binary_pairwise_logistic",
  "temperature": 1.0,
  "reduction": "mean",
  "dtype": "float32",
  "scope": "global_batch"
}
```

保持 `mean_{p,n} softplus(z_n-z_p)`；正式8rank、每rank8主图/8视图、accumulation1，全局64图/8正56负/448对，一次 forward/backward/optimizer/LR/EMA。默认 WORLD 可微 logits gather、nongrad labels gather及既有 DDP 归一化原样复用，不额外乘除 world size，局部统计不重复计全局 batch。不新增 margin、BCE 混合、hard mining、配对库或辅助优化器。

全部其余科学合同继承 R105/R107：公开 fresh `convnext_small`、固定 revision `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b` 和既有已验 Small 资产；512/原缓存/FP32、SD0.1/dropout0.2/LN1e-6、backbone LR1e-4、head/age 各1e-3、原 Adam/clip/weight_decay/age 统计/增强；固定9:1、splitseed20261024、原 manifest/FREEZE、G64P8、mixup关闭。训练 seed 改动不改变数据划分。

每800成功更新验证，最多24000；每完整周期 max(online,EMA) 内部 dev AUROC 严格刷新才重置，首轮 baseline、patience4/min_delta0/min_cycles1；warmup与EMA warmcopy绝对1600、首blend1601、cosine24000、原唯一 selector与单视图验证均保持。`p4` 是早停耐心，不改变采样 P8。

seed27 既有公开 fresh 工厂整体控制为 `91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67`。新 Git 正式原生 CPU 包一次实匹配、空 Adam；不加载 R105/R102 checkpoint，不另做 control-only。工程 train21600秒/total28800秒/GPU grant32400秒保持；准备20%与训练80%使用实时核验有效的互斥 CPU 池，不申请或延长实例资源。

## 有限评判与配对判据

| 固定训练 seed | BCE 基线：原 selector 所选点 | Pairwise：原 selector 所选点 |
| --- | --- | --- |
| 20261029 | R104 c3 EMA@2400；AUROC .9850823187728706，AP .8116753198258198，error_twice24778 | R107 **原件已核**：c4 EMA@3200；AUROC .9884405731067525，AP .8274326888154083，error_twice19200 |
| 20261027 | R105 c3 **online**@2400；AUROC .987161811506687，AP .8058457289282671，error_twice21324 | R108 待执行，不预填结果 |

R107 原件为6400次尝试=成功更新、skip0、8周期/bad4正常早停，TRAIN组件 completed/exit0、hard_deadline_met=true、signals为空；freeze含c1–8各online/EMA完整16条，c5–8均未超过c4 EMA。按预定 selected-policy 对 R104 的整体 AUROC 增量约 +.0033582543，AP约 +.015757369，支持本 seed、本选择政策下的有限改善。不能用相同 c4 行约+.0038替代主比较；c2两种variant的AUROC均为 .979230358908164，低于R104 c2的 .9827126362597548，因此不能称全部周期普遍抬升。不同停步与选点周期仍如实保留，不把它当等步数模型比较。

该改善不证明排序目标机制、显著性、稳定泛化或校准；同一固定 dev 经历多轮自适应选择，单 seed 内部新高仍有选择偏差。不同 seed 的 R105 只用于 seed27 的预定直接对照，不能用跨 seed 峰值判断方法优劣。固定难组分解未到，不声称共同难例已解决；其到来只影响解释，不作执行批准门、不改变整体选点。448对共享64样本，不是448独立样本；logit共同平移不受监督、异常标签及梯度尺度改变等 R107 反例继续成立。

R108 主判据是相对 **R105** 的完整内部 dev、原 selector 所选 AUROC。另一 seed 若未提高，则如实记两配对方向不一致；若提高，也只记两个预先固定 seed 同方向，不能由两次估计显著性或挑较好 seed 报告。AP独立分报，原固定20/其余正例错序贡献仅辅助，不能用 AP/难组挽救未达到的主目标。无论结果如何，不搜索新 seed、温度、混合权重或尺度，不改 selector/早停救结果。

本次仅文本消费 R107 内部两原件，远程根为 `exp-v3/rounds/r107-small22k-seed20261029-scale085-pairlog-24k-p4-9010-20261029/`：`SINGLE-MODEL-FREEZE.json` 原Pod SHA `ccf40d8eaa4b134c02c67651988a7c584179edb8576d5d88893776b7c9b0f435`；`evidence/TRAIN_RESULT.json` 原Pod SHA `b84ab2c676dad4ed76f54796e643d47ba9b6b108a935dabebdead9138a7c93be`。本机原样导回位于 `work/exp-v3-8card-review-20261028/r108-update/r107-terminal-export/`，没有本机解析、重哈希或新增计算。官方5849仅原流程一次最终分报，不用于本决定；A另实核完整工程终态与安全交接，不从训练完成推定 safe。

## 最短实施计划

- [ ] **A 最小配置与身份接线。** 新增 `configs/R108-SCIENCE.json`，更新 `runtime/bundle/retrain/experiment91.py` 的当前 profile/candidate及 `tests/test_candidate_binding.py` 既有配置 fixture；沿现有配置入口消费 seed27、91507控制、固定六字段 loss 和 round 身份。`classification_loss.py`、data、engine/gather 路径完全复用，不改数学、增强或分布式语义。
- [ ] **A 一次必要配置窄验。** 在实时有效准备分配内，沿既有 runner 核本轮 SCIENCE/身份/配置生成；不重 Gloo/math/像素/来源测试，不新增 control-only、独立 GPU 预演、旧全套或平台。B 无本轮新增实现文件。
- [ ] **A 唯一同步与接续。** 必要配置验收后一次 Git 同步/实读实际 tree；新正式原生 CPU 包自然匹配91507初态与空Adam，按实际清单冻结部署。A实核 R107 原一次官方自然结束、完整工程有效及进程/锁/设备安全交接后，唯一 activation/fresh launch；真实训练首更新确认64图、8P56N448对与新seed初态。无需文档批准或官方数值。

本次规划仅新增本页；不改 R107 冻结包、旧决定、分析 README 或过程状态，无 Pod/测试/Git 写入。本页核实后冻结交 A 一次必要集成，不追逐过程润色；R108 尚不标为已执行或通过。
