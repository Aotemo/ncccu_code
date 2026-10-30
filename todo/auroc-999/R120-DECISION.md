# R120：有界对称成对排序目标

**最终状态：完整工程有效，selected内部AUROC严格胜预冻R107；按原规则接固定seed27的R121复核，尚不升级方法基线。** 官方未达.996按政策未运行，A实际回执确认整轮safe；详见末尾。下文实施清单保留原预登记，不是当前待办。2026-10-31。

| 身份 | 唯一值 |
|---|---|
| order | 30 |
| 配置（A独占） | `configs/R120-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairsigmoid-v1` |
| candidate | `r120-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairsigmoid-9010` |
| round | `r120-small22k-seed20261029-scale085-pairsigmoid-24k-p4-9010-20261031` |

## 单一问题与数学合同

直接匹配R107 seed20261029，仅把纯pairwise logistic换为有界对称sigmoid目标。对完整全局批次的8正56负、448对，令`d=z_negative-z_positive`，采用`L=2.*mean(sigmoid(d))`。所有对等权，不删样本、不hard mining、不混BCE、不加margin、teacher、配对库或辅助优化器。

`job.loss`精确六字段如下；乘2锁在版本定义内，不增加可调系数：

```json
{"version":"binary-pairwise-sigmoid-v1","kind":"binary_pairwise_sigmoid","temperature":1.0,"reduction":"mean","dtype":"float32","scope":"global_batch"}
```

乘2使零差值处导数为.5，与softplus匹配；不等于全程梯度尺度、Adam或clip轨迹一致。旧logistic对差值的导数也有界，严重错序时趋近1，不能称其梯度爆炸。新导数`2*sigmoid(d)*(1-sigmoid(d))`在两端趋零，检验尾部影响抑制是否改善整体排序。FP32饱和可产生精确零梯度，真实难正例也可能被忽略，非凸优化可能得到更差解；finite或训练loss降低均不是收益证据。

已有dev错误集中不证明fit极端对主导梯度，更不证明错标；不按dev组设权、删改标签或回灌训练。R100 focal增强难样本BCE、R21的BCE+.25pairwise及R107的纯logistic均非本目标；已查历史未见本有界对称版本，不保证未归档历史全无。一般理论依据为[ICML2019对称损失原论文](https://proceedings.mlr.press/v97/charoenphakdee19a.html)，其噪声分布假设未在本项目证实，不套成本数据或医学收益结论。共同logit平移不受监督，校准与AP不作保证。

## 保持的实验与执行语义

其余全部沿R107：已验Small公开ImageNet22K→1K来源及固定revision、公开fresh/seed20261029、stride2/全参数可训、SD.1/dropout.2、单图prepared512与原缓存、scale[.85,1]及原增强、ImageNet归一化、FP32/noTF32、原fit-only age、固定9:1与清单、G64/P8/8rank各8/accum1、原LR/WD/clip1/AdamW/EMA、800成功更新一次内部评估、24000上限、patience4、绝对warmup/EMA1600。64主图64视图，一次forward/backward/optimizer/LR/EMA推进。

不启用R119输入扩展、R118 R50/BCE、SSL、freezeearly或stride1。预期完整公开fresh state仍为`50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，须正式CPU实际匹配，同时记录新loss非tensor身份、实际参数组及空Adam；不加载R107 checkpoint、不另control-only。

复用原`gather_pairwise_inputs`本体：默认全局WORLD可微logit gather、普通label gather，每rank同算完整mean；Gather backward SUM与DDP平均抵消，不乘除world_size。多数rank无正例，不得落入rank-local loss。单卡旧入口仍拒绝本次pairwise，保留正式8rank/accum1及无mixup的限制；日志保留64/8P56N/448对与实际global loss。

## 最短实施与必要证据（原预登记）

- [ ] **B独占** `runtime/bundle/retrain/classification_loss.py`与`tests/test_pairwise_sigmoid_loss.py`：新增显式版本，提供`PAIRWISE_LOSS_KINDS` tuple给A识别；原None/focal/logistic及gather本体不改。
- [ ] **A独占其余接线**：config/engine/distributed/experiment91/SCIENCE和生产binding统一消费两种已知pairwise种类，覆盖全局路由、限制及日志。R119未验代码由A精确保留patch并从后续生产路径隔离，不盲reset；目前不能声称隔离已完成。
- [ ] **仅三项新目标窄验及受影响绑定**：独立解析值/梯度、对称性与零点斜率；极值finite/平移/存在的零bias梯度及旧缺省；复用真实2进程Gloo结构，零正例rank、完整64=8P56N/448对，新loss共享参数梯度对完整batch参考。只跑受影响新项，不重复旧套件或独立GPU预演。CPU通过不冒充真实NCCL首更新。
- [ ] **A唯一正式链**：必要窄验→一次Git同步与精确冻结→正式原生包/实际初态→实时身份、资源及前驱safe→唯一fresh首成功。动态有效准备20%与训练80%互斥；不加文档或再次确认门，不热改旧包。

无新公开资产、缓存、模型参数或图像前向次数。按旧R107单图日志估满24k及原30次内部评估/保存约3–3.5小时，仅规划数量级，非新benchmark、不依赖早停。保持训练6h/整轮8h/grant9h、默认Small round12GiB、健康9C+I/启动free20GiB/运行floor16GiB及完整保存保护/300秒/atomic/prune/failfast；不增资源。最终单模型约198MB只是既有体量估计，实际≤500MB及指定硬件端到端100ms仍未由本轮证明。

## 预定判据与接续

先一次fresh；完整工程有效且原selector所选内部AUROC严格胜R107 **.9884405731067525**，才另冻既定seed20261027，对照R108 **.9872328538178018**。首跑未胜关闭本固定surrogate，不扫温度、scale、混合或seed；AP/固定难组/官方不救主判。两seed方向同向也不宣称显著或已消除dev自适应偏差。

仅完整训练及唯一checkpoint冻结后，selected内部AUROC≥.996才完整official5849一次；否则`not_run_by_policy/threshold_not_met`、metrics=null，不当失败。该门独立于seed27复核，中途partial不触发；不授权平台test产物或提交。总体raw AUROC .999目标保持。阶段结束由规划依据完整内部结果主动提出下一独立问题，不把分支停止当项目无限期暂停，也不在本页预占下一候选。

## 完整结果与预定复核触发

2026-10-31 07:03:05.983327北京A实际终态：completed/early_stopping，8000success=attempt/24000、skip0、eval10、bad4；完整freeze20条唯一selected为**EMA c6@4800，AUROC .9894773092062407、AP .8313831149805229、error_twice17478**，tensor SHA `f21bffae88acbf7c216c4141c0ef0b9d990a624e3e92a1baf750a501c415fc85`。已一次文本核`work/r120-20261031-a/progress-0700/round/SINGLE-MODEL-FREEZE.json`的selected与`round/evidence/TRAIN_RESULT.json`；TRAIN completed/exit0/hard_deadline_met、signals[]。

按A实际回执、由root转交：official=`not_run_by_policy/threshold_not_met`、evaluated=false、metrics=null，ROUND/done completed；owner247028Z不称reaped、原8rank退出、lockholders[]、deviceusers[]、8idle、scanDenied0，safe=true。规划未另读官方或查询资源。progress原Pod SHA `726d0ebd8a7ba25563fc838668bb12107a172d851d8990732274297974199937`，CPU guard SHA `d44b3ef52a7ff98419603c0ef49b9dc1290b13932100fa2d241e86f8571b4166`，仅记录交接值，未本机重算。

本次完整selected严格胜R107 **.9884405731067525**，成为当前固定9:1的单次内部新高，满足原定固定seed27复核条件；root已采用[R121](R121-DECISION.md)。比较阈值仍预冻R107/R108，不按本次峰值或partial移动。尚不证明两seed稳健、显著性、机制或.999可达，也未消除同dev反复适应偏差；不依AP或官方补救主判。R120原包、clock/token保持，selected checkpoint保护真实依赖；必要tokenfree结果归档由A统一处理，不重复训练/官方。本文至此收口停写。
