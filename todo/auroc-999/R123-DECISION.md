# R123：无年龄支路的匹配消融

状态：完整内部结果已核，2026-10-31 12:03:28 北京时间 A 实际终态确认安全退出；无年龄合同未胜两道预定门槛，分支关闭、不做 seed27。以下保留原预登记合同，最终证据和解释边界见末段。

## 唯一身份与科学问题

- order：`33`；配置：`configs/R123-SCIENCE.json`（A 生成）。
- profile：`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-noage-v1`。
- candidate：`r123-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-noage-9010`。
- round：`r123-small22k-seed20261029-scale085-pairlog-noage-24k-p4-9010-20261031`。

直接匹配 R107 / seed20261029，检验当前 Small、9:1、pairwise logistic 合同下，去掉年龄残差支路是否改善完整内部 dev 排序。原 logit 为 `f(image)+w·age`；正负年龄差进入 pairwise margin，也影响图像分支梯度的权重。这只是可证伪假设，未证明年龄捷径、标签噪声或当前瓶颈；不是把 R120 已训练权重在推理时置零。

最强已有反例是旧 R8 Tiny/BCE/8:2 年龄臂的点估计优于无年龄臂，且条件区间跨零；不能说年龄从未验证或应当有害。当前合同不同，这次结果也不能改写旧结果。删除年龄梯度会改变全局 clip1 范数和后续优化轨迹，这是干预本身的影响，不能声称梯度轨迹相同。既有 fit 联合输入 metadata 的 C=0 只排除严格联合键冲突，并非选择此消融的收益证据。

## 精确干预与不变量

唯一科学开关为 SCIENCE 顶层可选布尔 `age_enabled=false`，缺省为 `true`。共享 `resolved_age_spec` 从原 AGE 元数据深拷贝，只改变 `enabled`；原 contract、contract SHA、年龄分层固定 split 不变。resolved `job.age` 保留原 contract 且 `enabled=false`，不删除来源记录。candidate readout 为 `native_avgpool_final_LN_dropout_linear`；生产模型无 `age_weight`，optimizer 无 `age` role。

其余完整沿 R107：公开 ConvNeXt Small fe6b ImageNet22K→1K fresh、job seed20261029、全参数训练、stride2、单图 prepared512、training scale `[.85,1]`、原 `None` 光度增强、G64/P8（8 rank 各8主图，8P56N/448对）、FP32/noTF32、dropout .2、SD .1、原 backbone/head LR、WD、AdamW、clip1、EMA、固定9:1及原 selector。损失仍为原六字段：

```text
version=binary-pairwise-logistic-v1
kind=binary_pairwise_logistic
temperature=1.0
reduction=mean
dtype=float32
scope=global_batch
```

每800成功更新验证，最多24000成功更新，patience4，绝对 warmup/EMA1600不变；不启用 pairsigmoid、photoidentity、native local、SSL、冻结早层或新来源。无 N06/FLAIR 参数、teacher 或统计继承。

去掉2个年龄标量后，whole tensor SHA 必须是本轮正式工厂实测的新值，不能沿用 R107 的 `50c…`。公共源严格加载身份保留；共同图像/分类头的初始逐键同一性须有必要实际证据，不能仅靠同 seed 推断。新 whole、可训/optimizer 成员、空 Adam、无年龄非 tensor 绑定以及真实 first batch 的无 age features / 三元素 tuple，由正常正式 CPU 和真实8 rank路径核实；回执到前均不记通过。

## 最短实施与验收

- [ ] A 独占 SCIENCE、config、prepare、experiment91、生产绑定及 `tools/validation/check_candidate.py` 的必要适配；统一消费同一 resolved age spec，保留旧缺省合同、严格 checkpoint 重建及 eval/inference 身份。
- [ ] B 只新增 `tests/test_age_disabled.py` 两项 toy/synthetic fixture：attach/真实 optimizer/RNG；Dataset/unpack/forward/pixel及无年龄消费。A 负责第三项最小绑定 fixture：SCIENCE 缺省 true / 显式 false、保存严格重载与 eval/inference 绑定。不同人不共写文件。
- [ ] A 在当次有效环境与本人准备份额统一运行上述受影响窄验，再唯一必要 Git、精确部署、正式原生包、实时安全准入、一次 activation/fresh，并报告实际首成功。复用已有无年龄执行路径，不扩旧单卡功能，不重跑 age/loss/Gloo/source/cache/pixel 大套件，不另做 control-only 模型、GPU预演或 fit 诊断。

CPU17 是已交的当前分配线索，仍由 A 在使用时实核 UID/Ready/TTL/预算；不能沿用过期 CPU16。若当前有效120核配置仍成立，A准备0..11/12核、两执行准备0..23、训练24..119/96核互斥；实际值服从当次绑定，不把历史 mask 当永久默认。

默认 round 12GiB、启动 free20GiB、运行 floor16GiB、300秒保存/保护/atomic/prune/failfast、训练21600秒/整轮28800秒/grant32400秒均不变，不新建或延长云资源。仍64图单编码，去掉2标量不构成可宣称的加速。满额24000步约3–3.5小时仅据旧 Small 组件记录估算，非本轮 benchmark 或保证，不靠早停救预算；实际交付包≤500MB和指定硬件端到端100ms仍待实证。

## 预冻结判据与停止条件

完整且工程有效的原 selector 唯一 checkpoint，内部 AUROC **严格大于 R107 `.9884405731067525`** 才称本次同 seed 正向；还须**严格大于已记录单次 best R120 `.9894773092062407`**，才由协调方按条件安排固定 seed20261027 对 R108 `.9872328538178018` 的复核。本页不授权自动启动第二 seed。

仅胜 R107、未超 R120：有限正向、无新 best，本阶段结束；未胜 R107：关闭整个无年龄合同。不继续年龄分箱、非线性支路、正则权重或 seed 扫描。AP 独立报告，partial、AP、官方或平台分数均不能挽救主判；两 seed 也不证明显著、消除同 dev 自适应或保证 `.999`。

完整训练及唯一 checkpoint 冻结后，selected 内部 AUROC `>=.996` 才运行一次官方5849；不足则 `not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，整轮可正常 completed 并实际 safe。这与科学正向、单次 best 和第二 seed 门槛分开；不运行平台 test CSV。总体 raw AUROC `.999` 目标不变。

完整结果后由规划/协调方按上述条件收口并决定下一问题，不把关闭本分支当总体暂停；当前不排新候选队列。A 是唯一 Git/GPU 执行者，本文冻结后不追逐过程状态改写。

## 完整结果与分支收口（2026-10-31）

本次仅文本核对 `work/r123-20261031-a/progress-1200/SINGLE-MODEL-FREEZE.json` 的唯一 selected 与 `evidence/TRAIN_RESULT.json`：7200成功更新、9周期，early_stopping/bad4，TRAIN completed/exit0、hard_deadline_met=true、signals空、live_owned_ranks空。唯一 selected 为 **EMA c5@4000，内部 AUROC `.9867584356723914`、AP `.8255658133774761`、error_twice `21994`**；tensor `bc86edde5547c8069a2117b176fe4f172cf93ccc1b66a58986b5943462b7e7b3`。7200attempt=success、skip0/eval9以及整轮安全事实引用协调方转交 A 的实际12:03:28回执，不另查 Pod 或读取官方结果。

所选 AUROC 低于直接 R107 `.9884405731067525`，也低于单次 best R120 `.9894773092062407`。按原判据关闭整个无年龄合同，**no seed27**；不继续年龄分箱、非线性、正则权重或 seed 扫描。原缺省 `age_enabled=true` 路径继续作为比较基线；本结果不证明年龄普遍必需或任何临床因果。R120仍仅单次 best，不升级方法基线。

内部未达`.996`，A确认官方 `not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，无官方前向；TRAIN/ROUND/done完整。owner270599为Z（不称reaped），原8 rank270625..270632已不在proc，locks/deviceusers为空、scanDenied0、8busy0、actualsafe=true。原冻结包、clock与token不改。转交实证 pins：progress `362817e9b8d47223435f68332a9018dedb57caf4a03a1ba21374ba89c020a7b3`，CPU `302621f2b051ee83ee0050af483ba0a3f1aec6e40d5876aa7477d7045973ae53`；本文未重算或校验这些SHA。

本页收口并停写，交唯一A随必要结果归档。总体`.999`未达；下一独立问题仅由规划作有界文本评议交协调方，本收口不授权新轮、计算或实现。
