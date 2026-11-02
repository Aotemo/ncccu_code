# R167/R168 固定优化器对照 Implementation Plan

**Goal:** 在R166训练期间完成下一条可直接接续的最小准备；R167/order79只在完整R166严格超过R107/R120并安全结束后做原meanpatch seed27，否则跳过，独立R168/order80直接比较R107。

**Architecture:** R168回到原ConvNeXt Small/public fe6b/seed29/512/logistic/G64P8/age，唯一变化是原生torch.optim.AdamW所有既有组的betas从默认(.9,.999)改为固定(.9,.95)。仅检验较短二阶矩平滑时间尺度，不改变一阶矩、LR/WD/eps/AMSGrad/foreach/fused、采样、clip、EMA/24k-800-p4；不叠加DeiT/meanpatch、Lion或其他已闭分支，不声称梯度噪声机制已证。若阴性不扫beta/LR/seed。

**Tech Stack:** 原Python3.12/torch2.10，已有_optimizer/训练保存评估监控链。参考[当前所用PyTorch2.10原生AdamW定义](https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html)，文档仅支持参数/算法语义，不支持本任务AUROC收益；实际依赖/参数组以Pod检查为准。R137的Lion整体配方不等价，不能作为beta2单变量结论；当前记录未见同一固定对照。

- [x] 在`tests/test_adamw_policy.py`先用真实_optimizer观察忽略beta2新policy的RED；独立逐参数三步矩/更新公式、下一步保存恢复对照、None/省略原生默认值/RNG/5组兼容。
- [x] 生产改动仅`config.py`版本化可选policy/保存及互斥验证、`engine.py`原_optimizer一条原生AdamW构造分支、`experiment91.py`两身份与job精确绑定；`check_candidate.py`只核新组的实际betas/eps/amsgrad。不新建优化器、训练、状态保存或监控平台，distributed_engine/loss/EMA/models/inference不改。
- [x] 原`test_candidate_binding.py`加R167/R168及缺失/错误policy拒绝，写两SCIENCE，实际public fresh/5空组及组betas；R168原50c不变，R167原8108不变。旧loss/WORLD在原数学/测试/gather字节与engine除_optimizer外均相同时复用，不把改过engine整件说成未改。
- [ ] 受影响CPU检查及源码身份确认后实际Git同步/回读/精确技术包；R166保持98badc9冻结，准备dca3根23核0–22+peer1核23与训练96核24–119互斥。完整结果决定唯一分支，安全释放后沿既有入口一次接续，不重置旧clock/token/grant，不新forward/GPU预演或资源。

只用内部完整所选AUROC主判；AP/难组不能救主判，官方仍完整selected≥.996才一次，总目标raw .999未达。

当前[3目标/6编译、两public fresh与实际5组betas](../../records/preparation/adamw-beta2-20261102/README.md)均已通过，等待实际Git同步/回读及技术包。无新normalformal/grant/clock/GPU，R166继续98冻结。
