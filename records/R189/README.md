# R189：固定原生 DINOv3 Small CLS 单读出

完整结论已同步且R191实际首成功后，本轮冗余checkpoint已与R184R1按[精确18件清单](../storage/closed-r184r1-r189-20261102/README.md)清理；原selected身份/CSV/precision/侧车和元数据保留，不再加载或评估旧权重。

[完整结果](R189-RESULT.json)：7200成功更新/9轮内部dev/连续4轮未提升正常completed，唯一选点EMA c5@4000，AUROC `.977164111350996`、AP `.7450861591443787`、error_twice37930。严格超过R107与R120两门均false，官方按阈值未跑；R190 seed27不eligible。原freeze.status为null原样保留，不补造字段；模型tensor/原blob/CSV/precision SHA随精简结果封存。此固定CLS配方未胜锚点，不能据此外推所有CLS表征或临床原因。

[唯一安全终态观测](R189-COMPLETION.json)确认owner627828/start368273364为Z、八rank均missing、锁持有者/设备使用者为空、8卡busy0、denied0；CPU guard yrkhrbef/`00ac6caa9579c9473730680bbdd2951eba93fdd567be34db3eebb1aeac14f67c` exit0。封存后不重复safe或加载旧权重。之后独立[R191](../R191/README.md)已实际八rank首成功；末段下一轮未启动描述均为历史。raw .999未达。

[真实八rank初态及首自然更新](R189-START.json)已证：165全训、5空AdamW、whole `e347757c93f10e7eb80de9322980ac52a421b8a864c35bdeaddc736433c26227`；自然首25次成功=尝试、loss `.7190340161323547`有限、WORLD8/G64P8/448对、各rank P8。八PCI与训练96核mask匹配。21:39实观500success=attempt/0skip/eval0，尚无完整selector/官方结果，raw .999未达。

[实际初态门元数据](R189-INITIAL-GATE.json)保存了global_pool=`token`及精确vit_dino3_pool_policy；live FP32 RoPE1、checkpoint persistent-buffer0/空摘要，165张量/5空组及八rank原件SHA已核。模型初态tensor SHA仍与R184R1一致，pool属性由保存job/规范描述固定，冷图证据在原CPU准备。运行冻结Git `0f3b19e6fe6f52c14e8aaed48980d51966ec7317`/tree458e9c7f、SCI `11cf4bc7bc85822a6aa79a8670a4f09f232f4698f46db29ab391652b79747eb6`、SITE `23409f6207ff5335385c466860bdae71ae9a960ef3269e93862f672300590b47`。

[正常CPU19/19与精确Git部署](R189-PREPARATION.json)在R1训练期间完成；原4目标/5编译按精确源码复用，一次窗口核验/activation/direct launch均exit0，未GPU预演。正常guard _enxd0va、activation kuaew823、首成功wka6y0ix均0，小件归档0wvyw89f/`6e95ee70206ccb869b8430a1d6d0c2065cec19ca9255d5e520d5cb92b8f03534` exit0。

保持原R184R1的公开3bf4720a/seed29/512/ImageNet/logistic/G64P8/AdamW/LR/EMA/age/9:1/24k-800-p4，只将native final LN后的1024 patch均值改为第0个CLS token。全部1CLS+4register+1024patch仍参与自注意力，Dropout.2/Linear384→1/age各一次；训练/保存/推理复用原链路。依据为DINO公开CLS全局特征假设，尚无本任务AUROC收益或临床机制证明；后续Base代码未混入本轮0f3冻结。

R1完整双门false、R188不eligible、actualsafe后独立fresh接续。[真实事件间隔](R189-HANDOFF-GAP.json)：自R1整轮结束至dispatch94.058秒/最晚设备151.353秒/首25记录192.199秒；自最后训练日志分别129.289/186.584/227.430秒，均≤300。这里只分列事件间隔，连续DCU利用率未测；默认12GiB/free16GiB保持。

32e3唯一Git/GPU owner，未恢复/派生Agent；5de准备root0–11/保留peer12–23与训练24–119互斥。当前入口 `work/r137-20261101-a/r189-pod-context.env` 绑定ready/start-3。完整R189严格双门与safe决定R190同Small-CLS seed27资格，否则后续独立R191 Base-CLS；后两者仍仅CPU准备、尚未formal/clock/GPU。
