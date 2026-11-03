# R195：原 Small sigmoid 的固定 8000 次衰减跨度

完整结果同步b2aa1bd且R197真实运行后，[9个冗余checkpoint已清理](../storage/closed-r195-20261103/README.md)，逻辑6,733,279,627B；三KEEP/当前依赖保持，原CSV/precision/sidecar/index/freeze/锁均留。原selected张量与blob身份为历史证据，不再保留本轮权重。

**已完整早停结束，未达目标。** [完整结果](R195-RESULT.json)6400成功=尝试/0skip、8次评估、bad4；唯一内部选点EMA c4@3200，AUROC `.9861756478998568` / AP `.8220377066729916` / error_twice22962，严格双门均false，R196不eligible。未达内部.996，官方未运行；原freeze.status缺省null保留。实际6400结束，没有自然走到8000衰减尾点。

[安全终态](R195-COMPLETION.json)于11-03 01:18:45.481实核owner Z、八rank missing、锁holders/device_users空、denied0、8PCI全部busy0；[guardvjjjmd7h](R195-CPU-COMPLETION.json) / `eba272374205ec59c018f766dd1324af46030922915daa4a805ca6f156031c32` exit0，封存后不重查safe。selected tensor `ef345d24e69aa9c9c921a740e11d71628e1344b9dfbd30e1c8a7e8e598e0c8d7`、blob `799ed0f03eadbe597e325f23a51b061574a0d889f16a7609fc8f29a816c2ea28`；CSV05c9dc29 / precisionc4583d81均由原freeze登记。

已独立fresh[接续R197](../R197/README.md)，正常CPU/部署在本轮训练期间完成，新八rank/首25已实证。该较早衰减配方未胜R107/R120，不外推其他日程；无官方或逐样本标签改动。下文“尚无selected”等为初始运行时点。

11-03 00:32:32.511 UTC的[既有监控单次快照](R195-DEVICE-POINT.json)八绑定PCI全部可读、busy均100%；这是瞬时观察，未测连续利用率。[CPU回执](R195-CPU-DEVICE-POINT.json)khkjd_93/`16beaff5f764a9347bc278d7622340af6292b7af4bb1af3ed93e1ea4e785f70e` exit0，0 GPU API/forward/模型构建，未改运行。

[真实八rank初态及首自然25](R195-START.json)通过：345全训张量、5组空原生AdamW、whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`与formal一致；WORLD8/G64P8/448对、各rank8输入、有限loss `.931303858757019`。11-03 00:29最新475成功=尝试、0skip、eval0，八rank活跃；尚无完整selected或官方结果，raw `.999`未达。

直接原R120 public fe6b/seed20261029/Small/512/FP32/ImageNet/sigmoid，恢复原骨干LR1e-4、head/age1e-3/default AdamW；仅`lr_decay_updates=8000`。原warmup1600后沿原余弦在下一成功计数7999到达尾倍率.01，此后保持；实际最多24000成功更新、每800评估、连续4次无提升早停、EMA1600与唯一内部dev AUROC选点保持。未叠R193半LR或后续gamma。首25日志列示下一步两骨干组LR `1.625e-6`、其余三组`1.625e-5`；首25位于原warmup，不能把它写成GPU已实测8000边界。

冻结Git `693d48c732871314e163c7d4f95582f7bc0c23a9`/tree `6a0f3392534ccb00aedfc4e1dbf1b2ffa3bbd292`、SCI `57c1d89a61ef9e9e4306b680961c4e5cac7be5c13e6c3569deeb3a9e693a4841`、SITE `5d5c56ab87fcf59d50c76eb0db51e92e337402d9fed86dd082b9779ba5614144`。[正常CPU18/18与精确部署](R195-PREPARATION.json)在R193训练期间完成，完整LR边界producer本次实际通过；[6必要目标/6源码编译](../preparation/cosine8k-20261102/README.md)按[原件SHA复用](R195-CPU-REUSE.json)，未重跑未变模型/WORLD/EMA/P8/cold。准备、一次activation与首成功观测各[原CPU回执](R195-CPU-PREPARE.json)均0；[activation](R195-CPU-ACTIVATION.json)/[首成功](R195-CPU-FIRST-SUCCESS.json)分列，未额外GPU预演。默认12GiB/free16GiB保持。

公开原始revision fe6b358aa409414cdb522ae3d43a2c9922f7cc1b、converted SHA `7aaea5ce6fe6db953874d9792d3f1711a5cc0193a10d740e023108043918c610`，完整344张量来源与原R120一致；[实际初态门](R195-INITIAL-GATE.json)绑定新job/SCI。未用本任务旧checkpoint、teacher、医学初始化或官方调参。固定较早衰减仅是优化假设，不证明旧日程有bug、临床机制或收益。

R193完整两门false、R194不eligible且actualsafe后独立fresh接续。[实际间隔](R195-HANDOFF-GAP.json)：整轮结束至dispatch131.740秒、最晚设备初始化194.871秒、首25记录270.654秒，均≤300。从前轮最后训练日志起分别203.491/266.622/342.405秒，最后一项超过300；未测连续空闲或利用率，不冒称全程满足5分钟。前轮封存/退出、终态发现与准入处理、新八rank初态均计入这些真实时点；下一次缩短终态发现和准入之间的处理间隔，不跳过必要校验或改旧clock/token/grant。

唯一Git/GPU owner32e3，准备53ab/root0–11、保留peer12–23与训练24–119互斥；00:15实核peer idle/无activeTurn，未恢复/派生Agent。新只读定位`work/r137-20261101-a/r195-pod-context.env`由本轮实际初态/当前allocation字段生成，绑定start-3，不source旧轮环境。归档rxzzh62s/`ded814fcd2875fcfb902f345f78336af9404be277054335dfcfd2d00fdeefbc6` exit0。后续R196条件8k seed27/R197独立fit gamma仍只436ae5b技术包，当前693冻结不改。
