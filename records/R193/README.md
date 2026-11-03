# R193：Small sigmoid 骨干学习率减半

已完整结束并安全收口：[最终结果](R193-RESULT.json)为6400 success=attempt、0skip、8次内部评估、连续4次无提升早停。唯一EMA c4@3200内部AUROC `.9838854364466322`、AP `.8086886177323231`、error_twice26766；严格胜R107/R120两门均false，R194不eligible。官方5849未达`.996`门而未运行，raw `.999`未达；freeze.status原值null保持。

[安全终态](R193-COMPLETION.json)于11-03 00:22:38.256 UTC确认owner同birth为Z、八rank缺席、锁holders与设备users空、denied0、八PCI busy0；[CPU回执](R193-CPU-COMPLETION.json)wf2703qi/5e04046f…93f7b exit0。selected tensor `96e03c5c584e76d56da26d1c720e789f45b0a20b63b4c064c4fa7da487cdefca`、blob `07df6f1b3581130748963ad8a8b6ef8ae6b7e64391bca09344a577c093aea205`，CSV/precision身份存最终结果。此后独立接续[R195](../R195/README.md)；不重复查旧safe、forward或加载权重。下文为运行历史。

23:39:13.007 UTC的[既有监控单次快照](R193-DEVICE-POINT.json)中，八个绑定PCI全部可读、busy均99%–100%；这是瞬时资源观察，未测连续利用率。[CPU回执](R193-CPU-DEVICE-POINT.json)45w5_4x5/`04d1f06c4fbd0e1b8b09ad0624995689634cd2d2d3033c9b96ea66ee5c27180c` exit0，未调用GPU API、forward或构建模型，未改运行。

[真实八rank初态与首自然25](R193-START.json)已证345全训张量、5组空原生AdamW、whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`；首25成功=尝试、有限loss `.9362108111381531`、WORLD8/G64P8/448对、各rank8输入。23:35最新700成功/零跳步/eval0，八rank活跃；尚无完整selected或官方结果，raw `.999`未达。

直接R120原Small/公开fe6b/seed20261029/512/FP32/ImageNet/pair-sigmoid，只把骨干LR `1e-4→5e-5`；head与age仍`1e-3`，default AdamW、WD、采样、增强、EMA、warmup1600与原24k余弦保持。训练最多24000成功更新、每800内部评估、连续4次无提升早停，唯一checkpoint按完整内部dev AUROC选择；官方5849仍隔离。首25日志下一步两骨干组LR `8.125e-7`、其余三组`1.625e-5`，符合原warmup及20:1比例。仅检验此固定较慢骨干适配配方，不预判收益或临床原因。后续R195的8k衰减没有进入本轮。

冻结Git `2a094f43cd4a5a861c9019eaf67d54b69158732f`/tree `631937790dbcccab411962b5d98552a6d97bcd94`、SCI `0985f5763473658f599e72bb2dd93fc5d81e7a94bcdefed2ffe36092dec3463c`、SITE `b8f94d4162457fa3f9495f02b6d829a20d3e5616b6b8630d47b780bd7add7058`。[正常CPU18/18与精确部署](R193-PREPARATION.json)在R191训练期间完成；原[4必要目标/3源码编译与两fresh](../preparation/pair-sigmoid-lr-20261102/README.md)按[SHA身份复用](R193-CPU-REUSE.json)，没有重建训练或验证链。[准备](R193-CPU-PREPARE.json)、[一次activation](R193-CPU-ACTIVATION.json)、[首成功观测](R193-CPU-FIRST-SUCCESS.json)均exit0；未额外GPU预演。默认12GiB存储预算/free16GiB不变。

公开初始化原始revision `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b`、original SHA `c912f45cd738512978a900cc8aefade65bc44e507321f0144378a2d98aa5e914`、converted SHA `7aaea5ce6fe6db953874d9792d3f1711a5cc0193a10d740e023108043918c610`，未用本任务旧checkpoint/teacher。[实际初态门](R193-INITIAL-GATE.json)与保存job均绑定本轮配置。

R191完整两门false、R192不eligible、actualsafe后独立fresh接续。[实际事件间隔](R193-HANDOFF-GAP.json)：R191整轮结束至新dispatch123.052秒、最晚设备188.878秒、首25记录251.980秒；自前轮最后训练日志分别168.070/233.897/296.998秒，均≤300。连续DCU利用率未测；未改旧clock/token/grant或跳过必要校验。

唯一Git/GPU owner32e3，未恢复或派生Agent。准备allocation53ab/root0–11、保留peer12–23，与训练24–119互斥；当前只读入口`work/r137-20261101-a/r193-pod-context.env`绑定start-4。归档guard4_ht3sx1/`8104599af4dcb5c629ca0982f541c37501ef189c6e4e2c037bfcc8beb2c71921` exit0；现役2a冻结保持。
