# R191：固定原生 DINOv3 Base CLS

[结果已同步95b522b并在R193真实首成功后清理8个冗余权重](../storage/closed-r191-20261102/README.md)，逻辑10,281,294,086B。结果、预测、precision、sidecar与来源身份保留，三KEEP及当前依赖未变。

已完整结束并安全收口：[最终结果](R191-RESULT.json)为10400 success=attempt、0skip、13次内部评估、连续4次无提升早停，唯一EMA c9@7200内部AUROC `.9666919930498946`、AP `.6996772333504798`、error_twice55324。严格胜R107/R120两门均false，R192不eligible；官方5849未达`.996`门而未运行，raw `.999`未达。freeze.status原值null保持，不把它改写为completed。

[唯一安全终态](R191-COMPLETION.json)于23:26:18.970 UTC确认：owner同birth为Z、八rank缺席、锁holders与设备users均空、denied0、八PCI busy0；[CPU回执](R191-CPU-COMPLETION.json)q7yl2v32 exit0。selected tensor `496543053f17a429ab31ee98516c9a39d23ad971955e78f926592c37ecd6dec0`、blob `d80c09709dca3b79ba4f6a850f24399ca1bb4d22e24e258ff576c93f3834f098`，CSV/precision身份保存在最终结果。归档不重新forward、加载权重或重复查旧safe。其后按原固定分支接续[R193](../R193/README.md)；下文是运行历史。

22:51实观5700 success=attempt/0skip/eval7/bad0，八rank活跃，暂best EMA c7@5600内部AUROC `.9662970459643753`，尚无完整selected。22:50:42.506的[既有监控单次快照](R191-DEVICE-POINT.json)中，八个绑定PCI均busy100%、全部可读，非连续利用率证明；CPU guard tjvpi8rc/`747fdad3a75e40e8c60f256cb529df1db1fae8d3a0e5c925814e0fcfb3b40dd7` exit0，0 GPU API/forward/模型构建。观察准备现用53ab/root0–11/保留peer12–23，与训练24–119互斥；入口`work/r137-20261101-a/r191-renewed-pod-context.env`仍绑定原ready-preparefix/start-3。原a77/GPU allocation/clock保持，下文为启动历史。

[真实八rank初态与首自然25](R191-START.json)已证：165全训张量/85,641,987元素、5组空原生AdamW，whole `c3495f5af134d01496fee241554a8e6113972f554bccb899eb6184cb92c09325`；首25成功=尝试、loss `.6996408700942993`有限、WORLD8/G64P8/448对/各rank P8。22:14最新675成功/零跳步/eval0，八rank活跃，尚无完整selector或官方结果，raw .999未达。

[实际初态门元数据](R191-INITIAL-GATE.json)明确variant `vit_base_patch16_dinov3.lvd1689m`、features768、global_pool=`token`与固定CLS策略；live FP32 RoPE1、checkpoint persistent-buffer0/空摘要。12层/12头，1CLS+4register+1024patch参与自注意力，native final LN后仅取CLS，Dropout.2/Linear768→1/外层age各一次。与原Small-CLS保持512/ImageNet/logistic/G64P8/default AdamW/LR/EMA/age/9:1/24k-800-p4，仅换Base架构与匹配公开自然图像初始化；公开来源256不是比赛输入。未证明临床机制或AUROC收益。

运行冻结Git `a77d39563cbfaeef221aaa307051de441ff8e7d6`/tree `4c5ee8056397419a7bfc73986f893837501a754e`，SCI `f41f43a2f80fd93c1ffed36dc2a86bf1f22dc47a02610402850fbe25f310ef73`、SITE `de26a8812b7507e17e459157f4032ba23f2cad400fbad6054b98bb6827963b2d`。[正常CPU19/19与精确Git部署](R191-PREPARATION.json)在R189训练期间完成；原8必要目标/14源码编译按实际SHA复用，仅修原prepare对Base漏写162字段的已留失败，不重跑模型P8。一次窗口、activation、direct launch均exit0，未独立GPU预演。仅本精确round16GiB例外，默认/旧例外/free16GiB不变；容量情景不是DCU峰值或16GiB必成功证明。

公开初始化固定repo `timm/vit_base_patch16_dinov3.lvd1689m`，revision `c6a5fb7d12bbd3cf3b0079253141c3332aaed7da`，完整bin SHA `ed8a1265847086c913535fe7d84c0156a68dc98c0edd52298b8bb4bb7bdcbbb1`，原生162张量strict/exact/FP32来源证据见[准备归档](../preparation/vit-base-20261102/README.md)。未用本任务旧checkpoint/teacher或医学预训练。

R189完整双门false、R190不eligible、actualsafe后独立fresh接续。[实际事件间隔](R191-HANDOFF-GAP.json)：R189整轮结束至新dispatch88.641秒、最晚设备150.289秒、首25记录206.155秒；自最后训练日志分别126.377/188.025/243.891秒，均≤300。连续DCU利用率未测，未改旧clock/token/grant或跳过必要校验。

32e3唯一Git/GPU owner；未恢复/派生Agent。5de准备root0–11/保留peer12–23与训练24–119互斥。当前入口 `work/r137-20261101-a/r191-pod-context.env` 绑定ready-preparefix/start-3；活跃冻结不热改。归档guard16mtuls7/`a96227ebe6acd19e4f11db1ad1f194b7a2a076e2310822612a606094645a7999` exit0。
