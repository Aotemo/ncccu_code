# R217：原 Small 较长 EMA 窗口

13:33实观4900 success=attempt/0skip/c6/bad0、八rank活跃，暂EMA c6@4800内部AUROC .985093155735583 / AP .8024036697244497，尚无完整selected。13:33:18[原sysfs单次快照](R217-DEVICE-POINT.json)八绑定PCI均busy100%，[guard](R217-CPU-DEVICE.json)xiylcuee/5e7b17a5 exit0；非连续利用率测量。当前冻结部署保持，以下12:56为实际启动时点。

12:56[真实八rank与自然首25](R217-START.json)已证可运行，最近150 success=attempt/0skip/eval0。345全训、五组/各rank空优化器state、初态50c6f55c与正式CPU一致；首25 loss .6513809561729431有限，WORLD8/G64P8/448pairs，每rank8主图/8视图。尚无完整selected或收益声称，raw .999未达。

直接原R107 Small/publicfe6/seed29/512/ImageNet/原logistic/default AdamW-LR-WD/增强/age/globalclip/24k-800-p4，仅前1600次原EMA逐步复制后decay .999→.9999。既有更新、保存、加载和双variant内部dev唯一选点复用；首25处于复制阶段，不声称已观测1601之后真实混合或泛化收益。完整内部所选>=.996才官方5849一次。

冻结source`6fb491afaf8e5d61add04bfeacc04b932c0bb127`/tree`29540f5973693358f925139c291f92d5bd792cf9`，SCI92c087c8、actual SITE4c4f341a。[5目标4compile与两actual public fresh](../preparation/long-ema-20261103/README.md)按相同源码[复用](R217-CPU-REUSE.json)；新未启动lease2的[正式CPU18/18](R217-CPU-PREPARE.json)与[精确部署](R217-PREPARATION.json)在R215训练期间通过。manifestfe45a45b/cpu-prep91942c1d/deploymentbbaa1e4e；default12GiB/free16GiB及storage8f88dcad保持。

原stage1正常CPU已过但完整窗口不足，另绑未启动lease2并逐文件复用technical，旧grant/包/回执未改。新窗口guardgypbma6z/830aac5c、[一次activation](R217-CPU-ACTIVATION.json)txe4qea_/13cc61f0及显式八设备direct launch均0，四既有调用同SSH串行，无GPU预演或重复launch。start1/start2的starting_or_unknown保留，start3/guard3k44o29c/c5d2c257为实际首成功。generic非Lion身份不等于逐rank实测完整AdamW kwargs或moment数量。

R215完整EMA .9706920363977455，严格双门false/R216不eligible/官方未跑，唯一actualsafe后接续，旧safe不重查。[实际间隔](R217-HANDOFF-GAP.json)：整轮结束至dispatch106.410秒/最新设备170.204秒/首25记录242.089秒；末训练日志至三项142.303/206.097/277.981秒，均<=300。连续利用率未测，未跳必要校验或改旧clock/token/grant。

owner748423/start373775124，rank748435–748442，实际身份/mask见首成功原件。训练24–119/96核与准备d837/root0–11/保留peer12–23互斥，当前私有r217-pod-context.env/lease2/start3。32e3唯一Git/GPU，不恢复/派生/发消息；[R218条件同EMA seed27/R219独立Small12k](../../todo/auroc-999/R218-R219-PREPARATION.md)仍在20%池开发，当前冻结不改。
