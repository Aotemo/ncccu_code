# R207：固定逐配对平方 hinge 的独立 fresh

11-03 07:56实观5825 success=attempt/0skip/eval7/bad3，八rank活跃，暂best EMA c4@3200内部AUROC `.9865633703435679` / AP `.8282748154536138`，无完整selected。07:25[既有sysfs单次快照](R207-DEVICE-POINT.json)八PCI busy为99/99/99/99/99/96/85/54%，不代表连续利用率；[guard8z6biegt](R207-CPU-DEVICE-POINT.json) / `61967962a2b126190d228db39e68cb1b3572108e31c69d329724af25a4913ae8` exit0，无GPU API或forward。后备[R209 normal19/19及精确部署](../preparation/amsgrad-20261103/NORMAL-R209.json)已在本轮训练期间完成，未新clock/activation/GPU；R208仍仅technical，原78冻结不变。下文为初态时点。

11-03 07:11[真实初态及自然首25](R207-START.json)已证八rank、345全训、5空原生AdamW、whole50c6与formal一致；WORLD8/G64P8/448对、每rank8图，首25日志loss `.2398408204317093`有限、0skip。07:13实观375 success=attempt/0skip/eval0，八rank活跃。证明当前链路可运行，完整AUROC收益仍待本轮终态，raw .999未达。

独立直接R107 Small/publicfe6/seed20261029/RGB512/FP32/ImageNet/原AdamW、LR/WD及24k余弦，仅loss改固定`mean(relu(1+n-p)^2)/4`；完整448配对，无额外forward、参数、状态或历史分数。margin1与系数1/4不扫描；仅在d=0处匹配原softplus导数0.5，不声称完整梯度/优化轨迹相同。原采样、增强、age、最多24000成功更新/每800内部评估/p4/EMA1600及唯一内部AUROC选点保持；官方>=.996才一次的门沿现R120，历史R107不改。不叠hard-negative、AMSGrad、WD.05、低LR或其他policy，也无临床机制或收益证明。

冻结Git `78e80aa25701888b482e96de3066511de63f2def` / tree `f6bd57161068be6d658ca23af4471f4ebd2cf2aa`、SCI `a146cfe2855eebd6aa77bb9d572159df059c220f1cffa51ba076a6ce883bc1b9`、actual SITE `1718c7f3c2114d1209a2e2c09b5a184805525fcd073c30a4567958f9b068e000`。[原5必要目标4compile](../preparation/squared-hinge-20261103/README.md)按[精确SHA复用](R207-CPU-REUSE.json)，[normal18/18及精确部署](R207-PREPARATION.json)在R205训练期间完成。[窗口](R207-CPU-WINDOW.json)guardqnw1lqeh/12025df8、[一次activation](R207-CPU-ACTIVATION.json)guard7_uxz128/37795bed、[真实首成功](R207-CPU-FIRST-SUCCESS.json)guardvo7akkt5/20cbfd87均exit0；原入口显式可见8设备、direct launch一次exit0，没有独立GPU预演。默认12GiB/free16GiB，冻结storage policy93f8a86f保持。

[R205完整所选](../R205/R205-RESULT.json)内部`.9860335632776274`两门false，R206不eligible；同次actualsafe后接续。[实际间隔](R207-HANDOFF-GAP.json)：整轮结束至dispatch83.190秒、最晚设备146.507秒、首25记录218.187秒均<=300；末训练日志至三项168.217/231.534/303.213秒，最后一项超300如实保留。连续DCU利用率未测，旧clock/token/grant未改，必要检查未跳过。

唯一Git/GPU32e3；07:06实际只读核peer idle/activeTurn null，无Agent消息/恢复/派生。ec5/root0–11、peer12–23保留，与训练24–119/96核互斥，owner704201/start371711608用control112–119，八rank704220–704227各11核。当前`work/r137-20261101-a/r207-pod-context.env`按实际start-4/UID/ec5绑定。前三次starting/unknown原件保留，不当成功或重复启动；start-3已完成真实初态但尚无更新，start-4自然首25才证实完整链路。

[归档guardianhqsqi](R207-CPU-ARCHIVE.json) / `d302d72b08c073e438c357ad5b3edeea82e4f29d359b551f4f2194231c3f0db9` exit0，只读既有小型元数据，无新模型/checkpoint/forward/官方。后续R208条件同hinge seed27、R209独立原生AMSGrad仍仅d6两technical包，normal在本轮可行收尾前完成，不混入当前78。
