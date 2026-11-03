# R205：固定困难负例四分之一混合的独立 fresh

11-03 06:22[真实初态和自然首25](R205-START.json)已证八rank、345全训、5空原生AdamW、whole50c6与formal一致；WORLD8/G64P8/448对、每rank8图，首25 loss `.7473738193511963`有限、0skip。06:24实观425 success=attempt/0skip/eval0，八rank活跃。当前链路可运行，完整AUROC收益仍待本轮终态，raw .999未达。

独立直接R107 Small/publicfe6/seed29/RGB512/FP32/ImageNet/原AdamW与LR/WD/24k余弦，仅loss改固定`.5*mean_all_negative+.5*mean_stable_top14_negative`，逐负例项是8正例softplus均值，按WORLD负例原序稳定破同分。全部448对保留base梯度、112对多获梯度，无额外forward/teacher/dev名单。最多24k成功更新/每800内部评估/p4/EMA1600，完整内部AUROC唯一选点；显式官方>=.996门沿现R120，历史R107不改，不叠R203低LR或其他policy。

冻结Git `7ead1526ec629508277f0623a506bc42fbf0dee8` / tree `9a03c357677b629b3256d0531fe863c7f98a6f61`、SCI `7aaa297d8ec74aa76758294bc47dcf0891a6410c6c8fa729d64953b188e9ab4c`、actual SITE `10afb90397f6046e809a91a553c9a6e1926ba9d321d93760a19a1a421431d96f`。原[6目标5compile](../preparation/hard-negative-20261103/README.md)按[精确SHA复用](R205-CPU-REUSE.json)，[normal18/18及精确部署](R205-PREPARATION.json)在R203训练期间完成，[原CPU guard](R205-CPU-PREPARE.json) exit0。[完整窗口](R205-CPU-WINDOW.json)、[一次activation](R205-CPU-ACTIVATION.json)、[自然首成功](R205-CPU-FIRST-SUCCESS.json)guard均0，无独立GPU预演。默认12GiB/free16GiB保持，冻结storage93f8a86f消费。

[R203完整所选](../R203/R203-RESULT.json)内部`.9816433892721294`未胜两基线，R204不eligible；同次actualsafe后接续，旧safe不重查。[实际间隔](R205-HANDOFF-GAP.json)：整轮结束至dispatch146.790秒/最晚设备205.295秒/首25记录268.265秒均<=300；末训练事件至三项197.321/255.827/318.796秒，末项超300如实保留，连续利用率未测。原start-1/start-2仍为starting_or_unknown，未重复activation/launch。

唯一Git/GPU32e3；06:13/06:15实核peer idle/activeTurn null，无消息/恢复/派生。ec5准备root0–11、peer12–23保留，与训练24–119/96核互斥；owner696335/start371417650用control112–119，八rank696347–696354各11核。当前`work/r137-20261101-a/r205-pod-context.env`按真实start-3/UID/ec5绑定。[归档guardffy4868r](R205-CPU-ARCHIVE.json) / `c0b790bd475428f4466b1b73609f3b35eedb39ff920ee8ba9c613638d4745781` exit0，无新forward/模型/checkpoint或官方读取。后续R206条件同loss seed27/R207独立逐pair平方hinge仍78两technical包，R208/R209的d6两包为更后续准备，无normal/clock/GPU，不混入当前7ead。
