# R199：原 Small sigmoid 的固定 weight decay .05

11-03 02:49[真实初态及首自然25](R199-START.json)已证八rank/345全训/5空原生AdamW/whole50c6与formal一致，WORLD8/G64P8/448对、各rank8图，首25 loss `.9313032627105713`有限、0skip。当前运行尚无完整所选点或官方结果，raw .999未达。

直接R120 Small/public fe6/seed20261029/512/FP32/ImageNet/原sigmoid，仅原job.weight_decay改为.05；原分组顺序的实际WD `[0,.05,0,.05,.0001]`，age保持1e-4。原backbone1e-4/head-age1e-3/default betas/eps与原24000余弦保持，不叠gamma、8k跨度、半LR、wide目标或其他policy。固定.05是独立假设，不是作者微调配方复现或已证明的过拟合机制。最多24k成功更新/每800内部评估/p4早停/EMA1600及唯一内部AUROC选点不变。

冻结commit `82add932e59e61f8eb3e8a7f0a4b4cfe1dfd454d` / tree `89c22313619ddadd1e24139ef6d3878fa8b871e5`、SCI `cc0e2703c30184cf31d09f9dfe040e2f3ef781b185e682ff67ccbff930a6dccc`、actual SITE `a1aa771046388485335260bd0fff66f2a5348cef9380ed601b58f5924d5d5cc2`。[三必要目标三编译](../preparation/weight-decay-20261103/README.md)按[精确SHA复用](R199-CPU-REUSE.json)，[正常CPU18/18及精确部署](R199-PREPARATION.json)在R197训练期间一次完成，完整producer通过；[准备](R199-CPU-PREPARE.json)、[一次activation](R199-CPU-ACTIVATION.json)、[首成功观测](R199-CPU-FIRST-SUCCESS.json)guard均0。原始公开344张量完整来源及原模型/P8/EMA/cold/WORLD证据复用，实际训练fresh初态另见[initial gate](R199-INITIAL-GATE.json)，无旧checkpoint/teacher或独立GPU预演。默认12GiB/free16GiB保持。

R197完整10400/c13/bad4、EMA c9 AUROC `.9884706757809537`，严格R107门true/R120门false，R198不eligible；官方未跑、actualsafe后独立本轮。[实际交接](R199-HANDOFF-GAP.json)：整轮结束→dispatch90.726秒/最晚八卡设备初始化153.845秒/首25记录230.374秒，均≤300。末次训练日志起分别167.714/230.833/307.362秒，最后一项略超300；连续DCU空闲未测，不声称所有口径均已满足。旧clock/token/grant未改，没有跳必要检查。

唯一Git/GPU32e3；02:41实时核peer idle/无activeTurn，无Agent恢复/派生。准备53ab/root0–11、peer12–23保留，与训练24–119/96核互斥。当前私有`work/r137-20261101-a/r199-pod-context.env`按实际start-3/当次allocation与UID绑定，不source旧env。归档[guardgozp32sf](R199-CPU-HANDOFF-ARCHIVE.json) / `cefbd9d4524ca252df048234c14074be59932c5ae817b0b553ccab5025388bc0` exit0，无新模型/forward/官方。后续R200条件同WD seed27/R201独立wide2 loss仅8635b53两technical包，正常CPU仍待可行收尾前一次执行；R202/R203仅预登记，不混入现82冻结。
