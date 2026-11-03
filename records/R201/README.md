# R201：固定较宽 sigmoid 的独立 fresh

11-03 05:15[完整结束](R201-RESULT.json)：7200 success=attempt/0skip、9次评估、连续4次未提升早停，唯一EMA c5@4000内部AUROC `.9889535226751404` / AP `.8363466610846791` / error_twice18348；严格R107门true/R120门false，条件R202不eligible，固定wide2组合关闭。官方不足`.996`未跑，freeze.status原null保留。同次[终态安全证据](R201-COMPLETION.json)证明owner679171/start370666628为Z/非active、八rank missing、锁holders/device_users空、扫描denied0及八PCI busy0；[guardekweron4](R201-CPU-COMPLETION.json) / `4587399e901082b9c90f879ca4bccd3973ca1d845a0d334d8877224557633fe1` exit0，旧safe此后不重查，不重启本轮。独立[R203实际接续](../R203/README.md)已证八rank/165全训/5空组/首自然25有限；原8635冻结不改，下文为历史。

04:27[既有监控单次快照](R201-DEVICE-POINT.json)八绑定PCI busy均100%，[guard964izxpj](R201-CPU-DEVICE-POINT.json) / `8db676dffc99254fe65db3a7b1810118c448e4a58e3da1c042545f7a6864e05f` exit0；这是sysfs时点观测，不是连续利用率。最近1500 success=attempt/0skip/eval1/bad0，首次内部AUROC `.976776388907285`，无完整selected。04:28 Paseo实核仅32e3 running、peer a4 idle且activeTurn null，未恢复/派生或发消息；ec5两CPU池继续互斥。

11-03 04:17[真实初态及首自然25](R201-START.json)已证八rank、345全训、5空原生AdamW、whole50c6与formal一致，WORLD8/G64P8/448对、每rank8图，首25 loss `1.92756986618042`有限、0skip；当次已50 success=attempt、尚未内部评估。这里只确认当前路径能运行，完整AUROC收益尚待终态，raw .999未达。

直接R120 Small/public fe6/seed20261029/512/FP32/ImageNet/default AdamW/原LR/WD1e-4/24000余弦，只改固定`mean(4*sigmoid((n-p)/2))`；同448对、同模型及forward数，不叠R199的WD.05或其他policy。温度2和乘4固定，同d=0峰值导数不代表全部梯度或轨迹相同，极端尾部仍饱和，不跨目标比较loss数值。最多24k成功更新/每800内部评估/p4早停/EMA1600、完整内部AUROC唯一选点、selected>=.996才一次官方保持。

冻结Git `8635b53728f3b52829e44d9faa24434ec0c2197f` / tree `844d58670225afab2aca011223a0ba46b99909cc`、SCI `bb449eb85474e644f61e0f8e40bc9d2780d04d1dc13905fb4da115864c2774a8`、actual SITE `4f412fe0ef87385aa3181d2ff1c4fd06ed0f795ca8a9092af95ac087f79262d5`。[原5必要目标4编译](../preparation/wide-sigmoid-20261103/README.md)按[精确SHA复用](R201-CPU-REUSE.json)，window2[normal18/18与精确部署](R201-PREPARATION.json)在R199训练期间完成。[窗口](R201-CPU-WINDOW.json)、[一次activation](R201-CPU-ACTIVATION.json)、[首成功](R201-CPU-FIRST-SUCCESS.json)guard均0。原stage时窗不足的拒绝保留，只对新未启动window2生成绑定；旧grant/clock/token不重置，无独立GPU预演。默认12GiB/free16GiB保持。

[R199完整所选](../R199/R199-RESULT.json)内部`.9858601718742286`未胜两基线、R200不eligible，actualsafe后才接续。[实际间隔](R201-HANDOFF-GAP.json)：整轮结束至dispatch91.592秒/最晚设备初始化154.013秒/首25记录220.905秒，均<=300；末次训练日志至三项分别173.681/236.102/302.993秒，末项略超300，连续空闲未测。所有原始事件保留，不声称所有口径都已满足5分钟。

唯一Git/GPU32e3，无Agent恢复/派生。实核ec5准备root0–11、peer12–23保留，与训练24–119/96核互斥；owner679171/start370666628在control112–119，八rank分别11核。当前`work/r137-20261101-a/r201-pod-context.env`按真实start-5/UID与ec5绑定。归档[guard9qxkb7wn](R201-CPU-ARCHIVE.json) / `9bcbda08dd4c36d2935a1e8539bd0bbbaeb7013fa9948121c8bdba10be498f47` exit0，无新forward/模型/旧checkpoint读取。后续R202条件wide seed27/R203独立native ViT-B低骨干LR仅既有technical包，normal在可行收尾前按原入口准备，不混入当前8635。
