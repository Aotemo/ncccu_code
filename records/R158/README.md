# R158同架构Small公开384微调初始化

04:26:32.205244 UTC已核[完整结果](R158-RESULT.json)：6400 success=attempt、0skip、8次内部评估、连续4轮无提升正常早停；唯一EMA c4@3200 AUROC `.9864032241168177`、AP `.8270582470791915`、error_twice22584。均未严格胜R107/R120，固定公开初始化分支关闭，R159未formal/activation/GPU；不扫来源checkpoint或新seed补考，也不把本配方阴性外推为所有公开384初始化无效。完整selected不足.996，官方未运行。

[同次安全终态](R158-COMPLETION.json)已证：owner495002/start361771252现Z、8rank缺失，锁持有者和设备使用者空、扫描拒绝0、8PCI busy0。归档只读此前终态与freeze元数据，未重新安全扫描、加载checkpoint或forward；blob `0c4e1243713be6137cb0f460ba0cdb857684263d7014a8963d20919505f7b06f`、tensor `1247fe883ce09c285f8381d590e6ed850f8d7be713f53b5b3ab4dd128bb8fb4d`及原clock/grant保持。下文为启动阶段证据，当前接续独立R160。

03:36:39.657018 UTC[真实首成功观测](R158-START.json)：375 success=attempt/0skip/eval0，8rank匹配，345全训参数张量/5组空AdamW/whole `440ad1fff41c1166d0598a91b6e85b0cb04b4bc949bef195b40113476ccf7a96`与formal一致。首自然25 loss `.6516250371932983`，WORLD8/G64P8/448pairs、每rank8图；owner495002/start361771252，96核/8PCI实核匹配。没有GPU预演。

冻结Git `c10dd8cd49fd2aede5fbcc3ea744ebf6fcaff5ef` / tree `04355a555d3aa6de006fdc44c4adcdbe7eb4207b`，SCI `d4766e6dd4939d3563b691b0247a55dc30ebdc75519483bc7d7e4398bc0077f2`，SITE `496e7c46acd005410d17c7660bb21acd2b7dc139e4dab5c96e81248463ec54a5`。[必要CPU证据](../preparation/small384-20261102/README.md)精确复用；[唯一正常formal](R158-PREPARATION.json)18/18、88.209秒，精确部署13.590秒、SHA `0f920d72926a54a18861712aa2dbc2bac3c0310d8895c0d9924676fc4787c63a`，一次activation/direct launch0。[实际job/512几何与原logistic loss](R158-ACTUAL-GEOMETRY.json)已核，不叠R156损失；同架构不同公开初始化的收益不能单独归因于384像素。

[接续间隔](R158-HANDOFF-GAP.json)：前轮训练完成事件至dispatch227.177秒/设备288.850秒/首25记录342.239秒；前两项≤300、后者>300。以整轮ended计169.761/231.434/284.823秒。连续利用率未测，连续空闲≤5分钟未证，不用较短基点冒作更长事件间隔；未跳必要检查或重置旧clock/grant。

唯一Git/GPU owner32e3，训练24–119/96核；[aa0f准备分配](R158-PREP-ALLOCATION.json)根0–22/23核、用户另开peer23/1核，互斥、至07:33:47 UTC。当前入口r158-pod-context.env绑定START3；8rank初态与自然成功只证明代码可运行。原Small/FP32/age/增强/AdamW/LR/EMA/24k-800-p4/唯一内部选点保持，完整selected≥.996才官方5849一次，raw .999未达，继续。
