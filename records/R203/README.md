# R203：原生 ViT-B/CLS 的固定较低骨干学习率

11-03 05:18[真实初态与自然首25](R203-START.json)已证八rank、165可训练张量、5空原生AdamW、wholec3495f5a与formal一致，WORLD8/G64P8/448对、每rank8图，首25 loss `.7028408646583557`有限、0skip；05:19实观200成功0跳过，尚无内部评估。这里证明当前链路能运行，不是AUROC收益；raw .999未达。

独立直接R191原native ViT-B/DINOv3 publicc6a/seed20261029/512/FP32/ImageNet/原生最终LN后CLS/原logistic/G64P8/default AdamW，只将backbone_lr从1e-4固定降为1e-5。head/age1e-3、WD1e-4、原24000余弦/800评估/p4/EMA1600与全部165参数训练保持；原RoPE非持久buffer及5个prefix在注意力中保留，不叠wide2、hard-negative、强WD或其他policy。较低LR是独立固定假设，不证明特征漂移、临床机制或收益，也不是冻结骨干/发布者微调配方复现。

冻结Git `57348fb055945e5284afbae5a9fde1efd2c0ecd9` / tree `2c55ee69b285f55592ce4c15aead33cd4d1a196a`、SCI `33054ab03a6e88a3999e348359bcfff8516a816d8489fec2128e9ca1055e14c8`、actual SITE `fac19be4c8cae375bc6f68f5e2d53e897ce985138320141f1ec1a887d1dae0e6`。[原四必要目标三编译](../preparation/vitb-low-lr-20261103/README.md)按[精确SHA复用](R203-CPU-REUSE.json)，[normal19/19和精确部署](R203-PREPARATION.json)在R201训练期间一次完成；[窗口](R203-CPU-WINDOW.json)、[一次activation](R203-CPU-ACTIVATION.json)、[首成功](R203-CPU-FIRST-SUCCESS.json)guard均0。没有独立GPU预演，原未启动长名135字符技术失败及namefix证据保留，不将它算成科学阴性。新精确round16GiB/free16GiB保持，原storage policy187a713b由冻结包消费。

[R201完整选点](../R201/R201-RESULT.json)内部`.9889535226751404`仅胜R107、未胜R120，R202不eligible，同次actualsafe后接续。[实际间隔](R203-HANDOFF-GAP.json)：整轮结束至dispatch95.727秒/最晚设备初始化159.899秒/首25记录213.434秒；末次训练日志至三项164.604/228.776/282.311秒，均<=300。连续DCU利用率未测，旧clock/token/grant未改，必要检查未跳过。

唯一Git/GPU32e3；05:07实核peer idle/activeTurn null，无Agent恢复/派生或消息。ec5/root0–11、peer12–23保留，与训练24–119/96核互斥，owner687476/start371037834用control112–119，八rank各11核。当前`work/r137-20261101-a/r203-pod-context.env`按真实start-3/UID/ec5绑定。归档[guardkf2qrx48](R203-CPU-ARCHIVE.json) / `e84abb0dd58b4e6bbf78e4a8c959e4c82df54b0c8e75a2e0d1776e091730f8c3` exit0，只读既有元数据，无新forward/模型/checkpoint或官方读取。后续R204条件同native低LR seed27、R205独立固定负例top14-mix05仍仅7ead152两technical包，normal在可行收尾前准备，不混入当前573。
