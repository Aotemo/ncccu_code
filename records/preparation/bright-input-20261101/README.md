# 固定亮区裁剪与条件DINO seed27：CPU准备

两项必要目标实际RED两FAIL/0ERROR后，GREEN1 **2/2、0error、1.845秒**，6个受影响Python源compile通过，GPU未初始化。[CPU-CHECKS](CPU-CHECKS.json) SHA `2e8000bf2001fd54927964f5dea669599b06b3d1c009199018d295ae00157483`、[CPU-GUARD](CPU-GUARD.json) SHA `cd5da8fe58dd621bbb4762832811989c839f44612732d6404eb662b47c26e362`，原guard0和OUTPUT SHA实核一致。[RED](RED.json)/[GREEN](GREEN.json)原义保留。

第一目标覆盖固定半短边框、平坦几何中心、已知亮斑中心邻域、裁框边界/极小和窄图、原RGB逐像素裁切，以及实际Dataset的None旧像素、8draw/private随机流/全局RNG保持；真实serial/parallel/predict消费者均收到保存policy生成的同一输入。第二目标覆盖R142条件DINO27和R143固定prepared裁剪的配置/age/八rank/保存job身份，与各直接基线整件diff仅允许约定字段改变。R142仍固定head_seed27，预期whole9b4由未来正常formal实核，本目标没有构造第二次公开模型。

旧`prepare_image`、`PreparedImageCache`完整AST保持，原prepared/cache命名空间和原始数据读取不改；可选input preparation之后的增强/归一化AST保持。训练/模型/loss/optimizer/EMA/保存及消费者其余源码按与R141冻结0c一致的SHA复用，没有重已过旧suite、公开模型前向或GPU预演。仅config/data/preprocessing/experiment91、两现有测试与R142/R143 SCIENCE为本次实现范围。

同次只读资源快照记录R141八个真实PCI全部可见、瞬时busy44%–98%、每卡约11.7–11.8GB已用；这是逐卡瞬时sysfs读数，不能说整段利用率100%。准备在本人0–11/nice19/隐藏GPU，训练24–119互斥，旧0c运行未修改。R142资格等待R141完整selected双门及safe；R143尚无normalformal/activation/GPU。固定方案及预定原生细节后续边界见[接续计划](../../../todo/auroc-999/NEXT-PREPARATION-20261101.md)。
