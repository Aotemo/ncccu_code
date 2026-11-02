# R162：固定余弦16末端分类头

05:29:52.646149 UTC[真实八rank首成功](R162-START.json)：100 success=attempt、0skip/eval0，8PCI及96核mask匹配；345全训、5组初态空AdamW、whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`与formal一致。首自然25 loss `.6625226736068726` finite，WORLD8/G64P8/448pairs、每rank8输入。GPU链已实际可运行，完整AUROC收益待结果。

运行Git `608e71b524e5c1e6975cb9673c64899031d68edb` / tree `069bacc30f4b333586506f3f2cc4476b54d9b416`，SCI `5ddf40f0d29a3a2907a18db0ca7111fd926d16ee581ffde77fe515fa68a699d3`，SITE `4676781dfc0c8436e0722c4c647e06ed42018144ef3dd452cd41bb694ce2ab37`。[3新目标/7compile、两public fresh及旧6loss/WORLD证据](../preparation/cosine-head-20261102/README.md)按SHA复用；[唯一正常formal](R162-PREPARATION.json)20/20、88.500秒，精确部署10.437秒、SHA `548d31abb57d840f64475fb269771f341d81d3ce1db3882fceb147cd650e1c43`；一次activation/direct launch0，无独立GPU预演。

直接R107/public fe6b fresh/seed29/原RGB512及logistic，仅原Linear变为固定`16*cos(x,w)+原bias`，x位于原GAP/LN/Flatten/Dropout之后，原age随后一次。原参数对象、名称、RNG及optimizer组不变；参数SHA可相同而函数不同。余弦policy随job保存并由真实冷factory核无状态图；不是NormFace softmax训练复现，不声称难图或临床根因已证。原增强、AdamW/LR/EMA和唯一内部选点保持。

[实测接续](R162-HANDOFF-GAP.json)：旧训练完成事件至dispatch188.652秒、最晚设备证明250.588秒，均≤300；首自然25记录322.927秒，仍>300。整轮ended至三项分别153.201/215.137/287.476秒。技术包前置及收尾并行formal已落实，连续DCU利用率未测，不据设备证明宣称连续空闲≤300已证；无跳检查、重置clock/token/grant。

唯一Git/GPU owner32e3，owner511382/start362460276；训练24–119/96核，aa0f准备根0–22/23核、peer23/1核互斥。当前`r162-pod-context.env`绑定START3，冻结608不热改；最多24000成功更新/每800内部dev/连续4轮无提升早停，完整selected≥.996才官方5849一次。raw .999未达，继续。
