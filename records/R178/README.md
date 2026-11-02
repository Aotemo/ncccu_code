# R178：固定 pairlog 与逐图 BCE 各半

[完整结果](R178-RESULT.json)：6400 success=attempt、0skip、c8/bad4合法早停，唯一EMA c4@3200内部AUROC **.9861985259322498**、AP **.8266579317333352**、error_twice22924。两锚点均未超过，固定联合目标关闭，R179不eligible且未normalformal/clock/GPU；仅说明本固定配方/seed未改善。完整freeze状态原为null，原样保留；official正常not_run_by_policy/threshold_not_met/evaluated=false。owner Z、8rank absent、锁/设备使用者空、8卡busy0的原safe实证见[R178-COMPLETION](R178-COMPLETION.json)，归档未重查safe、未加载模型或前向。随后独立R180沿原入口串行接续；下文为启动时点。

[实际首成功](R178-START.json)：25 success=attempt、0 skip、eval0，八rank fresh/345全训/5空Adam与正常formal匹配，whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`与R107一致。自然首25 loss **.6673235893249512**有限，WORLD8/G64P8/448对，实际8PCI及训练96核mask匹配。冻结Git `313d5abeb57240159091b682c3948aec0244b310`/tree `1dd0d3cb97e76a4224bfeb12bd35c9e255cc9bd8`、SCI `1ab259410210ad148942755f577a42448c79709d3ef0f3798951e715665d85ba`、SITE `55b95cdf030707245693fef28b8833126d6718780cf367179c9e1131c019c50e`。[正常CPU19/19及精确部署](R178-PREPARATION.json)在R176训练期间完成；一次activation/direct launch0，无独立GPU预演。

直接R107原Small/公开fe6/seed29/512/ImageNet输入/G64P8/采样增强/AdamW/LR/EMA/age/24k-800-p4，只替换目标为`.5×mean448 softplus(z_n-z_p)+.5×mean64 BCEWithLogits(z,y)`，固定temperature1、pointwise_weight .5。消费同一全局logits及原WORLD gather/DDP归一化，原网络forward数不变；[3目标/5编译、真实两public fresh与原失败证据](../preparation/pair-bce-mix-20261102/README.md)分列。逐图项按采样P8/N56计算，不把它称为总体患病率校准或已证实的错误根因；完整内部AUROC为主判，系数不扫。

R176完整8800/c11/bad4、唯一EMA c7@5600 `.9834218552639342`，两门false/官方未跑/actualsafe，R177无资格、仅技术包未formal/GPU。[实测交接](R178-HANDOFF-GAP.json)：训练完成事件至新dispatch **144.757秒**、最晚设备 **211.711秒**、首自然25记录 **282.176秒**，三项均≤300；整轮结束至对应事件80.868/147.821/218.287秒。这是分立事件间隔，连续DCU利用率未测。未跳验证或重置旧clock/token/grant；正常验收已提前完成，完整终态/safe后直接沿原入口接续。

唯一Git/GPU owner32e3，14:05实核另一既有会话closed且未恢复/派生。准备719a root0–11/12核、peer保留12–23/12核，与训练24–119/96核互斥；CPU nice19/隐藏GPU，GPU入口显式八卡可见。当前入口`work/r137-20261101-a/r178-pod-context.env`绑定start-3。尚无完整selected/官方，raw .999未达。
