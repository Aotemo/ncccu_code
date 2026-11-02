# R174：原 CLIP Base 合同的独立 fresh 存储技术复验

[完整结果](R174-RESULT.json)：6400 success=attempt、0 skip、8次评估、连续4次无提升正常早停；唯一EMA c4@3200内部AUROC **.9855170013883353**、AP **.815694928342545**、error_twice24056，均按原完整选点合同保存。未严格超过R107 `.9884405731067525` 或R120 `.9894773092062407`，关闭本固定Base CLIP配方，R175无资格且未normalformal/activation/GPU；不外推其他CLIP来源或输入配方。[终态原件](R174-COMPLETION.json)证明整轮completed、官方不足`.996`未运行、owner退出/8rank缺席/锁及设备users空/8卡busy0，actualsafe true，不重复安全扫描。原freeze的status为null，照实保留；唯一所选tensor `9994028d954d43c3d4bc9745e49ff3f19f6f3cfa3daf4195a905b5c24b927949`，权重及依赖保留在exp-v3，不进Git。

[真实首成功](R174-START.json)：25 success=attempt、0 skip、eval0，八rank fresh初态/345全训/5空Adam与正式检查匹配，首自然25 loss **.7022261023521423**有限，WORLD8/G64P8/448对。运行Git`6ec73abd252535ae99f01bcfcda600a31d148c6c`/tree`073fc381735a2d701b49362b7c0f4c27a51390b3`、SCI`2c8d50fc38680613c4c7c5d2d02935b46c34f55bda73bd1fc92981276b2affec`、SITE`6ba09f8bb12d7c9bcff9a2eff44f60b3a8344a3fe1c209b185f58d0552edca11`。[正常formal18/18及精确部署](R174-PREPARATION.json)在R172训练期间完成；一次activation/direct launch0，初态whole`5d2ebdcfb799ed587d48dd680a4ce8540d9b29159c9944e4f452b18f6c6a56b2`一致，无旧checkpoint/clock/grant重置。

这是R170原固定Base/公开CLIP768f3849/seed29/ImageNet输入512/原pair-logistic/default AdamW/age/LR/EMA合同的科学delta0复验。R170工程failed、无完整selected，不能判科学阴性。本轮仅为新的精确round名消费原集中配置的16GiB额度；[实际冻结配置](R174-ACTUAL-GEOMETRY.json)已核，原默认12GiB、旧R170/R172 policy及free16GiB底线不变。来源384/squash/CLIP均值标准差仅为来源属性，比赛仍原ImageNet输入512，不把来源属性写成实际输入。无独立GPU预演。

R172已完整阴性/actualsafe，R173无资格。[真实间隔](R174-HANDOFF-GAP.json)：训练完成事件至dispatch **537.365秒**、最晚设备 **597.897秒**、首25记录 **664.967秒**；整轮结束至三项463.054/523.586/590.656秒，均超300，未满足5分钟。主要是共享NFS flock在LOCK_NB下仍阻塞，独立新目录也复现；登录节点heredoc临时文件满另有一次入口失败，未进入normalformal。条件清退仅针对本人尚未启动子进程的guard，原运行未接收信号；旧guard自行退出时清退检查安全中止。最小[CPU本地互斥锁修复](../preparation/cpu-local-lease-20261102/README.md)2目标/2编译通过并push后，原有终态/activation链继续，未跳必要检查。单次事件间隔不是连续DCU利用率测量。

CPU准备719a绑定为root0–11/12核、peer保留12–23/12核，与训练24–119/96核互斥；正常formal的旧dca3分配保留原事实。唯一Git/GPU owner32e3，历史入口`work/r137-20261101-a/r174-pod-context.env`绑定start-2；CPU guard为实际已push的aaf475a/f7eebb84，未热改本轮6ec运行包。完整结果已收口，独立fresh R176沿既有入口串行接续；raw .999未达。
