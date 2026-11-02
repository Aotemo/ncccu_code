# R176：Small DINOv3 公开骨干 fresh 初始化

[真实首成功](R176-START.json)：13:05实观50 success=attempt、0 skip、eval0，八rank初态/345全训/5空Adam组与正常formal匹配，whole `126ddef0d745516515716a252ad2e68847486b5163897470c51c8eed303ec333`。自然首25 loss **.7686490416526794**有限，WORLD8/G64P8/448对；实际8PCI及96核训练mask匹配。冻结Git `e9a7f1de44748a6b38dcb3ba4265d3689a30c0e3`/tree `1399aa092e30a68153669e4f7b1f0c736edcc311`、SCI `9af9a4f33ce090beab9f93ec9952b14f3a4ba63bc46e290fbf868cdb7820891c`、SITE `9631c0eb78de13ff8a12b8a7854b00cd82f1cad9fc4ecf2b4c3daa539cd730a9`。[18/18正常formal及精确部署](R176-PREPARATION.json)在前轮训练期间完成；一次activation/direct launch0，无独立GPU预演。

直接R107原Small/seed29/512/ImageNet输入/原pair-logistic/G64P8/default AdamW/LR/EMA/age/24k-800-p4，仅将完整公开骨干初始化换为自然LVD-1689M DINOv3固定 `c497e64d5d483b3ad42eac93e2ef9d1799f9547c`。342无公开分类头张量完整strict，生产二分类头及年龄保留原生RNG构造；[来源、许可和CPU证据](../preparation/small-dino3-20261102/README.md)另列。来源224/center仅是来源属性，比赛仍512；不预设临床机制或AUROC收益。实际冻结存储默认12GiB，free floor16GiB未改。

R174完整所选`.9855170013883353`、双门false且actualsafe，R175无资格，未normalformal/activation/GPU。[交接实测](R176-HANDOFF-GAP.json)：训练完成事件至dispatch **301.692秒**、最晚设备 **362.176秒**、首25记录 **442.919秒**，未满足5分钟；整轮完成至对应事件234.039/294.523/375.266秒。完成后首次观测时owner仍在收尾；再次取得完整selected与actualsafe后，立即窗口检查、激活和派发。后续在预定收尾前缩短观测间隔；不跳校验或重置clock/token/grant。上述为分立事件间隔，不是连续DCU利用率测量。

唯一Git/GPU owner32e3；13:03实时Paseo另一既有会话idle，未唤醒/派生。准备719a root0–11/12核、peer保留12–23/12核，与训练24–119/96核互斥；CPU任务nice19/隐藏GPU，GPU入口显式八卡可见。当前入口`work/r137-20261101-a/r176-pod-context.env`绑定start-6。尚无完整selected，官方按原门隔离，raw .999未达。
