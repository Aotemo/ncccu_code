# R168：固定原生 AdamW beta2=.95

[真实八rank首成功](R168-START.json)记录25 success=attempt/0skip/eval0，345全训/5空Adam/whole50c与正常formal匹配，原WORLD8/G64P8/448pairs、首自然25 loss `.6512339115142822`有限。八rank进程/start/argv、8PCI、控制112–119与rank24–111共96核均匹配。首成功guard `e1064bee92ceb963f999406099714270f863ffeb4d8166a292bdce0465d2fe80`；不是CPU或独立GPU预演替代实际训练。

运行Git `2683a8abb4fa72be7c612a14d17aeb1153510924` / tree `c23226b64e00a9fca02040774c986feafb3f601b`，SCI `d71e02eb026b8f4d689878ba653cc6d502bcedd858c5e04cb727f573e9ca2266`、SITE7ab53ce9；[唯一正常formal](R168-PREPARATION.json)19/19、94.032秒，精确部署18.185秒，一次activation/direct launch0。五组betas(.9,.95)/eps1e-8/amsgradFalse为正常CPU实测，GPU首成功未额外逐rank记录矩状态或betas；原构造源码与保存job经过实际部署核验。

直接原R107 Small/fe6b/public fresh/seed29/512/logistic，仅原生AdamW的二阶矩平滑系数变为.95，保留原LR/WD/age/采样/clip/EMA与24k-800-p4。未叠DeiT/meanpatch、Lion或其他分支；收益待完整内部AUROC，不以AP/难组救主判。官方仍完整selected≥.996才一次，raw `.999`未达。

[交接实测](R168-HANDOFF-GAP.json)：R166训练完成事件至dispatch247.039秒、最晚设备306.117秒、首自然25记录379.192秒；整轮ended至三项为211.463/270.542/343.616秒。dispatch≤300但设备记录略超，未证明连续DCU空闲≤300，不虚报达标；未跳检查/重置旧clock/token/grant。技术包前轮已完成，正常formal/部署及实际owner/rank启动仍占交接；后续将不依赖结果的正常CPU验收在训练期间完成，科学分支选择与GPU仍等待完整结果和actualsafe。

唯一Git/GPU owner32e3，peer仅CPU23；dca3根0–22/23核与peer1核合计准备24核，训练24–119/96核互斥，当前r168-pod-context.env绑定start-4。原R166完整阴性/actualsafe归档后不重复旧扫描；R167未formal/GPU。权重/数据/逐样本预测只在exp-v3。
