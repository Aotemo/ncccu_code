# R180：原生 ConvNeXt Large 与匹配公开初始化

17:23:59完整终态及actualsafe已核：6400 success=attempt/0skip/eval8/bad4，正常早停；唯一EMA c4@3200内部AUROC `.9880889738720829`、AP `.8240243312217729`、error_twice19784，两门false，关闭固定Large/dc2组合，R181不eligible且仅technical包、无normalformal/clock/GPU。官方正常threshold_not_met、未评估，freeze.status原值null保留。所选tensor480473fc…03833、blob372c103b…9ac12、freeze c12271df…88d19。owner同start为Z、8rank absent、锁和设备使用者空/denied0/8PCI busy0；[完整结果](R180-RESULT.json)及[唯一完整安全观测](R180-COMPLETION.json)不重查、不加载blob或新forward。独立[R182](../R182/README.md)已串行接续。下文启动及running数值均为历史时点。

[真实八rank及首自然成功](R180-START.json)已证：345全训/5空Adam初态与正常CPU匹配，whole `3360682ea3aed68301752371888352f31e7a12ab541f8d438e7df4bcd71450e0`；首25次成功更新loss `.7808462381362915`有限、WORLD8/G64P8/448对，各rank8输入，实际PCI及训练96核mask一致。15:59观测1425 success=attempt/0skip/eval1，16:01轻观测1525/eval1/bad0，暂最佳online/EMA c1@800内部AUROC `.9774085450655094`。这不是完整结果，raw .999未达。

冻结Git `28e19c1690d29b7d4ff2a8a63a1086c88cd0fe98`/tree `7886d68948807ac08906278f74fafe74b1c35b83`，SCI `9ea7429cbef86cc69b20cea37afdd2760043c17968401a5e662b9083740ba442`，SITE `6058fb0afc912881c814f000b114f575442230a340fea0026f5c32d5795aafbb`。[正常CPU18/18与精确部署](R180-PREPARATION.json)、一次activation/正式launch均exit0，无独立GPU预演。[公开完整344 strict、两fresh、5个受影响目标及11个唯一源码编译覆盖](../preparation/large22k-20261102/README.md)与原传输/冷加载/交接字段修正证据分列。原17deee5与r180原准备目录仅未启动阶段，未formal/clock/GPU；最终采用ready目录。

直接R107原512/ImageNet输入/seed29/原pairlog/G64P8/采样增强/AdamW/LR/EMA/age/24k-800-p4，仅骨干及其匹配公开ImageNet22K→1K初始化共同改为原生Large/dc2e53fa。完整原生网络、avgpool/final LN/Dropout.2/Linear1536→1/age一次；源224预处理仅为来源属性，不是比赛输入。不是容量纯因果消融，不叠加BCE或DINO/ROI等policy。新精确round存储32GiB、原free16GiB/clock/resource保持；实际峰值与完整指标仍待本轮。

R178完整6400/c8/bad4、唯一EMA c4@3200 `.9861985259322498`，两门false、官方未跑且safe，R179不eligible。此次[交接](R180-HANDOFF-GAP.json)未满足5分钟：前轮训练完成至dispatch **1512.943秒**、最晚设备 **1583.919秒**、首25记录 **1668.388秒**。准备未在训练期间全部完成，传输失败与分段重取、130秒fixture、冗余SCI交接字段修正以及后置Git/源码复制/正常CPU/部署共同造成延迟；不虚报DCU利用率、不跳校验或重置旧clock/token/grant。后续将不依赖结果的完整准备前移至本轮训练期间，收尾前完成正常CPU与精确部署。

唯一Git/GPU owner32e3；当前Pod内准备719a root0–11/12核、peer12–23保留，与训练24–119/96核互斥，GPU入口显式八卡可见。实际运行入口为工作目录r180-pod-context.env及ready/start-2；原件在exp-v3。完整selected严格过R107/R120才考虑固定seed27，完整selected≥.996才官方5849一次。官方未参与本轮训练/选点。
