# R139 固定局部对比度：完整结束，未达目标

实际6400success=attempt/0skip/eval8/bad4，正常early_stopping；唯一EMA c4@3200，内部dev raw AUROC **0.9844971227863999**、AP **0.8115500912154501**、error_twice25750，tensor `a5239bd83b66cde5a3818bada3e0281ee04b1e4630b64a597e8791b776ee05b1`。strict>R107与strict>R120均false，关闭精确sigma10/4/−4/128配方/no seed27；R140只技术准备，未activation、未训练。官方按`.996`门未达正常跳过，metrics=null，没有官方前向。

2026-11-01 16:12:24.951145 UTC唯一完整终态观测确认done/TRAIN/ROUND completed、exit0/signals[]/errornull，actualsafe=true/terminal_idle。owner同birth Z非reaped，八rank absent，同inode锁holders[]、设备users[]、denied0、8expected PCI seen/busy0；freeze.status原值null保留。[完整结果](R139-RESULT.json)、[终态原件](R139-COMPLETION.json)及[CPU观测树](R139-CPU-COMPLETION.json)均只复制既有小件，未重复旧safe、hash/load大权重或补前向。原selected/冻结d190/clock/grant/所有成本与依赖保护；后续已独立fresh接R141。

以下为当时启动实证；“尚无完整selector”等由上方终态覆盖。

2026-11-01 15:19:16.110554 UTC观测running375success=attempt/0skip/eval0。真实native初态345全训/5组/八rank空AdamW、whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`全部匹配formal。首自然日志update25/attempt25，finite loss`.7421902418136597`，WORLD8/G64/P8/448pairs。saved input policy与SCIENCE一致；未额外GPU预演、step1探针或官方前向。

冻结runGit `d19019c8fee9878a8b62c23647062a8563be5637`、tree `673a8b23cd1e176cb0c8e1498727e3d4073f6fec`，SCI SHA `17661d8bb45ecc9177654630b479d8d9fedb2dcef70a1c3f2352b21241a9307c`，SITE SHA `02c4cc8ca8fec97917c707a000f97e338ca484de99f33ff31ba9884abab9cb35`。已通过的两项受影响CPU目标与9源compile按源码/SCI/原回执SHA精确复用；本轮normalformal唯一18/18、94.017秒，实际Git blob/mode精确部署、唯一activation及direct launch均exit0。[准备](R139-PREPARATION.json)、[复用](R139-CPU-REUSE.json)、[部署](R139-DEPLOYMENT.json)与[远端commit回读](R139-REMOTE-COMMIT.json)分列。

owner407052/start357345776及8rank的PID/start/argv/CPU mask与8PCI均实核匹配，见[首成功原件](R139-START.json)。实核effective120，训练96核24–119、prep24核0–23互斥；唯一执行/Git/GPU owner32e3本人准备占0–11，12–23保留不用。当前只读入口为本机`work/r137-20261101-a/r139-pod-context.env`；原件在`exp-v3/technical-prep/r139-20261101-32e3`，不再以R137运行定位观察当前训练。

[接续间隔](R139-HANDOFF-GAP.json)实测R137安全终态观测到R139 dispatch185.204秒。旧训练实际结束早于终态观测，这不是完整DCU空闲时长或≤300秒证明；保留两个时点、首自然更新及必要formal/activation/初态等待，未虚报达标。小件CPU树见[正式准备](R139-CPU-PREPARE.json)、[activation](R139-CPU-START.json)、[首成功观测](R139-CPU-FIRST-SUCCESS.json)。

科学合同为[SCIENCE](../../configs/R139-SCIENCE.json)与[接续方案](../../todo/auroc-999/NEXT-PREPARATION-20261101.md)：直接R107/public fresh/seed29/AdamW，只改确定性prepared RGB局部颜色对比度，训练与所有评估/预测consumer使用保存policy；512/age/原增强/G64P8/pairlog/LR/EMA/24k800p4保持。完整selected须strict>R107`.9884405731067525`且>R120 singlebest`.9894773092062407`才固定seed27，否则关闭精确sigma10/4/−4/128配方。完整freeze≥.996才官方5849一次；当前没有完整selector或官方结果，raw .999未达。
