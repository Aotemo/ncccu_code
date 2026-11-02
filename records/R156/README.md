# R156固定困难正例损失：完整阴性结果

03:31:22.308151 UTC[完整终态](R156-COMPLETION.json)：6400 success=attempt/0skip/eval8/bad4，正常早停。[唯一EMA选点](R156-RESULT.json)c4@3200内部AUROC `.9860275427427871`、AP `.8255205020020818`、error_twice23208，tensor `02bfc24718ab07ae0cee91c97aaf6775c4ac949e1044022aee53c85cd864436a`。低于R107和R120两门，固定top2/.5合同关闭，R157未formal/activation/GPU，不扫描topk/权重/温度。官方因内部<.996按政策未运行；[实际guard](R156-CPU-COMPLETION.json)exit0，owner同birth为Z、8rank退出、锁/设备users空、8PCI busy0，actualsafe，不重查。旧clock/grant/selected依赖保持，后续独立R158已接续。

原封存20难正例组的后验分解：错序质量6623→8762，修复814/新增2953，净多2139；其余147正例2977→2842，净少135；总体9600→11604，净多2004并守恒。这是配对错序质量（同分半计），不是错图数，也不推所有困难样本方法必然无效。原汇总SHA `bd980cd44c7fab3675a270ba53c0999e174152baebebd1375bfb57f0bf35673d`，CPU23原guard SHA `b1d4336b27d04d3e1ed2f8856379e2fdb93af269016dd89423dc964b8328bc3d`；只用新selected CSV一次，无新图像/模型/官方或forward，名单未重选。以下为本轮原始启动记录。

2026-11-02 02:38:24.879743 UTC[首成功观测](R156-START.json)：running75 success=attempt、0skip、尚未内部评估。345个全训参数张量、5组空AdamW、whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`与R107及formal相同。首自然25记录loss `.7609652876853943`，WORLD8/G64P8、全部448pairs、每rank8图；owner486863/start361432472及八rank486882–486889的birth、argv、96核mask和8PCI身份匹配。没有独立GPU预演。

冻结runGit `874fefff678d99b2174df929331b9bb38f4b1ca6` / tree `c575ce4fb760000da02576815d1f016be57d9cf1`，SCI `ec65ee71c6a55c5009b9d79302462ae0fba48acc5030eee0b15b8b60a39fbfb9`，formal SITE `a65f8c59ae612b16cb7813a2938fdb98641a46714619a85098898974e1655876`。[必要6目标/6compile](../preparation/hard-positive-20261102/README.md)按精确SHA复用；[唯一正常formal](R156-PREPARATION.json)19/19、96.465秒，实际Git精确部署11.535秒、SHA `0ac6122801cb85b65c11c23c1731fe09e15f63afa7f8475dc005eed3b8c25a4d`，一次activation/direct launch exit0。部署原件路径与SHA及[实际保存job/准入几何和loss](R156-ACTUAL-GEOMETRY.json)均已核对，不用额外GPU hook。

本轮独立directR107/public fresh Small、seed29、RGB512，原采样/增强/age/AdamW/LR/WD/EMA保持；只让fit当前批内困难正例多获梯度：448个softplus配对先沿56负例求8个正例均值，再`.5全8均值+.5最大2个均值`，稳定WORLD正例顺序破同分。无dev名单/分数设权，无新forward/teacher或训练checkpoint。用户确认1=青光眼，不据此推根因或声称收益；完整总体内部AUROC为主，最多24k/800评估/p4/唯一online-EMA selector，完整selected≥`.996`才官方5849一次。

[接续间隔](R156-HANDOFF-GAP.json)：R154训练完成事件至dispatch220.662秒、至最晚八卡设备284.415秒、至首自然25记录352.513秒；前两项≤300，首记录>300。从整轮ended_at计分别187.493/251.247/319.344秒。没有连续利用率测量，连续空闲≤5分钟未证；不把设备初始化或较短基点冒作训练首成功。原clock/grant不重置，必要校验未跳过；正常formal与旧freeze收尾重叠，actualsafe后立即activation/direct launch。

唯一Git/GPU owner32e3，训练冻结24–119/96核；[aa0f准备池](R156-PREP-ALLOCATION.json)根0–22/23核、用户另开分析会话CPU23/1核，互斥且总24核，有效至07:33:47 UTC。各执行者自身任务串行，旧allocation不热改。当前入口`work/r137-20261101-a/r156-pod-context.env`绑定真实START4。启动证明代码可运行，不代表实验已完成；raw `.999`未达，继续训练及下一轮20%池准备。
