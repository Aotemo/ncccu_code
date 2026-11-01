# R137 固定 Lion10x：完整结束，未达目标

9600次成功=尝试、0跳步、12轮评估、连续4轮无提升正常早停。唯一selected为EMA c8@6400，内部dev raw AUROC **0.9854411426493483**、AP **0.8124829988951043**、error_twice24182，tensor SHA `47cbfd290f6a0a7543cba13a1439fa1e10ccc9b6bb3e757441c68bca5eaebb77`。strict>R107及strict>R120均false，关闭精确Lion10x、不做seed27；预留R138未activation、未训练。ROUND官方按`.996`门未达正常跳过，metrics=null；没有官方前向。

唯一完整终态观测为2026-11-01 15:11:29.724784 UTC：done/TRAIN/ROUND completed、exit0/signals[]，actualsafe=true/terminal_idle。owner同birth Z非reaped，八rank absent，同inode锁holders[]、设备users[]、denied0、8expected PCI seen/busy0；freeze.status原值null保留。见[结果](R137-RESULT.json)、[完整终态原件](R137-COMPLETION.json)和[观测CPU树](R137-CPU-COMPLETION.json)。归档只复制已有小件，没有重复safe、hash/load大权重或补前向。原selected与全部clock/grant/失败成本和依赖保护；后续已接R139。

以下为启动时实证；当时“尚无完整结果”等状态由上方终态覆盖。

2026-11-01 21:53:55.416945 +08:00观测为running，250成功=尝试、0跳步、eval0。首自然log为update25/attempt25，finite loss `.6603637933731079`，WORLD8/G64/P8/448pairs。真实八rank初态全部为Lion、固定policy及有效age合同、原345全训/5组/空optimizer state，whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，与本轮正式CPU验收逐组匹配。初态为空不代表更新后仍为空；自然rank_inputs投影未导出optimizer字段，后续moment数量未观察，不热改当前运行或补step1/GPU探针。

冻结runGit `7b48f495ecb85664037a03015b78216b7d7a31e9`、tree `7e1314d833b668172884532166d6fc870d7c4df5`，SCI SHA `a972186db69f98373f77356af728b0b5fa04483a350821fa68a6082a6df183f3`，SITE SHA `260b0814b4ad10a912530c5d1a3c40fb25530fe6ab22955ffe8b0f5c717c18d6`。原三项受影响CPU目标与8源compile经精确源码/SCI/原回执及输出SHA匹配复用，未重跑；本轮唯一normalformal 19/19（96.751秒）、实际Git blob/mode精确部署、一次activation及一次direct live launch均exit0。不存在独立GPU预演。

owner398779/start356837700和八rank398791–398798均实际PID/start/argv匹配；8PCI/64GiB级设备及rank CPU mask匹配。实核effective120核，训练96核24–119（control112–119/ranks24–111）、prep24核0–23互斥，本执行owner使用0–11，12–23保留不用。当前唯一Git/Pod/GPU执行Agent为`32e3b079-81b0-4f66-9604-b32491599f44`，未恢复旧Agent；只读入口为`work/r137-20261101-a/pod-context.env`，真实first-success为服务器`START-OBSERVATION-2.json`。原件位于`exp-v3/technical-prep/r137-20261101-32e3`，活跃冻结包和所有旧clock/token/grant/保护资产保持。

接续未满足≤300秒目标：[真实间隔](R137-HANDOFF-GAP.json)记录R136安全终态观察到新dispatch为6126.768秒，含本次接手前等待。该数是两个有凭据时点之差，不是连续设备利用率测量。本次有Git作者设置、SSH握手/传输等待及必要正式验收；没有跳过校验或虚报空闲达标。失败传输日志保存在本机`work/r137-20261101-a`，未执行第二次formal/activation/launch。

当前尚无完整内部selector、双门或官方结论，raw .999未达到。唯一科学合同见[R137决定](../../todo/auroc-999/R137-DECISION.md)；完整selected同时严格胜R107 `.9884405731067525`及R120 singlebest `.9894773092062407`才固定seed27，否则关闭精确Lion10x；完整freeze≥.996才官方5849一次。

小件：[启动](R137-START.json)、[正式准备](R137-PREPARATION.json)、[原CPU检查](R137-CPU-CHECKS.json)、[复用证明](R137-CPU-REUSE.json)、[正式准备CPU树](R137-CPU-PREPARE.json)、[activation CPU树](R137-CPU-START.json)。原始数据/public/prepared512、R130全部、R120/KEEP/R134R1/R135/R136 selected与全部失败成本继续保护。
