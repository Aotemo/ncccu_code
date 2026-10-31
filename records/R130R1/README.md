# R130R1：固定layerwise .8完整关闭

2026-11-01 04:44:14.857北京时间[唯一终态观察](COMPLETION-R130R1.json)确认completed/early_stopping：11200 success=attempt、0skip、14轮评估、连续4轮无提升。唯一selected为online c10@8000，内部dev AUROC **0.9850016436060114**、AP **0.7899286280804421**、error_twice24912。严格低于R107 `.9884405731067525`及R120 singlebest `.9894773092062407`，两门false，关闭这个固定.8方案/no seed27，不扫邻近factor/LR/group/seed；不推所有layerwise方法普遍无效。

完整TRAIN/ROUND/done均completed，TRAIN exit0/signals[]。完整selected低于`.996`，ROUND正常记录official not_run_by_policy/threshold_not_met/evaluatedfalse/metricsnull，单独official文件未见；未重复官方或前向。[RESULT](R130R1-RESULT.json) SHA `48aab334daa7557bf4810c858386c855215e5c0cc24189068067020abbcf885b`；COMPLETION SHA `b7a94a0ec8b2244c10fee62809d082c2e88cd800ee69e77f55ecfcddddc0e69b`、[CPU原件](TERMINAL-CPU-RECEIPT.json) SHA `29b4cfa20ec0db9c7316899ff2aab2180dc0336d5a4136ff44a3a09b125df40a`。

同次actualsafe=true/terminal_idle：owner333936/start350121193匹配现Z（不称reaped），8rank333948–333955全absent；lock holders[]、device_users[]、scanDenied0，8个expected PCI全seen/busy0。未为归档重查safe、hash/load checkpoint或重跑fixture；freeze selected的tensor SHA `6c921246106f352c7bbdcbf9ab86715e6f8036f719a672fe7301b24adaa23e7d`来自实际小件，不冒checkpoint blob重hash。train硬deadline字段和freeze顶层blob SHA未单独抽查。原R1301503及所有依赖、R120/KEEP继续保留。

后续按授权采用唯一[R131固定pointwise-r8](../../todo/auroc-999/R131-DECISION.md)，直接R107、公开fresh、独立身份；不继承本轮倍率、initial或Adam。以下保留本轮准备/启动阶段来源，旧“尚无完整结果”仅为当时状态。

用户授权按R130固定方案从头开始，本轮以order41独立身份运行。runGit为 `bc5d788b51bb2d1be3a9f76f6088e7c3cc3aabdd`，SCIENCE SHA为 `85549dfc4387681e9c81b3d6166bbbd7cd0be5bc51b028c723616ef749daa2ab`。原R130的1503步中断与产物保留，不用于新初始化或拼接曲线。

[工程准备回执](PREPARATION-RESULT.json)对应一次normal formal 20/20及673个源码文件的实际Git精确冻结；该阶段尚无clock/GPU。[启动回执](START-OBSERVATION-R130R1-1.json)为2026-11-01 03:12:11.386北京时间的后续实查：owner333936/start350121193及8rank333948–333955均身份匹配active；真实native初态345tensor/0BN/30组/空Adam，随后8rank自然首成功step1的WORLD finite、实际LR消费和345个Adam状态step1均已证明。此时status.json仍是running0/eval0初态快照，实际首成功来自metrics，不能将它解释为0个真实更新，也不推测25/125步。

首次外层运输漏传stdin，返回0并未调用真实入口；[修正前缺失证据](EMPTY-TRANSPORT-ABSENCE.txt)确认13项真实调用/变更标记全absent、activation/launcher此前均0。保留原命令和日志后，仅修正运输，完成第一次真实activation exit0与正确live launch exit0；没有重复activation、重新计时或改原R130。完整原件保存在exp-v3的 `technical-prep/r130r1-20261101-7c17`，private clock/token不入Git。

本轮准备/启动时尚无完整结果；最终selector、两门和安全状态以本页首完整收口为准。合同与来源见 [R130R1决定页](../../todo/auroc-999/R130R1-DECISION.md)；根的唯一heartbeat `9c70b2bc`继续协调后续轮次，不重复本轮completion。冻结runGit与原资产不随归档改变。
