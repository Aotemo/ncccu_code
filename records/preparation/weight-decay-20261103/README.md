# 原生 AdamW 固定 weight decay 对照准备

11-03 02:39后备R199已在R197训练期间完成[正常CPU18/18与精确部署](NORMAL-R199.json)，[guard6_i19uxd](NORMAL-R199-CPU.json) / `69ca7035b70c3002daf8498aa79bb07538871890df4511de6b9cb8b8865da054` exit0。原3必要目标/3编译按SHA复用，完整producer实际通过；345全训/5空AdamW/whole50c6，实际SITE `a1aa771046388485335260bd0fff66f2a5348cef9380ed601b58f5924d5d5cc2`，manifest b859b750、CPU-prep3899d844、deployment af75051d。Git82add93/tree89c2231、SCIcc0e2703、默认12GiB/free16GiB不变；尚无新clock/activation/GPU。只在完整R197决定分支并actualsafe后接续；下文技术包及待normal描述为历史时点。

00:58两[actual technical包](TECHNICAL-PACKAGES.json)已按GitHub实际回读commit `82add932e59e61f8eb3e8a7f0a4b4cfe1dfd454d` / tree `89c22313619ddadd1e24139ef6d3878fa8b871e5`生成。R198 SITEbc0058c4 / [guardsmw0mnep](CPU-TECHNICAL-R198.json)与R199 SITEfd68bf41 / [guardbmquvwro](CPU-TECHNICAL-R199.json)均0；归档dwxebtq8 / `795a7c3344c4f8158c49b87c05704efb69d541669716f9e2aba1ed43a36a9bed` exit0，实核无formal-cpu、gpu-inputs或真实round目录，没有新clock/GPU。旧R195/693及先行R196/R197/436冻结不改。

[CPU汇总](CPU-CHECKS.json) SHA `cfd51c12e8a30d46110abfed7ba1e520a5813120c2c5d4cb58d77818ec5e637a`：3必要目标通过、3份受影响源码编译、两actual public fresh；[最终guard](CPU-FINAL.json)z_kwsql7 / `09ff968abf04e3d7c9066ff5412b225c13fec3ba10d0fdf5fbaf8b1b0068ec3d` exit0。逐次UID/Ready实核、53ab/root0–11、nice19/隐藏GPU，与活跃R195训练24–119互斥；没有新Agent或计算资源。

[三个目标](BINDING-GREEN.json)覆盖R198原R197 gamma的条件seed27身份、R199独立R120仅job.weight_decay=.05及candidate显式身份、保存job合同、来源/原9:1/日程/早停/官方门保持、错误叠加拒绝、旧R196/R197兼容。真实原生_optimizer配Toy参数对象映射核五组、原LR/betas/eps/空状态/RNG；所有参数赋零梯度，实际一步与独立`old*(1-lr*expected_wd)`逐参数exact对照。不是生产八rank恢复证明。

[公开fresh摘要](PUBLIC-FRESH-SUMMARY.json)：R198 whole91507a65 / SCIed53707e，R199原whole50c6 / SCIcc0e2703，均345全训/5空原生AdamW/0图像forward。原顺序为backbone无衰减、backbone衰减、head无衰减、head衰减、age；R199实际WD为`[0,.05,0,.05,.0001]`，R198仍`[0,.0001,0,.0001,.0001]`。原engine、config、data、model/loss/EMA/DDP/保存/推理均未变，原Gamma像素和模型/P8/EMA/cold/WORLD按精确SHA复用。新checker条件仅编译及静态核对；完整producer待normal准备一次执行前，不称已通过。

原RED为新候选尚未登记，其余两个目标已通过；SSH返回255未带输出，但[原回执恢复](RED-RECOVERED.json)已核原aeuwxpt9 exit1、整树退出、无affinity违规及OUTPUT SHA，没有因传输中断重跑测试。GREEN为52lqaj_l / `f09ff1124ee80825141bcc9e7ce39116a5a0a16e5ceedb8e34aafbc62364accd` exit0。首次归档51myri0x exit1因汇总误写组顺序，原件保留；只修汇总预期、不改生产源码或重跑测试/fresh，第二次归档通过。

永久源码只在原experiment91登记/校验与原check_candidate的WD消费条件最小修改，另两SCI、两个候选测试目标及文档。私有原件`exp-v3/technical-prep/weight-decay-20261103-32e3`。固定.05是独立参数收缩假设；作者通用默认与其微调示例不同，参见[预登记计划](../../../todo/auroc-999/R198-R199-PREPARATION.md)，未证明过拟合、临床机制或收益。

本组没有normalformal/新clock/GPU，正常CPU/精确部署仍于前轮可行收尾前一次执行。先行R195/693与R196/R197/436不改，R198仅完整R197严格双门通过且actualsafe才GPU，否则独立R199。默认12GiB/free16GiB、24k/800/p4/EMA1600保持；完整内部selected≥.996才官方一次，raw .999未达。
