# 原生 ViT-B 骨干低学习率对照准备

[CPU汇总](CPU-CHECKS.json) SHA `46db60d116225a4e579a96f1fce4278b4ef1a67c0a537ef15af963bce675c397`：4必要目标、3份受影响源码编译、两actual public fresh通过；[最终guard6g4cx0gz](CPU-FINAL.json) / `1cc944acb2a1dd9acbfa2c8d3ff465782b22def60eb38b5ed7336c8c2e1c1eae` exit0。当前仍未Git冻结/技术包/formal/新clock或GPU，不把本准备写成已经启动。现役R199/82与既定R200/R201/8635冻结不改。

[先RED](CHECKS-RED.json)确因新候选尚未登记及R203仍落默认12GiB失败，两组旧候选目标先通过；最小登记/精确新round policy后[四目标GREEN](CHECKS-GREEN.json)。新身份逐项比较R202对R201仅seed、R203对R191仅backbone_lr，公源/结构/9:1/日程/早停/官方门及saved job保持，拒绝错误seed/LR/WD/loss/光度/衰减跨度/pool及来源。原预算函数fixture覆盖R203及旧16GiB、R202及旧默认12GiB、超cap及free15拒绝；所有旧32GiB键值另按policy完整相等核对。只给R203精确新round追加16GiB，与R191相同工程余量；默认12GiB/free16GiB及原runtime预算源码不改。不是实际写17GiB或峰值容量保证。

[两次真实公开fresh](PUBLIC-FRESH-SUMMARY.json)：R202 whole91507a65/345全训/5空AdamW，R203原wholec3495f5a/165全训/5空AdamW、162公开骨干张量逐件strict/exact/FP32。实际五组LR分别`[1e-4,1e-4,1e-3,1e-3,1e-3]`与`[1e-5,1e-5,1e-3,1e-3,1e-3]`，WD均`[0,1e-4,0,1e-4,1e-4]`，默认betas/eps与空状态核实；R203唯一live RoPE buffer仍非持久，原CLS/5prefix/512模型描述保持。0图像forward/无旧checkpoint或官方读取。

只改原experiment91两身份/新profile严格LR绑定与一条存储override，另两SCI和原candidate/storage窄测试。新SCI为R202 `c553710a847697d6bf68ce971e98457c81652a3e2f92c1af3924562cca326f25`、R203 `33054ab03a6e88a3999e348359bcfff8516a816d8489fec2128e9ca1055e14c8`。原模型/EMA/fresh/保存/推理及非持久buffer测试和R191证据SHA相同；config/engine/checker沿后续已通过的精确来源证据，不宣称其整个模块仍与R191相同。原wide loss数学、旧loss值/梯度与真实2rank Gloo按34706b04证据及相同源码复用，不重跑未变P8/EMA/cold/WORLD或加GPU预演。新候选完整producer仍待normal阶段一次实际运行。

R202仅完整有效R201严格胜R107/R120及actualsafe后复核；R203为独立direct R191，仅骨干LR十倍降低，其他全训/原优化/24k余弦均保持。[预登记假设](../../../todo/auroc-999/R202-R203-PREPARATION.md)不是发布者微调配方或已证明特征漂移/临床机制/收益，不扫描LR。私有原件在`exp-v3/technical-prep/vitb-low-lr-20261103-32e3`。逐次UID/Ready门、53ab/root0–11/nice19/隐藏GPU，与R199训练24–119互斥；无新Agent或资源，raw .999未达。
