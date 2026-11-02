# R161/R162必要CPU准备

[CPU-CHECKS](CPU-CHECKS.json) SHA `0bff876fb74bf3e35c2a4d0f3b277d2835c8deb80ad3ffb2029c0eb6cc285b54`：3项新目标/7个受影响文件compile、0failure/error。独立标量值与解析输入/weight/bias梯度、缩放/零输入有限性、原对象/RNG、native factory/age一次/5组空Adam/真实CheckpointStore→inference冷图两项11.628秒；候选配置身份目标另3.834秒。旧None分支保持原native Linear，新子类只在显式policy下复用原参数，没有新buffer/可训练scale或新前向链。原缺adapter的RED exit1保留；[最终guard](CPU-FINAL.json) SHA `e1857dac486a7976722011ec26ebf94816269ce536080e58dabe8905a08d3218` exit0。

原6项loss/WORLD目标按classification_loss、两个测试与engine/distributed_engine/EMA实际SHA一致复用，未重跑。新配置仅R161原R160固定seed27/order73、R162直接R107/seed29/order74/原logistic+余弦头；配置目标逐job、source、geometry、保存合同及基线差异核对通过，原R160冻结3ef未动。

[实际两public fresh](PUBLIC-FRESH-SUMMARY.json)：R161 whole `91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67`；R162 whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`。均344完整public strict/exact/FP32 finite后345全训/5空Adam/0buffer，R162真实descriptor含CosineLinear/固定policy。此阶段未图像forward/读训练checkpoint/官方/GPU；核心冷图检查使用合成32输入，未冒作新512吞吐或真实图像证明。

固定归一化头参考[NormFace](https://arxiv.org/abs/1704.06369)，本配方仍二元原pairlog和age，不是其softmax方案复现；scale16/eps1e-12预先固定。参数身份相同不表示前向相同，也不证明临床机制或收益。完整AUROC判定、.996官方门及raw .999目标见[合同](../../../todo/auroc-999/R161-R162-PREPARATION.md)。

全部计算在aa0f根0–22/23核、nice19、隐藏GPU，与peer23及训练24–119互斥。预算仅复用原Small512满24k×2历史情景24154.892秒<soft27750秒，新归一化算术未作GPU benchmark，不当相同计算图或实测上界；不靠早停或改科学合同救预算。现CPU准备完成，待实际Git同步/回读与技术包；未新normalformal/grant/clock/activation/GPU。
