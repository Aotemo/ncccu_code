# 早两stage local8 MixStyle：CPU技术准备

本阶段仅结果无关CPU准备，未注册新SCI/profile/order/round/grant，未normalformal、public fresh或派GPU。R133保持冻结runGit `a6f2e72e887ac57125a720ffbfb9ea76730162f4`；完整双门胜仍seed27优先，[NEXT授权与固定条件合同](../../../todo/auroc-999/NEXT-PREPARATION-20261101.md)是接续入口。CPU通过不自动采用此候选。

两项目标的执行版本分别保存，未将它们描述成同一次最终2/2全跑：

- RED两项FAIL、0ERROR，原因是固定helper尚不存在。
- GREEN-1中公式/解析梯度/私有RNG目标通过；保存回放目标在`EMA(model)`遗漏现有API必需rule处发生fixture ERROR（0FAIL/1ERROR）。原件SHA `c2c7a284ebcb536e220d29dd1e767cf8815c920d3649480e0843391aea53abcc`保留在摘要中。
- 仅修第二目标两行`EMA(model, RULE)`，GREEN-2只重受影响目标：1/1通过、0error、6.862秒、9源compile。SHA `97de50c8ea315242a236b2dc7724109606744d267c722db0fdf0c1cd3a815471`；已过第一目标未重跑，8生产源两次保持同SHA。规划一次只读审查未见本范围Critical/Important或待修Minor。

[CPU-CHECKS.json](CPU-CHECKS.json)保留两次真实统计、源SHA及两行fixture差异验证；[RED原件](RED-CPU-CHECKS.json)不覆盖；[CPU-GUARD.json](CPU-GUARD.json)为GREEN-2原guard（SHA `c90334b9a139116409f651df44b05e956b940da1098387916858fb5b4fbe8f77`），672绑定/A12核0–11/nice19/隐藏GPU/exit0，root12–23保留未用，与训练24–119互斥。[OPERATOR-AND-REPLAY.json](OPERATOR-AND-REPLAY.json)提取两目标实际打印的数学/RNG trace与保存回放证据，保留原stdout路径/SHA。

数学参考由独立Python标量空间均值、HW−1方差和解析梯度推导，覆盖zero variance/eps及partner-stat梯度为0；生产FP32分开mean/var并detach mu/sigma。私有hash/Random的gate、8Beta后shuffle在attempt/rank/micro/site上与独立stdlib oracle一致，含gate关闭的合法key；None/eval helper不消费全局Python/NumPy/Torch RNG，不混图像/标签/age。

第二目标真实经过配置、job_contract、CheckpointStore、`inference._load_model`与显式早stage小toy_factory，attempt6/success4恢复到caller context后，两处训练行为、输出和trace回放一致；online/rebuilt/原EMA模型eval均不混合。None原值、输入/参数梯度、原全局RNG退出状态及canonical identity保持；原SD/dropout的合法Torch采样不被误写成训练全程零RNG。native `initialize=False`单独核原345参数对象/名称/值和0buffer，没有public参数加载或完整Small图像forward。小factory保留原前两stage CNBlock、简化后段，不是完整公开模型、正式profile、完整Adam/EMA更新恢复或独立export交付实证；默认nonNone production hook仅经源码审阅。

context/trace是caller每attempt新建的临时对象，模型仅存固定policy/seed常量，无hook闭包、持久context/generator或额外parameter/buffer。只在原log_every25-or-skipped分支经原ctx.values进入rank_inputs，未增event、collective或forward。DDP kwargs/空trace语义仅有上游源码静读，未执行BW真实8rank；不外推复用非空trace或多micro。所有现有正式profile均禁止非空policy，None走原签名/native图且从配置/保存身份省略；单卡未支持入口明确拒绝。优化器、EMA、保存和通信算法保持。

按固定512输入及实际早stage通道，per-rank P8的site1/3单个全特征FP32张量分别`50331648`/`25165824 B`；两处均激活时`75497472`次全特征逐元素标量算术，各site另有独立mean/var归约、BC统计及lambda32B/perm64B设备传输。这不是峰值、额外峰值、FLOP总量或BW时耗实测，不含归约实现/backward/workspace/并存张量；完整24k+30双dev+save、吞吐、峰值及100ms仍unknown，不靠早停承诺或扩预算。无已证domainshift，早层统计可能含标签语义，local8类别偏差和历史负先验见NEXT；不承诺收益或raw`.999`。

必要源码与小件按授权同步，精确远端源包9pin核验原件留独立technical-prep，不为自引用另提交。原R1301503/R1/R131/R132/R133/R120/KEEP所有资产、clock/token/grant和运行包保持保护。
