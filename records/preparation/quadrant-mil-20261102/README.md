# R171/R172 固定重叠局部视野准备

R170运行冻结58bc761保持不变。本次只准备其后的R171条件同CLIP Base seed27与独立R172 Small/fe6/seed29。R172沿原R107输入、pair-logistic、G64/P8、AdamW、LR、EMA及age，仅在同一共享骨干内聚合全图512与四个重叠288角窗：`.5*global + .5*max(local4)`，age在聚合后一次。窗口来自同一最终增强/归一化张量，不重采样，不增加有标签样本；没有新增参数、buffer或初始化随机采样。保存job及冷构图保留版本化policy，不能仅凭相同张量SHA认为函数相同。

[CPU-CHECKS](CPU-CHECKS.json)记录3个受影响目标和6源编译；两项模型/数学目标按同源码复用刚通过的[CORE2](CORE2-CHECKS.json)，配置绑定目标最后实际执行通过。独立解析值、x/w/b梯度及真实同分梯度、原生None/RNG/参数身份、5空组、EMA、P8反传345个有限梯度与真实CheckpointStore→inference冷图一致均有断言。实际P8内部调用形状为8×3×512×512与32×3×288×288；age调用一次、未做optimizer更新、未初始化GPU。原loss/WORLD的6目标仅在engine、distributed、loss、EMA及相关测试源码完全相同时复用，没有新增GPU预演。

[公开fresh汇总](PUBLIC-FRESH-SUMMARY.json)为两次实际加载完整344张量并建立345全训/5空Adam组，原生betas均`.9,.999`，不读取旧训练checkpoint或图像、不forward。R171初始whole为`2d6f2eefb533ff0976071648b10c39ca91a6983c8155241201840ceb74d2512f`；R172仍为原R107的`50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`。R171仅在完整R170严格胜两个既定锚点且安全后有资格；否则独立R172，不能从暂时best决定。

保留真实失败成本：[RED](RED-CHECKS.json)两FAIL/0ERROR来自helper/policy尚未实现；[首次CORE](CORE-CHECKS.json)的两项断言虽通过，CPU整树peak runnable45超过分配23，原guard exit2，不能称该次整体通过。仅把CPU fixture改为11 intraop/1 interop、被动OpenMP等待与串行autograd后，原同源码两目标201.307秒、[guard](CPU-CORE-GUARD.json)exit0；训练80%池和科学合同未改。最后配置检查第一次运输因SSH登录节点根盘满而无法创建heredoc，尚未进Pod；随后直接调用同一Pod CPU guard，10.963秒目标通过且[最终guard](CPU-FINAL-GUARD.json)exit0。没有清理其他任务文件或跳过校验。

CPU P8保存tensor storage去重16,694,105,280字节不是DCU峰值或吞吐证明。输入像素面积比2.265625也不是性能上界：按既有R168 first25→6400完成元数据粗估24k为25,851秒，×1.25情景为32,314秒，超过原train hard27,900秒；原时钟/准入不扩限，完整24k可能受deadline截断。checkpoint形状不变，磁盘存储预算与VRAM分开核对。临床ROI、根因或AUROC收益尚未证明，完整内部AUROC为主，官方隔离及raw .999目标保持。

源码实际Git为`1859670356f1068ce57726b6e072485959154d4b`/tree`216951f23b3bb81e2c3a26357dc90a750c210d55`，已push/回读并精确部署。[两技术包](NEXT-READY.json)与[CPU整树](CPU-TECHNICAL-READY.json)exit0；独立后备R172在R170训练期间完成[正常formal18/18与部署](R172-PREPARATION.json)，一次28.855秒，[guard](R172-CPU-PREPARE.json)exit0，345全训/5空组/whole50c匹配。[受影响CPU证据精确复用](R172-CPU-REUSE.json)，没有重复跑已通过目标。尚无新grant、clock、activation或GPU；完整R170双门与actualsafe后才选唯一分支，R171尚未normalformal。

既有peer仅CPU23静态窄核六源码实际SHA一致，未发现阻断；没有独立复跑测试/模型/数据。其guard原件为dca3任务`a4e26231-4059-407f-b171-98e65362ec47-_5ch4zzd/CPU-RECEIPT.json`，SHA`43837689936a1b4ec4bf1f051d6fa2f89f090c03be7d582e5350121ce001f264`、exit0。初始化RNG保持的证据不代表训练RNG轨迹相同：额外一次network调用会消费原Dropout/StochasticDepth随机数，并改变实际计算和随机正则化过程。

原件在exp-v3独立`technical-prep/quadrant-mil-20261102-32e3`，准备dca3根0–22/23核、peer23/1核与活跃训练24–119/96核互斥。R170冻结58bc761不因这次源码/准备更新而改动。
