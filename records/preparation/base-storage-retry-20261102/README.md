# R173/R174：Base 存储 headroom 与固定科学合同复验

R172继续冻结Git1859670；本次仅准备条件同MIL seed27的R173，以及原R170科学合同delta=0的独立fresh技术复验R174。R170原6144步中断/无完整selected，不判科学阴性，不恢复其checkpoint/clock。新增工程配置只给R174新round登记16GiB，沿同尺寸R116既有例外；默认12GiB、失败R170与当前R172、free16GiB底线/VRAM85%不变。训练、模型、保存/prune、评估与监控实现未改。

[CPU-CHECKS](CPU-CHECKS.json)包括两项实际新执行目标：扩展SCI/candidate身份与原存储guard，11.919秒通过；3个受改Python源编译通过。原两项模型目标及6loss/WORLD目标按实际使用源码完全相同复用，**4个通过目标中2个为模型证据复用**，不是重复跑P8。新policy验证R116/R117/R174的13GiB可用、17GiB拒绝、free15GiB拒绝，并验证旧R170/当前R172及其他round保持12GiB。原存储[RED](RED-CHECKS.json)只有新R174缺例外的一FAIL/0ERROR，[guard](CPU-RED-GUARD.json)exit1；修复后[最终guard](CPU-FINAL-GUARD.json)exit0。

[公开fresh](PUBLIC-FRESH-SUMMARY.json)两次实际加载完整344张量并建345全训/5空Adam组，betas均`.9,.999`，没有图像forward/旧训练checkpoint/官方读取或GPU初始化。R173实际whole`91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67`；R174实际whole`5d2ebdcfb799ed587d48dd680a4ce8540d9b29159c9944e4f452b18f6c6a56b2`，与原R170相同。配置目标同时逐项比较R174与R170完整science（除profile/候选id/order/说明），确认输入、LR、loss、source及初态没有叠改；R173只改原R172的seed/身份。

[存储情景](STORAGE-HEADROOM.json)只读既有stat审计/小JSON及相同保存源码：事后round11,951,892,576字节，加一个现有最大完整blob1,401,945,034，原子写后prune的情景为13,353,837,610字节，超过旧12GiB但落在新16GiB内。last-good原资源回执free43,095,425,024/size11,951,887,320；这些不是故障瞬时峰值重建或严格全程容量上界。当前R172与旧R170没有改policy/清理现场，磁盘预算和DCU VRAM分开。

原件在exp-v3`technical-prep/base-storage-retry-20261102-32e3`。准备dca3根0–22/23核、peer23/1核与训练24–119/96核互斥，nice19/隐藏GPU。下一步同步Git、构建两技术包并在R172训练期间完成后备R174正常CPU/精确部署；尚未新normalformal/grant/clock/activation/GPU。完整R172选点才决定R173资格，否则独立R174；raw .999未达。
