# DeiT III 固定读出对照准备

[CPU-CHECKS.json](CPU-CHECKS.json)记录4个受影响目标、5源编译通过：新meanpatch两目标、原生默认CLS参考一目标及R153–R166配置绑定一目标。先观察缺policy的RED，随后真实CPU P8梯度/5空Adam组/EMA、原生与独立patch均值参考、同seed参数/RNG及默认None兼容、实际保存冷图均通过；原6loss/WORLD与未变推理/验收/来源代码按SHA复用，旧工厂函数原文亦核相同。最终guard `84c1fe0698022096fb6de3c0e29e91f07da0394a9d7b9333418f1d587c37cdeb`。

R165/order77仅在完整R164双门严格通过后复核原CLS seed27，实际fresh `8108ef45b018a9b0049084d90b13d1cfe7ec3a357c72e38059b79e0af1c79caa`。否则跳过，独立R166/order78使用同public8d59b281/512/原logistic，仅把原生最终LN后的1024patch平均（排CLS、fc_norm仍Identity），接原Dropout.2/Linear/age一次；没有新增参数或forward。其fresh仍 `64556b75f7fb9b0f1946089b5a0115de38348f049aba6252b49f3cd8bf5c9a93`，张量相同而函数不同，policy随job保存并由冷factory核实际图。两者均177全训/5空Adam/完整176 public严格加载，见[PUBLIC-FRESH-SUMMARY.json](PUBLIC-FRESH-SUMMARY.json)。

公共原件/许可及原生库身份复用[已核来源](../deit3-20261102/README.md)，不新下载或转换、不读真实图像/官方、无teacher/ROI/dev名单。既有12GiB默认预算、设备下限与24k-800-p4/时钟保留；旧CPU存储/历史时间情景是参考，不是新DCU峰值或吞吐上界。R164的f061冻结、GPU分配与clock不改；原生CLS已实际八rank成功不等于新meanpatch的GPU证明，后续仍沿现有唯一正常formal/精确Git部署和一次启动，由真实初态/自然更新证明。完整内部AUROC主判，AP或难组不救；无已证临床机制/收益，阴性不扫描pool/权重/温度/seed。
