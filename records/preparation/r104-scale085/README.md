# R104：训练尺度下限单轴接线

直接R103 Small/seed20261029对照，只新增`job.training_scale=[0.85,1.0]`；模型工厂、公开资产、seed和整体初态控制不变。历史未配置job不增加此字段或改变checkpoint身份；eval不使用训练范围。两训练入口消费同一字段，其他增强调用顺序与抽样数不变。

真实Pod CPU6 A份额完成5项必要检查：3项配置/绑定/旧身份检查与2项synthetic像素/RNG/eval检查，全部通过。旧默认与显式旧范围像素、RNG状态一致；新下限可达几何变小，抽样次数与后续状态不变；验证像素不变。技术包实际传递范围，但不构造模型、不前向、不初始化GPU、不创建正式CPU-PREP或clock。

[CONFIGURATION-CHECKS.json](CONFIGURATION-CHECKS.json) Pod SHA `55a06ab6481475511fe756ccf3bd119f604ba30e74b5a815441ce9d0cfc735c3`；[CPU-RECEIPT.json](CPU-RECEIPT.json) SHA `d8a0c565e246304a9ade9bc7624da16715c7cdf61e0388e8efdc654724e1ea14`。源码在检查期间未变，nice19/隐藏GPU/mask0..11/intra12/interop1。SCIENCE SHA `bc54059304f53043506a01c94b77b91e72d273e558e3168aa603d572a0c050d5`。

原件根为`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/r104-small22k-scale085-20261029-a`。只普通导回；未在本机解析、重哈希或运行测试。后续实际新Git原生正式包自然核对不变初态50c6f55c；本次不另建control-only模型或独立GPU预演。
