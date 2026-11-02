# R188/R189 原生 CLS 读出准备

[实际CPU汇总](CPU-CHECKS.json)：4个必要目标、5个唯一受影响源码编译均通过。原平均池化对CLS独立oracle的首项先真实数值RED（差0.50113356），最小实现后两个模型目标25.617秒通过；因描述函数新增policy，再跑共享RoPE/真实EMA/freeze冷图回归6.131秒通过；两新候选身份目标3.592秒通过。没有重复旧P8或扩旧全套。

原生timm1.0.27实际`forward_head(features)`与独立`features[:,0]` exact；新工厂与同seed原生参考的全部张量、头与RNG一致。只在原avg构造/完整public strict后设global_pool='token'，保留原final LN/fc_norm Identity、全部5个prefix及1024patch参与注意力，不重建参数/采样。默认None/省略的state/RNG/描述一致；真实load_config接受非None并省略None，job_contract保留新policy，真实CheckpointStore→inference冷图恢复token/readout/固定RoPE且输出exact。P8仅一次network forward、165有限梯度/5空Adam、真实noBN EMA首更新及age一次已测；没有额外GPU预演或DCU吞吐/峰值证明。

R188/order101条件meanpatch seed27：[公开fresh](R188-PUBLIC-FRESH-INITIAL.json)whole8c62435a、SCIb21417ee3a2b6b29ef5061a303cbe7a680592c8a47fcea270fd0e58209ca4c42。仅R184R1完整严格过R107/R120且safe才formal/GPU。R189/order102独立CLS/seed29：[公开fresh](R189-PUBLIC-FRESH-INITIAL.json)whole e347757c与原meanpatch相同，但保存policy/候选/描述明确函数不同；SCI11cf4bc7bc85822a6aa79a8670a4f09f232f4698f46db29ab391652b79747eb6。两者完整public162 strict/exact/FP32、165全训、5空Adam和真实noBN EMA首更新通过，0图像forward/0GPU/不读旧训练权重或官方。源3bf4720a/原b7e7b7d1/provenance6193971f及4个timm源码pin保持；不称逐文件pin整个transitive依赖包。

只改末端聚合，仍原512/ImageNet/原logistic/G64P8/AdamW/LR/EMA/age/固定9:1/24k-800-p4。原R186和R187/R184R1的f9/cd冻结包不改；[顺序与假设](../../../todo/auroc-999/R188-R189-PREPARATION.md)保留。官方源码的CLS预训练用途仅为假设依据，未证明临床机制或收益，亦非官方CLS+patch拼接分类配方。默认12GiB/free16GiB及官方隔离保持，CPU合成证据不代表GPU吞吐/峰值或AUROC。

根5de CPU0–11/nice19/隐藏GPU，与训练96核互斥。最终guard3b9l860x exit0，SHAa9b25c1c065d2c96a5a1c5690369af190ffc729d0f9a9f137c1ef459890beed7，原件在exp-v3/technical-prep/cpu-renewal-20261102-1651-32e3/tasks/32e3b079-81b0-4f66-9604-b32491599f44-3b9l860x/CPU-RECEIPT.json。零SHA仅留私有draft，Git配置为最终fresh实测值；后续Git/技术包及正常CPU/实际八rank分别记账。
