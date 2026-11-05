# Fit-only OOF 八 rank 入口：独立合成安全审阅

范围只新增 `tests/test_f05_oof_owner_safety.py` 与本说明；`f04_gpu_full_dev.py`、`f05_oof_gpu.py`、`configs/F05-OOF-GPU-20261105.json`、`tests/test_f05_oof_gpu.py` 均由 08ac 维护。该助手不派 GPU、不读取真实 OOF 或图像、不执行官方前向。每折必须依固定顺序各运行一次完整 fold-train 标定人群和 heldout 人群，不以仅 heldout 代替完整合同。

## 合成检查范围

- standalone 必须在 soup 与 OOF 之间恰选一种，拒绝任何官方/最终链参数；验证前轮安全终态、有效期限、peer 不同 Pod 与身份、peer 未终态时保留完整容量、容量边界和过期回执。
- 模拟八 rank 真实身份、device proof、parent 与 release；缺 rank、重复、设备错配、错 parent、启动信息/证明篡改均拒绝。此处不声称做了真实设备探测。
- 用明确 fixture 标记的三折、两角色及 online/EMA 合成包，让 producer 的 `make_scores` 对比 consumer 实际 score schema 并进入 `verify_fold_input`；核 train/heldout 不交叉、不含 dev、heldout 跨折覆盖 fit 恰一次。对选点、rank、fold、role、proof、TTA、严格加载和 fp32 声明，以及缺/重复/多余/错误行、NaN 均构造失败情形。
- 用合成 checkpoint、sidecar、site、runtime 和终态文件测试 owner 资产验证路径，故障包括错误格式、checkpoint 或本地副本篡改、来源文件改动及安全状态/折身份不符；其中庞大实际 population 用合成占位并模拟 roles/bound verifier，单独的三折角色测试负责真实逻辑。
- `prepare` 假调用在 execution UID 不符时应于写入前拒绝；唯一 selected 枚举覆盖缺失、重复、checkpoint/sidecar 孤儿、文件或目录链接、扫描报错和非法 JSON。未证明真实历史目录完整性，须在正式准备时重新核实。
- CPU 假 worker 用本地小型合成张量、伪 selected load 和假设备/guard，在模型身份或 dtype 错误时断言尚未访问数据；正常路径核 rank 0 两角色分片及 packet。另以伪反序列化载荷和小模型调用实际 CPU loader，检查缺失/多余 tensor 时 strict load 拒绝。没有实际图片读取、真实模型加载或 GPU。

## 当前可见的接线缺口

只读审阅 08ac 尚在修改的工作副本时，`f05_oof_gpu.py` 已陆续添加 inputs 准备、两角色 worker、双 score 聚合与父回执，`f04_gpu_full_dev.py` 正在接线。具体完成状态须以其提供的精确源码身份和集成测试证明；实际 OOF 数据尚未读取。

## 隔离验证结果

2026-11-05 B Pod UID `37e3fb34-c233-4294-b517-e98978cbd016`，12–23 mask、nice 19、隐藏设备。以已核 d559 源码树复制件为底，叠加 owner 的 `1101` 草稿快照和 d3ceb81 指定的五个 consumer 依赖，**不是完整发布树**。owner 的合成测试 **7/7**、本助手独立安全测试 **14/14**，`GPU_initialized=false`；guard 回执 SHA `1777673bf71d7c33f1421d6693bfcf4187a453b1662e541c5cb36e2017b6658e`。包含实际 `aggregate` 生成双 score/父回执后交 consumer 的纯合成测试、CPU 假 worker 和假 checkpoint strict load；没有真实图像、权重、OOF、GPU 或官方数据。

被测实际 SHA：`f04_gpu_full_dev.py` `b249c370624b2b98e572e4a27c15484f351e34ac805550f2f2a2ac846192014c`；`f05_oof_gpu.py` `10e4c0519f01e3f7aec548c4117c5b43ae6de73c13125df22dd65a06cc98f3ee`；配置 `87731284e57440d9edb2b7aa590f69618506ce6b511fb7859b9aedea75cf379a`；owner 测试 `e405d8fb020f9f0499d76c235b19c87d9e7783fae0ce17fb914adafe6207e776`；助手测试 `6ec403044ba83dc88e9670de4caa33cec588804702b08e8a88d8f75f289e72ed`；consumer `7a56b2f905254c2ade4fe56118857e0771646a4e95724b50f63dd74395c1938a`。之前 `1057` 草稿因 controller 括号语法错误导入失败，guard SHA `6203a76575507e70f7f94a36543f97724cffdbc0a8d28e648b4e1c80fb1bd3ea`，已保留，`1101` 使用修正后的新隔离身份。

`prepare` 测试直接覆盖 UID 不符时写入前拒绝与独立 selected 枚举；真实资产准备全路径尚未实证，合成通过不替代实际设备屏障、真实模型逐 rank 严格加载、完整角色输出、main 全树证明或 owner 的安全终态核验。
