# A2 CPU 准备状态（helper/cpu-prep-20261105）

已按 A2 合同准备代码草稿。本人受 guard 约束的 A/B 12 核 probe 分别通过，回执 `e42f51609c891b564db0be8304e6d6fd904b6fd9c2974ca87898a3d5024fd4c5` / `92eff8568aaef562b1cf24f4d3dfce07768436ad1a83a1b956de8a8e76ef79cc`，均仅显示 CPU、GPU 未初始化。A 源码传输超时后的远端 tar 处于 D 状态，原次与目标目录保留、状态 unknown；B 已复制已验证的基础树并叠加助手代码用于合成检查。当前没有抽样、盲包、粗框、模型前向或指标；固定 63/64 门是否通过为 **unknown**。

必要文件范围：新增 `tools/diagnostics/fit_disc_attention.py` 与 `tests/test_fit_disc_attention.py`；为复用既有盲包与对照，窄改 `tools/diagnostics/fit_disc_consensus.py`，增加上批 64 例排除、A2 seed 和准确的居中图像矩形记录，保留原入口默认行为。准备后依次封存无预测包和粗框，分别在独立进程执行原双线索对照及 R120 显著图，再记录实际回执。

B 准备池在经验证的 `00eb266` 树复制件上叠加助手代码后，旧测试快照 **6/6** 合成检查通过，guard 回执 `590ad0b77a6d14360a4dff208e69237c42ff3021574074ad041392872e2728ec`。实际 `fit_disc_attention.py` SHA `ea9f5a363f0a3e696198f96a52a27ed47047b2be729377a598b418647060efaa`，`fit_disc_consensus.py` SHA `50d95a5bc6e141fdb6b409a23e5dd8cf4350c0da90c916be395e8d826c695da3`，当时测试文件 SHA `5cee24e024828c41cb747ad7ebce4ca65de812606b53656de4bef6a993408ecd`。该批未覆盖后来新增的盲包篡改检查；真实 R120 加载、名单抽样和几何门均未运行。正式研究须由 08ac 审阅合并 main、push/API 并精确验证源码后执行。

08ac 后续只读核查报告：同一登录节点已未见该 namespace 的 `kubectl` 客户端；Pod 内原 tar 进程尚未复核，因此该次 A 传输的 Pod 端终态仍是 unknown，原目标保留，不据客户端消失断言完成。
