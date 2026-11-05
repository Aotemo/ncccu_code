# A2 CPU 准备状态（helper/cpu-prep-20261105）

已按 A2 合同准备代码草稿。本人受 guard 约束的 A/B 12 核 probe 分别通过，回执 `e42f51609c891b564db0be8304e6d6fd904b6fd9c2974ca87898a3d5024fd4c5` / `92eff8568aaef562b1cf24f4d3dfce07768436ad1a83a1b956de8a8e76ef79cc`，均仅显示 CPU、GPU 未初始化。A 源码传输超时后的远端 tar 处于 D 状态，原次与目标目录保留、状态 unknown；B 已复制已验证的基础树并叠加助手代码用于合成检查。当前没有抽样、盲包、粗框、模型前向或指标；固定 63/64 门是否通过为 **unknown**。

必要文件范围：新增 `tools/diagnostics/fit_disc_attention.py` 与 `tests/test_fit_disc_attention.py`；为复用既有盲包与对照，窄改 `tools/diagnostics/fit_disc_consensus.py`，增加上批 64 例排除、A2 seed 和准确的居中图像矩形记录，保留原入口默认行为。准备后依次封存无预测包和粗框，分别在独立进程执行原双线索对照及 R120 显著图，再记录实际回执。尚未在 Pod 运行测试、完成真实加载或检查依赖，草稿不能视为可运行或可部署。
