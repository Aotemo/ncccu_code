# A2 CPU 准备状态（helper/cpu-prep-20261105）

已按 A2 合同准备代码草稿；未获得当次已登记的 Pod CPU 分配前，仅做本地文本和代码编辑。当前没有抽样、盲包、粗框、模型前向或指标；固定 63/64 门是否通过为 **unknown**。

必要文件范围：新增 `tools/diagnostics/fit_disc_attention.py` 与 `tests/test_fit_disc_attention.py`；为复用既有盲包与对照，窄改 `tools/diagnostics/fit_disc_consensus.py`，增加上批 64 例排除、A2 seed 和准确的居中图像矩形记录，保留原入口默认行为。准备后依次封存无预测包和粗框，分别在独立进程执行原双线索对照及 R120 显著图，再记录实际回执。尚未在 Pod 运行测试、完成真实加载或检查依赖，草稿不能视为可运行或可部署。
