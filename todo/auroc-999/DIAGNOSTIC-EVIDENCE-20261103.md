# 诊断证据工具 Implementation Plan

**Goal:** 把用户授权的瓶颈复盘落实为可执行诊断，先判断已有提升是否稳定、哪些关键证据仍缺失，再为下一实验提供依据。

**Architecture:** 新增一个 CPU-only 工具，复用 `persistent_errors.pair_contributions` 与原 `n05_metrics.metrics`，读取固定 manifest、完整 selected freeze/CSV/precision。配对 bootstrap 对模型使用相同样本权重，支持图像分层抽样与显式分组抽样；仅报告条件区间。可选读取同 selected 张量的 fit 预测和人工原图/512结构审阅表；缺失保留 unknown，不自动诊断病因、不改变选点或训练队列。

**Tech Stack:** 既有 Python3.12、NumPy、unittest；所有解析、测试、哈希和分析均在经实核的既有 Pod 准备池。按 writing-plans / TDD / executing-plans 内联执行；用户已授权开发，无新增 Agent 或审批等待。

**Spec:** 本页同时保存设计；依据本会话对近几天实验的只读复盘与用户“根据这个开发”的授权。

## 文件与职责

- `tools/diagnostics/paired_evidence.py`：manifest/选点身份核验、配对区间、固定组错序质量、可选 fit/dev 和结构审阅证据、CPU-only CLI。
- `tests/test_paired_evidence.py`：独立穷举 AUC、同分与重复权重、配对/分组抽样、身份错配与 unknown 边界、合成端到端。
- 本任务文件：设计、分工及执行状态。
- `records/analysis/20261103-diagnostic-evidence/`：必要小汇总、实际 CPU 回执和解释。样本名单、预测、抽样向量、inventory 和原始结果只放 exp-v3。

执行者 a4 仅写上述新文件；原八件、活跃运行、runtime/config 和他人未提交文件保持原样。唯一 Git/GPU owner 仍为32e3，本会话不 add/commit/push/GPU。

## 科学与执行边界

- 仅内部 dev 的预定完整 selected；不得读取官方5849、改变 checkpoint、按 bootstrap 换点或将 dev 回灌训练。
- CSV 必须与 manifest 的名单/顺序/标签一致，freeze/预测/precision 与张量、更新数、manifest身份一致；错误即拒绝，不替换样本或改用其他模型。
- 图像分层 bootstrap 的单位是图像，分组 bootstrap 只消费显式 group CSV；没有患者ID不能声称患者级区间。正负配对不作为独立抽样单位。
- 区间条件于已有训练与选点，不能消除跨 seed 波动或反复使用同一 dev 的选择偏差，不作为新 GPU 流程门。
- fit 证据须属于同一 selected 张量和 manifest 的 fit-only 索引；缺失或未知不填成0。fit/dev gap本身不证明过拟合、标签噪声或可达上限。
- 结构审阅只是带来源的人工证据；技术清晰度不能代替视盘/目标征象可见性，工具不加载或诊断图像、不更改标签。

## 执行步骤

- [x] 写独立 oracle/边界测试，在独立 prep 观察预期 RED。
- [x] 最小实现；九项新测试通过（0.251秒），两个新文件编译通过。
- [x] 对既有 R107/R120/R201 完整选点完成一次预声明配对分析；固定难组沿用封存定义，不重新选组。
- [x] 归档小汇总、真实 CPU 回执和缺口；CLI与格式说明见 [交付记录](../../records/analysis/20261103-diagnostic-evidence/README.md)。本会话未 Git 写入，由唯一 owner 集成。

## 实际交付

实际使用实时实核分配 d8370ed3…32b66 的本人 CPUs12–23，与训练24–119互斥，nice19、隐藏GPU。私有 inventory／结果位于 exp-v3 的 `technical-prep/diagnostic-evidence-20261103-a4e2/analysis/`。样本、图像、权重、预测和inventory不进Git。

最终源码 SHA `ed211542a8987df86fcc88d9538c239bb913b56060b74c831be8bdd4c16bfe90`；九项测试 SHA `a9c2f6cde6b78142422952d913d418d002a2bf39bbc26aa86732c27c594ec0c8`。最终测试 guard `8e33a84e…1148e`、实际分析 guard `71eee806…2efec`、两编译及12份旧回执身份核验 guard `43bd388b…bd3bc` 均 exit0。详细完整身份见交付记录，不重复运行原项目测试。

R120−R107 AUROC差 +0.0010367361，配对95%区间 [−0.0011201506,+0.0038679227]；R201−R107和R201−R120区间也跨0。区间仅描述既有选点的图像抽样不确定性，未包含跨seed或dev选择偏差；点估计改善不足以证明稳定提升。R201相对R107固定20错序质量减少531、其余147增加105；局部修复伴随新增。fit/dev与结构审阅尚无实际输入，均unknown，病因未识别。工具不构成新训练准入门，不改变现有队列，不宣称总体目标完成。
