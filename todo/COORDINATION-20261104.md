# 统筹与执行分工（2026-11-04 起）

## 用户决定

2026-11-04 12:3x UTC 用户要求：停止 4e 的工作，由 c607 替代 4e 负责总体统筹但不做具体执行；执行由 c607 新增一个 Ter 执行 Agent 完成。此为用户对“不恢复旧Agent、不再派生”的明确例外授权，仅限一个执行 Agent。

- 4e（`4e0ce707-5c70-493a-8e54-796363eb740f`）：当前 turn 已于用户要求后由 c607 cancel，不再接受任务，不归档以保留历史；其后不再是 Git 写入者或 GPU owner。cancel 不影响 Pod 上已运行的训练进程。
- c607（`c6073541-7b4e-4b23-8fa6-c696e2d95733`，统筹）：决定实验方向、队列与优先级，审阅结果，向执行 Agent 下达指令并向用户汇报。不执行 Pod 命令、Python、测试或 GPU 计算，不提交 Git。只写本文件及其新建的统筹/提案文件。
- 执行 Agent（ID 见下方交接记录）：唯一 Git 写入者、两个 Pod 的唯一 GPU owner 及 CPU 准备执行者，承接 4e 的全部执行职责、文件与私有执行目录；按 AGENTS.md、docs/EXECUTION.md 执行。维护 todo/README.md 页首、轮次合同、records/**、源码、配置与测试；负责提交 c607 写的文本（不改其内容，有异议向 c607 提出）。

## 接管时状态（c607 只读核对，未在 Pod 上验证）

- 本地 HEAD 976ad8c，与 origin/main 一致。4e 遗留未提交：R240R1/R242 完整结果与归档回执、R244/R245 启动/进度回执、`records/environment/dual-pod-20261104/REPLICA-START-SUMMARY.json`、F02 登记 `todo/auroc-999/F02-HETEROGENEOUS-EQUAL-20261104.md`、`configs/F02-HETEROGENEOUS-EQUAL.json` 及 `tools/diagnostics/ensemble_dev.py`/`tests/test_ensemble_dev.py` 修改（4e 称五项窄检查已过、未同步、未读取 F02 结果）。
- R240R1 完整：selected AUROC 0.985423 / AP 0.826382；R242 完整：AUROC 0.983140 / AP 0.810391。均低于 R120，fit 适配族共 10 轮阴性。
- R244、R245（R120 配方新 seed）已在两 Pod 运行，最近回执分别约 25、300 次成功更新。
- F02 库存：11 个候选的 selected dev 预测可核，但其中 6 个 selected checkpoint 已缺失——这些成员只能做互补性研究，不能部署。
- c617 遗留未提交文本（`records/analysis/20261101-experiment-review/`、`20261104-experiment-review/`、`todo/auroc-999/EXPERIMENT-REVIEW-20261104-c617.md`）及 R134R1 旧 census 草稿（`tools/diagnostics/fit_census_*.py`、`tests/test_fit_census.py`、`todo/auroc-999/R134R1-FIT-CENSUS-20261101.md`、`tools/diagnostics/R134R1-FIT-CENSUS-20261101.json`）不属本次接管范围，不混入提交。

## 当前方向（c607 决定）

1. fit 表征适配族（R224–R242）停止新增轮次；R241/R243 等该族 seed27 复核不跑。
2. 主线转为融合：目标 raw AUROC 0.995 / AP > 0.9。融合成员必须有可加载的 selected 权重才算可部署。
3. 近期队列：R244/R245 跑完 → 按 F02 结果选择异构成员训练（不同骨干/通用公开预训练/输入视野），由执行 Agent 在 R244/R245 训练期间提出具体两轮方案，c607 批复；若 GPU 将空闲而批复未到，执行 Agent 按其已登记方案直接接续，不让 DCU 等待。

## 交接记录

- 执行 Agent：`08ac5df9-3ab6-41ad-b7b1-ecfba1f8ac92`，Ter执行 profile（codex gpt-6.1-terra，max，full-access），由 c607 创建于 2026-11-04；接管时间与首个提交由其在 todo/README.md 页首登记。
