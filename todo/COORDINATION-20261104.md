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

## 2026-11-05 新增 CPU 准备助手（用户批准）

- 助手：`516d5fca-10fc-4fd7-ae00-1d772f025c94`（Ter执行 profile），工作区 git worktree `/home/hao/.paseo/worktrees/0vbeo4hh/helper-cpu-prep`，分支 `helper/cpu-prep-20261105`。
- 职责：仅 CPU/准备池工作（A2 注意力定位可行性、汤模式评估代码、标签清洗/蒸馏训练代码与 CPU 测试）；不 activation/launch、不发训练信号、不派 GPU、不读官方。
- Git：只 push helper/* 分支；08ac 审阅后合并到 main，合并前不得部署。08ac 仍是唯一 GPU owner 与 main 唯一写入者。
- CPU：每 Pod 24 核准备池拆分互斥 12/12（08ac 0–11，助手 12–23），allocation 与 context 路径由 08ac 生成后登记于下方。

（08ac 在此登记助手 allocation SHA 与 context 文件路径）

### 2026-11-05 07:35 UTC 双 Pod 12/12 实核与上下文

- A UID `04203d31-96b5-415d-b126-a30f3ac6107f`、B UID `37e3fb34-c233-4294-b517-e98978cbd016`；当前两 Pod 分别实测 effective `120` CPU、cgroup quota `12000000/100000`、Python `3.12.7`，每 Pod 准备总额 24。两份新 allocation 均固定 08ac `0–11`、516d `12–23`、每人 12 线程，互不重叠；训练池使用其既有冻结 80% 分配，不热改。
- A allocation `/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/takeover-20261104-08ac/pod-a/F05-DUAL-PREP-ALLOCATION-v1.json`，SHA `d0eb7b718b747f7188ba93dc14bd335ef0032079ae0fbdfd31f5449ac36b31af`；B allocation `/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/takeover-20261104-08ac/pod-b/F05-DUAL-PREP-ALLOCATION-v1.json`，SHA `eef8b24fd7c99817ecbee503d52a0eef76f1996db447440263f21724727dd9fd`。均在 2026-11-05 13:32:22 UTC 前有效；到期前由 08ac 重新实测并发放，不沿用过期 allocation。
- 助手环境文件：[Pod A](/home/hao/Desktop/challenge/work/takeover-20261104-08ac/helper-pod-a-prep.env)、[Pod B](/home/hao/Desktop/challenge/work/takeover-20261104-08ac/helper-pod-b-prep.env)。它们只引用已有本机绑定、不复制凭据，并设置助手唯一 executor ID、对应 allocation/SHA 及独立 `NCCCU_HELPER_PREP_ROOT=/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/helper-20261105-516d`；源码/测试仅在助手 worktree，实际 Pod 产物只写其独立 technical-prep。每次仍须由 `tools/environment/pod_cpu.sh` 核 UID/Ready/容器、资源 policy 与 guard 后，在隐藏设备、nice 19 的 12–23 mask 内执行；不要沿用旧 24 核准备入口或用 0–11 运行助手辅助进程。
- 08ac 自用上下文为 `/home/hao/Desktop/challenge/work/takeover-20261104-08ac/pod-a-dual-owner-prep.env` 和 `pod-b-dual-owner-prep.env`；原 `pod-a-soup-prep.env`/`pod-b-train-prep.env` 已转接新额度，旧文件副本只作历史记录。08ac 通过助手上下文在 A/B 实际 probe，均为 12 线程、mask 12–23、GPU 未初始化，CPU 回执分别 `9fa98a5c480388e7d534c887b340fe4195e9aab5138f75702ab43b9ee1f6e985` / `35270da40dc88922f38975f4981ec9cde65dc34d7119e5f5ddd3fa5525f3e47e`；这两次 probe 由 08ac 执行，不冒称助手本人执行。
- 助手专责 A2、汤模式、清洗/蒸馏所需 `classification_loss.py`、`distributed_engine.py`、`data.py`、`config.py` 及其测试；08ac 专责三折 `f05_oof.py`、`f05_oof_assets.py`、`f05_folds.py`、`experiment91.py`、`pipeline.py`、`tools/activation/prepare.py`。共享接口由助手写入 `HELPER-*` 说明后由 08ac 接线；合并前不得部署助手代码。两方共用已有 SSH 并发限制，避免同时启动多条重 I/O。
