# F11 官方验证集并入训练（2026-11-06）

用户原话（c607 转达）：**"还有最后一次"** 提交机会；截止北京 23:45。

依据比赛规则第 3 条："验证集是否参与训练，由选手结合实际需求自主选择。验证集一经并入训练，不得再用于早停、checkpoint 选择或任何阈值、超参数调整；此类选择须在并入训练前完成。"

## 决定

- 训练集 = 原 fit 46256 ∪ 官方 5849 全量；**选点只用内部 dev 5140**（dev 从未进入训练）。
- 官方标签并入后不再用于任何统计、阈值、z 常数或模型选择；F11 的 z 常数与 Platt 仅由内部 dev 计算。
- 热启动 = R249 selected EMA（tensor `6251937d…`，file `f78ab30c…`，7200 updates），原配方 LR 的 1/10（backbone 1e-5、head 1e-4）、warmup 400、每 400 评估、上限 6000、patience 4、G64/P8、pairwise sigmoid 不变。
- 判定：F11 成员内部 dev AUROC ≥ 原成员 dev − 0.0005 即替换（收益预期在 test 侧，dev 只做"不变差"检查）。

## 实现（最小改动）

- `runtime/bundle/retrain/data.py`：新增 split `official_merge`——内部 dev 仍按 9:1 分组从原 train 行划出，fit = base_fit ∪ 全部官方 valid 行。
- `runtime/bundle/retrain/config.py`：`validate_split` 接受 `official_merge`（必须 `include_valid=True`）；新增 `F11_BINDING`/`F11_SEED`/F11 热启动常量与 `validate_job` 的 `f11` 分支；`_unknown` 增加 `f11`。
- `runtime/bundle/retrain/fresh.py`：F11 的 steps_per_cycle=400。
- `runtime/bundle/parallel_fresh_train.py`：入口接受 `official_merge`。
- `tools/diagnostics/export_warm_start.py`（新）：通用 selected-EMA 热启动导出（校验 tensor digest 与 update 数）。
- `tools/diagnostics/hp_setup.py`：新增 `--mode f11`（split=official_merge、seed 20261205、job 键 `f11`）。
- `tests/test_f11_official_merge.py`：manifest 语义（官方全部进 fit、dev 只来自原 train、两集合不交）与 split 校验。

## 执行记录

- 09:22–09:50 UTC：代码实现、常量填入、测试与部署。
