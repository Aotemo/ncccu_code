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

## Pod B：R269-F11（2026-11-06 10:0x–10:3x UTC）

- 按 c607 路线 2：`exec/hardpos` 合并 `origin/main`（`9510a1b`，含 `8d835d4`/`2da1ff7`）后，最小改动登记 R269 热启动：`config.py` 的 `F11_WARM_STARTS` 按 binding.version 查表（R249 `6251937d…`/`f78ab30c…` 保持不变；R269 `3fd623ee…`/`1ff8f94d…`，binding `f11-official-merge-r269-warmstart-v1`），`hp_setup.py` 依据 warm receipt 的 tensor sha 选 binding，`cmd_site` 从 science 文件读取 binding_version。commit **`8ced6861269a9bd664faf31a7c490aabf1e539aa`** 已推 `origin/exec/hardpos`。
- 回归：`tests/test_f11_official_merge.py` + `tests/test_hp_hardpos.py` 在 Pod B CPU **9 passed**，含 R249 原 binding 通过、R269 新 binding 通过、交叉错配/未知版本/多余键被拒。
- 配置/规格（Pod B CPU）：config `0acc32ea…`、fresh spec `b73c9868…`、plan `cd4eaa62…`、site `c8552318…`、spec receipt `73625a7e…`；fit=**52105**（46256+5849）、dev=5140/167、negatives=50424、steps/cycle=400、seed=20261205、G64/P8、LR 1e-5/1e-4、patience 4。热启动 = R269 selected EMA（ema@8800，tensor `3fd623ee…`，file `1ff8f94d…`）。source tree `f11-src-8ced686`（87 files，tree `e855e6a6…`，source_digest `347d8d1b…`）。
- 启动：10:3x UTC 8 rank 在位，首成功 25/25、0 skip，8 卡 90–100%；`--manifest` 不传（F11 由 `load_source`+`official_merge` 自建 manifest，入口会拒绝 include_valid+manifest 组合）。只读 cache-union 复用（51,396+5,849）。
- 完成后：fp16 严格回载 → draft4 dev 前向 → 报 dev AUROC/AP + 预测路径/SHA；**不用官方做任何统计或选点**（官方已并入训练）。
