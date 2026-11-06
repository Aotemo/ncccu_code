# F11 官方 5849 并入训练（2026-11-06）

用户原话（c607 转达）：**"还有最后一次"** 提交机会，截止北京 23:45。依据比赛规则第 3 条：验证集可并入训练，并入后不得再用于早停/checkpoint 选择/阈值或超参调整。

## 合同

- 训练集 = 原 fit 46256 ∪ 官方 5849 全量（52105 行）；选点 = 内部 dev 5140（从未进入训练）。
- 官方标签并入后不再用于任何统计、阈值、z 常数或模型选择；F11 的 z 与 Platt 只用内部 dev 计算。
- 热启动 = R249 selected EMA（tensor `6251937d…` / file `f78ab30c…`，7200 updates）；LR backbone 1e-5 / head 1e-4（原配方 1/10）、warmup 400、每 400 评估、上限 6000、patience 4、G64/P8、pairwise sigmoid。
- 替换规则：F11 成员 draft4 dev AUROC ≥ 原成员 dev − 0.0005 即替换。

## 实现（最小改动）

`retrain/data.py` 新增 split `official_merge`：内部 dev 仍按 9:1 从原 train 行分组划出，fit = base_fit ∪ 全部官方 valid 行。`retrain/config.py` 接受该 split 并新增 `f11` job 合同（`F11_BINDING`、seed 20261205、R249 热启动常量）。`fresh.py` 400 步/cycle；`parallel_fresh_train.py` 入口放行；`hp_setup.py` 增加 `--mode f11`；`tools/diagnostics/export_warm_start.py` 通用热启动导出；`tests/test_f11_official_merge.py` 覆盖 manifest 语义。

测试：改动前 2 failed（RED，`config.py:771 ValueError`）→ 改动后 2 passed（GREEN），均在 Pod A CPU 池执行。

## 训练结果（Pod A，8 rank）

- round `r249-f11-official-merge-20261106`，commit `2da1ff7`，10:57 UTC 开训，11:27 UTC completed，8 rank exit 0。
- **2000 成功更新后早停**（cycle 5；最佳在 cycle 1 = 400 updates）：内部 dev AUROC 0.9881937 / AP 0.8260，variant `online`。
- 训练 loss 从 1e-15 量级降到 1e-19：R249 已收敛，pairwise 梯度几乎为 0，权重变化很小（与 HP 观察一致）。

## 替换判定（draft4 口径）

| 项 | dev AUROC | dev AP |
|---|---|---|
| R249 原成员（draft4） | 0.9884580 | 0.8259 |
| **F11 成员（f11, draft4）** | **0.9883069** | 0.8256 |
| 差值 | −0.0001511 | — |

差值在 −0.0005 容差内 → **替换 R249**。三成员融合（F11 + R269 + SOUP-R120，dev 统计）：

- dev AUROC **0.9901817** / AP **0.84553**（F10 三成员 dev 0.9902299 / 0.84574，−0.000048）
- Platt：a=1.8340409484650222，b=−5.816911937923023
- z（dev）：R249-F11 mean −50.99365469747901 / std 18.480645618868394；R269 mean −13.113201576071983 / std 3.496907618627405；SOUP-R120 mean −17.23285260796547 / std 6.279514072682053

官方 5849 已并入训练，因此本轮不再有独立的官方验证集；选点、z、Platt 全部只用 dev。

## 最终提交（F11-final）

- 配置 `configs/F11-3MEMBER-INFERENCE.json`（Pod 运行时副本 SHA `d89151cb…`）。
- 成员：R249-F11 `294c7b0e…`（99,018,253 B）+ R269 `3f77e91d…` 99,018,253 B + SOUP-R120 `28f3fc55…` 99,018,317 B = **297,054,823 B**。
- 提交 `submition/F11-final/submit.csv`：5855 行 + 表头、CRLF、行序与 R120-98.45 一致、正类 199、SHA-256 `4edfb708…`。
- cost.py（Pod A 单卡 fp16，batch=1，5 预热，100×3 轮）：**84.3 ms**，与提交 0 不一致；T4 换算前向 53.7×0.53=28.5，整链 59.1 ms，×1.5 = **88.6 ms < 100 ms**。
- 官方 5849 已并入训练：本轮的 z 常数、Platt、成员选择全部只由内部 dev 5140 计算，未再用官方做任何统计或选择。

## F11b 候选（R269-F11 替换失败，仅供用户选择）

R269-F11（Pod B，commit `21f3da8` 的登记与结果）：selected @400，draft4 dev AUROC **0.9887958** / AP 0.8384368，对比原 R269（draft4 dev 0.9893111 / AP 0.8376689）Δ AUROC **−0.0005153**，比 −0.0005 容差多出 1.5e-5 → 按规则**不替换**，F11-final 保持原 R269。

按要求并行备好候选 F11b：成员 R249-F11 + **R269-F11** + SOUP-R120，同一 `predict_core`（fp16、draft4）、z 与 Platt 只用内部 dev。

- dev 融合 **AUROC 0.9899710 / AP 0.8449141**（比 F11-final 的 0.9901817 低 0.00021，因 R269-F11 的 dev 略降）
- z（dev）：R249-F11 −50.99365469747901/18.480645618868394；R269-F11 −16.2005928330638/3.9543984635757368；SOUP-R120 −17.23285260796547/6.279514072682053
- Platt：a=1.7939338796219115，b=−5.709569342135444
- 提交 `submition/F11b/submit.csv`：5855 行 + 表头、CRLF、行序与 R120-98.45 一致、正类 196、SHA-256 `25d8febb…`
- cost.py：**85.7 ms**、0 不一致；T4 前向 55.4×0.53=29.4，整链 59.7 ms，×1.5 = **89.6 ms < 100 ms**
- 配置 `configs/F11B-CANDIDATE-INFERENCE.json`（Pod 运行时副本 SHA `330180c8…`）

**建议**：dev 上 F11b 低于 F11-final；若用户只看 dev，应选 F11-final（R249-F11+R269+SOUP-R120）。
