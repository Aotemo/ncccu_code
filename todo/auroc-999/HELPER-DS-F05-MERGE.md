# DS 助手：helper/cpu-prep-20261105 → main 的 F05 合并清单

状态：**已并入 main `85ca56e`，并在合并后的全树通过终态回归（20/20、6/6、5/5、11/11）。** 基线为 `origin/main` `c304d67`。本文只记录逐项判断与理由；不回写任何未实核的结论。

## 判断口径

以当前 main 为基准，对 `origin/helper/cpu-prep-20261105`（43 个提交、`0701d0e`）逐文件比较。main 已通过 `c96305b` 合入与助手 `d1a9215` 等价的选员初版，故只保留 main 尚缺、且被已登记裁定要求的部分。沿用助手已在 B Pod 取得 20/20 合成回执的完整快照，不手工切分已验证文件，避免产生从未整体测试过的变体。

## 已等价存在于 main、未重复合并

| 助手内容 | main 现状 | 结论 |
| --- | --- | --- |
| `runtime/bundle/retrain/{config,data,distributed_engine,experiment91,pipeline}.py`、`runtime/bundle/train.py` | 已含等价实现 | 不合并 |
| `tests/test_f05_clean_distill.py`、`tests/test_f05_training_binding.py`、`tests/test_f05_oof_targets.py`、`tests/test_f05_oof_owner_safety.py` | 已含 | 不合并 |
| `configs/templates/R268-*`、`R269-*`、`configs/R268-*`、`R269-*`、OOF 目标与清理回执 | main 已有更晚版本（含 `F05-OOF-TARGETS-v5`、prune 回执） | 不合并，保留 main |
| `tools/diagnostics/f05_oof_targets.py` | **main 更新更晚**：`a8fb533` 加入 `CALIBRATION_THREADS=8` 与 `threadpool_limits`，保证标定与目标表可复现；助手版会删掉该固定 | **明确不合并**，冲突按 main 保留 |
| `tools/diagnostics/f04_gpu_full_dev.py` | main 已含后继改动 | 不合并 |
| `UPSTREAM-TRAINING-DATA.md` 其余改动 | main 已含 | 不合并 |

## 本次合并的内容

| 文件 | 变更 | 理由 |
| --- | --- | --- |
| `tools/diagnostics/f05_final_selection.py` | +513/−13（相对 main） | 补 main 缺的三折候选准入：`require_fold_candidate_census`、`validate_candidate_population`、`screen_candidates(base_ref/work_root)` 的 population 判定与 `manifest_mismatch_verified` 处理、`verify_round_export` 的 `fold_binding` 一致性、`freeze_selection` 的折普查门；另含 `verify_soup_export(include_vectors)`、`fuse_from_selected_packets` 与 `fuse` 子命令 |
| `tests/test_f05_final_selection.py` | +445/−6 | 上述功能的合成用例（折绑定、派生字节/双 SCIENCE 篡改、原 dev 同序、无关 manifest 明确排除、fp32 对照、rank 包完整性） |
| `tools/diagnostics/f05_fold_candidates.py` | 新增 282 行 | 三折候选发现：`discover`/`verify_augmented`，按裁定产生 `fold_binding`，原 F03 原件只读保留 |
| `tests/test_f05_fold_candidates.py` | 新增 234 行 | 上述发现的合成用例 |
| `todo/auroc-999/HELPER-F05-FOLD-CANDIDATES.md` | 新增 | 接口、血缘与剩余接线说明 |
| `todo/auroc-999/HELPER-F05-SELECTION.md` | +18/−2 | 三折发现裁定、选中成员 dev 分片接线草案 |
| `todo/auroc-999/HELPER-R268-R269-PREP.md` | +8 | main `58304` 精确树 CPU 合成回归记录（10/10、4/4、6/6） |
| `todo/auroc-999/HELPER-SOUP-IMPLEMENTATION.md` | +3/−3 | “待测”改为“已合成测试、待审阅合并”的措辞订正 |

## 未合并项及理由

- `tests/test_f05_oof_short_chain.py` 与 `todo/auroc-999/HELPER-OOF-SHORT-CHAIN.md`：测试对象是 08ac 私有的两份 shell wrapper 快照，仓库内无该产物，必须靠 `NCCCU_F05_SHORT_OUTER`/`NCCCU_F05_SHORT_POD` 指向仓库外文件才能运行；合入后既不能自跑也不能防回归。包装控制流仍由 08ac 自测，不并入 main。
- 助手 `f05_oof_targets.py`：见上表，会回退 main 的可复现固定，不合并。
- 旧五折、A2、汤模式等其余内容：main 已有对应实现或已由 08ac 接线，不重复引入。

## 需要 08ac 决策/接线的点

1. `fuse` 子命令（`f05_final_selection.py fuse`）是纯 CPU 归约，**尚无 producer**：`f05-selected-dev-packets-v1` 的八 rank 包需 owner 在现有 `f04_gpu_full_dev.py` 监督链产生。未接线前它只能拒绝或用于合成 fixture，不能被当作正式 F05 dev 指标入口；是否现在就部署由 08ac 决定，删除该分支不影响三折选员。
2. `f05_fold_candidates.discover` 目前**没有 CLI**，现有 F03 发现入口尚未调用它；按 `HELPER-F05-FOLD-CANDIDATES.md`，消费扩充库存前必须先调 `verify_augmented`，不能把未验证 dict 直接送选员。
3. `freeze_selection` 与 `main()` 新增的折普查门要求库存中 R265–R267 各出现一次（unknown 行也满足）。若实际 F03 库存缺少任一行，选员会按裁定直接停止而不是静默跳过。

## 验证状态

- 本机仅做只读文本核对：确认助手快照相对 main 只有加法、无 main 侧修复被覆盖；`paired_evidence._reference/_selected`、`retrain.site_binding.verify_rank_devices`、`retrain.parallel_evaluation.canonical_predictions`、`f04_size_ensemble.compare_predictions` 在 main 均存在；无 `f05_oof_gpu` 等已移除模块的引用。
- 2026-11-05 Pod A 实测：`git archive` 的 `382dd6d` 全树 3562 文件在 12–23、nice 19、隐藏设备、12 线程下运行，`tests/test_f05_final_selection.py` **20/20**、`tests/test_f05_fold_candidates.py` **6/6**、`tests/test_test_submission_aggregate.py` **5/5**，0 skip、0 failure，CPU guard exit 0，回执 `140d7e75c2ab613cd61ec856594639986b91ecb9431faf7260046b6942d7f48d`。Pod 内实算的源码 SHA 与本清单一致：选员 `328b6db475ac93499efc2d469ba898964a3dc69610ef70b1df0de1ae0e4e3d36`、折发现 `0e8c0ad4f41937a05146cca20096a6a8a18730d62a79c106a641823f9e823f10`、对应测试 `396fdf2dbe2e103b8feb6e5dce3333e86b1a2da3014ff95fffe82facef60094d` / `5d3ea3c4fcde359e4c3aee126e3a12f4cdcfea3143d88a30a4fe3834811ce40e`；基线 `f05_folds.py` `78dc9797…`、`paired_evidence.py` `ed211542…`、`site_binding.py` `e1e2198e…`、`parallel_evaluation.py` `fa5b9319…`。
- 这只证明合成 fixture 与纯函数行为，不证明真实三折资产、`verify_augmented` 在发现入口的接线、真实选员、fp16 分片或 `fuse` producer；上述仍需在最终 main 全树与真实资产上核验。

## 合并后终态回归（main `85ca56e`）

- DS执行2 已把本分支四个提交 `--no-ff` 合入 main `85ca56e`，前一条为 `e455f7c`（`runtime/run_round.py` 的 transient tree 目录消失容忍）。合并后 `git diff helper/ds-20261105 origin/main --` 对本分支六个代码/测试文件为空，即字节未变，未被合并改动。
- 2026-11-05 在 Pod A 以新 allocation `…/pod-a/CPU-ALLOCATION-20261105T1340Z.json`（SHA `43d8cf2877ab1529a21bf0ebf1bd7a797278bd947081a28d8e458bf6f355be89`）对 `85ca56e` 的 `git archive` 全树（3564 文件）跑终态回归：`test_f05_final_selection` **20/20**、`test_f05_fold_candidates` **6/6**、`test_test_submission_aggregate` **5/5**、`test_monitor_unknown` **11/11**，0 skip、0 failure，exit 0，CPU guard 回执 `900054133b6518292506f0ea3d8bddb314b4d6e9323b59603a5af4dd5d8eda30`（12 线程、mask 12–23、隐藏设备）。`test_monitor_unknown` 一项是对 `e455f7c` 合并结果的旁证，不是本分支内容。
- 仍未做：真实三折资产上的 `discover`/`verify_augmented` 集成、真实选员、fp16 分片、`fuse` producer；这些需要 08ac 接线后的真实资产与 GPU 回执，本项不声称通过。
