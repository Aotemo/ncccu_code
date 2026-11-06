# HP 难阳性上权与 OHEM 变体合同（2026-11-06）

在读取任何 OOF 权重表或产生任何 HP/OHEM 训练结果之前登记。执行者为 GLM 执行 Agent（Pod B 唯一 owner，分支 `exec/hardpos`）。协调者 c607 于 2026-11-06 04:0x UTC 确认设计。

## HP：难阳性行权重（任务 HP）

- 数据边界：只使用 F05 三折 fit-only OOF v5（`technical-prep/takeover-20261104-08ac/F05-OOF-TARGETS-v5/SUMMARY.json`，SHA 以 `configs/R268-A-SCIENCE.json` 的 `f05_targets.summary_sha256` 为准）。该 OOF 仅由原 fit 三折产生，未读 dev/官方/test；本合同不新增任何官方读取（训练与选点只涉及原 fit 与原 dev；官方前向仅用于授权后的最终成员验证）。
- 难例规则（预登记，在生成权重表之前固定）：
  - 规则 A：fit 正例按标定 OOF 概率升序取最低 15%（`floor(0.15 * n_pos)` 个；概率相等时按原 manifest 全局行索引升序破平）。
  - 规则 B：OOF 概率严格低于 fit 负例 OOF 概率最近秩（nearest-rank）95 百分位（升序第 `ceil(0.95 * n_neg)` 个值）的 fit 正例。
  - 难例集合 = A、B 中较小者；大小相等取 A。写明最终数量与选择。
- 权重：难例行权重 3.0，其余 fit 行 1.0；pairwise sigmoid 的正负对权重为两行权重乘积并按总对权重归一化（复用 R268 清洗机制，方向相反）。阈值与概率完全由 v5 OOF 独立重建（复用 `verify_f05_oof_for_training` 的标定重建），不手工指定。
- 训练合同：R120 selected EMA 热启动（复用 F06 的 `exported-fp32-state-dict-v1` warm start，张量 `f21bffae…`、文件 `2ac4a025…`）；训练集 = 原 fit 46256（不含官方 5849）；选点 = 内部 dev AUROC（原 5140）；每 400 更新评估、上限 6000 成功更新、patience 4、min_cycles 1、LR = R120 的 1/10（backbone 1e-5、head 1e-4）、G64/P8、pairwise sigmoid、512/512、ImageNet 归一化、age 同 R120/F06。seed `20261203`。
- 评估：selected 的内部 dev AUROC/AP；fp16 导出（严格回载）；在内部 dev 5140 与官方 5849 上 fp16 前向（官方此处仅作验证）；与 F04 成员组合报告 "R120→HP 替换" 与 "HP 加入 2–4 成员组合" 的 (dev+官方)/2，基准 F04-4member .98984，门槛 +0.0005。
- 是否进入最终提交由 c607 与 48a0f156 的 F09 整合决定；本执行者不生成 submission。

## OHEM：难配对挖掘变体（HP 之后，不占 HP 前的 GPU）

- 损失：pairwise sigmoid 加预登记 top-k 选项——每个 WORLD batch 内对所有正负对按当前对损失降序（稳定排序，行主序破平）取最难 top 比例（固定 25%，`floor(0.25 * n_pairs)`、至少 1）参与平均，其余对权重 0。top 比例 1.0 时与原 pairwise sigmoid 完全一致。loss 合同 `binary-pairwise-sigmoid-ohem-top25-v1`。本轮行权重全 1（无 f05 目标），不与 HP 行权重叠加。
- 训练合同：同 HP（R120 EMA 热启动、fit-only 46256、dev 选点、每 400 评估、上限 6000、patience 4、LR 1/10、G64/P8）。独立 seed `20261204`。
- HP 训练在 Pod B 达到安全终态后立即开训 OHEM（c607 已授权，无需再问）；结果同样做 dev/官方前向与组合评估。

## 最小文件范围（实现前登记）

- `runtime/bundle/retrain/classification_loss.py`：OHEM loss kind + f05_pairwise_objective 的 `hardpos` 模式（值域 {1,3}）。
- `runtime/bundle/retrain/config.py`：`validate_f05_training_binding` 接受 `hardpos`；`validate_job` 增加 HP/OHEM 热启动分支；HP 常量。
- `runtime/bundle/retrain/data.py`：`f05_hard_positive_rule`；`validate_f05_target_table`/`load_f05_training_targets`/`verify_f05_oof_for_training` 的 `hardpos` 支持（R268/R269 的 clean/distill 路径行为不变）。
- `runtime/bundle/retrain/fresh.py`：HP 的 steps_per_cycle=400。
- `runtime/bundle/retrain/distributed_engine.py`：f05 批损失 `hardpos` 值域。
- `tools/diagnostics/hp_targets.py`（HP 权重表/HP-SUMMARY/evidence 生成，CPU）；`tools/diagnostics/hp_setup.py`（config/spec/site，仿 F06）；`tests/test_hp_hardpos.py`。
- `configs/HP-AGE-CONTRACT.json`（自 F06 fold0 训练配置提取的原 fit age 合同，fit 46256 行不变）。
- 样本级概率/权重表只存 exp-v3；不进 Git。禁止医学预训练、test 图像/标签。

## 执行记录（2026-11-06）

- 04:1x–05:0x UTC：代码实现 + Pod B CPU 测试（allocation `exec-hardpos-glm-b`，guard 回执见 `technical-prep/hp-hardpos-20261106-glmB/alloc/tasks/`）：`test_hp_hardpos` 6/6、`test_f05_clean_distill` 4/4（含 WORLD 两 rank 梯度）、`test_f06_official_ft` 6/6；`test_f05_training_binding` 在该 Pod 环境对未修改 main 同样段错误（exit -11，同用例复现，预存环境问题，非本次改动）。commit `83b96e6`/`f335790`/`ffd9238` 推送 `origin/exec/hardpos`。
- 05:0x UTC：HP 权重表生成并独立验证（Pod B CPU，v5 OOF 重建）：规则 A=223、规则 B=108（负例 OOF 最近秩 95 分位 0.00330946）→ 取较小集合 **B：108 个难阳性行权重 3.0**，其余 46148 行 1.0。产物 `technical-prep/hp-hardpos-20261106-glmB/targets/HP-OOF-TARGETS-v1/`：HARDPOS-TARGETS-private.json `0c22aa5d…`、HP-SUMMARY.json `080c1ce5…`、evidence `rounds/hp-hardpos-fitonly-20261106/evidence/F05-OOF-VERIFICATION.json`。
- 05:2x UTC：site binding（`technical-prep/hp-hardpos-20261106-glmB/site/site-hp-v1/`，Pod B 实测 120 有效核、owner 96/验证 24、control 112–119）+ HP 训练配置 `cfg/HP-TRAIN-CONFIG.json` `d81b2d51…` + fresh spec 验证通过（8 rank、G64/P8、400/6000）。
- 05:3x–06:0x UTC 三次启动：① 缺 LD_PRELOAD NUMA 守卫失败（无状态创建）；② 直启 build_manifest 与封存 manifest split 键不一致（R268 证明 base manifest `3275aaa3…` 即训练 manifest）→ 改用 `--manifest` 直传 `assets/fixed-9010-stage1/manifest.json` `7cc0f8b7…`；③ 共享 `prepared512-v2-pillow12.3.0` 目录仅 1024 条不完整，改用 F06 只读 cache-union（114,490 文件，含全部 51396 train + 5849 official 条目）。06:0x UTC 第四次启动成功：8 rank 全部在位，8 卡 100%，0 skip。
- 结论 08:30 UTC 前；OHEM 于 HP 安全终态后同链启动（seed 20261204，loss `binary-pairwise-sigmoid-ohem-top25-v1`，无 f05 目标）。

## 2026-11-06 07:1x–08:0x UTC Pod B 新执行者接管与 HP 终结

- 交接：GLM `b39fbd2c` 会话在 workbuddy 重启后失效并归档；新执行者接管 Pod B（worktree `/home/hao/.paseo/worktrees/0vbeo4hh/exec-hardpos`，分支 `exec/hardpos`），不碰 Pod A、不推 main。
- HP 训练实际终结于 2800/6000（早停，连续 4 轮无提升，0 skip）。selected = **online c3 @1200**，内部 dev AUROC **0.9888102** / AP 0.81964；EMA 最好 c1 @400 AUROC 0.9887500。
- selected strict fp16 导出 + 独立回载通过：`eval-v1/HP-EXPORT`，model sha `6de5907e…`、fp16 tensor `ad1f0cb5…`、99,018,253 B；receipt `STANDALONE.json` 4fdfbc6e…
- draft4 前向（F09 同一 `src-54b9aac`、`NCCCU_DRAFT_DIVISOR=4`、单视图 fp16）完成 dev 5140 + 官方 5849：`eval-v1/hp-draft4-shards/`。
- draft4 两成员表（(dev+官方)/2，等权 z，dev 统计），`eval-v1/HP-DRAFT4-COMBO-v1.json`（receipt sha 见文件）：
  - HP 单成员 .9876880（dev .9887320 / 官方 .9866440）
  - HP+R269 .9897675（dev .9902347 / 官方 .9893002）← HP 最好两成员
  - HP+SOUP-R120 .9896714、HP+SOUP-R201 .9895654、HP+R250 .9895085、HP+R249 .9894155，其余更低
  - 基准 F09 最佳两成员 R249+SOUP-R120 **.9900255**（本工具复算完全一致，sanity 通过）；门槛 .9905255
- **HP 结论：最好 HP 两成员比基准低 0.000258，未达 +0.0005 门槛；HP 不作为替换/新增成员提交。** F09 R249+SOUP-R120 保持。
- OHEM：`cfg/OHEM-TRAIN-CONFIG.json` `1855f7b7…`（seed 20261204、`binary_pairwise_sigmoid_ohem` top25、无 f05 目标、R120 EMA 热启动）、`run/OHEM-FRESH-SPEC.json` `a9de036e…`、`run/OHEM-PLAN.json` `486785f2…`、site `site/site-ohem-v1b/SITE_BINDING.json` `cc7e67d5…`。07:5x UTC 8 rank 在位、首成功 25/25、0 skip、8 卡 100%；首个 50 更新已记录。训练中，结果待 >400 更新后评估。
- 启动失败保留：① 直接启（无任务 CPU mask）→ inherited mask 越界；② 缺 `TRAIN_IMAGE_CACHE_READONLY`；③ output_root 未预建。分别修正后第四次启动成功，未重置任何 clock/token/grant。
- 工具：`tools/diagnostics/hp_eval.py` 新增 `combo-draft`（读 F09 draft shards，两成员等权 z，sanity pair），commit `18e1eb5`/`379679d` 已推送 `origin/exec/hardpos`。

## 2026-11-06 08:0x–08:3x UTC OHEM 结果

- OHEM 早停于 **2000/6000**（patience 4，0 skip，连续 4 轮无提升）：dev online AUROC 400 .9888415 → 800 .9877350 → 1200 .9866573 → 1600 .9872931 → 2000 .9873376；EMA 800 .9888704 最高、之后回落。selected = index `best` = **online @400**，dev AUROC **.9888415** / AP .82502（EMA best .9888704 @800）。
- strict fp16 导出+回载通过：model `8000b78d…`、tensor `6a271acd…`、99,018,253 B；`OHEM-EXPORT/STANDALONE.json` `71a57e06…`。
- draft4 前向（同 F09 `src-54b9aac`、divisor 4、单视图 fp16，2 分片）完成：`eval-v1/ohem-draft4-shards/`；dev 两片 csv `575678f7…`/`f6ac2566…`，官方两片 `09b0fdc6…`/`614388e4…`。
- 组合（draft4，(dev+官方)/2，等权 z，dev 统计；`OHEM-DRAFT4-COMBO-v1.json` `83a9c19f…`；F09 基准 R249+SOUP-R120 .9900255 复算一致）：
  - OHEM 单成员 .9880125（dev .9886633 / 官方 .9873616）
  - OHEM+R269 .9896274（dev .9903720 / 官方 .9888829）← OHEM 最好两成员
  - OHEM+SOUP-R120 .9894275、OHEM+SOUP-R201 .9893529、OHEM+R250 .9893420、OHEM+R249 .9892887、OHEM+R134R1 .9891757，其余更低
  - **OHEM+R249+SOUP-R120（F09 两成员 + OHEM）三成员 .9898068**（dev .9896314 / 官方 .9899822）
- **结论：OHEM 最好两成员 .9896274、与 F09 两成员的三成员 .9898068，均低于 F09 基准 .9900255（分别 -0.000398 / -0.000219），未达 +0.0005 门槛。OHEM 不替换最终方案。**
- 机制观察：两轮都从已收敛 R120 热启动，fit pairwise 损失在数十步即 ~1e-7 以下、OHEM 到 2000 步 ~1e-13，fit 对梯度几近为零，dev 只随 EMA/LR 漂移；这是 HP/OHEM 均无法超过 R120 的直接原因。
- 预测文件（draft4 fp16，name/label/logit）：dev `eval-v1/ohem-draft4-shards/dev.OHEM.draft.shard{0,1}of2.csv`、官方 `official.OHEM.draft.shard{0,1}of2.csv`，SHA 见上；sidecar 记录 decode/rows/export_sha、test_label_values_read=false。
