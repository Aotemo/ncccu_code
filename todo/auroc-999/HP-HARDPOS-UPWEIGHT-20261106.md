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
