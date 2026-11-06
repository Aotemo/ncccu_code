# 最终提交材料包（F10 三成员，2026-11-06）

用户决定：主包改用 F10（平台 98.50），因历史最高分 F04 五成员在 batch=1 口径下 fp16 实测 209.6 ms（T4 换算约 166 ms，>150 ms 不予认定档）。

## 包内容

`submition/TEAM_最终提交材料.zip`（460 MiB / 461,373,078 B，SHA-256 `4b56fdeb…`），根目录含 `submission.csv`（= `submition/F10-final/submit.csv` 原文件，SHA `b70bd76f…`）。

- `code/`：predict.py / cost.py / model.py / utils.py / train.py / requirements.txt / retrain/（与仓库实现同源；`predict_core` 的 site 字段改为可选，便于脱离历史环境运行）
- `model/`：R249 `9544763a…`、R269 `3f77e91d…`、SOUP-R120 `28f3fc55…` 的 fp16 权重 + model.json + `F10-INFERENCE-CONFIG.json`
- `pretrain_model/`：上游 timm convnext_small.fb_in22k_ft_in1k（rev `fe6b358a…`，SHA `c912f45c…`，201,017,809 B），不参与推理
- `licenses/`：Apache-2.0 全文（`cfc7749b…`）、模型卡（`bbf1cf6f…`）、来源说明
- `模型核验材料与数据申明清单.xlsx`：按主办方模板 A1–F3 逐项填写 + F04 备注行
- `MANIFEST.json`：78 文件逐项字节/SHA；推理权重 297,054,823 B，包总 499,390,625 B

## 复测（包内 code，Pod A 单卡）

```
[ROUND] 85.00 / 85.43 / 84.67 ms  (pre≈23.3, h2d≈0.57, forward≈54.7)
[CONCLUSION] 最终平均耗时 85.0 ms，未超过上限；与 submission.csv 比对 100 张，不一致 0 张；推理精度 fp16。
```

T4：前向 54.7×0.53=29.0，整链约 59.3 ms，×1.5 = 89.0 ms（≤100 ms 档）。

## F04 对照（未采用）

F04 五成员 fp16 + 完整解码 batch=1：209.6 ms（pre 73.9 / forward 91.6）；Pred_Label 与 F04 CSV 0 不一致、|Δp| max 6.6e-4（≤1e-3），但 T4 ≈166 ms >150 ms，按新口径不予认定，故材料包改用 F10。
