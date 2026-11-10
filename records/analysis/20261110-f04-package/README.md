# F04 五成员最终提交材料包（2026-11-10）

用户/主办方要求："材料包中请提交与排行榜最高分 submission.csv 一致的 submission.csv，对应模型也请保持一致"。
本记录说明 `submition/TEAM_最终提交材料_F04-98.55.zip` 的组装、身份核对与计时回执来源。打包期间 Pod A 已释放，
按 c607 决定**本机不运行 Python、不重跑 cost.py**；延迟与一致性引用 2026-11-06 同一代码与权重在 Pod A 的实测。

## 包身份

- `submission.csv` = `submition/F04-fusion/submit.csv` 原文件：231,336 B，5855 行 + 表头，CRLF，201 正类，
  SHA-256 `3a9dc5b73d763ed048bf1127fbe367e3ad9d6446f3ccc7835081a7d68e24d2a6`（zip 内重新提取后一致）。
- 五成员 `model.pt` 与 `records/analysis/20261104-f04/F04-FINAL-FREEZE.json` 的 fp16 冻结导出一致：
  R120 `e0bc2c52…` 99,018,253 B、R134R1 `600cf0fd…` 99,018,253 B、R250 `8af616ad…` 99,018,253 B、
  R249 `9544763a…` 99,018,253 B、R135 `2a936d49…` 99,024,397 B；合计 **495,097,409 B**。
  五份 `SITE_BINDING.json` SHA 也与冻结记录一致。
- 推理侧车（5×model.json + 5×SITE_BINDING + F04-INFERENCE-CONFIG）67,617 B，推理合计 **495,165,026 B ≤ 500,000,000 B**。
  `pretrain_model/pytorch_model.bin`（201,017,809 B，上游 timm `convnext_small.fb_in22k_ft_in1k`，SHA `c912f45c…`）
  只作来源核验，不参与推理、不计入 500MB。
- 推理：fp16 autocast、完整 JPEG 解码、五成员等权 z 融合、Platt `a=1.9347664660965742 / b=-6.13690213106862`，阈值 0.5、无 TTA。

## 计时与一致性回执（2026-11-06 Pod A / DCU，未重跑）

来源：DS执行4 当次实测；本机 `sha256sum` 核对 Pod 侧计时包
`technical-prep/takeover-20261105-ds2/f11-20261106/f04pkg/` 与本包：

| 文件 | 两侧一致 SHA-256 |
|---|---|
| `code/cost.py` | `484e70d9e5ce1de939cee3232cc1ec60f96e249618978f500455b52e35a525c5` |
| `code/f04_core.py` | `07de24b5308f13e8d98f47aa955317f4889fddc47d353a337737b86d3a243a37` |
| `code/predict.py` | `4904b11c699b4391bfec9722b41dde0a8bd1d2fea11e61c520ef1536d68c92be` |
| `model/F04-INFERENCE-CONFIG.json` | `f049ed87821e97774d00ddd4ec6115238ea1f1400e84ae7e2f0c3ea583835f93` |
| `code/retrain/*.py`（55 个） | 逐文件一致（仅路径前缀不同） |
| 五份 `model.pt`、`submission.csv` | 与冻结记录及 `3a9dc5b7…` 一致 |

结果：batch=1 串行，5 张预热，100 张样本连续 3 轮，单轮 **209.08 / 209.32 / 210.43 ms**，
最终平均 **209.6 ms**；分阶段约 预处理 **73.9 ms** + 前向 **91.6 ms**；精度 FP16、完整解码。
T4 换算约 **166 ms**（前向 91.6×0.53=48.5 后整链约 166.5），×1.5 约 **250 ms**，>150 ms 属不予认定档。
一致性：Pred_Label 不一致 **0 张**，概率最大绝对误差 **6.57e-4**、mean **1.63e-5**（≤1e-3）。

注：包内 `f04_core.py` / `predict.py` 的 docstring 仍写 “fp32”，与 2026-11-06 计时包字节一致；实际前向为
`torch.autocast(fp16)`，A6/F1 按 FP16 申报，未为计时改动解码或预测。

## 清单（xlsx）

以 `submition/package-F10/模型核验材料与数据申明清单.xlsx` 为底，`unzip` 解包后**只用文本编辑**替换
`xl/worksheets/sheet1.xml` 中 E2–E23 与备注 B25 的 inlineStr 文本（23 个单元格），再按原 10 个条目顺序
`zip -X` 重打包；其余 9 个 XML/关系/style 文件与 F10 版逐字节相同，`unzip -t` 通过，`perl XML::Parser` 解析良构。
F04 事实：五成员列表；B3 逐成员增强（R134R1 MixStyle local8 sites[1,3] p=0.5 α=0.1、R249 gamma-reciprocal075、
R250 sigmoid T=2、R135 RGB+XY 五通道 stem、五成员固定每批 8 正例）；D1–D5 全部“未使用知识蒸馏”；
E 项五成员等权 z 融合、无 TTA、完整解码、fp16；F1 209.6 ms（>150 ms 档）、F2 0 不一致 / 6.57e-4、
F3 495,097,409 B。xlsx SHA-256 `9886a91ac0a944dcf52caf56ccbed86d5b84ae1062c7de8b91314db559185433`。

## 包与 MANIFEST

- 目录 `submition/package-F04/`：87 个文件，文件字节合计 **697,492,321 B**；`MANIFEST.json` 逐文件字节/SHA，
  已用本机 `jq` 解析并逐项 `stat`/`sha256sum` 复核 87/87 一致。
- ZIP：`submition/TEAM_最终提交材料_F04-98.55.zip`，**644,398,660 B**，SHA-256
  `a606b3626ff3b38750f154d699bffbd61d03d39e86739066a567f5ac01e9baaa`；`unzip -t` 无错误，
  包内 `submission.csv` 重新提取 SHA = `3a9dc5b7…`。未覆盖已有 `TEAM_最终提交材料.zip`。

## 说明

- 包内 5 份 `model.json` 去掉了训练期 `job.experiment91` 指针并重序列化数值（`model_sha256`、
  `checkpoint.tensor_sha256`、job 推理字段、SITE 绑定不变），故侧车文件 SHA 与 `f04-standalone-v1` 原侧车不同；
  权重 SHA 与冻结记录一致，2026-11-06 实测即用该实现。
- 官方 5849 未并入任何 F04 成员训练，只在冻结后用于选员与该方案唯一一次官方评估
  （AUROC 0.9894371867 / AP 0.8495332210）。
- 本次未运行 Python、未使用 GPU、未改任何权重或解码逻辑。
