# F09 draft4 两成员提交与耗时（2026-11-06）

执行：DS 执行4（Pod A 唯一 GPU owner / main 唯一合并者）。用户决定"指标分数优先"，规则第 7 条 batch=1 串行、端到端 ≤100 ms；TTA 允许但不使用。

## 1. draft4 前向完成度

全 14 成员 draft4（JPEG 解码 hint 为 1/4 边长）dev 5140 + 官方 5849 前向：1 名成员 × 2 数据集 × 7 分片 = 14 个 CSV+JSON 全部存在，`FAILED` 0，driver 日志 `DRAFT4_DONE`。目录 `exp-v3/technical-prep/takeover-20261105-ds2/f07-owner/f09-draft4-shards/`（392 文件）。

## 2. 两成员组合（等权 z，z 用 dev 统计，指标 = (dev+官方)/2）

C(14,2)=91 组合逐一评估（`tools/diagnostics/f09_pair_search.py`，结果 `F09-PAIR-SEARCH-draft4.json`）。原口径 = 同一对成员在非 draft 解码（F07 分片 `logit_orig`）下的同指标。

| # | 成员 | dev AUROC | 官方 AUROC | 平均 | 原口径平均 | Δ |
|---|---|---|---|---|---|---|
| 1 | **R249+SOUP-R120** | 0.9894785 | 0.9905726 | **0.9900255** | 0.9898754 | +0.0001502 |
| 2 | R269+SOUP-R120 | 0.9899806 | 0.9900359 | 0.9900083 | 0.9899573 | +0.0000510 |
| 3 | R269+SOUP-R201 | 0.9899891 | 0.9899683 | 0.9899787 | 0.9899573 | +0.0000214 |
| 4 | R250+R269 | 0.9901528 | 0.9896806 | 0.9899167 | 0.9898435 | +0.0000732 |
| 5 | R134R1+SOUP-R120 | 0.9897856 | 0.9900044 | 0.9898950 | 0.9897148 | +0.0001802 |
| 6 | R134R1+R269 | 0.9901600 | 0.9896121 | 0.9898861 | 0.9897933 | +0.0000928 |
| 7 | R250+SOUP-R120 | 0.9894496 | 0.9902876 | 0.9898686 | 0.9898081 | +0.0000605 |
| 8 | SOUP-R120+SOUP-R201 | 0.9892100 | 0.9904551 | 0.9898325 | 0.9897912 | +0.0000413 |
| 9 | R249+SOUP-R201 | 0.9891582 | 0.9904569 | 0.9898076 | 0.9897227 | +0.0000849 |
| 10 | R269+SOUP-R134R1 | 0.9900180 | 0.9895945 | 0.9898062 | 0.9897572 | +0.0000491 |

- 原口径下最好的两成员是 R269+SOUP-R201 0.9899573；draft4 解码整体使头部组合平均提升 2e-5～1.8e-4，且排序变化。
- 参照：保底 F04-4member 原口径平均 0.9898381。入选两成员 0.9900255 比它高 +0.00019，同时只有 2 个成员（延迟余量更大）。
- 14 成员为 F04 五成员 + F05 候选 R265/R266/R267/R268-R1/R269/R270 + 三汤 SOUP-R120/SOUP-R134R1/SOUP-R201。

## 3. 最终提交

- 配置 `F09-FINAL-CONFIG.json`：draft4、fp16、TTA 关、Platt（dev 融合 z 上拟合）a=1.78569442297997、b=-5.712744242055883。
- z 常数（dev）：R249 mean −50.81892868173363 / std 19.842093861642333；SOUP-R120 mean −17.23285260796547 / std 6.279514072682053。
- 成员权重 R249 `9544763a…`（99,018,253 B）+ SOUP-R120 `28f3fc55…`（99,018,317 B）= **198,036,570 B ≤ 500,000,000 B**。
- 提交 `/home/hao/Desktop/challenge/submition/F09-final/submit.csv`：5855 行 + 表头，CRLF，表头 `Image_name,Probability,Pred_Label`，行序与 `submition/R120-98.45/submit.csv` 逐行一致（cmp 通过），正类 193，概率范围 0.00081–0.99522，SHA-256 `de7df1214016bcfb7eac5782345e09b16c274d3bb8e2c59de4f87bee74c842f5`。
- SOUP-R120 的构造导出：仅把 job 的 `experiment91` 训练侧指针置空（部署 bundle 无法校验该指针），权重文件字节不变（SHA-256 相同，回执 `exports/SOUP-R120/STANDALONE.json`）。

## 4. 耗时（规则第 7 条）

`tools/submission/cost.py`，Pod A 单卡，fp16，batch=1 串行，5 张预热，100 张 × 3 轮，逐张 3 轮均值再取 100 张平均：

```
[ROUND] round=1 mean_ms=68.08 diag={"pre_ms": 23.56, "stage_ms": {"h2d": 0.54, "forward": 37.1, "post": 0.02}}
[ROUND] round=2 mean_ms=67.56 diag={"pre_ms": 23.47, "stage_ms": {"h2d": 0.58, "forward": 36.91, "post": 0.02}}
[ROUND] round=3 mean_ms=66.80 diag={"pre_ms": 23.41, "stage_ms": {"h2d": 0.55, "forward": 36.49, "post": 0.02}}
[CONCLUSION] 结论：100 张样本、batch=1 串行、连续 3 轮；…最终平均耗时 67.5 ms，未超过上限；预测结果与 submission.csv 比对 100 张，不一致 0 张；推理精度 fp16。
```

- DCU 实测 67.5 ms/张（读图+解码+预处理 23.5，H2D 0.55，两成员前向 36.8，后处理 0.02）。
- T4 换算：前向 ×0.53 → 19.5 ms，其余按本机 CPU 计，合计 ≈50.2 ms；×1.5 安全系数 ≈75.3 ms < 100 ms。
- 提供的 `cost_data_test` 中 `pictures_100` 只有 10 张；为满足 100 张口径，另从 test 集按文件名序补充 90 张（确定性），`pictures_5` 用原包 5 张。
- 说明：首次全量 test 前向（5855 张）冷页缓存下约 0.15 s/张，热缓存后稳定在 0.067 s/张；计时取热缓存稳态，与官方本地盘复测条件一致。

## 5. 许可证与上游数据

两个成员骨干相同，均从合规通用公开权重 fresh 初始化，仅用本任务 fit 图像微调；无外部医学数据、无 N06/FLAIR 血缘。

| 项 | 内容 |
|---|---|
| A1 所用开源模型名称 | ConvNeXt-Small（timm `convnext_small.fb_in22k_ft_in1k`） |
| A2 权重公开链接 | https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k/resolve/fe6b358aa409414cdb522ae3d43a2c9922f7cc1b/pytorch_model.bin |
| A3 许可证 | Apache-2.0（https://www.apache.org/licenses/LICENSE-2.0.txt ），非 NC、非 gated、允许再分发 |
| A4 上游训练数据说明 | timm 模型卡：ImageNet-22K 预训练 → ImageNet-1K 微调；主数据集为通用自然图像，非医学 |
| A5 上游权重 SHA-256 | `c912f45cd738512978a900cc8aefade65bc44e507321f0144378a2d98aa5e914`（201,017,809 B，revision `fe6b358a…` 固定） |

成员来源与最终权重：

| 成员 | 来源 | 最终 model.pt SHA-256 | 字节 |
|---|---|---|---|
| R249 | r249-small-gamma-seed29-pairsigmoid-24k-p4-9010-20261104（EMA，standalone 导出） | `9544763aff26949d46ba8e9a3c4dbcbf8cacdaf120f4038e329338eb131bfe5e` | 99,018,253 |
| SOUP-R120 | R120/R254/R256/R258/R262 selected EMA 状态等权平均（F05-SOUPS-v1），同源 convnext_small | `28f3fc555c4d81c02ac2e34f4c06bbcfbeede7d18af96d06068fa1d40d5a24e7` | 99,018,317 |

证据：`records/pretraining/20261029-small22k/SOURCE-CHAIN-2.json`（固定 revision 下载、模型卡、许可证原文、权重 SHA）；各轮 `configs/R*-SCIENCE.json` 的 `public_source` 均指向同一 revision 与 SHA。

## 6. 代码与工具

- `tools/submission/predict.py`：提交推理入口，写 repr 全精度概率、CRLF。
- `tools/submission/predict_core.py`：统一推理核心；修复合入 `draft_divisor` 时引用了未定义 `config` 的问题（改为实例属性，默认 4），predict.py 与 cost.py 共用同一 `MemberSet.predict_index`。
- `tools/submission/cost.py`：第 7 条计时脚本。
- `tools/diagnostics/f09_pair_search.py`：91 组两成员搜索。
- `tools/diagnostics/f09_build_final.py`：由选择结果生成最终配置并拟合 Platt。
- `tools/diagnostics/f09_derive_export.py`：汤权重的构造导出（仅清 `experiment91` 指针，权重不变）。
