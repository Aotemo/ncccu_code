# F10 三成员组合枚举（2026-11-06）

用户要求"尝试 3 模型"，仅 CPU、Pod A 准备池。数据全部复用已存 draft4 前向，不新增 GPU 前向。

## 数据口径

- 14 个 F09 成员：`technical-prep/takeover-20261105-ds2/f07-owner/f09-draft4-shards/`（dev 5140 + 官方 5849，7 分片）。
- HP 成员：`technical-prep/hp-hardpos-20261106-glmB/eval-v1/hp-draft4-shards/`（1 分片，只读；HP = HP hardpos 变体，R120 EMA 热启，内部 dev 选点，variant `online`）。
- 融合：等权 z，z 统计取内部 dev；指标 (dev AUROC + 官方 AUROC)/2。C(15,3)=455 组合全枚举（`tools/diagnostics/f10_triple_search.py`，结果 `F10-TRIPLE-SEARCH-draft4.json`）。

## 前 10

| # | 成员 | dev AUROC | 官方 AUROC | 平均 | dev AP | 官方 AP |
|---|---|---|---|---|---|---|
| 1 | **R249+R269+SOUP-R120** | 0.9902299 | 0.9902413 | **0.9902356** | 0.8457 | 0.8596 |
| 2 | R250+R269+SOUP-R120 | 0.9902190 | 0.9901238 | 0.9901714 | 0.8465 | 0.8560 |
| 3 | R134R1+R269+SOUP-R120 | 0.9902540 | 0.9900775 | 0.9901657 | 0.8465 | 0.8593 |
| 4 | R269+SOUP-R120+SOUP-R201 | 0.9900408 | 0.9902830 | 0.9901619 | 0.8449 | 0.8594 |
| 5 | R249+R268-R1+R269 | 0.9900601 | 0.9902450 | 0.9901526 | 0.8442 | 0.8586 |
| 6 | R249+R269+SOUP-R201 | 0.9901059 | 0.9901978 | 0.9901518 | 0.8420 | 0.8596 |
| 7 | R269+SOUP-R120+HP | 0.9903298 | 0.9899239 | 0.9901269 | 0.8426 | 0.8541 |
| 8 | R269+SOUP-R201+HP | 0.9902756 | 0.9899452 | 0.9901104 | 0.8401 | 0.8544 |
| 9 | R250+R269+SOUP-R201 | 0.9900938 | 0.9901127 | 0.9901033 | 0.8433 | 0.8564 |
| 10 | R134R1+R269+SOUP-R201 | 0.9901733 | 0.9900294 | 0.9901013 | 0.8434 | 0.8586 |

## 判定（已被用户决定覆盖）

- 基准 F09-final 两成员（R249+SOUP-R120）0.9900255；原换方案门槛 = 基准 + 0.0003 = 0.9903255。
- 最优三成员 0.9902356，仅高出基准 **+0.0002101**，未达原门槛。

## 用户决定（2026-11-06 10:45 UTC，c607 转达）

用户原话：**"换成3模型"**。最终方案改为三成员 **R249+R269+SOUP-R120**（draft4、fp16、等权 z、dev 统计、Platt 用 dev 融合分数）。

用户给出的理由（覆盖 +0.0003 门槛）：三成员 dev 0.9902299 / 官方 0.9902413 两集均衡（F09 两成员 dev 0.9894785 / 官方 0.9905726，dev 侧偏低），且 AP 两集均高于 F09（dev 0.8457 vs 0.8430，官方 0.8596 vs 0.8562）。因此以"两集均衡 + AP 更高"为优先，接受相对 F09 的 +0.00021 增益并换用三成员。

## 同轮其它事实

- Pod A 的 OHEM 变体按 c607 指令取消：停在准备阶段（已写 training config、validation CPU allocation、site 授权），未创建 round clock/token、未启动任何 rank、GPU 未初始化；`rounds/ohem-fitonly-20261106` 为空目录，`run/` 未生成。
- `exec/hardpos` 已审阅并以 `--no-ff` 合入 main（`ca2e80d`），合并提交已推送 `origin/main`；工作树无未完成合并。

## 最终提交（F10-final）

- 配置 `configs/F10-3MEMBER-INFERENCE.json`（Pod 运行时副本 SHA `606671c7…`）：draft4（divisor 4）、fp16、TTA 关、等权 z（dev 统计）、Platt a=1.838009597150887 / b=-5.849773033998915。
- 成员权重：R249 `9544763a…` 99,018,253 B + R269 `3f77e91d…` 99,018,253 B + SOUP-R120 `28f3fc55…` 99,018,317 B = **297,054,823 B ≤ 500,000,000 B**。
- 提交 `/home/hao/Desktop/challenge/submition/F10-final/submit.csv`：5855 行 + 表头，CRLF，行序与 `submition/R120-98.45/submit.csv` 逐行一致，正类 196，SHA-256 `b70bd76f12b1a9a413557f245ca78becd018bfdeeba65103b92cd68ed058a795`。
- R269 与 SOUP-R120 使用构造导出（仅清 job 的 `experiment91` 训练侧指针，权重文件字节不变、SHA 相同），回执在各 export 的 `STANDALONE.json`。

## 耗时（规则第 7 条，Pod A 单卡 fp16）

```
[ROUND] round=1 mean_ms=85.26 diag={"pre_ms": 22.96, "stage_ms": {"h2d": 0.58, "forward": 55.01, "post": 0.02}}
[ROUND] round=2 mean_ms=85.79 diag={"pre_ms": 23.5, "stage_ms": {"h2d": 0.58, "forward": 55.44, "post": 0.02}}
[ROUND] round=3 mean_ms=86.02 diag={"pre_ms": 23.35, "stage_ms": {"h2d": 0.58, "forward": 55.64, "post": 0.02}}
[CONCLUSION] 结论：100 张样本、batch=1 串行、连续 3 轮；单张端到端墙钟时间先按每张 3 轮取均值，再对 100 张取平均，最终平均耗时 85.7 ms，未超过上限；预测结果与 submission.csv 比对 100 张，不一致 0 张；推理精度 fp16。
```

T4 换算：前向 55.4 × 0.53 = 29.3 ms，合计 ≈59.7 ms；×1.5 安全系数 **89.5 ms < 100 ms**。

## 许可证（三成员同一骨干）

| 项 | 内容 |
|---|---|
| A1 名称 | ConvNeXt-Small（timm `convnext_small.fb_in22k_ft_in1k`） |
| A2 链接 | https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k/resolve/fe6b358aa409414cdb522ae3d43a2c9922f7cc1b/pytorch_model.bin |
| A3 许可证 | Apache-2.0（https://www.apache.org/licenses/LICENSE-2.0.txt ），非 NC、非 gated、允许再分发 |
| A4 上游训练数据 | timm 模型卡：ImageNet-22K 预训练 → ImageNet-1K 微调；主数据集为通用自然图像，非医学 |
| A5 上游权重 SHA-256 | `c912f45cd738512978a900cc8aefade65bc44e507321f0144378a2d98aa5e914`（201,017,809 B） |

| 成员 | 来源 | 最终 model.pt SHA-256 | 字节 |
|---|---|---|---|
| R249 | r249-small-gamma-seed29（EMA，f04-standalone-v1） | `9544763aff26949d46ba8e9a3c4dbcbf8cacdaf120f4038e329338eb131bfe5e` | 99,018,253 |
| R269 | r269-small-f05distill-seed20261202（EMA，f07-standalone-v1 fp16） | `3f77e91d7ca1dd12038b6b45a722985f44977a83eab80e517473af5a48ba9391` | 99,018,253 |
| SOUP-R120 | R120/R254/R256/R258/R262 selected EMA 等权平均（F05-SOUPS-v1） | `28f3fc555c4d81c02ac2e34f4c06bbcfbeede7d18af96d06068fa1d40d5a24e7` | 99,018,317 |

三成员均只在本任务 fit 图像上微调或由本任务模型合成，无外部医学数据、无 N06/FLAIR 血缘。
