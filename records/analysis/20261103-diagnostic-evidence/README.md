# 诊断证据工具与首次实际分析

用户在实验复盘后要求“根据这个开发”。已新增 [paired_evidence.py](../../../tools/diagnostics/paired_evidence.py) 与 [九项测试](../../../tests/test_paired_evidence.py)，复用原错序贡献函数和原指标函数。工具读取既有选点预测，报告配对 AUROC 区间、修复／新增／共享错序质量，以及 fit/dev、人工图像结构审阅证据是否齐备；不自动改变选点或训练队列。

首次运行覆盖固定内部 dev 的5140张图，167正例、4973负例。三轮 manifest、完整完成回执、selected freeze、CSV、precision、张量与更新数身份均已核验，指标复算一致。详细结果见 [SUMMARY.json](SUMMARY.json)，实际源码身份与 CPU 回执见 [VERIFICATION.json](VERIFICATION.json)。

## 实际结果

| 轮次 | 唯一 selected | 内部 AUROC | AP | 错序质量 |
| --- | --- | ---: | ---: | ---: |
| R107 | EMA c4@3200 | 0.9884405731 | 0.8274326888 | 9600 |
| R120 | EMA c6@4800 | 0.9894773092 | 0.8313831150 | 8739 |
| R201 | EMA c5@4000 | 0.9889535227 | 0.8363466611 | 9174 |

预声明三项比较；2000次正负类分层图像 bootstrap，PCG64 seed20261103，两模型每次使用相同样本权重。区间采用 percentile、NumPy linear 分位数。

| 候选−参照 | AUROC差 | 配对95%区间 | 修复质量 | 新增质量 | 共享质量 |
| --- | ---: | --- | ---: | ---: | ---: |
| R120−R107 | +0.0010367361 | [−0.0011201506, +0.0038679227] | 2987 | 2126 | 6613 |
| R201−R107 | +0.0005129496 | [−0.0007659324, +0.0019761503] | 2385 | 1959 | 7215 |
| R201−R120 | −0.0005237865 | [−0.0027370555, +0.0012450466] | 2216 | 2651 | 6523 |

区间均跨0。点估计有改善，当前分析不足以证明稳定增益，也不能证明方案等效或毫无作用。区间条件于这几个已训练、已选点的模型，不包含跨 seed 波动，不能消除反复使用同一 dev 的选择偏差；三项比较未做多重性调整。抽样单位是图像，未取得患者ID，不能称患者级区间。`fraction_delta_gt_zero` 是抽样统计量，不是 p 值或候选真实更优的概率。

原四锚点 R107/R108/R120/R134R1 定义的固定20正例保持不变，新轮不参与分组。R120相对R107净减少861错序质量，其中固定20减少794、其余147减少67；R201相对R107固定20减少531，其余147增加105，净减少426。修复与新增同时存在；局部难组不能替代完整内部 AUROC。所有数量都是正负例错序配对质量，同分计0.5，不是误判图片数，也不是独立观测数。

三个模型都没有提供同一 selected 张量的 fit 预测；来源可核验的人工原图／实际512视盘可见性审阅表也未提供。本次 `fit_dev`、`structure` 均为 `unknown`，`causal_diagnosis=not_identified`。用户明确正例对应青光眼，标签仅依据眼底照片；本工具没有医学复核，不把难例或图像质量指标当作标签错误证据。

## 输入与复用

清单和样本级材料仅放 exp-v3。通过当次实时核验、本人绑定的隐藏GPU CPU guard 传入以下目标脚本与参数；不得在本机或登录节点直接执行 Python：

```text
<部署根>/tools/diagnostics/paired_evidence.py
  --inventory <exp-v3内绝对路径>/INVENTORY.json
  --inventory-sha256 <清单真实SHA256>
  --output <已存在exp-v3目录>/新的报告.json
```

输入清单字段：

| 字段 | 合同 |
| --- | --- |
| `manifest` | `{path, sha256}`；固定 train 来源，fit/dev 不交叠且覆盖全部行；沿原数据入口的文件名 casefold 约定。 |
| `models` | 至少两个唯一 `id`；每个含固定 `{path, sha256}` 的 `freeze`、`completion`；只接受同轮根目录或 `evidence/` 下 completed 的 `TRAIN_RESULT.json`，selected更新数不得超过完成更新数。 |
| `comparisons` | 预先指定 `[候选id, 参照id]` 列表；工具不搜索最优组合。 |
| `bootstrap` | `replicates`（2至20000）、非负整数 `seed`、`confidence`；本次固定2000／20261103／0.95。 |
| `fixed_positive_indices` | 可选，预先封存的 dev 行位置；必须是唯一正例，不根据新结果重选。 |
| `fixed_positive_group_source` | 可选，原封存来源与摘要；本次归档原源SHA及旧四锚点定义。 |
| `bootstrap_groups` | 可选固定CSV `{path, sha256}`，列 `name,group_id`，每个dev样本按原顺序一行；由提供者说明组含义，不自动把内容组当患者ID。 |
| `structure_review` | 可选人工CSV `{path, sha256}`，格式见下文。 |

freeze 的 manifest、FP32／identity、selected张量、更新数和指标必须相符；CSV 名称／标签／行序必须与 manifest 一致，precision 与 selected 状态一致。只接受内部 `predictions/{ema,online}/dev-cycle-####.csv` 路径。SHA或身份错配、非完整轮次、官方路径均拒绝；不替换样本或改选 checkpoint。

可选 `models[].fit_prediction` 由 `{path, sha256, manifest_indices, precision:{path,sha256}}` 组成。路径限定 `predictions/{ema,online}/fit-selected.csv` 或 `fit-cycle-####.csv`，CSV列为 `name,label,logit`；`manifest_indices` 只允许 manifest 的 fit 行，顺序需一致。precision 必须指向同 selected 的 variant／identity view／tensor／successful_updates。报告样本数、是否完整fit、指标和 fit AUROC−dev AUROC；部分fit会明确标注，不把缺失填成0，也不凭 gap 直接诊断过拟合。本工具仅消费现有预测，不生成 fit 前向。

人工结构审阅CSV列为：

```csv
name,original_sha256,prepared_sha256,original_disc_visible,prepared_disc_visible,label_review,reviewer
```

每行需属于manifest，原图SHA与manifest一致，实际输入SHA合法，观察者非空。视盘可见性取 `yes/no/unknown`，标签审阅取 `agree/disagree/uncertain`。工具只统计“原图yes、处理图no”等人工观察，不加载图像、不自行核验处理图、不更改标签；输出明确 `images_independently_checked=false`。应由审阅者将已核实的原图和实际输入哈希一并提供，不能用清晰度代替目标征象可见性。

分组抽样只在提供完整组表时启用；每次对组抽样，并把同组权重同时用于两模型。单类抽样会被拒绝并记录；至多尝试10倍次数，不暗中退回图像抽样。输出是独占新建的必要汇总，不包含样本名称、分组成员或逐样本预测。

## 验证与限制

九项新测试实际通过（0.251秒），两个新文件编译通过；真实 CLI 已完成上述三项比较。测试覆盖独立重复样本穷举AUC oracle、同分、共享重采样与显式分组、selected与来源身份、失败／运行中拒绝、原evidence布局、casefold兼容、同selected fit约束及 unknown、人工审阅聚合。首次 RED 以及实际接入失败均保留，未把失败回执改写为通过。[最终编译与归档CPU原始回执](ARCHIVE-CPU.json) SHA为 `43bd388bf6cce8cf174e4a76c26f766c04dda0126efcbfbffa584f86f17bd3bc`，exit0。

实际接入修正了两个实现假设：完成回执位于 `evidence/`，以及真实文件名与CSV名字仅大小写不同。固定20名单第一次摘要比较失败来自 JSON 编码空格；采用原 `json.dumps(fixed)` 后与已归档摘要一致，四锚点组质量也逐项一致，成员和定义未改。

仅新增诊断代码与小汇总，不读取官方5849、原图、模型或checkpoint，不做新 forward；不修改活跃训练、runtime/config、原八件分析或训练队列。本工作未调用 Git 写入，由唯一Git owner统一集成。没有新增训练收益或达到0.999的声称。

配对重采样的方法边界参见 [SciPy bootstrap 文档](https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.stats.bootstrap.html)；本实现只使用NumPy，分位数定义见 [NumPy1.26 quantile](https://numpy.org/doc/1.26/reference/generated/numpy.quantile.html)。反复选模型的选择偏差不能由本次重采样消除，参见 [Cawley与Talbot（2010）](https://jmlr.org/papers/v11/cawley10a.html)。
