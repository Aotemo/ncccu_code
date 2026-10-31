# Fit 联合输入一致性：一次纯元数据核对

状态：2026-10-31 一次纯元数据任务已实际完成，`C=0`，原任务收口。只排除本页定义的严格封存 SHA+FP32 年龄联合输入冲突，不能据此证明标签正确、实际 cache 像素一致或拟合充分。本任务不是训练或 FIT256 补考，不占 R123/order/SCIENCE；以下保留原合同与最终结果，不授权新增执行。

问题：在封存的图像字节身份与原确定性输入合同下，当前 fit 是否存在“同一图像、同一实际年龄输入，却标签相反”的记录？按 content-group 保持划分不回答这个完整输入问题；既有八例标签传播审计也不代表全 fit。仅核监督约束的一致性，不预设冲突存在或主导泛化差距。

- **唯一范围与复用。** 先复用现成、精确对应本任务 fit 身份及联合 key 定义的统计；全 train 的 mixed-group 总数或仅同图计数不能冒充。否则在同一次预算内，只消费 R120 封存 manifest、原 age spec 及 R107 同一 9:1 的已有 identity receipt，对 `fit_indices` 指向的 46256 行统计。不重拆分、重算年龄统计或调用 `make_spec`；不为此调用会扫描目录的 source loader。不统计 dev，不读预测、图像、缓存、权重或模型，不导入 Torch，不做 GPU 或任何前向。
- **固定 key。** 使用既存 `sha256` 字符串作为封存字节身份，原 `age.features` 配合冻结 age spec 得到 standardized_age/missing 两维，并按真实模型输入转为有限 FP32 数值；key 为该 SHA 串与两维数值。`+0/-0` 数值相等，必须同 key，不能按不同位模式误拆。缺失年龄规则、原标签保持；不从文件名推患者，不用现有 group 代替 SHA，不重新 hash 图像。该 key 只基于封存身份与预期预处理，不验证实际 cache 像素。
- **一次预算与停止。** A 在实时有效 20% 本人份额、nice19、隐藏 GPU 的 guard 内执行；300 秒硬预算覆盖复用检查、身份、读取及聚合。身份缺项、不匹配、年龄特征非有限、超时或不能完整完成均为 unknown，结束且不重试、补样、扩大范围或另加探针。B 无任务；本页不授权规划或其他执行者计算。
- **最小聚合。** 只回 fit P/N 数、重复 joint-key 数及涉及行数、cross-label joint-key 数及其中 P/N 行数、`C=sum(p_k*n_k)`、身份/finite/elapsed 与小回执。重复 key 指该 key 总行数大于 1；cross-label 指该 key 同时有正负记录，`p_k/n_k` 为该 key 内正负行数。名单、年龄逐值、SHA 键、逐组表不导回或入 Git。C 是元数据定义的交叉标签对数，不是模型观测到的错序、AUROC 或独立样本数量。

预定解释：完整且 C>0，只表明在封存字节身份和确定性预处理成立的条件下，fit 中存在完整输入一致而监督相反的约束；确定性函数对这类输入应给相同分数。不能据此判哪个标签错、患者身份、临床原因，或给出内部 dev/平台指标上限。完整且 C=0，只排除这类最严格的元数据冲突，不排除近重复、未知标签语义、其它监督误差或信息损失，不证明标签正确或 `.999` 可达。unknown 不以部分计数补结论。

结果只交协调方决定下一问题；无论 C>0、C=0 或 unknown，均不自动授权改标、去重、设权、改采样、开启 loss 实验、修缓存、补跑 FIT256 或训练。原 FIT256 工程 unknown/0of256 和独立缓存核对的首 entry_missing 事实不变，不能将一个缓存项缺失解释成训练数据缺失。R107/R108 科学锚点、R120 单次内部 best `.9894773092062407`、完整内部 selected AUROC≥`.996` 才官方 5849 一次的独立门，以及总体 raw AUROC `.999` 目标保持；不授权平台 test，不保证收益。

**最终结果（协调方转交 A 实际回执，规划未重算）。** R107/R120 原 manifest 字节 SHA/digest、fit_indices SHA 与 age contract SHA 一致，原 `age.validate_spec` 通过。只统计 fit 46256 行：P1490/N44766；原 `age.features` 经 `np.float32` 转换、±0 数值合并，全部 FP32 有限。duplicate joint-key=0/涉及行=0，cross-label joint-key=0/涉及正=0/负=0，`C=sum(p*n)=0`。这不是模型前向或 AUROC 结果，也不是新训练瓶颈已获证明。

一次 metadata elapsed `2.163719s`，CPU17 当次本人 `0..11`/12、guard exit0；实时有效 CPU120、prep `0..23` 与 train `24..119` 互斥，UID `51fe…` Ready、Python3.12.7、nice19/hiddenGPU。以上仅为历史执行回执，不能沿用为新任务入口。没有 Torch、模型、图像、cache/stat、forward 或训练。原小回执目录 `work/r120-fit-input-consistency-20261031-a/receipts`；RESULT Pod SHA `cda8c1a8f8555bf97a8d44ec3871ed38d57971b8bd717aaf64073b366b4ada27`，CPU Pod SHA `6c4d515df5a71a8e47fa59f0b1f3c15539207c9b4eb21d9878efed83f3fc4de1`。

按预定 C=0 分支停止本核对，不延展元数据审计队列。只能排除这类严格元数据冲突，不能推出临床标签正确、实际缓存像素正确、优化充分、拟合失败、`.999` 可达或任一 dev/平台上限。原 FIT256 unknown 保持且不补跑；缓存核对仅首条 missing 已关闭。总体目标未达，下一科学问题由协调方另行决定，本结果不自动授权任何训练或数据修改。
