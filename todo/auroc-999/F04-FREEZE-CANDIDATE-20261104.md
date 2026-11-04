# F04 当前冻结候选清单（尚未最终冻结）

2026-11-04 c607 指定当前 F04 为最佳合规冻结候选。下列成员按已封存的原单模排序与真实 fp16 文件字节选出，合计 **495,097,409 B ≤ 500,000,000 B**；选择依据和回执 [F04](../../records/analysis/20261104-f04/README.md)。当前 `0.9902539582 / 0.8435746088` 是原 **fp32 dev 预测**的 AUROC/AP，不是 fp16 全 dev 结果，也未达到 `.995 / >.9` 目标。此文是候选登记，不是最终冻结或官方评估授权。

| 成员 | 唯一 selected EMA | 原 selected tensor SHA256 | fp16 文件 SHA256 | 实际字节 | 原 fp32 dev z 均值 / 总体标准差 |
| --- | --- | --- | --- | ---: | --- |
| R120 | c6 @ 4800 | `f21bffae88acbf7c216c4141c0ef0b9d990a624e3e92a1baf750a501c415fc85` | `e0bc2c52dcc62adcb8ab4f4fe47c8126b31a0939732deca3985d5757c8d99e83` | 99018253 | `-41.568375320352814 / 16.558998314435453` |
| R134R1 | c4 @ 3200 | `1442a625f8596cf8357a9c1d58f4c7b40d998d1dbe9dfc6eb880a193edf10462` | `600cf0fdd2db11da45b5e3e13fa9b2ccc941c0d4a6faf7c985be69d40a085661` | 99018253 | `-12.771333054145835 / 4.543301802872881` |
| R250 | c5 @ 4000 | `a0b54a0846a3949f4fa280196f93e724290574acc406c3ad7e4d62dda812c861` | `8af616adf5c02c4512189b404c2ca1f55b4169cb3f9dfafc661207b542cf4c0b` | 99018253 | `-53.73072970750586 / 21.385734528377952` |
| R249 | c9 @ 7200 | `6251937d3d778ed4be5d0cc11fc505fc4c2cc897d0ae18ff31ddfb6c8dce8a73` | `9544763aff26949d46ba8e9a3c4dbcbf8cacdaf120f4038e329338eb131bfe5e` | 99018253 | `-50.80369008629934 / 19.83424888353614` |
| R135 | c10 @ 8000 | `cfe343da157e330aa6aba024317a09826674ba63d441fd658402bac0f8126a39` | `2a936d4964c4948fccc90758fa5298373ddd5e73175d1a232b230605aadcfe5a` | 99024397 | `-18.079126347632577 / 5.606855365383561` |

唯一文件位于 exp-v3 的 `technical-prep/takeover-20261104-08ac/f04-exports-e68/<成员>/weights-fp16.pt`，每个文件包含推理所需 backbone/head/age 与持久 buffer；原模型、源代码、预处理和数据身份及严格加载回执在各成员 [`EXPORT.json`](../../records/analysis/20261104-f04/README.md) 中。原冻结源码 commit 依次为 `7922dd788f982bcb72d30a1a8e6d6e7761e42e88`、`49c8ad727073604135fc95f4984796e6b00a7808`、`2742968868e9c3e5cffd4ae35dbf5c5208eb320c`、`631ce2012e18c4ef6bee5f57e3fc6a6746195641`、`dbbe65531d511b91c89f63cd8a4cbba2e68e7303`。R135 是五通道 RGBXY stem，不能用普通 RGB 模型替代。每个成员都必须复用各自冻结的输入、normalization、age 和结构。

推理固定式为五个成员各输出一个单视图 logit，分别按相应内部 dev 均值和总体标准差标准化，再等权平均。表中常数来自原 fp32 dev logits，仅供当前标注为 fp32 预测的候选；fp16 文件在完整 dev 前向后应按原 F04 规则得到并记录实际 fp16 logits 的均值和总体标准差及融合指标，不凭小样本假定两套常数完全相同。

现有内部验证入口为 `tools/diagnostics/f04_predict_member.py`：调用各轮冻结的 `retrain.inference._load_model`、`ImageDataset` 与 age 路径，严格载入导出的 fp16 state 到 fp32 模型。它目前仍先依赖原 selected checkpoint 构造模型，并非可直接提交的五成员独立入口；不能把大 checkpoint 当成最终文件或声称该入口已完成。最终需用各自版本化结构/配置直接构造，仅读取上述五份文件并使用精确预处理、age 与固定 z 常数，完成无原 checkpoint 的严格加载及单视图等权推理验证，再测完整 dev 的 fp16 输入、指标和文件总量。现役结束后的既有 Pod GPU 仅执行内部 dev 全量验证，不读官方；官方5849只对最终冻结方案一次。

**独立加载准备范围（尚未完成）：** 新增 `tools/diagnostics/f04_standalone_member.py` 与对应窄测，复用已封存 F04 inventory/成员导出及各历史冻结 `retrain.inference._load_model`。仅在准备池从 selected 读取已验 job，生成带完整 age/结构身份的 `model.json`，对原 fp16 文件建立同文件系统硬链接 `model.pt`；不复制或修改权重。随后以导出目录独立严格加载，核 fp16 文件/tensor SHA、模型状态，以及同一固定合成图和 age 的单视图前向是否与直接装载 fp16 权重的历史模型一致。五个结果先逐成员留 exp-v3 小回执；不读取 dev/官方、不重新选成员、不进行 GPU 前向。此技术通过后才可作为全 dev GPU 验证入口的一个输入；GPU 入口仍须另做实际安全/锁/资源及发布校验，前轮终态时未备妥则直接启动已备妥后备轮。

CPU 窄测在旧源码缺独立导出工具时失败 `95522962`；草稿以模型工厂严格加载 fp16 导出并拒绝篡改 SHA 通过 `e4533127`。这仅是隔离单元证据，五个真实成员各自历史 runtime 的独立构造与合成前向尚待逐个实测，完整 dev 及最终融合入口尚未完成。
