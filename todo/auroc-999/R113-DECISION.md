# R113：冻结公开早层参数的有限对照与实施计划

**Goal:** 检验限制早层参数更新是否改善当前单模型内部 AUROC，直接匹配 R107，不继续 BN 校准分支。

**Architecture:** 复用公开 ImageNet22K→1K ConvNeXt Small，固定 `network.features[0:4]` 参数，后段、分类头及年龄分支继续训练。网络前向、训练模式和 SD 保持；EMA 从 online 的真实冻结参数名单逐次复制这些张量，其余参数沿原规则更新。

**Tech Stack:** 现有 PyTorch/TorchVision、8 rank DDP、全局 pairwise logistic、版本化 SCIENCE 和既有正式启动链。

**Spec:** 本页是唯一科学合同及短实施计划。协调方根据用户继续推进整体目标的要求选择此方向，不冒称用户点名了冻结早层。沿现有 A/B 执行分工，不新建 Agent、文档审批门或过程提交；本页冻结不代表检查通过或已开训。

## 科学身份和不变量

| 字段 | R113 |
| --- | --- |
| config | `configs/R113-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-freezeearly-v1` |
| candidate | `r113-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-freezeearly-9010` |
| order | 23 |
| round | `r113-small22k-seed20261029-freezeearly-24k-p4-9010-20261030` |
| 唯一科学新增字段 | `job.freeze_features_before = 4` |
| 直接匹配基线 | R107，seed20261029，c4 EMA @3200，内部 AUROC `0.9884405731067525`，AP `0.8274326888154083` |

配置从 `configs/R107-SCIENCE.json` 的科学字段派生，只改变身份和上述冻结字段；不能从 R112/R110 复制遗留的 R50、BN 校准或 SD0 设置。公开 Small revision `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b`、转换权重 SHA `7aaea5ce6fe6db953874d9792d3f1711a5cc0193a10d740e023108043918c610` 和既有 provenance/来源原件继续使用；不加载 R107 checkpoint，也不使用 N06/FLAIR 参数、teacher 或统计。

保持 seed20261029、单视图、512 prepared/input、ImageNet 归一化、`training_scale=[0.85,1.0]`、G64/P8、8 rank 各8主图、全局8正56负448对。原 loss 六字段为 `version=binary-pairwise-logistic-v1, kind=binary_pairwise_logistic, temperature=1.0, reduction=mean, dtype=float32, scope=global_batch`。backbone LR1e-4、head/age LR1e-3、原 WD/no-decay 分组、dropout0.2、SD0.1、最终及内部 LN epsilon1e-6、年龄合同和固定9:1清单全部保持；不改变采样、PIL 增强或 gather 算法。

每800次成功更新一次完整 dev，最多24000/30周期，warmup 和 EMA warm-copy 绝对1600、其后 decay0.999；内部 max(online,EMA) AUROC、严格提升/min_delta0、patience4/min_cycles1及唯一 selector 不变。工程 train21600/total28800/GPUgrant32400不变，不是测得吞吐或云实例期限承诺。`p4` 表示早停，正例配额仍 P8。

## 精确冻结和 EMA 接口

1. `freeze_features_before` 是最小可选字段：缺省完全不注入旧 job；出现时只接受整数4且模型为 `convnext_small`，不扩为任意深度或通用冻结表达式。`job_contract` 的现有完整复制保留该键，ex91 校验从绑定 SCIENCE 读取期望。
2. A 在 `models.py` 提供 `apply_feature_freeze(model, job) -> tuple[str, ...]`。缺省返回空 tuple，不改对象状态、模式或 RNG；字段为4时只对 `model.network.features[0:4]` 的参数设 `requires_grad_(False)`，返回排序后的完整冻结参数名。范围为 stem、stage1、下一 transition 和 stage2；`features[4:]`、最终 LN、分类头和 age 仍可训练。不调用前段 `.eval()`，不增加特殊 `no_grad` 前向，不关闭或删除 SD。
3. 工厂在原完整公开 strict 加载、新头及年龄构造之后应用 mask；`initialize=False` 的 checkpoint 重建也应用同一 mask，但不重新加载公开权重。冻结不消耗初始化 RNG。预期全量初态仍为 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，由新 Git 正式原生 CPU 流程一次实核，不能拿预期值代替实际结果或另做 control-only。
4. B 保持 `EMA(online, rule, *, state=None, successful=0)` 接口和现有 `state_dict` 六键不变；在 deepcopy 前，从 `online.named_parameters()` 中 `requires_grad=False` 的条目建立排序 tuple `self.frozen_parameter_names`。每次成功更新，将此名单与现有固定 BN copy 名单取并集并 `copy_` online 值；其余张量保留原 warm-copy/blend、skip/cursor 语义。不能从 EMA 模型的全 False 属性推名单，不能因禁止 optimizer 更新却允许 EMA 浮点混合漂移。恢复时名单来自已按 job 重建 mask 的 online 模型，不从 state_dict 猜测。
5. `requires_grad` 不在 state_dict 中。正式 CPU 与真实8 rank 初态必须分别证明：全量 state/来源身份、准确冻结和可训练名单、两者互斥且覆盖全部参数、optimizer 成员恰好为可训练集合且无重复、空 Adam、EMA 初态相同。全量342 backbone tensor/49,454,688 backbone 参数等结构计数继续表达全量模型，不能冒充可训练计数；新可训练数量由实际模型取得，不先填数字。可在既有 `candidate_normalization`/初态证明中承载可选冻结描述，不增加独立证明阶段。

## 假设、反例与边界

假设是全量微调可能损害部分公开可迁移表征，保留早层或能改善最终所选点；当前未证明遗忘、过拟合机制或损害具体发生在哪一层。N06 的卷积及 BN affine 本来可训练，本实验不是复刻 N06，也不能解释其99.32的来源。ImageNet 早层也可能需要眼底适配，固定它们可能降低性能；层之间的共同适应使切分本身可能增加优化难度。[迁移研究](https://arxiv.org/abs/1411.1792)只提供一般依据和反例。

只冻结权重，训练时前段 SD 随机路径仍存在；不声称冻结后是确定性特征提取器，也不声称完整训练轨迹与 R107 相同。完整前向保留，前段反向及相应 Adam 状态可减少，实际节省未经测量。采样与图像输入合同不变，不以新吞吐测试增加任务。

## 文件边界和短实施计划

规划仅本页，写完冻结；A/B 不共写同文件，旧子 Agent 不唤起。

- **A 独占：** `configs/R113-SCIENCE.json`，`runtime/bundle/retrain/{config.py,experiment91.py,models.py}`，`tools/validation/check_candidate.py`，现有 `tests/test_candidate_binding.py`。现行 ex91 是 R111/R112 R50 分支，须恢复本轮 Small 精确候选/来源/无 BN 分支，不能要求 Small 带 R50 `head_seed` 或 `bn_statistics`。`check_candidate` 的全量参数等于 optimizer 成员断言改为区分全量、冻结和可训练集合。必要的实际初态承载点是 `runtime/bundle/retrain/distributed_engine.py` 与 `runtime/r92_initial_gate.py`；若既有 normalization/组证明已完整承载则不额外改。`fresh.py`/`single91.py` 仅在现有 job/checkpoint 身份传播确实遗漏新键时最小适配，不扩旧入口或改算法。
- **B 独占：** `runtime/bundle/retrain/ema.py` 和新增 `tests/test_frozen_features.py`，仅文本编码；实际运行由 A 统一在新有效准备池执行。B 不改工厂、配置或生产组逻辑。

- [ ] A 重新动态发现现有 Pod、UID/Ready/依赖与实际 CPU 上限；CPU11 已过期，另建未来准备绑定与训练80%独立 grant，20%准备池互斥，不改历史分配，不申请实例或延长云资源。A 同时完成精确配置与 mask 接线，不等本文另批。
- [ ] B 先写约3项小张量 fixture，再做必要 EMA 改动：①真实 `apply_feature_freeze` + `_optimizer` 的范围、mode/RNG/state 保持和成员无重漏；一次小 optimizer step 冻结值不变、后段/head可更新；②冻结名单来源、warm-copy 后精确 copy 与可训练项 blend/skip/cursor；③缺省 no-op、空冻结名单、旧 EMA 数值路径和 state_dict 键兼容。无图像、预训练模型或新 Gloo 测试；真实 age 和完整结构由正式包承接。
- [ ] A 合并最终文本 snapshot，在有效准备池一次跑这3项及真正受影响的现有配置注入项。保留完整公开资产加载身份与必要模型证明，不重旧全套、像素、来源转换、pairwise/Gloo；失败仅修复具体受影响点，不扩拒绝矩阵。
- [ ] A 唯一 Git 同步实际 tree，原生正式 CPU 包实核50c初态及冻结/可训练组，按实际文件清单部署；实时独占资源核验后一次 activation/fresh。真实首成功更新报告64主图/64视图、8P56N448对及实际冻结名单/参数组，不用配置登记冒充训练成功，无独立 GPU 预演或重复 control。

## 阶段预算、判据和接续责任

本阶段最多两次 fresh、零校准。第一轮只做 R113 seed29：按完整内部 dev 的原 selector，AUROC 必须严格超过匹配 R107 `0.9884405731067525`，才接同一冻结范围的既定 seed27，对照 R108 `0.9872328538178018`；第二轮身份此时不冻结。AP 单独报告，固定难组仅描述，均不能挽救整体主指标不胜。工程不完整不得冒充科学失败。

首轮不胜则停止这个冻结范围，不扫层数、LR或新 seed；第二 seed 不胜则如实记方向不一致并结束该阶段，不自动加第三跑。两 seed 都胜也仅是有限一致证据，不能消除同一 dev 的自适应选择偏差或证明显著/泛化收益。官方与平台分数不参与科学取值，raw AUROC0.999目标未达不得称总体完成。

无论本阶段成功或失败，规划都必须在完整结果到齐后给出明确的下一问题决策及理由，不能把分支停止写成整个项目无限期暂停或长期空报；同时不以此条自动立项第三方向。A 保持唯一 GPU owner，历史冻结包不热改、不重启，结果与必要结论随必要提交统一归档。
