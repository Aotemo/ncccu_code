# 当前任务与接续

目标是在固定 9:1、单模型和内部 dev 唯一选点的合同下继续实验；raw AUROC 尚未达到 0.999，不称总体完成。自 R117 起，只有完整训练并冻结的所选内部 dev AUROC 达到 `0.996`，才运行一次官方 5849 验证；未达则记 `not_run_by_policy`。官方结果不用于训练、选点或下一轮调参。长期约束见 [AGENTS.md](../AGENTS.md)，执行入口见 [EXECUTION.md](../docs/EXECUTION.md)。

**当前已采用 [R124 固定内容加权汇聚](auroc-999/R124-DECISION.md)，order34，A/B已获必要实施，执行实证待A回执。** 直接匹配R107，只启用原 `content-attention-pool-v1` 全字典，恢复原enabled年龄；单编码/原缓存/pairlog等保持。旧R26仅探索正向、区间略跨0，R20/N11反证保留，不能称未试空间方法或已证收益。q零初始化但须在图；新增参数使whole SHA不同50c，实际共有stateexact/q0/optimizer/正式8rank/age tuple4/首成功均待实证。SCIENCE/Git/formal/activation未收到不填完成，必要准入齐A直接唯一fresh，不等文档审批。完整selected严格胜R107 `.9884405731067525`且超R120 singlebest `.9894773092062407`才条件固定27对R108 `.9872328538178018`；只胜前者则阶段结束，无pool/温度/维数/LR/seed网格，完整freeze`.996`官方门独立。

**[R123 无年龄消融](auroc-999/R123-DECISION.md)已完整结束并安全退出，分支关闭/no seed27。** 内部freeze唯一 EMA c5@4000，AUROC `.9867584356723914`、AP `.8255658133774761`、error_twice21994；7200成功/attempt相同/0skip/eval9/bad4，TRAIN completed/exit0。低于直接R107 `.9884405731067525`和单次best R120 `.9894773092062407`，不继续年龄分箱、非线性、正则权重或seed扫描；原缺省 `age_enabled=true` 继续比较基线，不推出年龄普遍必需或临床因果。内部selected未达`.996`，官方按政策未运行/metricsnull；整轮safe引用A实际2026-10-31 12:03:28回执，ownerZ不称reaped、原8rank退出、锁与设备使用者为空。原包/clock不改，必要归档由唯一A处理。

协调方已转交R123必要四件实际归档 main `2e17d3ee78f2350044f5f0fab6509e058f5f9039`、tree `9b8dca45fb46aac83be112c9bec11096d8c986ce`，远端四blob回读通过，tokenfree RESULT Pod SHA `b9bc6213a69c9b22660c59c913bd3c61d7d0f51ca564fe93f48425678d53ff8a`；本文未另查或重算。R123原决定页继续冻结。R120仍仅singlebest，R107/R108为科学锚点；R124不与R123作单变量比较，不延长metadata/fit诊断、缓存修复或已关闭分支。

**R122训练光度整组identity消融已完整结束并安全退出，主判据未通过。** [收口页](auroc-999/R122-DECISION.md)记录5600成功/0skip/eval7/bad4，unique EMA c3@2400内部AUROC `.9874796957462513`、AP `.8211659433777129`，低于R107 `.9884405731067525`及单次best R120 `.9894773092062407`。本分支关闭/no seed27，不拆三因子或扫描range/LR/seed；原None光度增强路径继续作为比较基线。官方未达`.996`按政策未运行，A实际10:02终态确认safe。协调方已转交A实际tokenfree归档：main `18dd3418ea978e90bfcfe8975016b5b8f4cc1f5f`、tree `6fa959e6d14f7ed5eaaa9db8864893f5969c1e2e`，仅result/README/R122页三件；本文未另查运行或校验原件。

**[R120固定256张fit诊断](auroc-999/R120-FIT256-DIAGNOSTIC.md)已一次结束为工程unknown，原任务关闭。** 据协调方转交A实际回执，128+128名单已private冻结，原EMA/blob/tensor严格加载通过；首readonly prepared-cache输入ValueError，0/256、wall `23.740717s`、主guard exit0，未超600秒。现有聚合仅定位为 `entry_or_source_fingerprint`，不可进一步区分，无AUROC/error指标或科学推断。unknown不是fit可分或不可分，也不代表标签/缓存全局结论；不重试、补样、换模型、修cache或原图fallback。必要三件归档已由A实际同步 Git `0ca69fab19f236977f1d54785644afde2133753e`、tree `e0863e33a1a3169482c92e0318ec2662dd789fba`；原含名错误/名单/logits不导出，原件不改。

**独立缓存绑定元数据核对已于约10:28完成，category=`entry_missing`。** 据协调方已读A脱敏RESULT，只定位原首批失败的1条：既定namespace下receipt/npy stat均不存在，source仅stat为regular；两冻结context的data/cache/metadata roots、ASSETS及prepversion/Pillow/size/dataset_root一致、全绝对路径，private cwd无影响。metadata耗时`.2383430847s`、总guard约18秒、CPU16本人份额guard0。未开不存在的receipt，未读图/npy字节/模型或前向/扫描/repair。只证明这一条缺cache，不知其余coverage或缺失原因，不推训练数据缺失或模型阴性；原FIT256仍unknown/0of256、无AUROC，不修后重跑或补样。完整边界与RESULT/CPU pins见[同页收口](auroc-999/R120-FIT256-DIAGNOSTIC.md)。协调方已转交A四件实际归档 main `3c07716fe7f82f895ad4b4fb56eab03ce6bfbe1a`、parent `0ca69fab19f236977f1d54785644afde2133753e`、tree `85f3e73600b207f8fe98e4fe4db0a636497d8b9e`，readback仅RESULT/README/原FIT页/todoREADME；本文未另查或校验原件。

**[Fit联合输入一致性元数据核对](auroc-999/FIT-INPUT-CONSISTENCY-20261031.md)已一次实际完成，C=0，任务收口。** 据协调方转交A回执，R107/R120原manifest、fit_indices及age身份一致，原age校验通过；fit46256=P1490+N44766，既存SHA+实际FP32两维age数值key（±0合并）全部有限。重复key/涉及行、cross-label key/其中P/N行均为0；metadata `2.163719s`，当次CPU17本人12核guard0，无Torch/图像/cache/stat/模型/forward/训练。只排除该严格元数据冲突，不能证明标签正确、实际cache像素或拟合充分，不能推出`.999`可达或dev/平台上限。原FIT256仍unknown且不补跑，首cache缺项不扩查。

协调方已转交该 metadata 任务实际归档 main `733869f7e08ae8c551b169342a9571d5daa1738f`、tree `b9e456f74bd60db3fe70efb67367573ad3836bf6`；本文未另查或校验原件。C=0不构成标签正确、拟合充分或后续方法收益的证据；原FIT256 unknown/0of256与首条cache missing均已关闭，不延展审计、不修后补考，不据这些metadata结果改标、去重、设权采样或修cache。当前独立训练接续见页首R124；R120仍仅单次best，R107/R108仍为科学比较锚点。

R120/R121固定pairsigmoid配对已完整结束，一胜一负，停止该分支主线升级。[R121](auroc-999/R121-DECISION.md)6400成功/0skip/eval8/bad4，unique EMA c4@3200内部AUROC `.9869089490433972`、AP `.8355471078657948`，未严格胜预冻R108 `.9872328538178018`；AP小高不救主判。官方未达`.996`按政策不运行，A实际08:31终态确认整轮safe。无第三seed/温度/scale/混合或近邻surrogate补考。必要归档已由A实核同步Git `ee5c1b9e470c4b1a46539a1cd9bff42c3f7cef50`、tree `638d1fb2eb421b975ddd93ffa4cc3d40f2a191ad`；原冻结产物不改，资源仍以执行当次实核为准。

[R120](auroc-999/R120-DECISION.md)已完整工程有效：8000成功/0skip/eval10/bad4早停，唯一EMA c6@4800内部AUROC **.9894773092062407**、AP **.8313831149805229**，仍为当前单次内部best，严格胜预冻R107 `.9884405731067525`。R121未复现方向，因此不升级正式基线，继续保留R107/R108比较锚点，不挑最好seed。未到`.996`，官方按政策未运行，A实际07:03终态确认整轮safe。原冻结包不改，selected及真实依赖暂保护；tokenfree结果已随实际Git `d1dbc05f4b65c675c9a97b25f3b9eb20b33d19e6`同步。

R113 冻结早期特征与 R114R1 最后下采样 stride1 均已完整结束，内部主指标未严格胜 R107 `0.9884405731067525`，各自分支按预定停止。原 R114 1426 步是工程失败，不充当完整科学结论。精简记录见 [R113](../records/R113/README.md)、[R114R1](../records/R114R1/README.md)及对应决定页。

R115 的固定 fit-only SSL1600 联合轮已完整结束，内部所选 EMA c4 AUROC `0.9863815501913928`，未胜 R107，分支按预定停止；原 once official 单报，不参与选参。必要归档见 [R115](../records/R115/README.md)。指定的[全历史复盘](auroc-999/EXPERIMENT-REVIEW-20261030.md)已完成，用户暂停后续训练的等待条件由协调确认解除。

R116 Base 配对完整内部所选 EMA c4 AUROC `0.987677169289011`，未严格胜 R107，分支不做 seed27；原一次官方已自然完成并单报，资源安全退出。精简结论见 [R116](../records/R116/README.md)。

R117 Swin V1 Base 配对完整内部所选 EMA c7 AUROC `0.9850076641408516`，未严格胜 R107，分支不做 seed27；内部低于 `0.996`，官方按政策未运行。round 已完成且资源安全退出。精简结论见 [R117](../records/R117/README.md)。

R118 DINOv1 ResNet50 来源替换已完整结束，内部所选 online c5 AUROC `0.970734180141627`，低于 R111 和预冻 T29，分支不做 seed27 或 A1 替补；低于 `0.996`，官方按政策未运行，round 已完成且资源安全退出。精简结论见 [R118](../records/R118/README.md)及[原决定](auroc-999/R118-DECISION.md)。

R119固定中心双输入容量核验通过，但唯一12CPU/64fit读取探针mean `.8380236838711426s`、p95 `1.0136382030323148s`均未达预定阈值，按合同终止当前实现，无训练、selected AUROC或官方结果；不能判为局部科学阴性或真实96CPU训练超时。必要小归档已由A同步至实际main `cb752047a741364c6ebe150065f0a76d3d90c482`，见[收口页](auroc-999/R119-DECISION.md)。不补考或建全fit/dev局部cache；未验输入代码的精确patch保留和生产隔离归A处理，尚不冒称完成。

当前执行 A `e62c5cb0-3516-4798-8b37-ed8da2d958ad` 是唯一 Git 写入者和 GPU owner；规划负责科学决定；执行 B 无 Git 写入、GPU、activation 或 launch 权。原执行者的历史回执保留，运行和 Agent 均不恢复。CPU 准备与训练分别使用实时有效 allocation 的互斥份额，入口须重新核 Pod UID、Ready、TTL、预算及独占状态。

源码与必要精简结论维护在本仓库；数据、模型、图像、样本预测和运行原件保存在 `/public/home/acfwvv9ejn/SothisAI/exp-v3`。当前 Pod、解释器、CPU 和设备状态均由 [动态发现工具](../docs/EXECUTION.md)实时核验，不把本文记录当默认配置。本机仅做文本、Git、传输和展示；解析、哈希、测试及模型执行在受控 Pod 内。
