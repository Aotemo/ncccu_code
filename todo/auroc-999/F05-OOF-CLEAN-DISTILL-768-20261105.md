# F05 三折 OOF、清洗、蒸馏及 768 合同

以下规则在读取任何新 OOF、清洗模型、蒸馏模型或 768 模型结果之前登记。只使用比赛提供的原 fit；原 5140 dev 始终不参与折模型训练、fit 标签清洗或蒸馏标签构造，官方不读取。所有模型 fresh，使用经来源审计的通用公开预训练，不使用 N06 或外部医学数据。每轮保留唯一内部 dev selected 权重及最小加载依赖；失败和未知如实记录。

## 三折及 OOF

- 原固定 9:1 manifest 的 46256 个 fit 索引和 `group` 是唯一分折范围；复用 `retrain.data._partition_groups` 的内容组/标签/年龄/尺寸分层，三等份，分折 seed `20261121`。每个原 fit 行恰好在一个 held-out 折，组不跨折，所有折与原 dev 不重叠；保留原行、标签、路径与 SHA，不改原 manifest。实际折大小及每折正负数在读取 OOF 之前封存并核实正例均衡。
- 预留 R265–R267 对应折 0–2；使用 R120 公开初始化与原训练 seed `20261029`、pairwise sigmoid、输入、G64/P8、优化器、EMA、评估和早停合同，唯一必要数据差异是 fit 去掉该折（各训练约 2/3 fit、留出约 1/3）；每折 age 统计仅从该折训练子集重算，避免使用 held-out 样本。原 dev 仍用于预定的该折唯一选点，明确其选择影响；held-out 折不参与模型训练或选点。初始接续按 A→R265、B→R266；之后一 Pod 跑 R267、另一 Pod 跑 R270，按已备妥及安全容量条件接续，三汤评估置于交接空档。
- 每个折模型仅对其 held-out fit 单视图前向一次，合并必须精确覆盖所有原 fit 行一次、模型/折/文件 SHA 与标签对齐，缺项、重复或损坏为失败/unknown。OOF 原始 logits、标定软标签与逐样本映射只存 exp-v3；不进入 Git。
- pairwise 损失不固定 logit 的绝对零点，直接对原始 logit 取 sigmoid 并以 0.95 判定会受任意平移影响。为每个折固定一个仅用该折训练子集的标定：按类分别以 seed `20261121 + 折号` 随机抽取最多 1024 行，单视图计算该折 selected 模型的分数；按训练子集真实类比例给抽样行加权，拟合一维 `LogisticRegression(C=1000, solver='lbfgs', max_iter=1000, tol=1e-8, random_state=20261121)`，并用此映射转换该折 held-out 分数。两类缺失、拟合未收敛或输出非有限值均为失败，不改阈值或退回原始 sigmoid。标定模型和模型训练均不接触本折 held-out 标签；标定样本属于该折训练集，可能过度自信，须明确报告，不称作医学确诊或严格校准概率。封存抽样索引及摘要、类比例、系数、截距和收敛状态。

## 置信筛查与清洗重训

事前固定二分类 confident-joint 规则：对每个 OOF 行令 `p1=该折训练子集拟合的标定映射(logit)`、`p0=1-p1`；对类 `j∈{0,1}` 以原标注为 j 的 fit 行计算 `t_j=max(0.95, mean(p_j | observed_label=j))`。当某行 `p_j≥t_j` 时将其记为预测类 j 的高置信项（若两个类同时符合则取概率较大者；完全相等且均符合记 unknown，不任意决断）；其余为未确认项。统计 `C[原标注, 高置信预测类]`，非对角线行标为可疑。该规则是基于 OOF 的置信分歧候选，不等同专家确诊错标；报告阈值、矩阵、数目、比例与来源，不手工改标签。清洗主方案只对上述 fit 样本赋固定权重 0.25，其他为 1；原 R120 pairwise sigmoid 的正负对权重为两样本权重乘积，按权重和归一化。无候选时仍按原合同如实报告，不能事后改阈值。新模型 fresh，数据、age、优化器和唯一 dev selector 沿 R120，科学配置及影响范围另作正式部署绑定。

## 任务内 OOF 蒸馏

教师仅使用上一节完整、逐样本折外且来源合规的 OOF 分数及对应折的训练子集标定映射；本合同不引入外部教师或 F04 官方输出。固定温度 1，学生为 fresh Small，训练损失为原 R120 pairwise sigmoid 加 `0.25 × BCEWithLogits(student_logit, OOF_标定软标签)`，软标签只查询原 fit 行。主方案先训练 1 个学生；第二个学生仅作为事前登记的独立 seed 复验，使用同一损失与教师，不据错例改变配比。所需教师文件 SHA、完整索引及每行排除该折的证明须在启动前核实。

## 768 多分辨率

一轮 R120 pairwise sigmoid/Small，输入与 prepared 尺寸固定 768，其余科学值沿 R120；训练采用未见于配置及合同的新 seed `20261122`，按总合同同时改变 seed，故不能将表现差异完全归因于视野。原 R152 曾以同类 Small、768、G64/P8 完整运行，可作为资源可行性历史线索，不作为本轮效果保证。对 fit/dev 建新版本化只读 768 缓存或经身份核验的既有缓存；不覆盖原始数据/512 缓存、不读官方。启动前实核显存、存储、CPU、正式检查及源树；如 G64/P8 不可行，记录实际原因，不跳安全门。

## 最小范围与实施门

拟新增 `configs/F05-FOLDS-3.json`、R265–R267 的版本化 SCIENCE、`tools/diagnostics/f05_oof.py` 及对应测试；必要时小范围修改既有 `runtime/bundle/retrain/data.py`、`experiment91.py`、`pipeline.py`、`distributed_engine.py`、`classification_loss.py`、activation/CPU 验证中对固定 manifest 的绑定。清洗、蒸馏和 768 分别另建版本化配置并复用这些同一训练/保存/评估/监督链。任何新增实际影响文件在实现前补登记；不新建并行训练平台。用户要求尽快占满两 Pod，但正式 CPU、Git 同步、实际设备与容量门不满足时不得派发或虚报可运行。

**历史五折，已被三折决定取代：** fit 分组五折生成核心曾在 Pod B 完成 3 项合成检查（CPU `740c667c716c4e9d3914b7b65039ce0ae512b2a0d33e074b9d096c630ce7afbe`）。

历史五折真实分组及逐样本索引/age 封存 SHA `a637bac2034ff53a8564ba8f298ca9cf6215af9b9e4d9839bbff86cd0290203f`，CPU `20bcf6e0fd4fd3aa8e287c8e1871b1aab7cfa23f5274f58b76ca7871b848b576`；保留原件及任何已派生元数据作审计，不删除或覆盖。用户改变折数时尚无任何五折训练或 OOF 结果，旧分组作废，不能用于训练或新 OOF。

接入核查发现原八卡训练入口明确拒绝非空 `audit_indices`。各折训练 manifest 因此从新三折封存分组派生：保留该折 fit 与原 dev、`audit_indices=[]`，held-out 索引只留在 SHA 固定的独立三折封存文件供训练结束后的 OOF 使用；训练 manifest、fold age、原始 manifest 及封存分组在 CPU 和正式配置中逐项绑定。保持原 DDP 的 audit 禁令，不在训练过程计算折外输出。仅新增相应派生与绑定校验，不改变早停或 selector。

派生文件最小范围增加 `tools/diagnostics/f05_oof_assets.py`、`runtime/bundle/retrain/f05_folds.py` 和对应窄测试：前者从已验证原始 manifest、原冻结元数据和新三折封存重算并生成每折 manifest/age/contract/freeze；后者让既有 `experiment91.py` 只在显式版本化 F05 折绑定时接受这些文件，并独立复核 fit、原 dev、held-out、组隔离与 age。未启用 F05 绑定时保留原 `FREEZE_SHA` 检查。`tools/activation/prepare.py` 必须验证新折元数据源自原迁移资产，不把迁移身份匹配简单移除。每折配置与精确部署仍须先 commit、push、18/18 和容量核验。

R265–R267 共用一个包含三折 profile 的源码 commit，先核验并部署 R265/B；同一 Pod 的后续折复用同一全树源码快照，按各折 SCIENCE 和数据身份重新完成正式检查。版本化 SCIENCE 的 `f05_fold` 必须固定版本、折号、原始 metadata 根、原始 freeze/manifest SHA、三折封存路径/SHA、派生 freeze SHA；运行时验证这些字段、当前 site 的派生根与 round/profile 对应关系，并重算原始/派生/封存身份、角色和 fit-only age。正式准备同时将原始 metadata 根与现有迁移回执中的 stage1 根精确比对；非折 profile 不得携带该字段。

三折改动的合成检查在旧五折代码下预期失败（CPU `008d9f1c3795c0a8018d48ece1397ec584227ff23d41b4fe8c202ef955bd052f`），修改后 11/11 通过（CPU `d7eadb396860397865ef547579c317b32ccbd10e0415fe3e0b3cb4acfe1de2b9`），覆盖一次 held-out、组隔离、正例均衡、fold age 及错误身份拒绝。这只证明合成样本和元数据核心，真实三折分组、正式训练绑定、CPU 18/18 与部署尚未完成。

补充旧五折状态拒绝检查后，三折、元数据派生和角色绑定 11/11 再通过（CPU `fd49a4c8f3c8548403d95ae2e6de19b05d48da4f77de603b72c286fc1ce528f2`）；仍待真实分组及完整正式链。

真实三折现已封存：fold 0/1/2 的 fit 行数 `30836/30838/30838`、held-out 行数 `15420/15418/15418`、held-out 正例 `497/496/497`，每折原 dev 均为 `5140` 行、`167` 正例。封存 SHA `eb880273832b81be61830a069d5ae9ba8dc2e52be4af2f911f7a6b1ec9e7f6d8`、CPU `964a401929db96ff5d7e18226f42d49f31d95fcd8eeb212e0bb4db06bd476954`，生成代码 push/API `00eb266007fc8b4e9fd95ecd0417a44307f955eb`；不代表正式折元数据/训练入口、18/18 或部署已完成。按当前工程量及间歇 API 延迟，R265/R266 正式备妥初估 60–120 分钟，随实际检查更新。

实际三折训练元数据已由 push/API `5cdef18e0e8d0fc49a452a2d3d87b4cfedc22850` 及 B 端完整树 `e3fc1284c4b02d121f134701c497dc9c732125705ba94d647642af13b079fef0` 重算生成；派生回执 `efb38262a1a3a68e0c678e6991200e4c96c953ea9b138ef249209d6294ccfca9`。各折 `FREEZE.json` SHA 依次为 `3e6a7236b6e21f7691acb9e9b8f9348242a804af745c86fd97d9ba4733853864`、`feb5500ec7c4be639e4829626952c0f6088a54ee2b17fd11b62ae369a1f2b6ad`、`239ca36f700a7554e383de790aa6403572a7706518dcff4101844eec7266e9b0`；R265 fit/held-out 正例 `993/497`、固定 dev 正例 `167`。这些事实不等于正式训练入口或部署通过。

元数据派生入口增加对三折配置、原始 manifest 和新封存 SHA 的同时校验；合成及入口窄测 12/12（CPU `c8e8688f4fc0f512d8c7c46828913aa4d139bec91104d89d87eca334dd98d1d3`）。真实派生仍待版本化源码 push、精确树与实际校验。
