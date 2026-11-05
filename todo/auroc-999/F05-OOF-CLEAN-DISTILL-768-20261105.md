# F05 三折 OOF、清洗、蒸馏及 768 合同

以下规则在读取任何新 OOF、清洗模型、蒸馏模型或 768 模型结果之前登记。只使用比赛提供的原 fit；原 5140 dev 始终不参与折模型训练、fit 标签清洗或蒸馏标签构造，官方不读取。所有模型 fresh，使用经来源审计的通用公开预训练，不使用 N06 或外部医学数据。每轮保留唯一内部 dev selected 权重及最小加载依赖；失败和未知如实记录。

## 三折及 OOF

- 原固定 9:1 manifest 的 46256 个 fit 索引和 `group` 是唯一分折范围；复用 `retrain.data._partition_groups` 的内容组/标签/年龄/尺寸分层，三等份，分折 seed `20261121`。每个原 fit 行恰好在一个 held-out 折，组不跨折，所有折与原 dev 不重叠；保留原行、标签、路径与 SHA，不改原 manifest。实际折大小及每折正负数在读取 OOF 之前封存并核实正例均衡。
- 预留 R265–R267 对应折 0–2；使用 R120 公开初始化与原训练 seed `20261029`、pairwise sigmoid、输入、G64/P8、优化器、EMA、评估和早停合同，唯一必要数据差异是 fit 去掉该折（各训练约 2/3 fit、留出约 1/3）；每折 age 统计仅从该折训练子集重算，避免使用 held-out 样本。原 dev 仍用于预定的该折唯一选点，明确其选择影响；held-out 折不参与模型训练或选点。初始接续按 A→R265、B→R266；之后一 Pod 跑 R267、另一 Pod 跑 R270，按已备妥及安全容量条件接续，三汤评估置于交接空档。
- 每个折模型仅对其 held-out fit 单视图前向一次，合并必须精确覆盖所有原 fit 行一次、模型/折/文件 SHA 与标签对齐，缺项、重复或损坏为失败/unknown。OOF 原始 logits、标定软标签与逐样本映射只存 exp-v3；不进入 Git。
- pairwise 损失不固定 logit 的绝对零点，直接对原始 logit 取 sigmoid 并以 0.95 判定会受任意平移影响。为每个折固定一个仅用该折**全部训练子集**的标定：按原 manifest 全局行索引升序，对该折每个训练行用该折 selected 模型做一次单视图前向；不抽样。原合同按训练集真实类比例修正分层抽样，使用全量训练集时此比例修正得到各行权重均为 `1`，均值为 `1`、总和为该折训练样本数；原合同没有反比例类别平衡。以这些权重拟合一维 `LogisticRegression(C=1000, solver='lbfgs', max_iter=1000, tol=1e-8, random_state=20261121)`，并用此映射转换该折 held-out 分数。两类缺失、训练索引缺漏/重复、拟合未收敛或输出非有限值均为失败，不改阈值或退回原始 sigmoid。标定模型和模型训练均不接触本折 held-out 标签；标定样本属于该折训练集，可能过度自信，须明确报告，不称作医学确诊或严格校准概率。封存完整训练索引身份、类比例、权重归一化、selected 模型及预测分片身份、系数、截距和收敛状态；样本级索引与输出只留 exp-v3。该全量及归一化裁定在任何 OOF 或标定分数产生前作出，替代原每类最多 1024 行抽样的文本。

## 置信筛查与清洗重训

事前固定二分类 confident-joint 规则：对每个 OOF 行令 `p1=该折训练子集拟合的标定映射(logit)`、`p0=1-p1`；对类 `j∈{0,1}` 以原标注为 j 的 fit 行计算 `t_j=max(0.95, mean(p_j | observed_label=j))`。当某行 `p_j≥t_j` 时将其记为预测类 j 的高置信项（若两个类同时符合则取概率较大者；完全相等且均符合记 unknown，不任意决断）；其余为未确认项。统计 `C[原标注, 高置信预测类]`，非对角线行标为可疑。该规则是基于 OOF 的置信分歧候选，不等同专家确诊错标；报告阈值、矩阵、数目、比例与来源，不手工改标签。清洗主方案只对上述 fit 样本赋固定权重 0.25，其他为 1；原 R120 pairwise sigmoid 的正负对权重为两样本权重乘积，按权重和归一化。无候选时仍按原合同如实报告，不能事后改阈值。新模型 fresh，数据、age、优化器和唯一 dev selector 沿 R120，科学配置及影响范围另作正式部署绑定。

## 任务内 OOF 蒸馏

教师仅使用上一节完整、逐样本折外且来源合规的 OOF 分数及对应折的训练子集标定映射；本合同不引入外部教师或 F04 官方输出。固定温度 1，学生为 fresh Small，训练损失为原 R120 pairwise sigmoid 加 `0.25 × BCEWithLogits(student_logit, OOF_标定软标签)`，软标签只查询原 fit 行。主方案先训练 1 个学生；第二个学生仅作为事前登记的独立 seed 复验，使用同一损失与教师，不据错例改变配比。所需教师文件 SHA、完整索引及每行排除该折的证明须在启动前核实。

## 768 多分辨率

一轮 R120 pairwise sigmoid/Small，输入与 prepared 尺寸固定 768，其余科学值沿 R120；训练采用未见于配置及合同的新 seed `20261122`，按总合同同时改变 seed，故不能将表现差异完全归因于视野。原 R152 曾以同类 Small、768、G64/P8 完整运行，可作为资源可行性历史线索，不作为本轮效果保证。对 fit/dev 建新版本化只读 768 缓存或经身份核验的既有缓存；不覆盖原始数据/512 缓存、不读官方。启动前实核显存、存储、CPU、正式检查及源树；如 G64/P8 不可行，记录实际原因，不跳安全门。

## 最小范围与实施门

拟新增 `configs/F05-FOLDS-3.json`、R265–R267 的版本化 SCIENCE、`tools/diagnostics/f05_oof.py` 及对应测试；必要时小范围修改既有 `runtime/bundle/retrain/data.py`、`experiment91.py`、`pipeline.py`、`distributed_engine.py`、`classification_loss.py`、activation/CPU 验证中对固定 manifest 的绑定。清洗、蒸馏和 768 分别另建版本化配置并复用这些同一训练/保存/评估/监督链。任何新增实际影响文件在实现前补登记；不新建并行训练平台。用户要求尽快占满两 Pod，但正式 CPU、Git 同步、实际设备与容量门不满足时不得派发或虚报可运行。

三折终态后的 fit-only 前向拟在现有 `tools/diagnostics/f04_gpu_full_dev.py` 增加显式受限模式，必要的角色及聚合校验放 `tools/diagnostics/f05_oof_gpu.py`、配置 `configs/F05-OOF-GPU-20261105.json`、窄测 `tests/test_f05_oof_gpu.py`；复用既有八设备监督、一次性 claim、设备/CPU/容量门与历史 runtime，不新建并行监督链。每折以原 manifest 及封存 folds SHA 固定训练/held-out 两个不重叠 population，依次对该折唯一 selected 在两个人群各做一次单视图前向，dev/官方均不可进入；私有分片及聚合结果须绑定原索引、角色、原标签、fold、selected 文件与张量身份、各 rank 设备证明和两种 population 的封存身份。标定仅消费完整训练角色，held-out 仅经已封标定映射转换。进入真实前向前须完成合成拒错测试、Git push、全树、fold 来源/严格加载与现场安全门；前向安排在交接空档，不延误下一折。具体输入/输出字段需与 `f05_oof_targets.py` 的独立血缘校验逐项对齐；样本级文件只留 exp-v3。

实施顺序：① 对已封 roles 和既有 rank 分片写合成 RED，包括错误折、训练/held-out 交叉、dev 混入、分片缺漏/重复、错误模型/设备证明和非有限值；② 新辅助先复用 `f05_folds.verify_roles` 与 `parallel_evaluation.canonical_predictions` 验证角色及分片并输出版本化私有格式；③ 在现有 supervisor 添加互斥 fit-only 模式，限定单 selected、8 rank、两个固定人群、identity view，拒绝官方参数；④ 以真实 fold 来源、safe 终态、checkpoint 和严格加载证明准备只读输入，在 Pod 运行绿测与原 F04/汤入口回归；⑤ 版本化源码 push/API 与 Pod 精确树后，仅由 GPU owner 在不阻塞下一折的空档运行一次并封存失败/成功回执。助手消费端独立复核结果，不接受仅凭 producer 自报身份。

OOF 评估准入补充：三折之后的清洗/蒸馏正式包依赖尚未生成的 OOF，不能要求一个虚构的新训练轮 ready。fit-only 模式可只读复用当次 Pod 的既有正式准备包作为设备及 80/20 CPU 绑定（如另一 Pod 已实际认领同轮而本 Pod 的对应变体从未 activation），该包只用于评估资源身份，不派发、不激活、不修改其轮次；必须另核其 CPU/部署回执和有效 grant、实时 Pod UID/Ready、当前 Pod 前轮 actualsafe 与设备/实例锁释放、另一 Pod 的真实运行和剩余容量预算、共享盘至少 20 GiB 加评估输出余量。评估用 fresh 独立一次性 claim、限定租期和路径，拒绝官方/开发集/TTA；任何现场状态 unknown 即不派发。只在 `f04_gpu_full_dev.py` 的显式 OOF 分支和 `f05_oof_gpu.py` 增加所需检查/窗口验证，不改现有 F04、汤或训练分支。原有 next-ready「轮次目录尚不存在」和训练完整租期检查不能被无条件放宽；新增分支须有隔离假调用与错误身份、过期、容量不足、peer 未知、锁/设备未释放的 Pod CPU 拒错回归后才可使用。

**历史五折，已被三折决定取代：** fit 分组五折生成核心曾在 Pod B 完成 3 项合成检查（CPU `740c667c716c4e9d3914b7b65039ce0ae512b2a0d33e074b9d096c630ce7afbe`）。

历史五折真实分组及逐样本索引/age 封存 SHA `a637bac2034ff53a8564ba8f298ca9cf6215af9b9e4d9839bbff86cd0290203f`，CPU `20bcf6e0fd4fd3aa8e287c8e1871b1aab7cfa23f5274f58b76ca7871b848b576`；保留原件及任何已派生元数据作审计，不删除或覆盖。用户改变折数时尚无任何五折训练或 OOF 结果，旧分组作废，不能用于训练或新 OOF。

接入核查发现原八卡训练入口明确拒绝非空 `audit_indices`。各折训练 manifest 因此从新三折封存分组派生：保留该折 fit 与原 dev、`audit_indices=[]`，held-out 索引只留在 SHA 固定的独立三折封存文件供训练结束后的 OOF 使用；训练 manifest、fold age、原始 manifest 及封存分组在 CPU 和正式配置中逐项绑定。保持原 DDP 的 audit 禁令，不在训练过程计算折外输出。仅新增相应派生与绑定校验，不改变早停或 selector。

派生文件最小范围增加 `tools/diagnostics/f05_oof_assets.py`、`runtime/bundle/retrain/f05_folds.py` 和对应窄测试：前者从已验证原始 manifest、原冻结元数据和新三折封存重算并生成每折 manifest/age/contract/freeze；后者让既有 `experiment91.py` 只在显式版本化 F05 折绑定时接受这些文件，并独立复核 fit、原 dev、held-out、组隔离与 age。未启用 F05 绑定时保留原 `FREEZE_SHA` 检查。`tools/activation/prepare.py` 必须验证新折元数据源自原迁移资产，不把迁移身份匹配简单移除。每折配置与精确部署仍须先 commit、push、18/18 和容量核验。

R265–R267 共用一个包含三折 profile 的源码 commit，先核验并部署 R265/B；同一 Pod 的后续折复用同一全树源码快照，按各折 SCIENCE 和数据身份重新完成正式检查。版本化 SCIENCE 的 `f05_fold` 必须固定版本、折号、原始 metadata 根、原始 freeze/manifest SHA、三折封存路径/SHA、派生 freeze SHA；运行时验证这些字段、当前 site 的派生根与 round/profile 对应关系，并重算原始/派生/封存身份、角色和 fit-only age。正式准备同时将原始 metadata 根与现有迁移回执中的 stage1 根精确比对；非折 profile 不得携带该字段。

R265/B 第一次正式准备中，技术准备通过但 `prepare_gpu_site.py` 的原始 relocation 路径匹配拒绝了派生折 metadata；失败日志 SHA `cba8e60b899436f3ea27604113ffe18959077dce08381815e915ca5cfffce0cb`，未进入正式 CPU/部署。额外受影响文件仅为 `tools/activation/prepare_gpu_site.py` 与现有折绑定测试：仅在已注册折上复核派生 metadata 与 relocation 的原始 stage1、封存分组和 SHA 后替代该单项路径匹配，其他资产路径和非折行为不变；旧失败目录保留。

三折改动的合成检查在旧五折代码下预期失败（CPU `008d9f1c3795c0a8018d48ece1397ec584227ff23d41b4fe8c202ef955bd052f`），修改后 11/11 通过（CPU `d7eadb396860397865ef547579c317b32ccbd10e0415fe3e0b3cb4acfe1de2b9`），覆盖一次 held-out、组隔离、正例均衡、fold age 及错误身份拒绝。这只证明合成样本和元数据核心，真实三折分组、正式训练绑定、CPU 18/18 与部署尚未完成。

补充旧五折状态拒绝检查后，三折、元数据派生和角色绑定 11/11 再通过（CPU `fd49a4c8f3c8548403d95ae2e6de19b05d48da4f77de603b72c286fc1ce528f2`）；仍待真实分组及完整正式链。

真实三折现已封存：fold 0/1/2 的 fit 行数 `30836/30838/30838`、held-out 行数 `15420/15418/15418`、held-out 正例 `497/496/497`，每折原 dev 均为 `5140` 行、`167` 正例。封存 SHA `eb880273832b81be61830a069d5ae9ba8dc2e52be4af2f911f7a6b1ec9e7f6d8`、CPU `964a401929db96ff5d7e18226f42d49f31d95fcd8eeb212e0bb4db06bd476954`，生成代码 push/API `00eb266007fc8b4e9fd95ecd0417a44307f955eb`；不代表正式折元数据/训练入口、18/18 或部署已完成。按当前工程量及间歇 API 延迟，R265/R266 正式备妥初估 60–120 分钟，随实际检查更新。

实际三折训练元数据已由 push/API `5cdef18e0e8d0fc49a452a2d3d87b4cfedc22850` 及 B 端完整树 `e3fc1284c4b02d121f134701c497dc9c732125705ba94d647642af13b079fef0` 重算生成；派生回执 `efb38262a1a3a68e0c678e6991200e4c96c953ea9b138ef249209d6294ccfca9`。各折 `FREEZE.json` SHA 依次为 `3e6a7236b6e21f7691acb9e9b8f9348242a804af745c86fd97d9ba4733853864`、`feb5500ec7c4be639e4829626952c0f6088a54ee2b17fd11b62ae369a1f2b6ad`、`239ca36f700a7554e383de790aa6403572a7706518dcff4101844eec7266e9b0`；R265 fit/held-out 正例 `993/497`、固定 dev 正例 `167`。这些事实不等于正式训练入口或部署通过。

元数据派生入口增加对三折配置、原始 manifest 和新封存 SHA 的同时校验；合成及入口窄测 12/12（CPU `c8e8688f4fc0f512d8c7c46828913aa4d139bec91104d89d87eca334dd98d1d3`）。真实派生仍待版本化源码 push、精确树与实际校验。

**11-05 R265 完成与 R266 接续：** R265/B 完整 `6400/6400`、0 skip，selected EMA c4 内部 dev AUROC `0.9865212265996862` / AP `0.8166606110304513`，实际安全终态 `9649b8a2e7ee7f8a44ed5c7785142dfa9484ca7517997158ab019e6977ab69cc`。R266/B 已 8 rank 初态/空优化器/CPU/设备绑定通过，首成功 `25/25`、0 skip（`5b7a5e91d01d6495fbda950293ab84b9a31186efbbd07f70b99a234847fa5cf7`）。done→dispatch `177.199 s`、→首更新 `337.866 s`（`a9b6d4fe7b8aad99061996d9519870d4b392fee68bd81aef228b61cdffa1611b`），无持续 GPU idle 采样。期间一次启动观察 kubectl exec EOF，记 unknown；后续以新观察得到上述首成功。间隔脚本第一次因路径不存在 exit2（CPU `478bd40e964ac1e9151ef934c84393b6d7588c76885fdde4877c0989429e652b`），重新传入既有只读脚本后获得上述真实回执。不因内部单模低于融合门而跳过 R265 的 OOF 职责。R270/A 仍运行，R267 待前轮安全后接续。

**11-05 OOF 角色/分片窄测：** 新的 fit-only 角色与八 rank packet 归约辅助先做合成 RED（缺实现，CPU `05f80451212a49ece9795bf0125ec46d957c59142fd6faf2afaa337a57cd2d7a`），实现后 4/4 通过（CPU `03d504da4acd1e1d9cde41b4007c9d4726777266ecaef6d876d68f75b0b43e6a`；源码 SHA `0cd9f9ca63414e6d6abbedfc704a77c98406481ebd093b4d14d9ddda2cf65939`、测试 SHA `b446b1c1437f17fc717062b0dd396c0cc65538637eb794b47ee22fb82f671ed6`）。检查已封 train/held-out/原 dev 互斥、原索引顺序、完整八 rank，以及分片缺失/重复、错误模型/设备证明、TTA/官方标志、非有限分数拒绝。此结果仅证明合成辅助，owner/worker 接线、真实 fold selected 血缘与任何实际 OOF 前向尚未完成。

**11-05 R270 完成与 R267 接续：** R270/A `8000/8000`、0 skip，selected EMA c6 内部 dev AUROC `0.9861684232580485` / AP `0.8380459555583255`，actualsafe `c57962efbdd76d04ed4a38f2dd6cc681e2663bf0d1d96b377d4013624454819e`。R267/A 8 rank 初态/空优化器/CPU/设备绑定通过，首成功 `50/50`、0 skip（`de6e40e311994de19ec9c0efa6eeec323cea229969b745c60f4923c7bffe0fa8`）；done→dispatch `193.135 s`、→首更新 `385.975 s`（`17a5d7e88d80cb9a2214c1a73668b2f2df6355807bf1cee99dfeece8c20099e9`），无持续 GPU idle 采样。R266/B 同时运行；两轮均须安全终态后才能对各自 selected 产生 OOF。

**11-05 R266 完成与 OOF 入口隔离检查：** R266/B `7200/7200`、0 skip，selected EMA c5/4000，内部 dev AUROC `0.9855266342440797`、AP `0.8237989124176052`，actualsafe `4ead456541c29e1438d0db326be06f682ac444d61ac64a1058a1b22e7834dcbd`。10:49 曾显示 terminal 但 `done` 未出现、`safe=false`（`7741516c9edbf8dcaee56f4561907b3dcedc825b484c1ec13edab0f6e531fdae`），未据此接续。对 OOF 两角色、原 selected/sidecar、本地副本、八 rank 证明和 standalone 安全/容量门，在 B 准备池执行隔离草稿合成测试 7/7 与 14/14（CPU `1777673bf71d7c33f1421d6693bfcf4187a453b1662e541c5cb36e2017b6658e`）；先前入口语法失败 `6203a76575507e70f7f94a36543f97724cffdbc0a8d28e648b4e1c80fb1bd3ea` 保留。真实资产绑定、已 push 精确树、完整 prepare 路径与现场 GPU 准入尚未完成，尚无真实 OOF 或官方前向；B 从安全终态至本次记录未派发，须另记后续实际事件间隔。
