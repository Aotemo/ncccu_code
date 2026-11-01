# R137：固定 Lion10x 配方

**完整结果已关闭此精确策略。** 9600success=attempt/0skip/eval12/bad4正常early_stopping，唯一EMA c8@6400 AUROC`.9854411426493483`、AP`.8124829988951043`，strict>R107和strict>R120均false/no seed27；R138仅预留技术包，未activation/未训练。2026-11-01 15:11:29.724784 UTC唯一完整终态确认done/TRAIN/ROUND completed及actualsafe=true，官方低于`.996`按政策未跑，freeze.status原值null不改。[完整结果与实际安全小件](../../records/R137/README.md)已归档；不重查旧safe或旧blob。当前GPU已独立fresh接R139，原7b冻结包/clock/grant和依赖保持。下文running/未完成均属历史阶段。

实际接续已成功：runGit`7b48f495ecb85664037a03015b78216b7d7a31e9`/tree`7e1314d833b668172884532166d6fc870d7c4df5`，原3目标及8源compile精确SHA复用，唯一normalformal19/19、精确部署、唯一activation与direct live launch均exit0。21:53:55.416945实查running250success=attempt/0skip/eval0，八rank实际Lion/5组/空初态/有效age/whole50c匹配formal，首自然log25 finite/WORLD8/G64P8/448pairs。当前owner32e3与真实PID/start/argv、96/24互斥池见[启动记录](../../records/R137/README.md)。自然rank_inputs遗漏optimizer投影，后续moment数未观测；保持冻结包，不另做GPU探针或重启。当前无完整selector/双门/official；以下“尚无formal/启动”均为历史阶段。

2026-11-01本次用户重新要求持续推进至指标、必要最小修改无需审批。实时Paseo仅新owner `32e3b079-81b0-4f66-9604-b32491599f44`，其接管唯一Git/Pod/GPU职责；不恢复历史Agent。原R137尚无formal/activation/launch，沿同一order49/round fresh合同接续，不重启旧轮。新准备目录 `technical-prep/r137-20261101-32e3` 和新CPU allocation `a3650fbdcd7931b5c34b371a8b31804da9b200931dd669150a0af8d01cc62212` 只绑定未启动阶段，旧CPU证据/成本原样。历史规划槽12–23保留不用，本owner0–11；后续训练使用实时80%互斥池。必要代码仅以下既有八文件；另随本次提交精简AGENTS及接续文档，未完成census/analysis保持隔离。原算法GREEN2两项、身份GREEN1一项和8源compile已找到，精确核SHA后复用，不重跑已通过目标。当前尚未证明正式CPU/GPU成功。

用户在根完整呈现八既有文件、一个Optimizer类和版本化age有效优化合同后明确“开始训练”；授权reference `43593082-ae75-4c90-9299-c135bcb1058f`。唯一A负责源码/Git/Pod/GPU，根和既有规划分工只读。R137/order49使用独立fresh身份；旧R136已完整关闭/no27，不重查其safe或读取旧checkpoint。原六项未获准草稿/analysis保持隔离。

profile `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-lion10x-v1`；candidate `r137-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-lion10x-9010`；round `r137-small22k-seed20261029-scale085-pairlog-lion10x-24k-p4-9010-20261101`；prep `technical-prep/r137-20261101-7c17`；配置`configs/R137-SCIENCE.json`。

直接R107 RGB3/public fe6b fresh/seed20261029，预期345全训/原50c/实际原5组/空optimizer state，由本轮唯一normalformal重新实证，不另构造公开控制模型。唯一方法变化为固定Lion(.9,.99)整体配方：所有原组initial_lr÷10、非零WD×10，SCI写有效backbone1e-5/head1e-4/age1e-4/非零WD1e-3/零WD0，构造器不得再次缩放。按已裁剪raw gradient执行符号方向、decoupled decay和单exp_avg；无epsilon、二阶矩、bias correction或假Adam step。保存恢复严格绑定算法/超参/命名成员/位置/shape/FP32/有限性，不把Adam状态接入Lion。

age源VERSION/make_spec/统计与原AGE.json不改；新`age-linear-lion10x-v1`仅携有效LR/WD和`source_contract_sha256`，还原原version/LR/WD及删除来源字段后必须精确回原源摘要。新job持有效新摘要，旧binding仍引用原AGE来源；原age特征、两标量初始化、manifest和缓存身份不变。其他R107科学值全部保持：固定9:1/splitseed20261024、512/scale085/原增强、G64P8/FP32-noTF32/purepairlog、原clip1、相对warmup1600/cosinefloor.01、EMA、24k成功/800eval/p4。GC等关闭optional不叠加。此为联合配方检验，不把收益解释为纯sign效应或已定位Adam问题。

获准源码/测试仅config.py、engine.py、distributed_engine.py、experiment91.py、age.py、check_candidate.py、test_gradient_clip_policy.py和test_candidate_binding.py；生产新增仅engine.Lion一个Optimizer类。复用原训练/保存/EMA/推理/入口/监控，不新增source/module/helper/依赖/执行器或调用层；超范围先给用户具体最小差异。必要验证仅两算法及一身份目标，真实RED后最小GREEN/受改源compile；原Gloo、CheckpointStore和globalgather/clip复用，不重旧GC/suite或GPU预演。FP32近零sign分歧须真实暴露，不宽容差、不加去噪阈值。

阶段状态：已核main a0f15d7、R137/order49未占用、同既有Pod UID/Ready和Python3.12.7；新独立CPU allocation `19eb9260df8e482aeb70c0b139e2484ab2bac2bc090075885fcbf1d3314497ac`，有效至1793548835.011004。实测effective120/prep24，A12 mask0–11、root12–23保留；后续训练80%整树和prep20%必须再实时实核互斥，不默认核数/TTL，不修改任何旧allocation/clock/grant。当前尚无本轮GREEN/formal/activation/训练首成功。

实际最小commit/nonforcepush/readback后才精确Git blob/mode部署；唯一正常formal证明publicstrict/50c/345/真实Lion类与组/空state/新age/onlineEMA/defaultfactory strict，继而新资源准入、独立grant/clock/absence/snapshot、唯一activation/正确direct uncontracted GPU launch。真实8rank初态及自然log25证明首成功，不补step1/event/collective/forward或独立GPU试跑。到真实首成功后才切current定位和归档启动小件。

只用现有实例，原6h训练/8h轮/9hgrant、Small12GiB/free20floor16/300s及保存预算不放宽；无物理扩资源/续租/关闭实例/新Agent或timer。所有R130/R120/KEEP/R134R1/R135/R136 selected、public/prepared512及失败成本继续保护。单moment减少一份持久状态，不保证总峰值减半、BW吞吐或满24k时耗；不靠早停兜底预算。

完整工程有效selected同时strict>R107 `.9884405731067525`和R120 singlebest `.9894773092062407`且actualsafe，才固定seed20261027对匹配R108 `.9872328538178018`；否则关闭精确Lion10x配方，不扫beta/倍率/LR/WD/seed救分。工程失败记unknown，不伪科学阴性；完整freeze≥.996才官方5849一次，内部/官方/平台分列，raw .999未证不称总体完成。
