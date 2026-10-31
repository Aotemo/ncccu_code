# R130R1：固定0.8逐层学习率的独立fresh合同与实施步骤

**用户已明确授权：“不需要续训，直接重头开始训练 R130 即可”。本轮采用原R130科学方案，以独立R130R1身份从公开初始化、step0及空Adam开始。** 用户随后授权每30分钟报告、每5分钟检查完成并持续推进后续训练；旧暂停、fresh未授权及禁止定时观测的条件已被覆盖。当前不另设设计或文档批准门，A可并行完成代码、Git冻结和实际准入后直接启动；本页不冒称新轮已运行。

> 执行者：现有Sol A按 `superpowers:executing-plans` 落实下列最小步骤；不另派Agent。规划仅编辑本页，A汇集、唯一Git写入并唯一串行执行Pod/GPU。勾选只代表已有对应实证，不代替回执。

**目标：** 完成被用户中断的固定逐层LR方案，获得独立、完整的内部选点和按门限允许的一次官方结果。

**实现：** 复用现有Small、layerwise分组、保存、直接训练和官方隔离链，仅新增R1科学身份及其现有正式验收路径。原R130源码、clock、token、grant、启动标记、checkpoint和全部依赖原样保留，不建继承桥或新平台。

**技术栈与依据：** 现有Python/PyTorch八rank FP32训练链；[R130科学合同](R130-DECISION.md)、[原SCIENCE](../../configs/R130-SCIENCE.json)、[接续审计](../DEVICE-RESUME-AUDIT-20261101.md)、[执行规范](../../docs/EXECUTION.md)。本页将writing-plans步骤合入项目决定页，不另建spec或审批文件。

## 独立身份与授权边界

| 字段 | 本轮固定值 |
| --- | --- |
| order / config | `41` / `configs/R130R1-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-layerdecay08-r1-v1` |
| candidate | `r130r1-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-layerdecay08-9010` |
| round | `r130r1-small22k-seed20261029-scale085-pairlog-layerdecay08-24k-p4-9010-20261101` |
| newprep（相对exp-v3） | `technical-prep/r130r1-20261101-7c17` |
| 直接科学对照 | R107 seed20261029，内部AUROC `.9884405731067525` |
| 单次best实用门 | R120 seed20261029，内部AUROC `.9894773092062407`；不升级为已复现基线 |
| 固定第二seed参考 | R108 seed20261027，内部AUROC `.9872328538178018` |

根协调 `b2a18d8d-628c-4c7c-9cf7-9345e736e0ac`、规划 `6b8ad6f4-d29a-4647-8fcf-d986715575f0`、Sol A `7c17043e-8381-43e9-b185-6e142634ad72` 配对于当前server `srv_KqW7XmJdXxGJ`；无B、无子Agent，不恢复旧Agent。非源码产物仍置于用户指定exp-v3。本机仅文本编辑/读取、Git、传输和展示；Python、解析、哈希、模型和测试仅由A在当次核验的既有Pod内执行，不新建物理资源、不关闭实例、不续租。

R130 order40仍是用户中断：1503success=attempt、0skip、eval1、无最终selector、TRAIN_RESULT、ROUND_RESULT、freeze或official，R107/R120两门为未判。新轮不读取旧1503或旧initial checkpoint作为初始化，不拼接曲线、不将旧1503补作新轮前缀、不隐去原消耗；新fresh也不把原轮改写成科学阴性。原one-shot无合法重入入口的审计结论保持，用户此次授权通过独立新轮落实。

## 科学值全部沿原R130

新SCIENCE与R130仅允许profile、candidate id/order及说明文字不同；初始化、job、数据、schedule和official policy等科学字段逐项相同。公开来源固定ConvNeXt Small ImageNet22K→1K revision `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b`，转换权重SHA `7aaea5ce6fe6db953874d9792d3f1711a5cc0193a10d740e023108043918c610`。使用公开权重与原生head初始化重建原native初态，预期whole tensor SHA `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，345tensor、0BN、空Adam；本轮实际结果须由新formal及真实训练证明，不能转抄R130为新实跑。

- 单模型、固定9:1 fit46256/dev5140、seed20261029、原AGE合同与GAP/finalLN/Dropout/Linear+age；单图prepared512、scale[.85,1]、原旋转及光度。不是R127去旋转、R128头预热或R129的22K-only来源。
- G64/P8、8rank每rank8primary、全局64view/448pairs、WORLD可微gather、FP32/noTF32；原binary pairwise logistic六字段保持。SD `.1`、dropout `.2`、AdamW/WD `1e-4`、一次globalclip1、全参数从成功步1起更新。
- `convnext-small-layerwise-lr-decay08-v1`与原12层映射固定：backbone `initial_lr=1e-4*0.8**(13-layer_id)`，head/age `1e-3`；原global successful cursor schedule再乘一次全局factor。保留实际名称、参数顺序、30组身份和原WD规则；finalLN仍属backbone，不重乘层倍率、不重分层、不冻结任何层。
- 最多24000次成功更新，每800步完成online/EMA配对内部dev，连续4轮内部AUROC无提升早停；原1600 warmup与EMA前1600 copy/1601起blend合同不变。沿原唯一selector及并列规则，不按AP、partial、官方或平台另选点。早停是正常完成，不强求跑满24k；中断或工程失败不是完整科学结果。

## 当前已核环境与证据界限

规划只文本读取A导回的[DEVICE-R130-OBSERVATION.json](../../../work/device-resume-20261101-a/receipts/DEVICE-R130-OBSERVATION.json)，观察时间 `2026-10-31T18:41:21.606972+00:00`。该回执证明当时namespace/Pod/UID/container/Python五项与旧R130 SITE一致，解释器3.12.7、有效CPU120；A实际prep12核mask0–11/nice19/隐藏GPU且新prep TTL有效，12–23为根保留份额、未执行。训练计划mask24–119共96核，回执明确 `training_grant_created=false`，不能冒称已获新GPU grant。

当时8张BW3000 `[1d94:6320]`、KFD名beimeizhou，每卡显存68702699520B/busy0；device_users为空、扫描拒绝0、原锁holders为空。cgroup memory max/current分别506806140928/123449794560B，磁盘free54034169856B。这是该时刻的实核快照，实际启动仍核当前身份、份额TTL和资源，不把旧safe或Agent idle代替准入。

旧last `checkpoint-806c3832c05e487d88c938245ac1d3e2.pt` stat792176842B，index广告SHA `08ac68114696b7bd698fb8e70658f7cfbeecf994088a023a115b7aa0591eb377`，sidecar与index一致；previous/best/initial/protected、source/manifests/cache/data、KEEP三模型和R120 selected均stat存在。未hash/load大blob，stat不等于数据完整性或模型可加载证明；1503游标来自runtime status，不在index。原implementation已过期，训练/整轮deadline及原GPU grant观察时仍有效，但grant owner不匹配新A；旧prep本人entry TTL未追到，保持unknown。无需为新fresh追加旧checkpoint加载。

资源比例仍集中读取版本化RESOURCE-POLICY并经env注入：训练owner/ranks/loader/control/helper整树最多 `floor(4*effectiveCPU/5)`，准备合计最多 `floor(effectiveCPU/5)`，两池互斥且总额不超有效CPU；当前120对应96/24上限，不把核数固化成新环境默认。只为本轮未启动阶段新建绑定，旧allocation/grant/clock不改。

独立新轮维持6h训练/8h整轮/9hgrant、Small12GiB round cap、启动free20GiB/运行floor16GiB、健康9C+I与准备余量、300s atomic/prune/failfast及全部warm/bestonlineEMA/lastprevious/initial保护。旧R130全部依赖、R120 selected/source与KEEP-MODELS继续保护；N06仍只研究隔离。新轮时钟不延长旧轮预算；新吞吐、满24k耗时、500MB最终包体及指定硬件100ms尚未实证。

## 文件边界与最小实施步骤

规划仅本页；A拥有其余文件。生产改动限定 `configs/R130R1-SCIENCE.json`、`runtime/bundle/retrain/experiment91.py`、`tools/validation/check_candidate.py`，新增唯一身份窄验 `tests/test_layerwise_lr_r1_identity.py`；README/接续审计及必要小回执由A同步。沿既有activation/资源/部署工具注入新身份，不另建训练源码、恢复工具或预算平台。

核对重点共五项：旧order40继续精确接受；新order41独立且科学值全等；新profile进入原layerwise正式检查；本轮真实初始化不继承旧Adam/checkpoint；当前资源与新一次启动绑定相符。前三项由唯一身份窄验和normal formal覆盖，后两项由既有部署/真实8rank入口覆盖，不另建拒绝case矩阵或GPU预演。

### 1. 独立身份与现有formal路径

- [ ] A创建上述新SCIENCE；`experiment91.py`增添精确R1 profile与candidate/order41映射，同时保留旧R130映射及全部科学校验。
- [ ] `check_candidate.py`以最小 `layerwise_profile(profile)` 谓词精确枚举R130与R130R1，供main与窄验共用；两profile均执行现有30组identity/空Adam及strict reload检查，不只改SCIENCE文件名而漏正式路径。
- [ ] 在A自己的新prep、有效Pod CPU guard内，只运行 `test_layerwise_lr_r1_identity.py` 中 `test_new_fresh_identity_preserves_science_and_existing_formal_route`。该项核实际`science_configuration/candidate`接受旧40/新41，除profile/id/order/note外整件科学相等，seed20261029、`initial_checkpoint=None`及正式layerwise路由；记录改映射前实际未支持失败和修正后的实际PASS，不造失败或重测已经通过的算法。

窄验命令在已部署源码根、Pod guard内为 `python -m unittest discover -s tests -p test_layerwise_lr_r1_identity.py -v`，`python`须替换为当次已核解释器绝对路径；不得在本机直接执行。通过口径是该唯一测试实际成功，非登记文件存在。R130原3项算法/保存fixture不重跑；现有normal formal仍对本轮实际配置/来源/初态/组产生新回执。

### 2. 正常formal、同步与精确部署

- [ ] A沿现有prepare与`check_candidate.py`完成本轮normal formal，明确新config/profile/实际source身份、native初态、30组与空Adam；未跑与失败如实保留，不写成R130旧20/20已覆盖新轮。
- [ ] A汇集实际最小diff（含本页），提交并push唯一仓库，回读远端commit/tree，再按该commit精确打包部署到本轮newprep/formal root；记录实际文件SHA、source digest、SCIENCE身份和部署回执。代码尚未同步不得GPU执行，不热改旧包。
- [ ] A按当下实核身份与RESOURCE-POLICY为本轮创建有效GPU grant及两池互斥绑定，保留准备池并行；核旧精确进程已退出、锁/设备占用、储存余量及声明资源时限。沿现有薄链准入，不增加文档批准或全体prep退场条件。

### 3. 新轮一次启动与真实首步

- [ ] 只在新R130R1 root执行一次activation、一次live launch，产生本轮新clock/token；不重复旧入口或重置旧标记，不用环境RESUME绕过fresh合同。
- [ ] 由真实训练入口证明8rank设备/CPU身份、native345/0BN/初态空Adam/30组一致；首自然成功核WORLD finite、scaled initial_lr乘原cursor的actual LR与345tensor Adamstep1。初态证据和首步证据分列，不追加forward/optimizer probe，不要求每个低LR参数都有可见FP32位移。
- [ ] 记录实际进度、eval次数、跳步、保存状态及资源；观测失败记unknown，不据此重启或判设备空闲。complete后才进入冻结/官方/安全退出判定。

### 4. 完整结果、后续分支与唯一观测

- [ ] 工程完整且原selector冻结后，用完整精度internal AUROC判两门：严格大于R107 `.9884405731067525`且严格大于R120 `.9894773092062407`。两门均过后，最新持续推进授权已覆盖固定seed20261027复核；建立独立科学身份、同方法仅换seed，并对照R108 `.9872328538178018`，不再等例行批准。第二seed完整严格胜参考才考虑方法升级，不能挑最好seed掩盖退化。
- [ ] 任一门未过则关闭本固定方案，不扫factor、层分组、LR或seed；科学门过/实用门未过须分列，不能统称科学阴性。分支关闭后由规划有限评议一个后续方向，再由A依必要证据串行推进；不恢复已关闭方法补考、不自动展开候选网格。工程中断/失败无完整selector时不判两门false，先处理真实原因，不重复旧once入口。
- [ ] 只有完整freeze的唯一selected internal AUROC达到 `.996`，才运行完整官方5849一次；完整selected不足时才记 `not_run_by_policy/threshold_not_met`。官方不参与选点、早停、年龄统计、难例挖掘或调参，不重复官方前向。
- [ ] 根负责唯一已active heartbeat `9c70b2bc`，`*/5`、Asia/Shanghai、无expiry/maxruns；内含每5分钟完成小检查及每`:00/:30`公开报告，完成后先核终态与实际安全再派下一轮。A不另建timer/heartbeat，规划不自行轮询。该监测授权不等于新建物理计算资源或续租。
- [ ] A归档必要精简结果及真实安全/部署回执，规划按完整内部结果作上述分支判断；保留所有受保护资产，不因新轮完成擅自清理旧R130。

总体目标仍为raw AUROC `0.999`，内部dev、完整官方5849与用户平台成绩分别报告。内部达到`.999`只证明内部值，不宣布官方或平台达标；完整一次官方raw达到`.999`才可报告官方目标达到，平台目标只有平台实际成绩才能证明，未经授权不生成或提交平台test CSV。未达到对应目标且没有新停止指令时，按上述完整结果分支持续推进；不把本页落盘、启动成功或某个阶段完成当总体目标完成。

本页完成的是已授权合同与可执行步骤，不是新formal、Git同步、activation或训练成功回执。上述实施状态由A后续真实结果接续，不为计划文案新增批准门或重复验证。
