# 设备交接定稿：R130按用户要求即时暂停

**最终停止状态：2026-11-01 01:43:49.896北京时间，R130已按用户要求中断并实际安全停止；当前无活跃训练，原heartbeat已删除。** 本页为迁移交接定稿，不是准备草稿或完整科学结果报告。根已文本核读A停止小原件，本文据转交实证记录；规划未另查Pod、重hash或加载模型。本次停止归档随Git历史定位，最终commit由协调方回执，不为记录归档自身SHA再造提交。

## 最新指令与恢复边界

用户最高优先新指令为：**“直接暂停本轮训练,等我迁移设备后继续”**，已覆盖“本轮结束后暂停”。R130已立即停止，不等待自然训练完成或官方评估。所有后续fresh、seed27及新方法评议暂停，无自动续训或下一轮。R130仅1503成功/eval1，没有最终selector，不判双门false或科学阴性。

**必须由用户迁移设备后明确恢复。恢复前仅只读接续材料，不启动训练、seed、GPU评估或诊断。** 当前实例未关闭、资源未续租；不申请新资源。总体raw AUROC 0.999尚未达到，用户暂停不是总体达标、方法穷尽或所有方向无效。旧文件的持续推进、完成本轮或条件seed27派发文字均受此次即时暂停覆盖。用户日后要求恢复也不等于允许重置原绝对时钟、token或lease；能否续接须先核保存schema与原deadline/启动约束。

## 新设备唯一入口与已实证版本

唯一源码仓库：[Aotemo/ncccu_code](https://github.com/Aotemo/ncccu_code)。新设备先只读取得实际最新main，依次读仓库[AGENTS.md](../AGENTS.md)、[todo入口](README.md)、[EXECUTION.md](../docs/EXECUTION.md)和本页，确认读到此即时暂停定稿。不能把下列运行commit当作永远最新，也不能凭已删除heartbeat的大段旧prompt接续。

本次已实证R130冻结runGit/main为 `c8c610dd5ee42e9de99473a83c15ae14b8189178`，parent `5f5072ca2c7b05fa5a9a7a15a85c0c053dee8623`，tree `d05cec3880fc327d716789950f5451969cc52669`，12path/12blob remote exact。实际部署666文件，deployment SHA `f3880f28688535ff5b24efed5a8a3995a1bbc1e7c2bf1cf11867a919f2fe63f2`。该commit是运行源码，不冒充随后停止归档的commit。

上一个完整科学轮R129已归档main `5f5072ca2c7b05fa5a9a7a15a85c0c053dee8623`，parent `04cdb75c170476a7fc80febeb4b4940b74cfa156`，tree `b3a34b522bb9b81db5224f253665a7e4cdac97db`，4path/4blob remote exact。本次停止归档包含 `records/R130/{R130-RESULT.json,README.md}`、R130决定页、todo入口及本页共5件；**提交由Git历史定位，最终commit由协调方回执**。运行版本、结果归档与模型/回执SHA各自分列。

## 本轮R130冻结身份

原冻结科学合同见[R130-DECISION.md](auroc-999/R130-DECISION.md)及运行Git中的SCIENCE。下列身份供审计和迁移阅读，不是再次activation/launch授权。

| 字段 | 固定值 |
| --- | --- |
| order / config | `40` / `configs/R130-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-layerdecay08-v1` |
| candidate | `r130-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-layerdecay08-9010` |
| round | `r130-small22k-seed20261029-scale085-pairlog-layerdecay08-24k-p4-9010-20261101` |
| readout | `native_avgpool_final_LN_dropout_linear_plus_age` |
| newprep相对路径 | `technical-prep/r130-20261101-e62a` |

直接R107 seed20261029、原Small22K→1K fe6b358a fresh，唯一变化为 `convnext-small-layerwise-lr-decay08-v1`：backbone initial_lr=1e-4×0.8**(13-layer_id)，head/age仍1e-3，原global successful schedule再乘一次global factor。原作者Small分组映射保留实际network.前缀，finalLN属backbone，实际head为network.classifier.3；原多维layer_scale的WD判据不变。全参数从step1训练，原globalclip1/EMA/数据与pairlog/单图512/G64P8/800max24000p4不变，None扩展恢复原路径，不继承R129来源、R128头预热或R127去旋转。

本轮已实证native345tensor、0BN、初态空Adam，whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，backbone/head/age元素49454688/769/2；30个新组identity通过。真实首成功另有30组actual LR消费及345tensor Adamstep1，不能只由同初态推断执行；初态Adam0更不能冒充1503步后仍空。24k+30eval约3–3.5h仅历史Small估计，本轮未跑满不能据此宣称实测；最终500MB包体与指定硬件端到端100ms仍未实证。

本轮分工为A唯一生产集成/Git/Pod/GPU，B仅 `runtime/bundle/retrain/layerwise_lr.py`、`tests/test_layerwise_lr.py`，A其余production/config/SCIENCE及第三绑定fixture；规划仅三份交接文本。3/3窄验已由A实际执行通过，20/20正式通过。角色及旧Agent ID属于历史授权，不能作为新设备默认配对或自动恢复旧Agent的指令。现在仅完成停止归档，不继续实验。

## 已完成工程与用户停止实证

| 项目 | 实际证据与边界 |
| --- | --- |
| 三项窄证据 | 3/3 PASS；B两项CPU `d9c1fb61db1321b7fd720e20cc4c44344b1fdd32a692133b671c13b82cba763b`；A新增savedAdam-ID子集检查后仅第三项重测PASS，CPU `8bddc5b7c1a8ccc7649a3c3294213370150819dc87cdf67e4c3178a3713f390c`，非先前训练失败 |
| 正式与初态 | formal20/20、native345/50c/Adam0/BN0与30组identity；formal CPU `d050e3d2d0402c11a16938f045478fa969c3186ef6f75a7283c7df841dc9583a` |
| 唯一启动 | 原授权reference `2e1c4016-d1c9-49d7-ab38-c474da99a643`；唯一activation0、正确live launch0，未再activation/reclock |
| 首成功与START2 | 01:31:26.116北京时间125success=attempt/0skip/eval0、8rank start match；首自然成功WORLD finite、30组actual LR=scaled initial_lr×原cursor、345tensor Adam1 |
| START2原pin | START2 `6c963b1296a6fb15efdd98cddd4428b026c53cb747b9d9b6ab23f59261ce32e7`；CPU `4d53581be17893705ce34b4d1f51a93926b6347a6f49d1d86f7a80e8e170e997` |
| 用户停止 | 一次已核身份supervisor SIGTERM于01:41:51.393发送；发送前1500/eval1，01:43:49.896观测1503success=attempt/0skip/eval1、interrupted/owner_stop_requested |
| Controller分类 | train-component timeout/exit130、done.failed/component not completed是用户中断的既有分类，非earlystop/科学阴性/完整训练结果，原件不改 |
| 最终选点与官方 | TRAIN_RESULT、ROUND_RESULT、freeze和official均未产生；无最终selector/AUROC，不判双门false，不写threshold_not_met |
| 实际安全 | 01:43:49.896北京时间：owner326810/start349511910现Z非reaped，supervisor absent，8rank326822–326829 absent，locks[]/device_users[]/scanDenied0/8busy0，actualsafe=true |
| Heartbeat | 根实际delete_heartbeat删除唯一 `1c2c8c3b`，返回 `{success:true}`；不再定时查询，不创建替代timer |

现有非Git产物基址为 `/public/home/acfwvv9ejn/SothisAI/exp-v3`；本轮prep相对路径见身份表，原round为 `rounds/<上表round>/`。停止导回小件在工作区 `../work/r130-20261101-a/stop/` 共3件，根已文本读 `STOP-OBSERVATION-R130-1`；STOP SHA `104e04e8bd725b2c54e2710ad9201e988f25ed0c5416dcb42638c2eb20aaf7bd`，CPU SHA `0a67b442051396ddb066561a30635072c01e1e716a989469addbb929c8b312e3`。只转引已有原件，不重查算；不把未产生的TRAIN/ROUND/freeze列成可用最终结果，不输出token或私有clock内容。

## Checkpoint现状与迁移续接限制

提交index的精确相对exp-v3路径已从[A结果README](../records/R130/README.md)纯文本引用：

```text
rounds/r130-small22k-seed20261029-scale085-pairlog-layerdecay08-24k-p4-9010-20261101/outputs/r130-small22k-seed20261029-scale085-pairlog-layerdecay08-24k-p4-9010-20261101-8c-convnext_small-s20261029-b64-p8-fp32/convnext-small/checkpoints/index.json
```

该index所在 `checkpoints/` 目录的last文件为 `checkpoint-806c3832c05e487d88c938245ac1d3e2.pt`，index广告SHA `08ac68114696b7bd698fb8e70658f7cfbeecf994088a023a115b7aa0591eb377`、792176842 B。根转交A RESULT确认保存cursor1503/eval1/0skip；文件存在，sidecar与index一致，previous/best/initial也存在，protected含best_ema/best_online/snapshot-800。**本次没有重hash或load大blob**，故此SHA是提交index记载，不冒称迁移时实际字节复验已通过；路径是该冻结运行的实际位置，不是新环境默认。

interrupted记录在save成功后写入，冻结代码的payload合同含model/optimizer/EMA/distributed_rng/successful+attempted/cycle。cursor1503来自提交元数据，仍不是已逐字段reload验证。保留全部R130续接依赖、原source/manifest/cache及R120/KEEP-MODELS，不清理、不搬改原件。

底层 `run_distributed` 支持resume，但要求原 `absolute_deadline_at` 完全一致；原one-shot owner/clock与grant没有暂停。**不能承诺换设备后直接续训，也不能把用户明确恢复当作重置clock/token/lease或再次activation/launch的授权。** 用户恢复后先只文本核checkpoint提交信息、保存schema、原deadline和一次启动约束，再决定是否存在合法续接；不假定有剩余时限、不修改历史回执或原冻结包。兼容fixture通过不等于这份大checkpoint已完整恢复成功；本文不复制具体private clock/grant epoch或内容。

## 指标、分支结论与保护范围

| 科学锚点/已记录best | 内部AUROC | 解释 |
| --- | --- | --- |
| R107 seed29 | .9884405731067525 | 当前同seed科学比较锚点，原GAP+age/pairlog合同 |
| R108 seed27 | .9872328538178018 | 固定第二seed比较锚点 |
| R120 seed29 | .9894773092062407 | AP .8313831149805229，EMA c6@4800；仅单次best，R121未胜R108，不升级为方法基线 |
| R129 seed29 | .9884188991813276 | AP .8233186216195478，EMA c4@3200；6400成功/0skip/eval8/bad4，2026-11-01 00:36:08.609已完整safe，双门false/no27 |

R130原完整selector的internal5140 strict>R107科学门及同时strict>R120实用门保留，固定27比较锚点R108不变。**本轮用户中断无最终selector，两门未判定、无seed27资格结论；当前不派任何后续训练。** AP、partial或官方不能救主判，不factor/分组层数/LR/seed补考。原完整训练并冻结唯一checkpoint后selected>=.996才官方5849一次的政策保持，但本次未达到完整冻结，不生成official、不记threshold_not_met。内部5140、官方5849与用户平台成绩分别报告；未授权平台test CSV或提交，raw.999未达是用户暂停而非总体达标。

近期已关闭的固定分支包括BN替换、早层冻结、stride1、固定1600 SSL、Base/Swin/DINO来源、sigmoid两seed、光度identity、去age、content pool、stage3残差、roleclip、旋转identity、1600头预热及22K-only来源；具体边界与非等价反例见[入口历史](README.md)及各决定页。R127是有限正向但未刷新best，不能改写成双门false；R119是12CPU读取筛查失败，不是局部科学阴性或真实训练超时。Tiny、SAM、L2-SP和EMA教师有界评议未采用，不能推出普遍无效。暂停期间不重开近邻补考、FIT/cache/C0诊断或自动候选队列。

用户已说明True_Label按单张图可见征象，来源是用户说明，未独立验证标注；不推无噪声、标签错误或.999可达。原增强是rotate-expand后整图fit画布，不是random-resized-crop。无新的临床、规则或样本审计门。

永久保留N06、R17、R8-C25各一个唯一EMA及最小加载身份，权威入口为[KEEP-MODELS.json](../records/storage/KEEP-MODELS.json)及[models](../models)。N06研究隔离，禁止作为比赛初始化、teacher、BN或训练参数；FLAIR等外部医学参数禁用保持。当前保留全部R130续接产物/实际依赖与R120真实selected/source，**此次safe不授权清理**。模型、图像、数据、cache、checkpoint、样本预测、私有clock/token与凭据不进Git或本文。

## 新设备接续步骤与运行限制

1. 先只读实际最新Git、AGENTS/todo/EXECUTION和本停止定稿，保留此次用户即时暂停指令。没有用户迁移后的明确恢复指令，不运行训练、seed、GPU评估或诊断，不依据“目标未达”自行续跑。
2. 读取R130停止聚合、原提交index/schema和启动合同：这是用户中断，不是完整科学轮。安全事实只在01:43:49.896实核时成立；未来执行不能用旧安全快照或Agent idle代替动态准入。原heartbeat已删除，不恢复旧prompt定时查询。
3. 用户明确恢复后，先文本判断保存状态与原绝对deadline、one-shot启动约束能否合法续接，再由明确的新接续安排确定执行；不承诺直接resume，不重置clock/token/lease，不擅自再activation/launch或改成新fresh，不默认seed27。旧主机绝对本机路径、Pod/Python/CPU allocation/Agent ID仅为历史证据，不恢复旧Agent或当作新默认。
4. 任何实际执行仍需工具动态discover既有Pod、核UID/Ready/解释器/依赖/effectiveCPU/本人entry TTL。Mac只文本、Git、传输与展示；Python/配置解析/hash/模型/测试只由唯一执行A在核验Pod guard内完成，不在Mac或SSH登录节点算、不装项目依赖。
5. prep合计20%和训练准入owner/rank/loader/control/helper整树80%用实时有效CPU的互斥mask，不默认120核/12核/旧TTL。过期份额只能按授权在既有Pod新建必要future软件绑定，不热改旧allocation/grant/clock，不新建物理资源、续租或关闭实例。
6. 保持6h训练/8h整轮/9hgrant、Small12GiB、free20GiB启动/16GiB运行floor、健康9C+I与prep余量、全部warm/bestonlineEMA/lastprevious/initial及真实依赖保护、300s atomic/prune/failfast。暂停没有重置这些预算或原绝对截止时间。代码先实际Git同步再按清单精确部署，不热改旧冻结包、再次activation/launch、重置clock/token或重复官方。历史EXECUTION片段不得覆盖最新即时暂停及原R130合同。

原唯一heartbeat `1c2c8c3b`已由根实际delete_heartbeat成功删除（`success:true`），不再定时查询，当前无活跃训练。新设备不沿旧prompt自动恢复heartbeat，不创建监控、轮询器或训练队列。当前实例未关、资源未续租；交接定稿与结果归档不会自动恢复训练，也不制造新的科学或文档批准门。
