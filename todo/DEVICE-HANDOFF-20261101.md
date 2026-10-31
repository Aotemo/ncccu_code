# 设备交接：R130之后暂停后续训练

**草稿：R130尚在必要准备，未收到本轮formal/Git冻结/activation/launch/首success或最终结果。最终实际终态、安全、归档与heartbeat停用回执到齐后再定稿。** 此草稿是接续材料，不是启动门，也不证明本轮已运行或原设备已完成收尾。

## 最新指令与恢复边界

用户最新明确：**“本轮结束后,先暂停后续训练,给出一个交接文档,在另一个设备进行”**。协调方将本轮明确为R130：允许完成必要准备、一次fresh、内部训练及符合既定阈值的一次官方评估，随后实际安全收口。正常完成或不可重试工程终态后，暂停所有后续fresh、seed27及新方法评议。即使R130双门通过，也只记录复核资格，不自动执行seed27。

**必须由用户在另一设备明确恢复。恢复前仅只读接续材料，不启动训练、seed、GPU评估或诊断，不自动派下一轮。** 不关闭当前实例、不申请资源或续租。总体raw AUROC 0.999尚未达到；按用户暂停不是宣告目标完成、方法穷尽或所有方向无效。旧文件中持续推进或条件seed27派发的历史文字均受本次暂停指令覆盖。

## 新设备唯一入口与已实证版本

唯一源码仓库：[Aotemo/ncccu_code](https://github.com/Aotemo/ncccu_code)。新设备先取得实际最新main，依次读仓库[AGENTS.md](../AGENTS.md)、[todo入口](README.md)、[EXECUTION.md](../docs/EXECUTION.md)和本页，核对本页是否已由草稿更新为最终版。不能把下列历史commit当作永远最新，也不能凭旧heartbeat的大段prompt接续。

本草稿收到的最新完整归档证据为R129：main/commit `5f5072ca2c7b05fa5a9a7a15a85c0c053dee8623`，parent `04cdb75c170476a7fc80febeb4b4940b74cfa156`，tree `b3a34b522bb9b81db5224f253665a7e4cdac97db`，4path/4blob远端正文exact。RESULT SHA `8cd605602e30985aa3e252e6381a730e193d57cb59204ad33c651af7ff282062`，归档README SHA `85e9faafc8d314e678b6072618caf6d9e389bdc0ef8a80ab89da6007750f214a`；以上仅转引根交A实证，未重查或重算。

R130 runGit、完整tree、部署清单/回执、结果归档commit：**待A实际回执**。不能填R129的runGit/归档Git或旧模型初态，也不能把文档已提交当作训练已启动。

## 本轮R130冻结身份

完整科学合同以[R130-DECISION.md](auroc-999/R130-DECISION.md)及A实际冻结的SCIENCE为准；SCIENCE生成、正式绑定尚待回执，本页不声称文件已验收。

| 字段 | 固定值 |
| --- | --- |
| order / config | `40` / `configs/R130-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-layerdecay08-v1` |
| candidate | `r130-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-layerdecay08-9010` |
| round | `r130-small22k-seed20261029-scale085-pairlog-layerdecay08-24k-p4-9010-20261101` |
| readout | `native_avgpool_final_LN_dropout_linear_plus_age` |
| newprep相对路径 | `technical-prep/r130-20261101-e62a` |

直接R107 seed20261029、原Small22K→1K fe6b358a fresh，唯一变化为 `convnext-small-layerwise-lr-decay08-v1`：backbone initial_lr=1e-4×0.8**(13-layer_id)，head/age仍1e-3，原global successful schedule再乘一次global factor。原作者Small分组映射保留实际network.前缀，finalLN属backbone，实际head为network.classifier.3；原多维layer_scale的WD判据不变。全参数从step1训练，原globalclip1/EMA/数据与pairlog/单图512/G64P8/800max24000p4不变，None扩展恢复原路径，不继承R129来源、R128头预热或R127去旋转。

原345tensor/0BN/空Adam、whole期望 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5` **待本轮实际核**；同初态不能证明逐层倍率被消费。预计模型参数/Adam/EMA元素不增，但更多组有调度成本；24k+30eval约3–3.5h仅历史Small估计，最终500MB包体与指定硬件端到端100ms仍未实证。

当前授权协作边界为：A唯一生产集成/Git/Pod/GPU；B仅 `runtime/bundle/retrain/layerwise_lr.py`、`tests/test_layerwise_lr.py` 两文本文件；规划仅R130决定页、todo入口和本交接文档，经根handoff。B两文件已文本冻结并交A，不等于测试通过。A负责其余production/config/SCIENCE和第三绑定fixture，不共写B文件。角色及旧Agent ID属于本轮历史授权，不能当作新设备默认配对或自动恢复旧Agent的指令。

## 实证索引与最终待补项

| 项目 | 草稿时状态 / 最终需补内容 |
| --- | --- |
| 授权与准备 | 根采用R130；A唯一授权reference `2e1c4016-d1c9-49d7-ab38-c474da99a643`、ack `56b3c366`；准备在途，非运行证据 |
| 三项窄证据 | B映射/唯一覆盖/WD/None/RNG，B独立AdamW及schedule参考，A SCIENCE→组身份→strict保存加载→LR/EMA/RNG；实际结果与回执待补 |
| 正式与初态 | actual formal项目数/PASS、新非tensor组身份、345/50c/空Adam、实际参数角色数量待补；不得抄旧formal数量 |
| Git与部署 | R130 runGit/完整tree/实际文件清单与部署receipt待补，和最终archive Git分列 |
| 唯一启动 | 两处120s freshness薄链、一次activation/唯一launch、真实8rank初态/firstbatch/首success待补 |
| 策略真实消费 | 首自然成功的各组实际LR、全role Adamstep1及成员身份待补；不要求每个FP32参数可见非零位移 |
| 内部终态 | success/attempt/skip/eval/早停、唯一selected版本/cycle/step/AUROC/AP/error、tensor与blob分别待补 |
| 官方 | selected完整冻结后>=.996才一次5849；实际运行或not_run_by_policy等待补，无selected不能写threshold_not_met |
| 安全 | 实际owner/rank/锁/设备/终态及观测时间待补；Agent idle不是safe，owner Z不能称reaped |
| 归档与停用 | R130精简结果/本页定稿的实际Git回读，以及根对唯一heartbeat的实际停用回执待补 |

现有非Git产物基址为 `/public/home/acfwvv9ejn/SothisAI/exp-v3`。本轮计划prep相对路径见上表；最终冻结包/部署/观察导回目录的精确路径待A回执。计划终态索引为 `rounds/<上表round>/SINGLE-MODEL-FREEZE.json`、`evidence/TRAIN_RESULT.json`、`evidence/ROUND_RESULT.json`、`control/done.json`，**这些目前仅为计划位置，存在性及实际路径待最终回执确认**。仓库 `records/R130/` 的最终归档也待实证，不虚填已存在。只在交接中记录脱敏聚合及必要身份；含token的TRAIN等原件不能整件带入交接文档或公开输出。

## 指标、分支结论与保护范围

| 科学锚点/已记录best | 内部AUROC | 解释 |
| --- | --- | --- |
| R107 seed29 | .9884405731067525 | 当前同seed科学比较锚点，原GAP+age/pairlog合同 |
| R108 seed27 | .9872328538178018 | 固定第二seed比较锚点 |
| R120 seed29 | .9894773092062407 | AP .8313831149805229，EMA c6@4800；仅单次best，R121未胜R108，不升级为方法基线 |
| R129 seed29 | .9884188991813276 | AP .8233186216195478，EMA c4@3200；6400成功/0skip/eval8/bad4，2026-11-01 00:36:08.609已完整safe，双门false/no27 |

R130完整原selector所选internal5140 strict>R107才科学正向，同时strict>R120才取得固定27对R108的复核资格。**最新暂停覆盖后续派发，即便有资格也不执行。** AP、partial或官方不能救主判；不factor/分组层数/LR/seed补考。完整训练并冻结唯一checkpoint后selected>=.996才做官方5849一次；不足正常记not_run_by_policy/threshold_not_met/evaluated=false/metrics=null，safe独立核。内部5140、官方5849与用户平台成绩分别报告；未授权平台test CSV或提交，raw.999目标仍未达。

近期已关闭的固定分支包括BN替换、早层冻结、stride1、固定1600 SSL、Base/Swin/DINO来源、sigmoid两seed、光度identity、去age、content pool、stage3残差、roleclip、旋转identity、1600头预热及22K-only来源；具体边界与非等价反例见[入口历史](README.md)及各决定页。R127是有限正向但未刷新best，不能改写成双门false；R119是12CPU读取筛查失败，不是局部科学阴性或真实训练超时。Tiny、SAM、L2-SP和EMA教师有界评议未采用，不能推出普遍无效。暂停期间不重开近邻补考、FIT/cache/C0诊断或自动候选队列。

用户已说明True_Label按单张图可见征象，来源是用户说明，未独立验证标注；不推无噪声、标签错误或.999可达。原增强是rotate-expand后整图fit画布，不是random-resized-crop。无新的临床、规则或样本审计门。

永久保留N06、R17、R8-C25各一个唯一EMA及最小加载身份，权威入口为[KEEP-MODELS.json](../records/storage/KEEP-MODELS.json)及[models](../models)。N06研究隔离，禁止作为比赛初始化、teacher、BN或训练参数；FLAIR等外部医学参数禁用保持。R120真实selected/source、R130活跃产物及实际依赖按原保护规则保留至安全终态与结果核实，不在交接中自行清理。模型、图像、数据、cache、checkpoint、样本预测、私有clock/token与凭据不进Git或本文。

## 新设备接续步骤与运行限制

1. 先只读实际最新Git和本页定稿状态。确认用户是否已明确恢复；没有恢复指令就停留在材料接续，不运行训练、seed或GPU诊断，不依据“目标未达”自行续跑。
2. 核最终R130结果、工程失败如有、实际safe与heartbeat停用回执是否齐备；unknown不推空闲，不用旧Agent idle或旧安全快照代替实时准入。草稿缺项由原协调方按实际回执一次补齐，不凭空补值。
3. 用户恢复后，后续科学选择与执行身份由明确的新接续安排确定，不恢复旧round、clock/token、旧Agent或默认seed27。旧主机绝对本机路径、Pod/Python/CPU allocation/Agent ID仅为历史证据，不是新设备默认值。
4. 任何实际执行仍需工具动态discover既有Pod、核UID/Ready/解释器/依赖/effectiveCPU/本人entry TTL。Mac只文本、Git、传输与展示；Python/配置解析/hash/模型/测试只由唯一执行A在核验Pod guard内完成，不在Mac或SSH登录节点算、不装项目依赖。
5. prep合计20%和训练准入owner/rank/loader/control/helper整树80%用实时有效CPU的互斥mask，不默认120核/12核/旧TTL。过期份额只能按授权在既有Pod新建必要future软件绑定，不热改旧allocation/grant/clock，不新建物理资源、续租或关闭实例。
6. 保持6h训练/8h整轮/9hgrant、Small12GiB、free20GiB启动/16GiB运行floor、健康9C+I与prep余量、全部warm/bestonlineEMA/lastprevious/initial及真实依赖保护、300s atomic/prune/failfast。代码先实际Git同步再按清单精确部署，不热改旧冻结包、再次activation/launch、重置clock/token或重复官方。历史EXECUTION片段不得覆盖最新用户暂停及R130专属合同。

原唯一heartbeat `1c2c8c3b`当前仍按5min轻量completion、公开:00/:30服务本R130，无第二timer；A执行/查询在途不追加，无活跃轮不查旧safe。**停用尚待根在本轮完整收口、实际安全和交接后操作并交回实证，现在不能写已删除/已停用。** 新设备不沿旧prompt自动恢复heartbeat，不创建监控、轮询器或训练队列。最终定稿只补实际回执，不以交接文档制造新的本轮启动批准门。
