# R129：同Small架构的ImageNet22K-only公开初始化对照

本轮已完整工程有效并安全退出。A实际2026-11-01 **00:36:08.609北京时间**：6400success=attempt/0skip/eval8/earlybad4，completed/early_stopping；唯一EMA c4@3200内部AUROC **.9884188991813276**、AP **.8233186216195478**。低于预冻R107与R120两门，根关闭固定22K-only×R107合同/no seed27，不source epoch/384源/LR/seed补考。来源、初态、窄验、唯一activation、包装命令失败及一次真实launch分阶段实证见末节；不把失败记录覆盖成全部firsttry成功。以下保留原预登记合同，结果归档由唯一A处理，本文不新增批准门。

## 唯一身份与问题

| 字段 | 固定值 |
| --- | --- |
| order / config | 39 / `configs/R129-SCIENCE.json` |
| profile | `single-convnext-small-in22konly-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-v1` |
| candidate | `r129-convnext-small-imagenet22k-only-24k-p4-seed20261029-scale085-pairlog-9010` |
| round | `r129-small22konly-seed20261029-scale085-pairlog-24k-p4-9010-20261031` |
| readout | `native_avgpool_final_LN_dropout_linear_plus_age` |
| newprep | `technical-prep/r129-20261031-e62a` |
| 唯一来源repo / tag | `timm/convnext_small.fb_in22k` / `fb_in22k` |
| 已实证唯一source revision | `4cde7ae59c8f7fcff3f9547de274819f54863dfd` |
| 新显式format | `torchvision-small22k-only-full-head-state-dict-v1` |
| training_data | `ImageNet-22K pretraining (no ImageNet-1K fine-tuning)` |
| 唯一activation授权reference | 根对A执行派工 `ec9c95a4-1be4-4eb2-8521-eb83ac85b067` |

直接对照R107，唯一科学变化是公开初始化由22K→1K改为上述22K-only来源；比较同一Small部署结构下两种公开预训练recipe/权重的迁移净效果。未取得同ancestor谱系证明，不能称精确撤销同一checkpoint的1K阶段，也不是纯类别数量因果。可能保留不同的可迁移表示，也可能因缺少1K微调而更难优化；未证当前1K遗忘、标签噪声或预训练语义瓶颈，不称22K类别更多必然更好。

支持一次有限对照的依据是问题可辨认、部署结构及下游算法不增加变化，不是已有收益证明或仅因未做过。R92→R93正向含finalLN变化，旧R16的22K→1K未胜其旧1K参照，R118来源替换亦阴性；这些非等价反例说明来源效果不单调，不能跨合同相减或用作本源胜据。定向历史检索未见等价完整对照，不等全历史从无。Tiny专项评议已结束、暂未采用，非Tiny来源禁用或普遍无效；本轮不是Tiny、R128解冻或已关闭方法的改名续跑。

[公开模型卡](https://huggingface.co/timm/convnext_small.fb_in22k)声明原作者ImageNet22K预训练及Apache2.0；[原作者源码](https://raw.githubusercontent.com/facebookresearch/ConvNeXt/main/models/convnext.py)列出Small深度[3,3,27,3]、宽度[96,192,384,768]及独立22K checkpoint路径。它们只支持公开来源/结构文字，不是当前字节、下载或strict验收。用户已说明True_Label按单张图可见征象；该说明不等独立标注验证，不推无噪声或`.999`可达，不新增医学、规则PDF或样本审计门。

## 固定来源、完整公头与初始化边界

根转交A `85adad07`的早期实证：唯一revision为上述`4cde7ae59c8f7fcff3f9547de274819f54863dfd`，fixed API为public/ungated并声明Apache2.0；代理503失败原件保留，raw-byte relay后已在Pod解析身份通过。当时API广告bin SHA/265124753bytes仅为广告，尚不是实际权重验收。API raw Pod SHA `1d560d7b32429ba9da56ad4f064d2dfb356539e6a90437132ccbde728c0c6018`，pin回执SHA `2c9ce02548b90ee2fc0f7efb2c7c82aad6e574642f0c21448480b0dd7ca558bf`，CPU回执SHA `680239f73977e9db33c40f77ce86ccb617343b31b12bc8815adb79881eb226ab`；只转引，不本机重算。后续A已实证原bin实际SHA `11b2027c5b6f0b94b98a0e67b6d8bf9a4d73f0079921db7e1fe50b849780e844`、265124753 B，转换SHA `c4e529be17e8eada4bdeb7ff34e69739ea743e7ed83ae1541294ed0ea1474f50`，公开分类数K=21841及完整公头strict通过；不将后续实证倒写成早期已验。

A按原合同从该固定revision元数据与实际张量绑定来源身份；`public_source.num_classes`为实证K=21841。本文不补猜本次转交未列的provenance/strict receipt SHA或预处理字段值。不换revision，不沿旧22K→1K provenance/format冒配新源。不并列其他tag、不医学源/N06/FLAIR/teacher/BN或旧训练权重。

源factory及TV完整K类模型先strict/exact/FP32有限加载与映射，再移除公开分类头；不先删head后strict=False。复用`strict_load.target_key`的stem、stage/downsample/block映射，仅gamma允许[C]→[C,1,1]。A已实证344公开tensor（342backbone含finalLN＋2源head）、41个LN、36个GELU及SD、40个conv、0 BN；LN eps1e-6及768末特征沿原固定结构合同。`head.norm`映射`classifier.0`并保留公开LN参数；仅丢`head.fc`两tensor，替换为原binary，再附原age。源大分类头不进入optimizer、比赛训练或推理。

生产仍按R107原TV1000构造顺序建立Small。仅新format分支，在保存/恢复CPU RNG的局部上下文（例如`fork_rng(devices=[])`，不初始化GPU）内临时换成K类public head并完成全源strict加载，然后回原顺序创建binary head/age。不能直接改构造num_classes后任其多耗RNG，导致比赛head也变；旧1000/缺省路径数值与RNG保持。必要toy及实际factory局部证据核binary/age初值和构造后RNG守恒，不另建full控制模型。

新whole初态已由A实证为`515f4e6340cc2c8c07934dbd971977c810f1aa89fee181da6af2d0787464540a`，不是旧50c。生产binary为345state＝342backbone＋2head＋1age，初态空Adam；真实首150成功更新与初始化分别记录，不能称150步后Adam仍空。`initialize=False`只复建binary并strict载入训练checkpoint，不再建立公开大head、读取公开源或访问网络，保持保存与推理身份一致。

## 直接R107保留合同

seed20261029，全参数训练、stride2、原GAP/finalLN/dropout/Linear＋age enabledtrue及原AGEcontract；512原prepared单图、scale[.85,1]、G64/P8、每rank8primary/8view、全局64view/448pairs。原binary-pairwise-logistic六字段为version `binary-pairwise-logistic-v1`、kind `binary_pairwise_logistic`、temperature1、reduction mean、dtype float32、scope global_batch，沿原WORLD可微gather。

SD.1/dropout.2、FP32/noTF32、backbone LR1e-4/head与age各1e-3、原WD/AdamW/globalclip1/EMA、固定9:1、每800评估/max24000/patience4、绝对warmup/EMA1600及原selector不变。`head_warmup_policy=None`、`training_rotation=None`、`training_photometric=None`、`gradient_clip_policy=None`、`attention_pool=None`、`stage3_residual=None`，training_views原缺省。不沿R128预热策略、R127去旋转、R126roleclip、旧q读出、R122去光度或R120sigmoid；不改engine/data/loss/sampler/gather/EMA/训练算法。

原增强是mirror/三photometric/rotate-expand后整图fit画布，不是random-resized-crop；本轮不改像素路径、不新cache/原图扫描/ROI/多视图/辅助阶段。源卡预处理作为来源身份核实，不替换R107候选预处理。

## 最短实施边界与三类必要证据

B仅根任务`483a40da-3ffd-45a3-ad05-a7eea064ecab`划定的三个来源文本文件（新来源工具、strict转换及对应窄fixture），不执行、不改旧来源常量冒新身份。根转交B `c3d9e359`三件freeze后，经根全文文本核及`99432522`实质交A；B文本完成与A后续真实3/3窄验分开。A独占其余production/config/models/experiment91/SCIENCE/正式绑定与必要binding fixture，唯一Pod/Git/GPU。规划仅本页和todo入口，不同人不共写；通过根实质handoff，不直接联系A/B。

1. 新来源固定revision/license/API及实际原blob/转换SHA、完整K头strict映射、normalization/non-tensor验收；小mapping fixture可并入，不追加拒绝矩阵。复用映射不等新source已验。
2. 临时公头RNG隔离、仅源fc丢弃及旧缺省兼容的小synthetic fixture；新binary/age初始化及构造后RNG必须可比，不另造full控制模型。
3. SCIENCE→新source/fresh factory→保存/initializeFalse推理绑定窄fixture。正常formal实际数量、新whole/原binary结构/角色组/空Adam另沿原链，真实八rank初态和首success记录新来源消费，不抄R12820/20。

不重旧source/loss/Gloo/cache/pixel/EMA大全套，不独立GPU预演、额外forward/model/数据probe。失败只由文件所有者窄修未过项，已过不重跑。A已实际完成3/3窄验、18/18正式验收、精确Git/部署、新初态、唯一activation与一次真实launch及首150成功；不抄R128正式数量，不以早期API回执代替后续完整来源验收。这些是已完成的正常验收，不新增用户、临床或文档批准门。

## 执行、预算与停止规则

仅现有A/B/规划，不新Agent/子Agent/资源/lease/monitor/poller/训练定时器或自动队列。Mac只文本/Git/rawtransfer；全部Python/配置解析/hash/model/tests由A工具动态discover既有Pod、核UID/Ready/Python/依赖/effectiveCPU/本人entryTTL后guard执行，不本机或SSH登录节点计算/装依赖。prep合计20%与训练准入owner/rank/loader/control/helper整树80%互斥，不默认核数。CPU23 TTL1793460024.284仅历史，有效才用，过期只同Pod必要future24软件allocation，不改旧allocation/grant/clock或续租。

原6h训练/8h整轮/9hgrant、Small12GiB round cap、启动free20GiB/运行floor16GiB、健康9C+I/reserve/parallelprep及全部warm、best online/EMA、last/previous、initial和R120真实selected/source保护、300s atomic/prune/monitor/failfast保持。新源包含大head的原blob、转换与临时副本按真实字节计存储；模型卡66.3M不是训练binary参数量。去源head后binary同Small，单编码前反向/IO/Adam/EMA/训练checkpoint体量预期不增但须actual；满24k+30eval约3–3.5h仅历史Small估计，无新源吞吐、500MB实际交付或指定硬件100ms保证。不靠早停、降batch/精度/尺寸/步数、扩时或删保护救限。

R128已完整safe且归档，不查询旧轮status/model/GPU。新root/newfuture binding/newclock须A当次动态核锁/设备/存储和整树资源，长准备与传输完成后沿原薄链两处120s实时preflight→一次activation→唯一fresh；根对A的上述派工是唯一授权reference，本文不产生第二授权或新增等候。精确新Git同步/部署与必要验收齐后A直接执行。启动后复用现有observer及唯一heartbeat`1c2c8c3b`，5min轻量completion、公开:00/:30，A在途不追加query，无活跃轮不查旧safe；不再手写round轮询脚本/建timer。

完整工程有效、原selector所选internal5140 AUROC **strict>R107 .9884405731067525**才同seed科学正向，且 **strict>R120 singlebest .9894773092062407**才根另派固定seed20261027对 **R108 .9872328538178018**。只第一门为有限正向endstage/no27；未胜关闭本固定22K-only×R107合同，不source epoch/384源/LR/seed补考。只首seed，无自动27；AP/partial/难组/official/platform不救AUROC或选参。R120仍仅singlebest，不升级方法基线。

完整结束并由原selector冻结唯一checkpoint后，selected内部 **>=.996**才原official5849完整一次；未达正常completed，记`not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，safe另实核。工程启动前无selected不能写threshold_not_met。无平台testCSV/提交；raw`.999`未达持续协调，不保证收益、显著性或可达，不称总体暂停/完成。本页freeze后由根交A随必要源码统一同步，不继续下一评议或候选队列。

## 完整实际结果与收口

规划一次本机文本核原freeze的selected/顶层blob及TRAIN限定字段：唯一EMA c4@3200，internal5140 AUROC `.9884188991813276`、AP `.8233186216195478`、error_twice19236；selected tensor SHA `173a9c1cf719310d502f44c4a9762c55df9796b3ca69bd0cd2c89842e3299671`，checkpoint blob SHA `c816cf8776a2ee1bd75ebcd7c49809c5e28720fd321c67781ff34db16c8a7f52`。TRAIN completed/exit0/6400成功/hard_deadline_met=true/signals[]，未整件打印含token TRAIN。其余attempt6400/skip0/eval8/earlybad4、ROUND/done completed与00:36:08.609实际终态沿根转交A回执。

科学门strict>R107 `.9884405731067525`与实用门strict>R120 `.9894773092062407`均false。固定22K-only×R107合同关闭/no27，不source epoch/384源/LR/seed补考；AP不救。不推出来源普遍无效、同ancestor的1K遗忘因果或统计显著等价。R107/R108比较锚点不变，R120仅singlebest，raw`.999`未达；本阶段关闭不是总体暂停。

运行Git为`04cdb75c170476a7fc80febeb4b4940b74cfa156`，parent `19efc9884d3d5a010e1244888af3e1b2c1f35a0f`，tree仅已转交短值`70e96098`不补猜；658blob+mode精确部署。实际原bin/转换、K=21841/完整head strict、3/3窄验、18/18正式、新whole/345state/初态空Adam、真实首150和完整6400分别记录，不混成同一阶段或借初态声称后续无Adam状态。

工程错误保留：唯一activation exit0后，第一次外wrapper误传`formal-cpu`而exit1，runpy尚未进入真实launcher，当时五项launch marker均absent。根另以`89e468b7-ee93-497b-8238-83fbd746a0a8`明确允许同activated root、正确live round_root/launch_once.py的一次真正调用，实际exit0；没有再次activation/reclock或第二次真实launch。错误与absence原件保留，不改写成所有命令firsttry0。

selected不足`.996`，official正常`not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，未新增官方前向。A同次实际safe=true：owner318913/start348860089现Z只称Z不reaped，8rank318925–318932 absent，锁/设备使用者空、denied0/8busy0，非agent idle推断。当前无活跃训练，不查询旧safe轮。raw为`work/r129-20261031-a/progress-0035/rounds/r129-small22konly-seed20261029-scale085-pairlog-24k-p4-9010-20261031/`下freeze和`evidence/TRAIN_RESULT.json`，不是旧平铺。A统一生成records/R129两个小件并随本页/todo入口一次必要四件归档；本次未获archive Git，不冒填或另造自引用提交。
