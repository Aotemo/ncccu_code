# R129：同Small架构的ImageNet22K-only公开初始化对照

根已正式采用本轮，仅授权一次seed20261029 fresh。根转交A已实证唯一source revision及fixed API身份，B三个来源文本文件已freeze并交A；权重实际字节/完整strict、正式CPU、Git/部署、activation/launch、八rank初态或首成功尚无回执。公开分类数K、preprocessing、实际原blob/转换/provenance/strict receipt SHA及新whole初态仍待A在既有Pod正常实证，不能填1000、22000或50c，也不能将API广告SHA冒充实际权重。本文不是启动批准门，必要准入齐后A直接执行；不注册第二候选或自动seed27。

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

根转交A `85adad07`实证：唯一revision为上述`4cde7ae59c8f7fcff3f9547de274819f54863dfd`，fixed API为public/ungated并声明Apache2.0；代理503失败原件保留，raw-byte relay后已在Pod解析身份通过。API广告bin SHA `11b2027c5b6f0b94b98a0e67b6d8bf9a4d73f0079921db7e1fe50b849780e844`、265124753bytes仅为广告，不是actual weight验收。API raw Pod SHA `1d560d7b32429ba9da56ad4f064d2dfb356539e6a90437132ccbde728c0c6018`，pin回执SHA `2c9ce02548b90ee2fc0f7efb2c7c82aad6e574642f0c21448480b0dd7ca558bf`，CPU回执SHA `680239f73977e9db33c40f77ce86ccb617343b31b12bc8815adb79881eb226ab`；只转引，不本机重算。K/preprocessing/actual bin/full strict/newwhole均仍未证。

A后续只从该固定revision元数据与实际张量绑定原blob/转换/provenance/strict receipt SHA、精确K与normalization；`public_source.num_classes`显式绑定实证K。不换revision，不把广告SHA当实际bytes，或沿旧22K→1K provenance/format冒配新源。不并列其他tag、不医学源/N06/FLAIR/teacher/BN或旧训练权重。

源factory及TV完整K类模型先strict/exact/FP32有限加载与映射，再移除公开分类头；不先删head后strict=False。复用`strict_load.target_key`的stem、stage/downsample/block映射，仅gamma允许[C]→[C,1,1]。Small结构期望为344公开tensor（342backbone含finalLN＋2源head）、41个LN eps1e-6、36个GELU及SD、768末特征、无BN，均待新源actual核。`head.norm`映射`classifier.0`并保留公开LN参数；仅丢`head.fc`两tensor，替换为原binary，再附原age。源大分类头不进入optimizer、比赛训练或推理。

生产仍按R107原TV1000构造顺序建立Small。仅新format分支，在保存/恢复CPU RNG的局部上下文（例如`fork_rng(devices=[])`，不初始化GPU）内临时换成K类public head并完成全源strict加载，然后回原顺序创建binary head/age。不能直接改构造num_classes后任其多耗RNG，导致比赛head也变；旧1000/缺省路径数值与RNG保持。必要toy及实际factory局部证据核binary/age初值和构造后RNG守恒，不另建full控制模型。

新backbone/finalLN与whole初态由A实算，不能沿50c。生产binary预期仍345state＝342backbone＋2head＋1age，角色成员与空Adam通过正常formal和真实八rankfresh核实；相同结构不能代替新来源消费实证。`initialize=False`只复建binary并strict载入训练checkpoint，不再建立公开大head、读取公开源或访问网络，保持保存与推理身份一致。

## 直接R107保留合同

seed20261029，全参数训练、stride2、原GAP/finalLN/dropout/Linear＋age enabledtrue及原AGEcontract；512原prepared单图、scale[.85,1]、G64/P8、每rank8primary/8view、全局64view/448pairs。原binary-pairwise-logistic六字段为version `binary-pairwise-logistic-v1`、kind `binary_pairwise_logistic`、temperature1、reduction mean、dtype float32、scope global_batch，沿原WORLD可微gather。

SD.1/dropout.2、FP32/noTF32、backbone LR1e-4/head与age各1e-3、原WD/AdamW/globalclip1/EMA、固定9:1、每800评估/max24000/patience4、绝对warmup/EMA1600及原selector不变。`head_warmup_policy=None`、`training_rotation=None`、`training_photometric=None`、`gradient_clip_policy=None`、`attention_pool=None`、`stage3_residual=None`，training_views原缺省。不沿R128预热策略、R127去旋转、R126roleclip、旧q读出、R122去光度或R120sigmoid；不改engine/data/loss/sampler/gather/EMA/训练算法。

原增强是mirror/三photometric/rotate-expand后整图fit画布，不是random-resized-crop；本轮不改像素路径、不新cache/原图扫描/ROI/多视图/辅助阶段。源卡预处理作为来源身份核实，不替换R107候选预处理。

## 最短实施边界与三类必要证据

B仅根任务`483a40da-3ffd-45a3-ad05-a7eea064ecab`划定的三个来源文本文件（新来源工具、strict转换及对应窄fixture），不执行、不改旧来源常量冒新身份。根转交B `c3d9e359`三件已freeze，经根全文文本核后由`99432522`实质交A；未执行、不称PASS，不再等待B交稿。A独占其余production/config/models/experiment91/SCIENCE/正式绑定与必要binding fixture，唯一Pod/Git/GPU。规划仅本页和todo入口，不同人不共写；通过根实质handoff，不直接联系A/B。

1. 新来源固定revision/license/API及实际原blob/转换SHA、完整K头strict映射、normalization/non-tensor验收；小mapping fixture可并入，不追加拒绝矩阵。复用映射不等新source已验。
2. 临时公头RNG隔离、仅源fc丢弃及旧缺省兼容的小synthetic fixture；新binary/age初始化及构造后RNG必须可比，不另造full控制模型。
3. SCIENCE→新source/fresh factory→保存/initializeFalse推理绑定窄fixture。正常formal实际数量、新whole/原binary结构/角色组/空Adam另沿原链，真实八rank初态和首success记录新来源消费，不抄R12820/20。

不重旧source/loss/Gloo/cache/pixel/EMA大全套，不独立GPU预演、额外forward/model/数据probe。失败只由文件所有者窄修未过项，已过不重跑。已核API/revision不冒充实际权重；K/preprocessing/实际source/新whole、三类窄证据、正式/Git/部署/activation/八rank/首success仍待A实际回执。这些属于正常必要验收，不是用户、临床或文档批准门。

## 执行、预算与停止规则

仅现有A/B/规划，不新Agent/子Agent/资源/lease/monitor/poller/训练定时器或自动队列。Mac只文本/Git/rawtransfer；全部Python/配置解析/hash/model/tests由A工具动态discover既有Pod、核UID/Ready/Python/依赖/effectiveCPU/本人entryTTL后guard执行，不本机或SSH登录节点计算/装依赖。prep合计20%与训练准入owner/rank/loader/control/helper整树80%互斥，不默认核数。CPU23 TTL1793460024.284仅历史，有效才用，过期只同Pod必要future24软件allocation，不改旧allocation/grant/clock或续租。

原6h训练/8h整轮/9hgrant、Small12GiB round cap、启动free20GiB/运行floor16GiB、健康9C+I/reserve/parallelprep及全部warm、best online/EMA、last/previous、initial和R120真实selected/source保护、300s atomic/prune/monitor/failfast保持。新源包含大head的原blob、转换与临时副本按真实字节计存储；模型卡66.3M不是训练binary参数量。去源head后binary同Small，单编码前反向/IO/Adam/EMA/训练checkpoint体量预期不增但须actual；满24k+30eval约3–3.5h仅历史Small估计，无新源吞吐、500MB实际交付或指定硬件100ms保证。不靠早停、降batch/精度/尺寸/步数、扩时或删保护救限。

R128已完整safe且归档，不查询旧轮status/model/GPU。新root/newfuture binding/newclock须A当次动态核锁/设备/存储和整树资源，长准备与传输完成后沿原薄链两处120s实时preflight→一次activation→唯一fresh；根对A的上述派工是唯一授权reference，本文不产生第二授权或新增等候。精确新Git同步/部署与必要验收齐后A直接执行。启动后复用现有observer及唯一heartbeat`1c2c8c3b`，5min轻量completion、公开:00/:30，A在途不追加query，无活跃轮不查旧safe；不再手写round轮询脚本/建timer。

完整工程有效、原selector所选internal5140 AUROC **strict>R107 .9884405731067525**才同seed科学正向，且 **strict>R120 singlebest .9894773092062407**才根另派固定seed20261027对 **R108 .9872328538178018**。只第一门为有限正向endstage/no27；未胜关闭本固定22K-only×R107合同，不source epoch/384源/LR/seed补考。只首seed，无自动27；AP/partial/难组/official/platform不救AUROC或选参。R120仍仅singlebest，不升级方法基线。

完整结束并由原selector冻结唯一checkpoint后，selected内部 **>=.996**才原official5849完整一次；未达正常completed，记`not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，safe另实核。工程启动前无selected不能写threshold_not_met。无平台testCSV/提交；raw`.999`未达持续协调，不保证收益、显著性或可达，不称总体暂停/完成。本页freeze后由根交A随必要源码统一同步，不继续下一评议或候选队列。
