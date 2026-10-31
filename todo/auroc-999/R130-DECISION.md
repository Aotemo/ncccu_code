# R130：Small固定0.8逐层学习率衰减

**最终停止状态：用户已将指令改为『直接暂停本轮训练,等我迁移设备后继续』，覆盖此前“本轮结束后暂停”。R130已按用户要求中断并实际安全停止，当前无活跃训练。** 2026-11-01 01:43:49.896北京时间停止观测为 `interrupted/owner_stop_requested`，success=attempt1503/skip0/eval1。没有完整训练结果或最终selector，不判双门false、不作科学阴性或earlystop结论。所有后续fresh、seed27及新方法评议暂停，须用户迁移后明确恢复；当前实例未关闭、资源未续租，raw.999未达不是总体完成。

本页保留原R130冻结科学合同和实际工程过程，停止实证由根已文本核读的A `STOP-OBSERVATION-R130-1` 转交，规划未另查Pod、解析、hash或加载模型。正式20/20、3/3窄验、唯一activation/launch及真实首成功均已实证，不能继续记为准备中。设备接续见[最终停止交接](../DEVICE-HANDOFF-20261101.md)；恢复前须先核原checkpoint/schema/deadline及启动约束，不能承诺迁移后直接resume，也不能重置原clock/token或再次activation/launch。

## 用户中断、安全与保存实证

一次已核身份的supervisor SIGTERM于01:41:51.393发送，发送前1500成功/eval1；01:43:49.896停止观测为1503成功/attempt相同/0skip/eval1。owner326810/start349511910现Z，仅称Z非reaped；supervisor absent、8rank326822–326829 absent、locks[]/device_users[]/scanDenied0/8busy0，actualsafe=true。`train-component timeout/exit130` 与 `done.failed/component not completed` 是原controller对用户中断的既有分类，不改原件，也不据此判模型失败。TRAIN_RESULT、ROUND_RESULT、freeze及official均未产生，无最终selected；官方未运行的原因是用户中断，不能写 `threshold_not_met`。

last已提交checkpoint为 `checkpoint-806c3832c05e487d88c938245ac1d3e2.pt`；index广告SHA `08ac68114696b7bd698fb8e70658f7cfbeecf994088a023a115b7aa0591eb377`、792176842 B，文件存在、sidecar与index一致。根转交A RESULT确认提交游标1503/eval1/0skip，previous/best/initial均存在，protected含best_ema/best_online/snapshot-800。未重hash或load大blob；interrupted记录在save成功后写入，冻结payload合同含model/optimizer/EMA/distributed_rng/successful+attempted/cycle。这里只证保存合同和提交元数据，不冒称逐字段reload验证。精确exp-v3提交index路径见[交接定稿](../DEVICE-HANDOFF-20261101.md)及[A结果README](../../records/R130/README.md)。全部R130续接依赖及R120/KEEP-MODELS保留，不清理。

底层 `run_distributed` 有resume能力，但要求原 `absolute_deadline_at` 完全一致；原one-shot owner/clock与grant没有暂停。用户恢复指令也不等于允许重置时钟、token或lease。迁移后先文本核checkpoint、保存schema和原deadline/启动约束，再由明确授权决定合法续接；当前不resume、relaunch或再次activation。此处不复制具体private clock/grant epoch或内容。

停止小件目录为工作区 `../work/r130-20261101-a/stop/`（3件）；STOP SHA `104e04e8bd725b2c54e2710ad9201e988f25ed0c5416dcb42638c2eb20aaf7bd`，CPU SHA `0a67b442051396ddb066561a30635072c01e1e716a989469addbb929c8b312e3`，均仅转引原回执。根已实际 `delete_heartbeat` 删除原唯一heartbeat `1c2c8c3b`，返回 `{success:true}`；不再定时查询，不创建替代heartbeat/poller。本次停止归档包含 `records/R130/{R130-RESULT.json,README.md}` 与本三页；提交由Git历史定位，最终commit由协调方回执。

## 唯一身份与对照

| 字段 | 固定值 |
| --- | --- |
| order / config | `40` / `configs/R130-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-layerdecay08-v1` |
| candidate | `r130-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-layerdecay08-9010` |
| round | `r130-small22k-seed20261029-scale085-pairlog-layerdecay08-24k-p4-9010-20261101` |
| readout | `native_avgpool_final_LN_dropout_linear_plus_age` |
| newprep | `technical-prep/r130-20261101-e62a` |

原直接对照R107 seed20261029，使用公开Small ImageNet22K→1K `fe6b358a` fresh；不是R129的22K-only来源及其LR补考，未加载历史训练checkpoint。本轮正式已实证native whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`、345tensor、0BN、初态空Adam，backbone/head/age参数元素49454688/769/2。初态Adam0只指初始化，不是1503步后Adam仍空。

唯一科学变化是按深度缩放backbone学习率；所有参数从成功步1开始active、requires_grad=true、Adam step=s。`head_warmup_policy`、`training_rotation`、`training_photometric`、`gradient_clip_policy`、`attention_pool`、`stage3_residual`均为None，training_views保持原缺省。原GAP/finalLN/Dropout/Linear+age、age enabled及AGEcontract、fulltrain/stride2、prepared512单图、scale[.85,1]、G64/P8、每rank8primary/全局64view/448pairs、原WORLD可微gather全部不变。原loss六字段为 `binary-pairwise-logistic-v1` / `binary_pairwise_logistic` / temperature1 / reduction mean / dtype float32 / scope global_batch；SD.1、drop.2、FP32/noTF32、base backbone LR1e-4、head/age LR1e-3、原WD/AdamW/一次globalclip1/EMA、固定9:1、800/max24000/patience4、绝对warmup/EMA1600及原selector均不变。

## 固定策略与实际名称

`job.layerwise_lr_policy`与`candidate.layerwise_lr_policy`精确同为 `LAYERWISE_LR_RULE`：

```json
{
  "version": "convnext-small-layerwise-lr-decay08-v1",
  "kind": "optimizer_group_layerwise_lr_decay",
  "mapping": "facebook-convnext-small-group12-tv-v1",
  "layer_decay": 0.8,
  "num_layers": 12,
  "backbone_scale": "layer_decay**(13-layer_id)",
  "head_scale": 1.0,
  "age_scale": 1.0,
  "schedule": "original_global_successful_cursor",
  "weight_decay": "original_per_parameter_rule",
  "parameter_membership": "unique_trainable_optimizer_parameters",
  "precision": "fp32"
}
```

缺省None严格保留旧数值、RNG及分组路径，不建通用逐层策略平台或factor扫描。实际封装名称保留 `network.` 前缀，映射如下：

| 实际模块 | layer_id |
| --- | --- |
| `network.features.0` stem | 0 |
| `network.features.1`三个block | 1 |
| `network.features.2`下采样、`network.features.3`三个block | 2 |
| `network.features.4`下采样 | 3 |
| `network.features.5`第j个block，j=0..26 | `3+j//3` |
| `network.features.6`下采样、`network.features.7`三个block | 12 |
| `network.classifier.0` finalLN | 13 |

原head实际为Dropout之后的 `network.classifier.3`，不是classifier.2；`age_weight`在顶层。沿原role识别head，finalLN仍是backbone。backbone组的scaled initial_lr为 `1e-4 * 0.8**(13-layer_id)`，head/age仍各1e-3。原 `_apply_lr` 继续以scaled initial_lr乘原global successful cursor factor，不能再乘一次层倍率。stem倍率大于0，末feature stage倍率.8，finalLN倍率1，不冻结任何层。

在原 `(head,decay)` bucket内按layer_id确定性细分，保留组内原named_parameters顺序；全部可训optimizer参数恰归一组，不set重排、重复或漏项。WD判据严格沿原 `parameter.ndim > 1` 且非bias；TV重塑为多维的layer_scale原本走decay，不能照作者1D gamma改变其WD。finalLN仍backbone/no_decay，age原WD不变。所有梯度继续参与同一次globalclip1；低LR也减少AdamW实际衰减幅度，这是固定深度倍率整体效应，不称纯feature保护、等平均LR或已证浅层需要保护。

保存与严格加载绑定新group metadata、完整参数名与顺序、layer/scale/base_initial_lr/scaled initial_lr/WD，以及state param-ID的位置对应。不能只凭组数或PyTorch按位置load推断身份正确；恢复后不能按role把倍率覆盖掉。普通resume保留loaded group及initial_lr，旧stage分支的role重写不在本轮扩展范围。不改stage.py/旧单卡age；本轮仅fresh及小型保存恢复兼容fixture，不授权实际resume/relaunch。

## 信息价值与反例

本问题检验同一公开表征、同一Small训练合同下，全层持续更新但按深度缩放更新幅度的净效用。原作者[optim_factory.py](https://raw.githubusercontent.com/facebookresearch/ConvNeXt/main/optim_factory.py)给出Small [3,3,27,3]的分组，[main.py](https://raw.githubusercontent.com/facebookresearch/ConvNeXt/main/main.py)使用12层分组和 `decay**(13-i)`，[engine.py](https://raw.githubusercontent.com/facebookresearch/ConvNeXt/main/engine.py)消费全局schedule×组倍率；[TRAINING.md](https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md)的22K→1K示例使用0.8。只借其深度倍率，不复制head_init_scale、WD、LR、AMP、mixup或epoch。示例含Base等，不是当前Small医学pairlog收益证据，也不证明当前feature distortion、临床瓶颈或0.999可达。

定向历史未见此精确合同，不等于全历史从未做过。R94是非当前Tiny/BCE合同，不能泛称统一降LR皆阴性；R96将R95的backbone LR1e-4降至5e-5，AUROC .983341180097075低于R95 .9866645153288838。R113永久冻结早层 .9838264352051979、R128前1600步仅head/age .9882093845688875亦未胜R107。这些是必须保留的非等价负先验；深度缩放可能妨碍浅层必要适应，不能因机制不同就忽略反例。收益未知，不由“未试过”保证正向。

## 独占边界与三项必要窄证据

- B仅 `runtime/bundle/retrain/layerwise_lr.py`、`tests/test_layerwise_lr.py`：第一项核名称映射、唯一全覆盖、原WD、None exact及无tensor/RNG副作用；第二项用真实AdamW小toy对独立逐参数参考，检查scaled initial_lr×原cursor在warmup/cosine边界的参数、矩、step及WD。
- A负责其余production/config/ex91/SCIENCE/正式绑定和第三项：SCIENCE→实际建组identity→小型保存/严格load→下一步LR/EMA/RNG与连续参考，验证完整group身份及param-ID位置映射。规划仅本决定页、todo入口和设备交接文档，由根实质handoff，彼此不共写。
- 本轮只执行上述三项必要窄证据及正常formal，没有把B文本freeze冒作执行结果，不重source/loss/Gloo/cache/pixel/EMA大全套、拒绝矩阵、控制模型或独立GPU预演。实际结果分阶段列于下段，暂停后不再追加测试。

实际3/3窄验PASS：B两项CPU SHA `d9c1fb61db1321b7fd720e20cc4c44344b1fdd32a692133b671c13b82cba763b`；A新增savedAdam-ID子集检查后仅第三项重测PASS，CPU SHA `8bddc5b7c1a8ccc7649a3c3294213370150819dc87cdf67e4c3178a3713f390c`，不是先前训练失败。正常formal20/20、30个新layerwise组identity通过，formal CPU SHA `d050e3d2d0402c11a16938f045478fa969c3186ef6f75a7283c7df841dc9583a`。

运行Git/main `c8c610dd5ee42e9de99473a83c15ae14b8189178`，parent `5f5072ca2c7b05fa5a9a7a15a85c0c053dee8623`，tree `d05cec3880fc327d716789950f5451969cc52669`；12path/12blob remote exact，实际部署666文件，deployment SHA `f3880f28688535ff5b24efed5a8a3995a1bbc1e7c2bf1cf11867a919f2fe63f2`。这些是冻结运行源码与部署，区别于本次停止归档提交。

本轮原A执行授权reference `2e1c4016-d1c9-49d7-ab38-c474da99a643`；唯一activation exit0及正确live launch exit0。START2实际01:31:26.116北京时间为125success=attempt/0skip/eval0、8rank start匹配；真实首成功WORLD finite，30组actual LR按scaled initial_lr×原cursor消费、345tensor Adamstep1。START2 SHA `6c963b1296a6fb15efdd98cddd4428b026c53cb747b9d9b6ab23f59261ce32e7`，CPU SHA `4d53581be17893705ce34b4d1f51a93926b6347a6f49d1d86f7a80e8e170e997`。这是实际自然首步消费，非仅同初态50c推断；小LR可能FP32舍入，不要求每个参数可见非零位移，不追加forward/optimizer probe。初态、首步、START2快照与中断1503步分列，不互相替代。

## 预算、运行与预冻判据

原6h训练/8h整轮/9hgrant、Small12GiB round cap、启动free20GiB/运行floor16GiB、健康9C+I/reserve/parallelprep、全部warm/bestonlineEMA/lastprevious/initial及R120真实selected/source保护、300s atomic/prune/monitor/failfast不变。参数、Adam矩及EMA元素不增加，仍同一完整单编码前后向与IO；更多paramgroup调度非免费，满24k+30eval约3–3.5h只属旧Small估计。精确新耗时、最终500MB交付及指定硬件端到端100ms均未证，不靠早停、降batch/精度/尺寸/步数、扩时或删保护救预算。

A唯一Git/Pod/GPU；所有Python、配置解析、hash、模型与测试原只在工具动态discover既有Pod、核UID/Ready/Python/依赖/effectiveCPU/本人entry TTL后的guard运行，Mac仅文本/Git/rawtransfer，不回退本机或SSH登录节点计算、不装依赖。prep合计20%与训练准入owner/rank/loader/control/helper整树80%按实时CPU互斥，不默认120核或固定份额。CPU24及所有旧allocation仅历史，迁移不得默认有效。原Git实际同步、精确部署、资源/身份准入及120s薄链已用于本轮唯一启动；现在用户即时暂停，不再以原启动授权派发，不改旧allocation/grant/clock或续租，不新Agent/资源/monitor/poller/timer，不重复查询已safe轮。

原预冻科学门仍是完整工程有效原selector所选internal5140 AUROC strict>R107 `.9884405731067525`，实用门为同时strict>singlebest R120 `.9894773092062407`，固定seed27比较锚点R108 `.9872328538178018`不动。**此次用户中断没有最终selector，双门均未作判定，不记false，不产生seed27资格。** 原不factor/分组层数/LR/seed补考及AP/partial/难组/official/platform不救主判的限制保留；没有本轮科学阴性结论。

原完整内部结束并冻结唯一checkpoint后selected>=.996才允许官方5849一次的政策不变，但本轮未到完整freeze、未生成official，不能套用threshold_not_met，也不新增官方或平台test CSV/提交。当前已即时暂停并actualsafe，无活跃训练；后续fresh、seed27、新方法评议全部停止，待用户迁移后明确恢复，并先判断原保存与绝对deadline/启动合同能否合法续接。总体raw.999未达，不能称总体完成。

原唯一heartbeat `1c2c8c3b`已由根实际删除，返回 `{success:true}`；不再定时查询或建立新heartbeat/poller。本页最终停止记录freeze后不自行轮询或拓下一方案，不改变冻结运行源码、原clock/token或任何原始失败/停止记录。
