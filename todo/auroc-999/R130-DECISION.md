# R130：Small固定0.8逐层学习率衰减

**根已采用，仅一次seed20261029 fresh；当前为必要准备，尚无本轮正式验收、Git冻结、activation、launch或首成功步实证。** A执行授权reference为 `2e1c4016-d1c9-49d7-ab38-c474da99a643`，已ack `56b3c366`。B两文件经 `44065732` / `91153d8d` 文本冻结、根回读后交A，未运行不称PASS。本页是已采用合同，不是额外启动审批门。

**最新用户指令优先：『本轮结束后,先暂停后续训练,给出一个交接文档,在另一个设备进行』。本轮明确为R130。** 只完成本轮必要准备、唯一fresh、原内部训练、按阈值的一次官方评估及实际安全收口。正常结束或不可重试的工程终态后，暂停一切后续fresh、seed27及新方法评议；即使双门通过也只记录复核资格，不派seed27。等待用户在另一设备明确恢复。不关闭当前实例、不新建资源或续租；暂停不等于raw AUROC已达0.999。设备接续见[交接草稿](../DEVICE-HANDOFF-20261101.md)，最终回执待补。

## 唯一身份与对照

| 字段 | 固定值 |
| --- | --- |
| order / config | `40` / `configs/R130-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-layerdecay08-v1` |
| candidate | `r130-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-layerdecay08-9010` |
| round | `r130-small22k-seed20261029-scale085-pairlog-layerdecay08-24k-p4-9010-20261101` |
| readout | `native_avgpool_final_LN_dropout_linear_plus_age` |
| newprep | `technical-prep/r130-20261101-e62a` |

直接对照R107 seed20261029，恢复原公开Small ImageNet22K→1K `fe6b358a` fresh；不是R129的22K-only来源及其LR补考，不加载历史训练checkpoint。原初态whole期望 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`、345tensor、0BN、空Adam，backbone/head/age参数元素分别49454688/769/2，均须本轮正常formal及实际fresh核实，不能将期望冒充已测。

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
- A负责其余production/config/ex91/SCIENCE/正式绑定和第三项：SCIENCE→实际建组identity→小型保存/严格load→下一步LR/EMA/RNG与连续参考，验证完整group身份及param-ID位置映射。规划仅本决定页、todo入口和设备交接草稿，由根实质handoff，彼此不共写。
- 只执行这三项必要窄证据及正常formal，不重source/loss/Gloo/cache/pixel/EMA大全套、拒绝矩阵、控制模型或独立GPU预演。真实失败由所有者窄修未过项；B文本完成不是执行PASS，旧formal数量不得冒填本轮。

正常formal须给实际项目数量、全部通过、新非tensor组身份、native345/50c/空Adam。真实8rank首个正常成功步需证原source/firstbatch、实际组LR已消费、全role Adamstep1；只消费首自然已有日志，不加forward/optimizer probe或边界query。早层小LR可能FP32舍入，不能要求每个参数都有可见非零位移。同初态50c本身不证明LR策略执行。当前上述实际回执均待A。

## 预算、运行与预冻判据

原6h训练/8h整轮/9hgrant、Small12GiB round cap、启动free20GiB/运行floor16GiB、健康9C+I/reserve/parallelprep、全部warm/bestonlineEMA/lastprevious/initial及R120真实selected/source保护、300s atomic/prune/monitor/failfast不变。参数、Adam矩及EMA元素不增加，仍同一完整单编码前后向与IO；更多paramgroup调度非免费，满24k+30eval约3–3.5h只属旧Small估计。精确新耗时、最终500MB交付及指定硬件端到端100ms均未证，不靠早停、降batch/精度/尺寸/步数、扩时或删保护救预算。

A唯一Git/Pod/GPU，所有Python、配置解析、hash、模型与测试只在工具动态discover既有Pod、核UID/Ready/Python/依赖/effectiveCPU/本人entry TTL后的guard运行；Mac仅文本/Git/rawtransfer，不回退本机或SSH登录节点计算、不装依赖。prep合计20%与训练准入owner/rank/loader/control/helper整树80%按实时CPU互斥，不默认120核或固定份额。CPU24仅历史，有效才用；过期仅同Pod必要future软件份额，不改旧allocation/grant/clock或续租。必要Git实际同步、精确部署、资源/身份/锁设备实核及原两处120s薄链通过后，一次activation/唯一fresh直接执行，不等文档批准。不新Agent/资源/monitor/poller/timer，旧safe轮不再查询。

完整工程有效原selector所选internal5140 AUROC strict>R107 `.9884405731067525`才同seed科学正向，且strict>singlebest R120 `.9894773092062407`才取得固定seed27对R108 `.9872328538178018`的复核资格。**本轮之后用户已要求暂停，双门通过也只记资格，不派seed27。** 只过第一门则有限正向/endstage/no27；未胜则关闭此固定0.8映射合同，不factor/分组层数/LR/seed补考。AP/partial/难组/official/platform不救主判。

完整内部结束并由原selector冻结唯一checkpoint后，selected>=.996才原official5849完整一次；低于则正常 `not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，safe另实核。工程未启动无selected时不能写threshold_not_met。无平台test CSV或提交。本轮正常完成或不可重试工程终态后停止后续训练与新评议，待用户另一设备明确恢复；总体raw.999未达不称完成。

原唯一heartbeat `1c2c8c3b`目前仅服务R130，沿5min轻量completion/公开:00和:30规则，A查询在途不追加，无活跃轮不查旧safe。根将在本轮完整收口、实际安全及交接后停用，**实际停用回执待补，现不能记已停用**。本页冻结后不自行轮询或拓下一方案。
