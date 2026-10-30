# R114：保留更密的末阶段空间网格

2026-10-30，用户已明确持续推进后续训练直至目标，目标为 raw AUROC≥0.999，不保证可达。协调采用本合同；A/B 已接精确边界并可并行准备，本文不是额外批准门。规划只写本页，A 唯一集成、Git 写入与 GPU 执行，新 B `e62c5cb0-3516-4798-8b37-ed8da2d958ad` 负责两件窄文本；不恢复旧 B 或新增 Agent。

## Goal 与取舍

检验：在相同512输入、同一公开 fresh Small 和完整微调条件下，减少最后一次空间降采样，能否改善按原策略选出的整体内部 AUROC。直接预登记基线为 **R107、seed20261029**，selected c4 EMA@3200 AUROC `0.9884405731067525`、AP `0.8274326888154083`。不从刚失败的 R113 作单变量对照，不因平台或官方分数改选基线。

R113 完整内部16条已核：6400/8cycles/skip0，c4 ONLINE@3200 AUROC `0.9838264352051979`，未过该基线，freeze4 分支结束、第二 seed 不启动。其负结果不诊断表征遗忘或特定病因。

固定64 fit 原图几何核说明该样本256个局部均不需上采样、线性原生像素保留倍率约2；不是临床信息或全体数据保证。当前放弃全图加四局部的五编码设计：它会增加训练、dev和单图推理的编码次数，完整时限及T4成本不足以辩护。本轮不读取原图、不建缓存，也不把512缓存裁片称高分辨细节。R114只改变已有512特征的后续计算网格。

旧 R20 空间残差、R26内容加权pool、整图768及历史中层/局部实验不是“未试”。本轮没有重复其读出或多模型设计；改变的是末阶段入口的步幅。已读历史未见这一匹配合同，但不能声称由此已确定信息瓶颈。末层更密仍可能只产生冗余相关位置，原GAP仍可能稀释局部信号；公开权重适配的空间尺度和有效感受野也会改变，完全可能退化。

## 唯一配置身份与 Architecture

| 字段 | 固定值 |
| --- | --- |
| SCIENCE | `configs/R114-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-laststride1-v1` |
| candidate | `r114-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-laststride1-9010` |
| order | `24` |
| round | `r114-small22k-seed20261029-scale085-pairlog-laststride1-24k-p4-9010-20261030` |
| 唯一科学 delta | `job.final_downsample_stride = 1`；旧缺省保持2 |

只把实际 `model.network.features[6][1]` 的一个 Conv2d `stride=(2,2)` 改为 `(1,1)`；原384→768通道、2×2 kernel、padding0、dilation1、groups/bias及参数对象保持，后续3个 CNBlock 不变。对512输入，上游32×32进入该层后，按卷积公式预期末阶段网格由16×16变为31×31；**这是几何推导，不是已完成真实 forward 的证据**。仍沿原GAP→final LN→dropout→linear，加原age残差，只有一个骨干、一个输出和一个checkpoint。

[TorchVision 0.25 官方实现](https://docs.pytorch.org/vision/0.25/_modules/torchvision/models/convnext.html)给出了Small的3/3/27/3层及阶段间2×2 stride2结构。这里只用它说明实现与算术估计，不采用其默认SD或另换TorchVision权重；本项目显式SD仍为0.1。不会同时改 dilation/padding 来补感受野，也不称“仅增加采样点而其余函数完全相同”。该变化同时影响空间采样、边界处理的覆盖以及末阶段有效感受野，属于这一预定结构干预的组成。

## 保持的 Spec 与来源身份

从 **R107** 构造科学配置：Small、训练seed20261029、ImageNet22K→1K公开资产 revision `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b`，converted SHA `7aaea5ce6fe6db953874d9792d3f1711a5cc0193a10d740e023108043918c610`、provenance SHA `8ea3e957e2381a40e2f8fc5feae5df31d0d8b33faf3bdd3fa5749e64ae98368b`。不加载R107/R113 checkpoint、N06/FLAIR参数或旧BN/teacher。公开医学预训练规则核查独立，不阻挡此既有非医学来源方向。

- 原512 prepared/input、单视图、原增强及 `training_scale=[0.85,1.0]`、固定9:1/原splitseed与清单、age fit统计及验证像素不变；不引入裁块或新图像来源。
- G64/P8、8rank各8主图/8视图，8P56N/448对，原可微WORLD gather和DDP归一化；原六字段纯pairwise logistic温度1、FP32/mean/global_batch，不混BCE或加新loss。
- 原backbone LR1e-4/head与age1e-3、WD1e-4、dropout0.2、SD0.1、LN1e-6、Adam/clip/EMA保持。所有参数可训；**省略R113的 `freeze_features_before`**，也不注入R50专用head_seed/BN字段。既有EMA在空冻结名单下保持原路径，不重改EMA。
- 每800成功更新完整dev，max(online,EMA) AUROC、strict/min_delta0/patience4/mincycles1、上限24000，warmup和EMA绝对1600不变。train21600/total28800/GPUgrant32400，80%训练与20%准备互斥池按新环境实核，不沿用过期分配或申请新实例。

先按未改的公开1000类结构 strict 加载，再替换原binary/age头，最后应用步幅。原source strict/handoff作为公开原结构证据保持不变；生产步幅另由SCIENCE和实际non-tensor证明绑定。所有state tensor的名称/shape和值预计保持R107 fresh整体 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，由新Git正式原生CPU流程实际确认，不额外control-only。**相同tensor hash不证明stride正确，也不表示初始logit或训练轨迹相同。** 正式CPU、8rank真实初态分别记录实际目标模块kernel/stride/padding/dilation及参数组/空Adam；不能把expected31当已观察shape。

fresh、checkpoint重建、推理均由同一factory消费新字段，`initialize=False`也应用。只保存state_dict不包含stride，因此旧缺省2合同不改，新job身份不得遗漏该字段。不得从public权重重新覆盖训练checkpoint，或因相同50c把两种非tensor结构混同。

## 真实代价与硬边界

只增加最后一次transition及末3个block的空间面积，前3/3/27个block和整个输入链不变；变化部分面积比为31²/16²，约3.75。按卷积/线性乘加的结构估计，整网算术量约增加四分之一，而非五次编码；此估计不包含全部算子、通信、IO或硬件效率，**不是测得的wall/显存**。[官方模型元数据](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.convnext_small.html)描述的是原224模型，也不能直接作为本轮512吞吐回执。

R107 已有训练项约0.425秒/成功步可作同环境数量级参考；在近似算术比例的假设下，24000步训练项约3.5小时，说明在6小时上限内有合理执行空间，不构成耗时承诺，dev/保存/初始化和整轮另受8小时约束。新鲜执行的实际显存与耗时由正常首更新/原日志报告，不另加GPU预演。若OOM、实际硬期限或工程异常导致未完成，保留失败/未完成，不降低batch、精度、周期或自动延长资源救场，不伪称科学阴性。

参数量与单模型数量不增，但实际推理激活和运算会增加。**单T4端到端100ms仍未实证**，当前训练设备结果不能替代T4；内部获胜也不等比赛部署达标。部署约束不在本页自动放宽，不以未知成本自称合规或申请新资源。

## A/B边界与短实施计划

规划仅本页，完成即freeze；A/B不共写文件，旧R113已封归档不混新材料。

- **B独占新增两件：** `runtime/bundle/retrain/spatial_stride.py`、`tests/test_spatial_stride.py`。API `apply_final_downsample_stride(model, job) -> None`，顶层不导Torch；缺省立即无动作，显式只支持本轮Small的整数1（排除bool），确认真实目标是原2×2/stride2/pad0/dilation1 Conv2d后，唯一从job赋值 `stride=(value,value)`。不换模块/参数、不改变mode/RNG/其他结构。B仅文本；实际测试由A统一在新有效准备份额执行。
- **A独占：** 新SCIENCE、`config.py`/`experiment91.py`配置候选绑定、`models.py`factory调用、必要的 `fresh.py`/`single91.py`身份传播和原candidate fixture；既有非tensor证明的真正承载点如 `tools/validation/check_candidate.py`、`runtime/r92_initial_gate.py`、`distributed_engine.py`，只在需要时最小接线。原Gloo/loss/data/sampler/EMA/输入缓存算法不改，不建立通用结构变换平台。

- [ ] B约3项小toy fixture：参数对象/state/mode/RNG保持且实际32输入目标层16→31；真实forward/backward与独立functional stride1值/梯度一致；缺省no-op及严格state重载后通过job重建同stride/输出。无实际Small、预训练、图像、Gloo或拒绝矩阵扩展。
- [ ] A一次新有效CPU池验收B3项与真正受影响配置/非tensor绑定项；公开来源、旧数学、像素与旧全套不重复。失败仅修实际受影响点，不能把文本完成称GREEN。
- [ ] A唯一Git同步实际tree，按实际清单部署，新Git正式原生包确认50c与真实stride1/全部可训/组/空Adam；沿既有安全交接后一次activation/fresh。真实8rank初态与首更新核64主图/64视图、8P56N448对及实际non-tensor字段；不重启、不重置clock。

## 判据、预算与主动接续

本阶段先1fresh，工程有效并按完整内部dev原selector选出的AUROC必须**严格大于R107 `0.9884405731067525`**，才固定相同干预的既定seed27，对照R108 `0.9872328538178018`；第二轮身份本页不提前冻结。AP独立分报、原固定难组仅辅助且不必为它增加本次任务，均不能救整体主指标。不得改选同周期行、换EMA/online或按官方/test选择。

首跑不胜停止laststride1分支，不扫描stride/dilation/分辨率/seed/LR；即使两seed同向，也不证明显著性、机制、普遍收益或消除反复使用同dev的偏差。raw AUROC未达0.999不称总体完成，外部平台分数仍与内部/官方分别解释。

持续推进责任保留：完整结果到齐后，规划主动据实际内部与工程证据决定下一独立问题并交A，不再等待用户说“继续”。若本轮不胜，不把原图局部几何可行偷换成已可承担五路训练；下一问题先据已经完成的来源规则核查和实际新吞吐判断可用来源或输入设计，给具体有限合同而非长期空报。该接续责任不自动批准第三跑、医学权重或新资源；任何具体方向仍须固定匹配基线与单一可证伪干预。

## R114 工程失败与 R114R1 fresh 重试

据协调方与执行 A 的实际终态报告，原 R114 在1426次成功更新/1426次尝试、skip0时被监控终止；仅完成第1次dev评估，online/EMA均为 AUROC `0.9744079105011373`、AP `0.7513749128110692`。15秒周期monitor捕获 `FileNotFoundError` 后发TERM，未达到早停或完整选点终态，也未执行official。A已核设备idle、原锁释放、owner/rank无存活。**本次是工程失败，partial指标不构成stride方向的科学失败，不用于触发第二seed或改选参数。**

协调已授权一次窄修后的新fresh工程attempt **R114R1**，不是恢复训练。现有 `prepare.py --round-id` 与科学profile/candidate分离，run_name也由round_id生成；因此保持 `configs/R114-SCIENCE.json`、原profile、candidate及 `experiment_order=24`，新round固定为 `r114r1-small22k-seed20261029-scale085-pairlog-laststride1-24k-p4-9010-20261030`。用新round/SITE、实际SCIENCE SHA、新Git及DEPLOYMENT共同绑定工程attempt；科学文件若字节未变，其SHA可以相同，无需为制造新SHA而改科学字段或增加新候选注册。

根已文本定位两条待A窄修的并发路径：monitor的临时名规则未覆盖 `CheckpointStore.save` 的 `.checkpoint-*.tmp`，以及合法prune可删除 `checkpoint-32hex.pt`及其sidecar；这不等于已证明本次异常的确切pathname。A仅修真正受影响的文件计量/监控路径并验收相应窄测试，不把所有文件或资源异常一概忽略；规划不修改实现，也不重跑loss/Gloo/像素/来源等无关项。

R114R1从同一公开fresh工厂零步开始，**不加载任何原R114 checkpoint、不继承Adam/EMA游标或1426步进度**；原job、sampler、输入/增强、seed29、stride1、loss/P8、LR/WD、SD/dropout、800/24000/patience4/绝对1600及每attempt的21600/28800/32400秒上限全部保持。新Git正式原生流程实核50c tensor及独立non-tensor stride/全部可训/空Adam，再由当前唯一执行A `e62c5cb0-3516-4798-8b37-ed8da2d958ad` 核实时身份和资源后一次activation/launch。旧终态、原clock/token、冻结包和失败证据原样保留，新attempt有自己的clock/token；不重置或覆盖原attempt。

失败与重试各自记录真实耗时、计算及准备成本，并在本阶段累计，不能把R114R1记作首attempt或免费补跑；本次授权不自动扩为R114R2或改变实例资源。official仅在新attempt完整内部freeze后执行一次。方向主判仍等待R114R1工程有效的完整内部结果，按原selector严格比较R107 `0.9884405731067525`；AP与partial不能替代。SSL仅保留后续文本备选，此时不推进或冻结；本补充不是新的批准门。

## R114R1 完整内部终态与分支收口

2026-10-30 一次文本核对 `work/r114r1-20261030-a/progress-2030/SINGLE-MODEL-FREEZE.json` 与 `TRAIN_RESULT.json`：7200次成功更新/7200次尝试、skip0、9次dev评估、bad4早停，freeze完整18条；原selector唯一选择 **c5 ONLINE@4000，AUROC `0.9875399010946536`、AP `0.8091074656953037`、error_twice20696**。EMA最佳为c6@4800，AUROC `0.9875025737786443`、AP `0.8230870989975856`，不据此改换所选variant。TRAIN组件exit0、hard_deadline_met=true、signals=[]，组件wall `4002.716028690338` 秒；整轮20:30:51已safe由A实际核验报告，规划未读取官方指标。

完整selected AUROC未严格超过预登记R107 `0.9884405731067525`，因此**结束laststride1分支，不启动seed27，不扫描步幅/分辨率**。该结论只否定本次合同的主目标支持，不泛化为空间信息无用。原R114的1426步工程失败与本次fresh分别保留、累计成本，partial不混入主判。总体目标未完成；下一独立问题由已采用的[R115 fit-only自监督续训合同](R115-DECISION.md)接续，不修改本轮冻结产物。本文至此停止过程更新。
