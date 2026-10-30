# R116 文本候选：ConvNeXt Base公开表征配对

**状态：指定分析已完成并合并，科学合同保持，交协调统一接续。** 2026-10-30，用户要求先等分析Agent `70c0b329-6e2f-4357-bd62-c81a37565835` 完成再开始下一轮；该Agent现已正式交付[全历史复盘](EXPERIMENT-REVIEW-20261030.md)，规划已文本读取。报告支持本页唯一Base配对，未改变候选、直接基线或停止条件，分析等待条件已满足。沿用户已有持续推进授权，由协调统一向现有A/B交接，不再增加用户确认门；规划不直接派GPU。报告与本页都不是资源safe证明，R115原once是否结束及新轮实时准入仍由唯一A核实，不据旧快照猜测。

复盘中的原生局部、其他公开家族及配对不确定性分析仅保留为后续方向，不据此增加本轮实验或CPU任务。两预定seed均胜才考虑升级方法基线，不以较好seed掩盖另一seed退化；这仍不等显著性证明。官方和平台数值不参与本页科学选择。

## 唯一问题、推荐与历史边界

推荐比较同家族**更宽的Base及其对应公开ImageNet22K→1K checkpoint**，直接锚定预登记R107 Small/seed20261029，而非刚失败的R115。问题是：保持原输入、损失与优化时钟，更宽网络和相应自然图像公开表征的组合，能否提升整体内部所选AUROC。架构、公开checkpoint、fresh head维度及构造随机消耗成对改变，不能称纯参数量因果；不从平台99.32或官方值挑参数，不使用N06/FLAIR、医学教师或旧训练权重。

本次对仓库 `records/`（含历史EXPERIMENT-CONCLUSIONS）、`models/`、`configs/`、`todo/`、`tools/`、`runtime/` 的md/json/py文本检索未发现普通ConvNeXt Base同路线记录。这是已查范围内未见，不保证未归档的外部历史从未做过。Tiny→Small两既定seed曾一微升一微降，已构成“简单加大模型必然有效”的反证；Base也可能增加方差、耗时却不改善病征排序。

本轮不重复普通R50/BN替换、冻结早层、stride1或1600步SSL，不扫LR/dropout/SD/温度/seed。只推荐Base这一项；五路全图加局部虽然有原生像素几何可能，当前满额训练及端到端部署成本仍更难约束，不在本页增加第二实验。

直接主对照：**R107 c4 EMA@3200，AUROC `0.9884405731067525`，AP `0.8274326888154083`**。R115完整c4 EMA AUROC `0.9863815501913928` 未胜该基线，其具体SSL分支已结束，不能把R116较弱对照改成R115来宣称进步。

## 保留的身份与科学合同

| 字段 | 待接续的固定值 |
| --- | --- |
| SCIENCE（仅A生成） | `configs/R116-SCIENCE.json` |
| profile | `single-convnext-base-in22k-ft-in1k-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-v1` |
| candidate | `r116-convnext-base-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-9010` |
| order | `26` |
| round | `r116-base22k-seed20261029-scale085-pairlog-24k-p4-9010-20261030` |
| 科学变化 | R107 Small及其公开checkpoint → Base及其对应公开checkpoint |
| 源repo/tag | `timm/convnext_base.fb_in22k_ft_in1k`，非`_384`、CLIP或ConvNeXtV2变体 |

其他全部从R107继承：seed20261029、固定9:1/原splitseed与fit/dev清单、原age fit统计、512 prepared/input、ImageNet归一化、原单视图增强/scale `[0.85,1.0]`、原stride2、全参数可训、SD显式0.1/dropout0.2、内外LN eps1e-6、FP32/noTF32、G64/P8/8rank各8/accum1、8P56N448对及默认WORLD可微gather、原纯pairwise六字段温度1、backbone1e-4/head与age1e-3/原WD和norm-bias例外/clip/Adam/EMA。

仍每800成功更新评估完整dev，最多24000，warmup与EMA绝对1600、原max(online,EMA) AUROC strict/min_delta0/patience4/mincycles1。保持train21600/total28800/grant32400秒，不复用SSL阶段，不带`ssl_recipe`或derived资产，不带freeze4/stride1/R50 head_seed/BN字段。不改标签、划分、采样、年龄、选点、官方once或运行中的R115包。

## 来源与资产：已有事实、尚缺实物

[timm官方模型卡](https://huggingface.co/timm/convnext_base.fb_in22k_ft_in1k)明确该tag由作者在ImageNet22K预训练、ImageNet1K微调，Apache2.0，约88.59M参数；是独立非医学公开来源。卡中列出的224训练/288测试预处理并不是本轮512候选预处理，本轮沿R107，不暗改为源卡resize/crop。

本机现有Small来源/转换工具可以复用映射机制，**不等于Base权重已存在或验收通过**。本次未读权重、下载、解析API、计算SHA或加载模型。Base immutable revision、原bin SHA/bytes、转换后SHA、provenance及新fresh整体tensor目前均缺实际回执，不抄Small的fe6 revision/7aae权重或50c初态。

本候选必要来源流程限定为同一个repo/tag：先取得一次公开API metadata并固定其immutable commit，随后所有config/README/bin均从该commit获取，只选一种权重格式（沿现工具的`pytorch_model.bin`），核公开/非gated、许可与ImageNet来源、实际bytes和LFS SHA；不边训练边追main。实际pin写入SCIENCE并在唯一正式Git前冻结，不把空值视作来源验收完成。不得因失败换成医学源、其他tag或另一recipe。

该来源增加新的合规公开自然图像预训练资产，不增加本项目fit/dev以外的训练图像或医学数据。来源类别、论文指标、许可证声明与当前Pod实际内容是不同证据，后者仍必须由A/B在其有效准备份额实际完成。

## 结构与最小真实接线

[TorchVision 0.25原始实现](https://docs.pytorch.org/vision/0.25/_modules/torchvision/models/convnext.html)给出Base四阶段宽度128/256/512/1024、深度3/3/27/3；与Small同深度、通道增宽。仍原生GAP→finalLN→Dropout0.2→新Linear(1024,1)+原2参数age，只一个骨干/输出/checkpoint。构造时必须传SD0.1，不采用库的Base默认0.5。预期同36个SD、41个LN、0BN/0buffer及full1000头 `[1000,1024]`，均须以新源实际CPU strict/结构结果确认，不能把预计写成已测。

当前代码不是只改一个model字符串就能执行：`config.MODELS`、`models.create_model`只列Tiny/Small，公开22K provenance分支也只显式支持Small；`experiment91`候选/来源绑定仍写Small/768。与此同时，已有 `strict_load.target_key(..., depths=(3,3,27,3))` 的名称映射不依赖通道数，`check_candidate`参数组及归一化证明多处已消费SCIENCE结构，可最小扩Base显式分支，不重写网络、采样或loss。

由协调统一向现有执行者派工，规划不直接打断A；最小文件边界如下：

- **B来源窄项**：`tools/pretraining/base_source.py`、`base_strict.py` 及一件小mapping fixture；复用现成source_chain/target_key/check_tensor，固定实际Base repo revision并严格映射full1000逐键FP32 finite/exact、missing/unexpected空、原non-tensor配置。默认Tiny/Small映射不改，不下载两种权重或重验Small源。
- **A生产集成**：R116 SCIENCE、`config.py`、`models.py`、`experiment91.py`及原candidate fixture；fresh/checkpoint/inference和原初态门仅在真正需要处消费同Base结构/来源。factory先完整public1000 strict，再按原native初始化替换binary头、零age；`initialize=False`重建Base并加载训练state，不覆盖回public。loss/gather/EMA/data/sampler/通信用已有已验实现，SSL分支本轮缺省不进入。
- 必要CPU证据为新来源严格转换与结构、新Base生产fresh实际初态/完整参数组/空Adam；原生正式包自然包含推理roundtrip。新head为1024维，整体hash必须实际生成，绝不能沿50c或加载Small/R107/R115 checkpoint。仅新mapping与受影响配置项，不扩旧全套、Gloo/像素/loss复验，不加独立GPU预演。

## 满额预算可行性与部署风险

来源卡同表给出Base约15.38 GMAC、Small约8.71 GMAC（224模型），比约1.77；同深度、同512网格且通道由96系列到128系列，主要点运算按宽度平方增加，约1.77倍可作为算术数量级。**这是结构估算，不是Base在当前硬件的训练或推理benchmark**，也不直接移用卡中的AMP/RTX3090吞吐。

当前已有R115监督阶段仍是原stride2 Small：6400步training约2696.797秒，8次eval约151.650秒，27次checkpoint约146.232秒，监督进程wall约3066.229秒。SSL另计，本轮不加入SSL。将监督wall按完整24000步/30评估周期外推，再用约1.8的宽度成本倍率作预算，约**5.75小时**；该量已包含按比例估计的dev/保存/其他监督开销，不是只拿训练步耗时，也不依赖p4提前停来满足满额24k。

结论是**满额有窄余量的可执行候选，非保证完成**：初始化、保存增长、通信/算子效率可能耗尽余量；若wall实际接近Small的2倍，满额可能超过6小时。8小时整轮还须容纳freeze及官方一次，9小时grant不等于可把训练延长到9小时。实际日志须分别报告train/eval/save及剩余时限；遇OOM、硬期限或工程失败如实记未完成，不私改batch、精度、步数、周期或资源救场，不用partial选点当科学失败。不增加独立GPUbenchmark或新实例。

约88.6M×4字节的full1000 FP32 tensor约354MB，换binary头后略小，因此**单份最终选中模型**在500MB内有合理空间。实际部署还要计config、age统计、代码要求的其他配套解压体积；训练optimizer和online+EMA多份训练checkpoint不能被当成合规推理bundle一起加载，最终只带唯一selected模型。实际包体仍由必要交付回执核，不把估算写成已过限制。

**未来R116工程存储合同：**协调已批准仅本轮round软件存储上限为**16GiB（17179869184字节）**，由A通过最小集中版本化配置及未来绑定消费；未指定的旧合同仍保持12GiB，R115及历史冻结包不改。健康保存/prune路径的瞬时并存上界按`9C+I`加其他round文件计（`C`为含model/Adam/EMA的完整训练checkpoint，`I`为独立空Adam初态），具体体积由A按实际Base参数和保存格式核实；prune失败可能留下额外文件，不能将该估算当无条件硬上界。启动前A须实核当前实例磁盘能覆盖本轮真实峰值、运行中至少16GiB空闲floor及并行准备/新来源资产余量；原启动20GiB空闲要求也保留，单独满足20GiB不代表已覆盖全程增长。此处不声称实盘已通过，不新增物理磁盘、云资源或续租；所有快照/best/last/previous/initial保护、300秒保存、atomic/prune/fail-fast及6h/8h/9h均不变。这是训练存储工程合同，与最终单模型500MB限制分别核实，不增加文档审批门。

Base单视图激活随通道增长、参数/Adam/EMA随参数量增长；历史Small双视图曾运行不等于本轮显存通过，实际峰值以正常fresh为准。指定硬件端到端100ms仍**未知**，更宽模型可能不满足；训练设备吞吐不能替代该资格。不得凭未知推理速度宣称部署合规，也不为本轮自动增加TTA/量化/第二模型或放宽比赛时限。

## 接续条件、有限预算与失败责任

指定分析现已完成并合并；来源和代码准备可沿既有授权接续，不以本页或分析报告替代实际回执。A须在R115原official自然完成且实际safe后，按新Git/源清单、实时环境/80%与20%互斥池、正式原生包与新初态，唯一fresh启动；不热改旧allocation/clock/token或重做官方。

阶段先1fresh。只有工程有效、完整内部按原selector所选AUROC **严格大于R107 `0.9884405731067525`**，才另行冻结同一Base干预的既定seed20261027，对照R108 `0.9872328538178018`。第二轮身份本页不提前注册，不选最好seed；AP单报、固定难组只作描述，不用官方/test/AP救主指标。正结果也不证明容量机制、普遍收益或消除固定dev自适应偏差。

首跑不胜关闭此Base+公开checkpoint组合，不自动升级Large、换384权重、调LR/SD或继续搜seed；若来源/全预算工程失败，区分未完成与科学阴性。分析等待解除后的每个完整结果均由规划主动形成下一独立问题决策并交协调，不长期空报；本页不预批第三方向。raw AUROC未达0.999，整体目标未完成。

## 最终内部结论（2026-10-31）

已文本核对本机 `work/r116-20261030-a/progress-0130/SINGLE-MODEL-FREEZE.json` 与 `TRAIN_RESULT.json`：6400 successful=attempted、skip0、8次完整dev/16条、bad4正常early_stopping；唯一selected为EMA c4@3200，AUROC **0.987677169289011**、AP **0.8222910002960596**、error_twice **20468**。TRAIN completed/exit0、hard_deadline_met=true、signals为空。这是完整内部结果，不是partial；未严格胜R107 **0.9884405731067525**，关闭此Base组合，不做seed27或相邻扫描，不泛化为更大模型/空间信息无效。

原件记录training 3471.292秒、evaluation 234.987秒、checkpoint 228.560秒/27次，训练组件wall 4079.423秒；不将其早停耗时当满24k成本。协调01:31:21北京时间的资源消息为原once official仍pending、ROUND/done未出、owner仍持锁/device有用户，safe=false；本页没有读取官方指标或追加资源查询，不推断随后状态。整体目标尚未达到，下一独立问题另由协调采用，原官方自然运行和实际安全交接继续由A负责。
