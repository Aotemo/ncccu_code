# R117决定：Swin-B公开表征配对

**状态：根已采用，用户已明确评估协议，口径等待已解除；现有A/B完成必要准备后按原链唯一fresh。** 2026-10-31。实际来源、CPU/formal、前驱safe及新Git绑定仍以回执为准，不称已注册或已启动。唯一身份/order27、科学合同及严格胜才seed27规则不变。R116完整内部未胜已收口，原official/safe仍以A实际为准；本页不推断其已退出，也不建立第二候选。

## 两方向取舍

**推荐Swin-B及对应ImageNet22K→1K公开checkpoint。** 检验窗口注意力、跨窗口信息交互及其公开表征组合，在相同512输入与R107监督合同下是否改善整体排序。它与ConvNeXt加宽不同，但改变架构、公开权重、归一化位置及fresh头构造轨迹，不能称纯attention或容量因果。当前限定历史/源码文本未见Swin完成记录，不保证未归档历史全无。旧DINOv2-S14单模内部约0.979925、Tiny→Small两seed混合、R116阴性均是反例；换Transformer没有必胜证据。

另一方向是Small共享骨干的**全图512+一个原生局部512**，图像级聚合后只输出一次，最多两次编码而非恢复五路。固定64fit几何原件已证明2×2区域长边512–2736、全部无需上采样、约2倍线性原生保留；它没证明哪个单局部包含病征，也没验证全训练集原生局部缓存。固定中心容易漏偏心内容；随机局部训练配固定中心评估会引入额外分布变化。当前512缓存不能提供新原生细节，新增原图解码/局部缓存、图像级聚合及正式推理接线并非免费。旧ROI/局部/768/attention和R106双全图负结果不等同本方法，却限制乐观外推。因此本轮不靠dev挑裁框，也不先实现两路模型；保留其为后续输入信息问题，本页不派额外几何扫描。

## 推荐固定合同

| 字段 | 协调采用的唯一值 |
| --- | --- |
| 配置/顺序 | `configs/R117-SCIENCE.json`（仅A生成）/27 |
| profile | `single-swin-base-ms22k-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-v1` |
| candidate | `r117-swin-base-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-9010` |
| round | `r117-swinbase22k-seed20261029-scale085-pairlog-24k-p4-9010-20261031` |
| 源 | `timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k`，非V2/384/医学来源 |
| 直接对照 | R107 Small/seed20261029，selected AUROC `0.9884405731067525`、AP `0.8274326888154083` |

除架构/对应来源及其必要原生结构外，沿R107：固定9:1清单与splitseed、训练seed20261029、512 prepared/input及已有图像缓存、ImageNet归一化、单视图scale[.85,1]与其余增强、G64/P8/8rank各8、FP32/noTF32、纯global-batch pairwise logistic原六字段/temperature1/448对、backbone1e-4/head与age1e-3、原AdamW/clip1/WD规则、800/24000/p4/绝对warmup及EMA1600、原整体dev selector。全参数可训，不带freeze4、stride1、SSL、旧checkpoint或医学teacher；内部/AP/官方/平台分开。

结构建议固定Swin V1：patch4，embed128，depths[2,2,18,2]，heads[4,8,16,32]，window7，shift交替0/3，MLP ratio4，attention/MLP dropout0，SD线性末值.1；最终原生LN(eps1e-5)→GAP1024→Dropout.2→fresh Linear(1024,1)+原age残差。LN eps/位置属于源架构，不能硬套ConvNeXt的1e-6或pool后LN。参数实际来源加载后新建binary头，原生`nn.Linear`初始化；不假称整体初态仍50c。监督seed相同不等同Torch随机轨迹相同。

沿现`engine._optimizer`的**实际**规则：ndim>1且名不以`.bias`结尾者衰减1e-4，norm/bias向量不衰减；二维`relative_position_bias_table`因此衰减，不偷偷采用另一份Swin no-decay recipe。包装器将新binary线性置于`classifier`下，保持last_linear分组；不要误把公开1000头留在优化器或让新头走backbone LR。

## 最新评估协议（用户已明确）

用户已明确：**99.6%指内部5140 dev AUROC≥0.996；“测试”指官方5849验证集评估。** 自R117起，完整训练结束并按原selector冻结唯一checkpoint后，只有该selected内部AUROC≥0.996才对其完整official5849评估一次；中途partial即使达到门槛也不触发官方。不新增fit指标前向，不授权平台test CSV生成或提交。

未达门槛时官方状态记`not_run_by_policy`、原因`threshold_not_met`，不记失败、不补造官方指标；内部与其他工程要求完整时，整轮可正常`completed`，实际safe仍须资源退出证据。内部5140 dev仍每800成功更新按原max(online,EMA) AUROC selector及patience4选点/早停；固定9:1、24k上限、绝对1600及总体raw AUROC≥0.999目标不变。**严格胜R107才进行既定seed27复核，与0.996官方评估门槛独立**，不能混为一个条件；官方/test仍不用于选参。旧R116已绑定的原once自然收口，冻结包和运行不热改，其实际终态由A核实。

## 来源、512与必要缺口

[官方timm模型卡](https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k)标MIT、ImageNet22K→1K、约87.8M参数、224下15.5GMAC。许可/非医学类别有公开依据，**immutable revision、实际原bytes/SHA、转换产物和当前Pod可用性尚未核得**。只固定该repo一个commit、一种权重格式；缺口由必要来源验收补齐，不用N06/FLAIR或另一个tag救场。

[TorchVision 0.25源码](https://docs.pytorch.org/vision/0.25/_modules/torchvision/models/swin_transformer.html)支持窗口补齐/还原，512输入的分阶段网格为128/64/32/16，window7不整除时须走该既有padding路径；relative-position表不因输入分辨率插值。采用显式`SwinTransformer`构造参数，避免`swin_b()`内部固定SD.5与外传.1冲突。这是源码支持，不是本轮512运行已通过。

建议转换为上述TorchVision原生结构，完整public1000参数逐键strict/exact，明确patch-merging排列、qkv、MLP及norm对应；不得只看shape接近。源可能不保存可重建relative-position index，而目标有整数buffer：这些须以确定性构造单独登记/核相等，不能伪称源包含它们或将其计为BN统计。转换后在224做一次合成输入的source/target full1000 FP32数值对照；实际512 shape、padding与有限输出在必要CPU结构验收中确认，不添加独立GPU预演。数值对照不通过则报告映射/实现未完成，不能直接开训或改科学值。

## 最小文件边界与窄验

协调已将以下必要准备分别交现有A/B，规划不直接联系A/B：

- B仅新`tools/pretraining/swin_source.py`、`swin_strict.py`及一件`tests/test_swin_source.py`，复用来源工具但不复用ConvNeXt的通道/键映射假设。固定来源、映射/静态buffer及源目标数值对照交A；无Git/GPU权。
- A新增`retrain/swin_model.py`及SCIENCE，接`config.py`/`models.py`/`experiment91.py`、原正式候选检查与必要fresh/checkpoint/inference路径。复用data、pairwise/gather、sampler、clock及通信优化；不改其算法。现`ema.py:78`把noBN等同无任何buffer，`single91.py`只抽BN buffer，均须增加**显式Swin静态整数index**身份分支；旧ConvNeXt/R50默认不变。整数index原样copy且全程不变，relative bias是可训参数按原EMA混合，不把24个预期index叫24个BN。
- 只三类新增意义验收：①完整源映射与224数值对照；②必要512结构/非整窗padding及实际head/参数组/静态buffer身份；③EMA与保存重载保持index精确、参数照原公式，原缺省兼容。实际全模型来源/初态与formal包证据复用同一次，不另control-only、不重Gloo/loss/像素全套。逐项范围由A/B按最终接口落实，不能凭本页把任何验收记为通过。

## 成本、停止条件

R116实测6400步训练组件wall4079.423秒（含初始化/评估/保存），按完整24000/30周期线性外推约4.25小时，是**ConvNeXt Base描述性成本参考，不是Swin benchmark**。Swin卡224计算量同级，但512窗口padding、attention矩阵、算子效率和buffer通信会增耗；以高于该参考约25%–35%的工程情景估算为约5.3–5.7小时，超过约40%余量便可能触6小时。倍率是风险情景，非测得吞吐或上界；不依赖早停救预算。保留单次train6h/total8h/grant9h，实际正常训练报告耗时/峰值；OOM/期限失败如实记工程未完成，不私降batch/精度/图像或加资源。

约88M FP32单份模型约350MB，最终500MB需含实际buffer/config/age等核实；这不证明端到端指定硬件100ms。窗口注意力可能比卷积慢，100ms仍未知、不从当前训练卡推合规。

**R117工程存储批准：**根已明确批准仅未来R117通过现有集中版本化资源政策及未来绑定使用16GiB round软件上限；旧未指定默认及旧冻结包不改。A须按Swin实际参数/buffer与保存格式核实健康路径`9C+I`（9份完整训练checkpoint加独立初态）、其他round文件及并行prep/新源余量，并实核当前磁盘同时满足原启动free20GiB、运行floor16GiB；不能把Base估计抄成Swin实测，prune异常残留也不能隐去。保留全部保存保护、300秒、atomic/prune/fail-fast及6h/8h/9h，不新增物理资源或续租。此为软件配额批准，不是实际磁盘、最终包体或时限已通过的证明，不增加文档审批门。

阶段先1次fresh，只有工程完整且按原selector整体AUROC严格胜R107，才另固定seed20261027对R108 `0.9872328538178018`复核；AP/难组/官方不救主目标，不挑最好seed。首跑不胜关闭这个Swin-B+公开checkpoint组合，不扫窗口/尺寸/LR/seed。现有反复dev自适应偏差保留，不能推显著性、机制或.999可达。

按用户新指令，R117完整及其可能的条件复核全部收口后，下一独立问题转为[用户指定分析的DINOv1 R50来源方案](N06-FRAMEWORK-SOURCE-PLAN-20261031.md)：只用官方非医学公开backbone，复用标准R50/R111下游合同更换来源，N06/FLAIR权重、teacher、BN统计继续隔离。最多seed29及条件seed27；具体轮次及T29/T27在分支启动前固定，分别取R107/R108与届时对应完整R117结果的较高者，无对应R117则用旧参考、不补跑。R111仅作来源对照参考，官方≥0.996门独立；主源训练前工程不可用才由根另决策A1，科学阴性不跑替补。原生局部保留为更后备选；本段不抢轮次/order，不改变或打断R117，不表示新来源已验收或后续训练已启动。

根已采用本页并交A/B必要准备，用户评估口径已明确，原activation等待解除。必要来源/CPU/formal、R116真实safe及新Git绑定完成后，按原链直接唯一fresh，无需再问用户或等待本页确认。本文不称已注册或已启动，不证明当前safe，也不替代资源准入；本次文本更新后停写。
