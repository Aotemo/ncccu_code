# R119决定：全图与固定原生局部，共享骨干单输出

**最终状态：容量核验通过，唯一固定64fit读取探针未达到预定阈值；root已确认终止本实现，未进行训练。** 2026-10-31。下文保留原采用合同与条件链作为历史，不再授权生产缓存、正式包或fresh。此次是准备池工程筛查失败，不是局部科学阴性或模型AUROC失败；完整证据见末尾收口。

| 身份 | 唯一值 |
|---|---|
| order | 29 |
| SCIENCE（A独占） | `configs/R119-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-nativecenter-v1` |
| candidate | `r119-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-nativecenter-9010` |
| round | `r119-small22k-seed20261029-scale085-pairlog-nativecenter-24k-p4-9010-20261031` |

## 问题与固定输入

假设：在完整视野仍可见的前提下，给同一Small骨干额外的原生中心细节，能改善整体排序。它同时改变输入信息、两个尺度的共享训练和融合，**不是纯信息因果**。直接比较R107 seed20261029，而非较弱R118。

唯一可选字段为`job.native_local_view="center-half-meanlogit-v1"`，缺省None保持旧单图合同。该版本固定如下，不暴露裁框/融合权重扫描：

1. 从每张**原始sealed图像**取EXIF校正后的RGB，经现有保守黑边裁剪得到未缩放的C，尺寸W×H。全图分支严格复用现有`prepare_image`的像素语义，最长边最多512、BILINEAR缩小；缓存命中仍使用原全图缓存。
2. 局部从C裁出中央半宽半高：`cw=max(1,W//2)`、`ch=max(1,H//2)`，`left=(W-cw)//2`、`top=(H-ch)//2`，框为`(left,top,left+cw,top+ch)`；之后最长边最多512、同BILINEAR。不再次对局部做黑边裁剪，不从prepared512裁放大，不借定位器或dev难例定位置。
3. train/dev及未来有条件官方使用相同固定裁区和双输入。train两分支分别以同一个原sample-key augmentation_seed构造局部RNG，沿原8次随机调用/scale[.85,1]与全部增强；全图像素应保持旧单视图身份，局部应用相同随机取值。eval均原identity/居中letterbox，不建增强RNG。两图都按原ImageNet归一化输出512。小原图若发生最终插值放大，不能称增加原生信息；生产缓存构造只记录聚合数量，不替换或删除样本。

固定64fit已有几何支持约2倍原生线性保留、无需上采样，但它来自2×2区域，既不是全fit证明，也未证明中心含病征。中心裁区可能漏偏心视盘、改变上下文并引入噪声；保留全图不能保证融合不退化。旧ROI/768/attention/stride及R106双随机全图阴性均作反例，不能说局部从未试过。另一可省成本的办法是训练时随机局部替换全图、eval仍全图，但会混入局部图像标签和train/eval视野差异，本轮不选。

## 只算一个图像级目标

Dataset对train/eval均返回`[2,3,512,512]`，仍只一份label/index/age，batch为`[B,2,3,512,512]`。共享同一公开fresh Small网络和同一个binary head，模型内部按sample-major展平一次编码；两路图像logit重排后取算术均值：`z_image=(z_global+z_local)/2`，**再只加一次age残差**。随后对64张主图像的64个logit用原global pairwise logistic，仍8P56N/448对、一次loss/backward/optimizer/LR/EMA推进。

不平均概率、不分别赋局部标签或分别计算loss；不沿R106的`flatten_training_views`复制label/age。`training_views`保持缺省，新的双输入在模型内部处理，8rank各8主图/16编码视图，全局64主图/128视图；仅一次DDP外层forward。共享头使局部噪声也能污染图像排序，这是反例，不把它称TTA或模型融合。Torch dropout/SD因2B消耗而与R107不同，不能宣称相同随机训练轨迹。

其余完全沿R107：已验Small公开22K→1K源、seed20261029、全参数训练/stride2、SD.1/drop.2、同fit-only age、固定9:1、FP32/noTF32、G64/P8、pairlog六字段温度1、原LR/WD/clip/AdamW/EMA、800/24000/p4/绝对1600。无新模型参数或学得的融合权重，预期完整初态仍50c；必须由新Git正式CPU实际核同state及新增非tensor双输入/融合身份，不以hash代替输入证明，不另control-only。

## 缓存与唯一待补的CPU证据

现`PreparedImageCache`只含确定性整图缩小结果；即使文件已存在，也不能由它补出原生局部。需要独立、带版本及sealed源SHA/原文件fingerprint/Pillow/尺寸/crop规则的**局部缓存namespace**，原全图缓存及历史receipt不覆盖。train与dev合计51396图各多一份uint8 RGB，正方形上界每图786432字节，合计约40.4GB十进制，另计npy头、receipt、目录和原子写余量；实际保持长宽比通常更小，但不能据64图外推成已核磁盘。fit约36GB、dev约4GB只是量级。新缓存放已声明shared-assets资产范围，仍计当前实例磁盘，不靠移出round目录隐去存储需求。

A于05:09实际摘要、由root转交：当前磁盘可用126804295680B（118.095703125GiB）；fit46256、dev5140、train manifest51396；prepared512既有回执记51396 arrays、31286003712B、mean608724.49B。逐图actual shape/dtype/bytes仍unknown，本次未扫描或读图；该回执不替代当前全图namespace完整性证明。

root已授权A在当前有效20%本人份额核局部每图786432B加npy/receipt/目录/atomic的完整上界、Small健康checkpoint峰值9C+I、运行16GiB floor及并行准备实际余量；原启动free20GiB条件保留。上述磁盘摘要本身不等于容量已通过。摘要不足或空间不足则不能开训，不私清保护、不扩物理资源、不把64几何中旧路径0命中当全cache缺失。

若容量成立，root已授权**一次10分钟、无模型/无GPU的固定64fit最终双图读取探针**：复用已定64名单，不扩样、不读dev/official；在A当前有效20%本人份额内，一次构造临时局部cache，最终pair reader按8路、每路8主图读取/增强/打包，最多首遍＋7重复共8遍。首遍、重复、启动、构建耗时分报，另报实际npy/receipt字节。原图一次用于局部构建，后续从cache读；不清OS页缓存，不称cold、GPU或全训练集性能证明。无模型计时只用于排除明显CPU瓶颈。

预定筛查：容量满足完整上界，且8遍aggregate的64主图完成时间mean≤0.60秒、p95≤0.70秒；首遍/重复仍分别报告。该阈值以已有双视图约0.736秒/步为数量级参照，不是GPU性能保证。失败或unknown须报告具体缺口并停止此实现，不扫描workers/codec/裁框、不补考。通过后已获条件授权，仅做**一次必要生产局部缓存构造**覆盖fit/dev，报告实际数量/字节/耗时并设训练只读；随后必要窄验、唯一Git、正式原生包、精确部署和实时安全准入齐备即一次fresh，不增加文档或再次用户确认门。准备耗时独立如实报告，不称免费或增开一套GPU时限。当前尚未证明probe/cache/SCIENCE/Git/GPU完成。官方仅在.996门满足时按相同规则准备输入并评估一次，不提前构造平台test产物。

## 满24k与交付代价

A已核旧实际日志字段如下，均为秒；data_wait单列沿原日志口径，不另加成独立总耗时。

| 运行 | 成功步 | training | 每成功步training | data_wait | eval | checkpoint | process wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| R106双随机全图 | 8800 | 6473.757 | .735654 | 6.266 | 204.444 | 184.990 | 6955.308 |
| R107单图 | 6400 | 2718.788 | .424811 | 4.698 | 146.153 | 136.526 | 3066.314 |

此前引用R106的7103.695秒是TRAIN **component wall**，不是上表process wall6955.308秒；二者不可混用。这两轮早停长度不同，也不是新原生双输入benchmark。沿已有R106训练数量级，24000步约17656秒；把保存/其它开销外推并按30次双编码dev计入，原规划约**5.5小时量级**、距离6小时仅约半小时，仍是非实测估算。不能把旧data_wait直接复制，新原图准备、cache读取、GPU kernel和实际验证可能耗尽余量；不靠p4早停满足满24k预算。若探针明显跟不上或满额预算不能守住，停止此具体实现，不能缩batch/精度/步数或扩大6h8h9h来救。

共享Small参数一份，最终FP32模型约198MB量级，双输入不等于两份模型，仍须实际交付≤500MB；缓存不是提交权重，但正式推理需从原图完成两路输入，解码/预处理+两编码可能超过指定硬件端到端100ms，当前未知，CPU探针/现训练卡均不能宣称达标。不开独立GPU预演，不先许诺部署合规。

## 已采用的最小分工与停止条件

root已派B只独占新增`runtime/bundle/retrain/native_local.py`和`tests/test_native_local.py`，固定原生crop/preparation及均值融合的无参数helper API已交A/B。A独占其余data/preprocessing双输入与局部cache接线、models内部单次2B编码、配置/ex91及训练/eval/冻结/inference消费，并负责实际计算、唯一Git/GPU。实际计数日志必须64主图/128视图/448对。所有入口统一版本字段，旧缺省单图和缓存namespace不变；无新label/age schema或通用多视图平台，不共写文件。

仅三类必要窄fixture：①合成原生高频图证明中心crop在resize前、全图旧像素不变、整数裁框及train/eval RNG；②toy共享线性网络验证先均值image-logit再age及单pairloss的值/梯度/主label数，排除R106复制目标；③cached/uncached双输入像素一致、源/规则身份隔离及缺省兼容，正式包自然检查同state与推理重建。不重来源/Gloo/旧数学套件，不另GPU预演；沿root已交的精确接口与上述文件边界实施。

科学阶段先1次fresh；完整selected AUROC严格胜R107 **0.9884405731067525**才另冻seed20261027对R108 **0.9872328538178018**复核。AP、困难组及官方不救主目标；首跑不胜关闭这个固定中心双输入组合，不扫中心大小/位置/融合权重或seed。完整冻结selected≥.996才official5849一次，否则正常`not_run_by_policy/threshold_not_met`，与seed复核独立。原6h/8h/9h、默认Small 12GiB round cap、保存保护/300秒/atomic/prune/failfast不变，新增共享cache磁盘另实核。

若科学阴性，下一问题应区分“固定中心覆盖不足”与“共享均值输入组合无益”，现有结果不能判别；规划先据既有训练与实际成本决定是否仍值得另一独立输入设计，不能自动将失败改写成四局部/定位器任务或再扫来源。若当前工程缺口不闭合，向root报告具体空间/吞吐缺口并停止该实现，不无限补探针。全局目标仍持续，阶段失败不记总体完成。本页按root本次采用与条件授权冻结；容量→一次探针→一次生产cache→必要绑定→唯一fresh由A原链推进，未取得的工程回执保持未完成，不以页面成文替代实证。规划不另派任务或增加新门。

## 最终工程收口：不触发训练

仅一次文本核本机`work/r119-20261031-a/probe-result/{CAPACITY-R119,PROBE-R119}.json`。05:22:05北京容量筛查通过：free126804295680B、local上界42103603200B、atomic819200B、目录/prep reserve2147483648B、Small健康9C+I估7517229768B、运行floor17179869184B。原回执`remaining_after_all_estimates_bytes=75035159864`实际为**扣运行floor之前**的余额，字段名不能解读为扣floor后；此澄清不修改原件、不改变容量通过结论，也未重新测量或计算。

05:25:25.877525北京唯一探针使用原固定64fit名单/manifest、最终ImageDataset pair-reader、准备池12CPU、8路各8主图，输出`[8,2,3,512,512]`，包括完整增强/归一化/打包。8遍完成时间原件依次约1.013638、.923815、.954262、.753227、.767484、.771925、.756119、.763719秒；mean **.8380236838711426>.60**、p95 **1.0136382030323148>.70**，`passed=false`。临时64份npy共37792256B、receipt38280B，构建19.723秒、整体29.113秒；guard exit0表示执行完成，不等于性能通过。

Probe原Pod SHA `6e7b80ebf143faa7defdfd154b58033c2dc325cf6af0134fb246660b5746afdc`；guard原Pod SHA `6b42654650c44a2bb9c0f5ab9a07a7d8b8841c5e7ebf1d3777a0d2189391bab5`（按A/root交接记录，未本机重算）。无模型/GPU/dev/official，未清OS缓存；结果只否定本次12核准备池筛查，**不能推出实际96CPU训练已超时、全fit性能或局部信息无效**。

按原冻结条件关闭本实现：不补考、不扫workers/codec/裁框，不建全fit/dev局部缓存、不做正式包/activation/训练。没有selected checkpoint、AUROC或官方结果；A新代码尚未正式验收/Git，B两helper为文本冻结、未跑tests，均不冒称GREEN或删除改写。root报告当前无新GPU运行。总体目标仍未达；下一独立问题另以有限文本建议交root，本页不占新轮次、不将局部失败改成原地重试。
