# R183/R184 准备计划：条件 Large DINOv3 复核与固定原生 ViT-S

18:00 实际受影响 4 目标及 11 个唯一源码编译覆盖通过；原生模型 2 目标、两身份共 1 目标、精确存储 1 目标均有 guard0，原 RED 与 fresh 辅助脚本语法错误原件保留。实际两 public fresh：R183 whole c51c4d8b、SCI1f116cd6；R184 whole e347757c、SCIa231b7d0，分别345/165全训、5空Adam，0图像前向/0GPU。源严格验收与旧默认兼容按不变源码复用；原P8/保存冷图未重复。下方模型/绑定/存储项实现已完成，Git冻结及技术包待完成，尚无新normalformal/clock/GPU。

必要 repo 外范围仅复用已执行的 PREPARE-LARGE-DINO3-NEXT.py：加入 R183/R184 两文件名、对应 4目标/11编译证据路径及 R184 默认12GiB期望；旧已执行文本保留为历史 .txt，不改任何现役冻结包或启动链。既有 START observer 的初态数量本已消费 formal 的 trainable_tensors，默认Adam无需适配，不另改监控。满24k沿用原Small512×2情景24154.892秒作为参考、低于27750软预算；不依赖早停，不称ViT实测吞吐或上界。CPU保存storage3314983264B及checkpoint张量容量情景均不是DCU峰值或保存并发峰值保证。

本计划由唯一Git/GPU owner32e3直接沿executing-plans实施，不派生Agent、不增加资源或审批门。当前R180冻结28e19c1、后续R181/R182已冻结技术源码2a128733，均不热改。详细来源先验见[ViT核查](VIT-DINO3-SOURCE-AUDIT-20261102.md)。

目标：把一个自然图像DINOv3 ViT-S/16组合准备为R182之后的独立fresh；单模型、原R107 seed29/512/ImageNet输入/原pairlog/G64P8/采样增强/默认AdamW/LR/EMA/age及24k-800-p4保持。只改原生架构及匹配公开初始化，不叠加Large/ROI/MIL/其他policy，不扫pool、来源、LR或seed。R141/R164/R166/R176阴性保留，不预设AUROC或临床机制收益。

架构：直接调用已安装且pin来源的timm1.0.27原生`vit_small_patch16_dinov3.lvd1689m`、pretrained=False、512、num_classes=0、发布者avg/SD.1；完整162公开张量strict加载。保留1CLS+4register在自注意力中，末端原final LN eps1e-5后排5prefix平均，fc_norm/原head均Identity；原native forward输出384，再单个Dropout.2+Linear384→1，原age残差一次。不得移植ConvNeXt的LN1e-6/buffer0假设；原FP32非persistent rope.periods[16]须实核并由同原生factory冷构图重建，不量化成BF16或移除。仅当前测试需要的库字节/非张量语义固定，不重建训练、保存、推理、观察或启动链。

R183/order95是R182同Large自然DINOv3配方固定seed20261027；完整工程有效R182唯一selected严格胜R107及R120、actualsafe后才有GPU资格，失败不normalformal/clock/GPU。R184/order96是上述独立ViT-S/seed29，仍按完整内部唯一selector主判；工程失败不当科学阴性。官方5849隔离，完整selected≥.996才一次；raw .999仍未达。

必要精确范围与检查：

- [x] 来源/原生库先验：固定3bf4720a、86,407,250 B/LFS b7e7b7d1、实际元数据public/non-gated、完整许可25d122eb与配置已核；随机factory构造/162state/实际非persistent buffer已核。没有strict加载或forward，不等同可运行候选。
- [x] `tools/pretraining/deit3_source.py`在原native来源入口增一个显式profile，默认/原DeiT3常量与行为保留；新profile只核固定repo/revision/bin/许可、完整162原生strict及非张量语义，不转换、不丢未知张量。旧签名真实RED后新来源strict通过，原三参数默认实际完整176/full1000 strict与原provenance语义相同通过；以两条真实来源执行代替重复常量字典的unit断言，未增加test_pretraining_sources.py。models.py仅先加必要库常量/字节检查函数。
- [ ] `runtime/bundle/retrain/models.py`增加唯一原生构造/公开来源验证与描述必要分支，`config.py`/`fresh.py`仅增加该精确kind/format/model；`inference.py`只扩原冷图白名单与描述，原保存链不变。`tests/test_deit3_model.py`增加实际factory参数/RNG/5空组/age一次/原始P8反传与真实CheckpointStore冷图，覆盖非persistent RoPE重建及prefix均值；无真实图像/官方或GPU预演。若复用同源码图证据，精确注明范围，不重复原DeiT3 suite。
- [ ] `experiment91.py`/`tools/validation/check_candidate.py`/`tests/test_candidate_binding.py`登记两固定身份及来源/seed/互斥、165全训与1个明确buffer的新分支；R183保持Large342/345/buffer0。两SCI只在实际public fresh后填身份，不虚构SHA。原engine/distributed/loss/EMA/persistence/age/入口/监控不改；源无关代码按精确SHA复用，不新增通用门。
- [ ] 若R183需要Large32GiB，仅现`ROUND-STORAGE-POLICY.json`及原函数窄目标给其精确新round例外；R184默认12GiB容量情景实际核算，free16GiB及原例外不改。实际P8 CPU storage不是DCU峰值/上界；满24k成本需按既有窗口合理准入，不能靠早停兜底或降batch/精度。
- [ ] CPU原件仅写独立`exp-v3/technical-prep`，5de56a8f准备root0–11/12核、peer12–23保留，训练24–119/96核互斥。受影响检查/真实fresh、Git push回读和精确冻结部署在前轮训练期间完成，终态及安全后一次串行接续。

本页新增时仅有repo外的来源审计和此文本计划，没有生产改动、新formal、clock或GPU。当前R182接续准备优先于本页；实现发生额外必要范围时先在此写明，不增加平台或重复已通过验收。

17:28实际R182已完成八rank/345初态/5空Adam及自然首25有限、50成功0skip，[当前运行](../../records/R182/README.md)冻结2a128733不改。R180完整阴性且safe，R181无资格。本页来源已strict验收；17:17原生构造/RNG/prefix均值/age、P8全165有限梯度/5空组/EMA与真实CheckpointStore冷图两个目标通过（guardcx50nx26/d0a202e5…e740 exit0），原缺构造函数RED保留。CPU保存storage3314983264B，不是DCU峰值/上界。当前仅models/config/inference与现test_deit3_model.py补必要分支；initialize=True、fresh/两SCI/候选绑定及精确部署尚待，不能称R184可训练或有收益。
