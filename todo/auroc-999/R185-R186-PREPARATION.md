# R185/R186 准备计划：条件 ViT-S 复核与原生 CAFormer-S36

18:49 当前前三项已实际完成：现来源入口两真实路径通过、两个模型目标分别取得通过证据、两候选身份共1目标通过，9个源码编译覆盖；[原件](../../records/preparation/caformer-20261102/MODEL-CPU-CHECKS.json)保留所有RED/优化器定位失败。只复测P8，首目标/原ViT图证据按相同函数正文复用。R185 SCI039715e4/whole8c62435a，R186 SCI37befe80/wholed2369b38，均实际public fresh/5空Adam/0image forward；初态占位未进Git。待同步冻结/技术包，尚无新normalformal/clock/GPU。

18:32必要范围收口：原factory/MLP对象/RNG/独立手工与原native前向/单drop/age目标已实际通过；同任务P8目标在现_optimizer的classifier命名筛选处失败，尚未forward。只在engine.py既有_optimizer的last_linear定位中给caformer_s36增加network.head.fc.fc2分支，原其他模型筛选、5组/LR/WD/AdamW构造正文保持；不改优化算法/训练链。仅复测失败P8目标，已通过且未受影响的第一个目标按原69n0jyve任务ok证据复用，整任务原exit1保留。下文“engine不改”由本项精确例外覆盖。

执行沿既有 executing-plans，由32e3唯一Git/GPU owner实施；用户已授权最小必要修改，无审批或新增Agent。当前R182冻结2a128733；R183/R184技术源码ab43f542及两技术包保持。此计划不会改变两轮既定顺序；当前只是更后续准备，无新formal/clock/GPU。

目标仍为单模型 raw AUROC .999。R185/order97仅在完整工程有效R184的唯一所选内部AUROC严格过R107/R120且actualsafe后，复核同ViT-S/3bf来源固定seed20261027；否则不normalformal/clock/GPU。R186/order98为固定独立CAFormer-S36/seed20261029，直接原R107的512/ImageNet输入、scale .85、G64/P8、448对原pairlog、AdamW/LR/WD/EMA/age、固定9:1及24k-800-p4。完整内部AUROC主判；官方5849隔离且所选≥.996才一次。没有临床机制或收益结论。

选定一种不同架构与其匹配自然图像公开初始化，不扫checkpoint/输入/池化/LR/seed。既有ConvNeXt/DINO/DeiT阴性保留；ConvNeXtV2的NC来源仍排除。EfficientNet备选涉及不同BN轨迹，当前选无BN的CAFormer混合卷积/注意力架构。组合试验不能纯归因架构或预训练。

固定来源：timm/caformer_s36.sail_in1k_384@695903aebe60ba2a96868cedd9129ea256a1f96e，ImageNet-1K，public/non-gated/Apache2；157,357,817 B/LFS SHA424a3a029dd1931ace03ac0c4697e4d760447c917d2042a82677ae11083ed592。已实际395 full1000 strict/exact/FP32 finite、0forward/0GPU，原件tensor061e1227；native timm1.0.27/metaformer.py e3d95b0f，源384仅为来源属性，比赛仍512。上游64c534ab完整Apache许可SHAc71d239d已核。最初只读审计脚本语法错误及修正后guard均保留。

架构约定：完整原生395张量先strict加载，保留stem、[3,12,18,3] stages、[64,128,320,512]特征、native avg/LN512及公开head的512→2048 MLP/激活/LN2048。仅最终fc2 2048→1000替换为原生Linear2048→1；不能调用会重建公开MLP的reset_classifier。构造drop_rate=0，外层head.drop维持0，仅内层head.fc.head_drop=.2（须按原native forward确认在最后Linear之前），SD .1按native规则；age外层一次。保留公开MLP是此原生架构的一部分，不能称只有512→1线性头。无BN/静态buffer/位置插值，不额外forward或加入ROI/teacher等。预计原393参数张量+新2+age1=396，最终以实际fresh为准。

最小文件与交付：

- [ ] tools/pretraining/deit3_source.py只给既有native来源函数加caformer显式profile，API/CLI默认deit3不变；models.py加必要5文件库pin、来源/构造/描述分支。先真实新profile strict与原默认兼容；不另建下载/转换平台。
- [ ] models.py/config.py/inference.py增加原生二元/冷图分支，tests/test_caformer_model.py仅两个有独立oracle的必要目标：原factory/完整公开MLP与RNG/一次drop和age，以及P8×512反传/5空组/EMA/真实CheckpointStore冷图。不重跑未改的ViT/旧suite；CPU storage不是DCU峰值/吞吐上界。
- [ ] experiment91.py/check_candidate.py及test_candidate_binding.py只登记R185/R186来源/seed/互斥/原生描述与保存身份，两SCI仅填实际public fresh SHA；复用prepare.py，确有必要才增其分支。原engine/WORLD/loss/EMA/保存/启动/监控不改。
- [ ] 两独立technical-prep包、源码/配置实际SHA、受影响CPU证据、Git push/远端回读/精确部署依旧走原链；默认12GiB容量及满24k时间情景核清，不以早停兜底，不加独立GPU预演。只在当前前驱安全终态后串行GPU。

准备使用现5de56a8f绑定：root0–11/12核，peer12–23保留，训练24–119/96核；nice19/隐藏GPU，同一root不并发CPU任务。现有NEXT准备辅助脚本只扩两文件名/证据路径与该轮精确数量，保留已执行原文，不复制启动、保存或监控链。计划外必要范围先写本页；当前现役/ab43冻结源码不热改。
