# R181/R182：Large 固定复核与同骨干的自然图像 DINOv3 初始化

唯一活跃R180/order92使用冻结28e19c1/SCI9ea7429c，真实八rank/345全训/5空Adam/自然首25有限已证；16:48实观4350 success=attempt/0skip/eval5/bad1，暂最佳EMA c4@3200 .9880889738720829，未完整，raw .999未达。当前32e3唯一Git/GPU；719a准备root0–11/12核、peer12–23保留，与训练24–119/96核互斥，当前既有peer只做结果汇总；不新建或恢复Agent/资源，不热改R180。

R181/order93：只有完整工程有效R180唯一selected严格大于R107 .9884405731067525及R120 .9894773092062407，且actualsafe后，才运行完全相同Large/dc2/512/原logistic/G64P8/优化EMA配方的seed20261027。否则不normalformal/clock/GPU。

R182/order94：独立fresh、原R180 Large/seed29/job/512/ImageNet输入/原logistic/G64P8/采样增强/AdamW/LR/EMA/age/24k-800-p4，只改公开初始化为`timm/convnext_large.dinov3_lvd1689m`，固定一个实际registry revision及LFS。它是自然图像LVD-1689M DINOv3来源，完整342 backbone、无预训练分类头，原生Large图保持，Native1000头的初始化RNG消费保留后再换原二元头，age一次。完整内部AUROC为主判，对照R107/R120及同骨干R180；不叠加BCE/ROI/其他policy，不扫来源/输入/LR/seed。Small DINOv3 R176 .98342是明确负先验；Small/Large与两公开来源形成有界描述比较，不宣称纯容量或纯预训练目标的因果分解，也不预设收益。

按用户明确无需审批直接实施最小计划：

- [x] 单次公共registry固定0f8b496d/bin LFS be34ed20，94段整件785050722B核对完成；完整许可、两factory strict/完整342有限FP32及非张量语义、转换194e4ac3/provenance0a6674dd均实核通过，无来源前向数值对照。
- [x] 精确源码范围：现`tools/pretraining/small_source.py`和`small_strict.py`增加独立largedino3 profile，仅读取profile的model/features/widths/文件名等差异，沿用原342映射/许可/非张量语义；旧Small API/CLI默认不变。`runtime/bundle/retrain/models.py`复用既有DINOv3公开backbone加载/临时Identity与恢复原头路径，新增精确Large format及来源pin；`config.py`/`fresh.py`只扩该精确Large初始化kind/format允许范围，`tools/activation/prepare.py`沿原位置注入342身份。`experiment91.py`/`tools/validation/check_candidate.py`仅加两固定身份/seed/source/互斥及保存选点绑定，现Large forward/optimizer/WORLD/EMA/保存/推理/启动/监控不改，不另建平台。
- [x] 现`tests/test_pretraining_sources.py`/`tests/test_candidate_binding.py`加受影响Large342与两新身份的窄目标；现Large测试扩public-headless初始化/RNG/真实冷加载的必要范围，复用来源无关的大模型图/P8已通过证据。不重复130秒旧全历史fixture。两SCI在真实fresh后固定SHA。现`ROUND-STORAGE-POLICY.json`仅给两精确新round32GiB，旧default/free16GiB不变，沿原存储函数窄验。
- [ ] 所有不依赖最终结果的来源/代码/CPU/配置/Git及technical包在R180训练期间完成；收尾前适时完成唯一正常CPU及精确部署，避免过早9小时软件绑定失去8.75小时clock余量。完整终态/safe后一次activation/direct launch，以真实八rank/自然首更新证明GPU可运行，无独立GPU预演。20%仅准备，80%训练，两池互斥；不改活跃clock/token/grant。

16:44已通过5受影响目标/12项编译、两public fresh及真实来源handoff；[真实证据](../../records/preparation/large-dino3-20261102/README.md)保留RED/fixture错误与精确复用范围。R181 whole24988a65/SCIc94d274c，R182 whole864775ca/SCIc7f2e7f3。现Git同步/两technical包准备，无新formal/clock/GPU，R180原冻结继续。

医学预训练权重/医学教师仍按既有规则排除，N06研究权重不用；本方向仅核同家族自然图像公开来源。公开页依据为[发布者模型卡](https://huggingface.co/timm/convnext_large.dinov3_lvd1689m)，精确许可与原件已经CPU实核通过。原始权重、图片、checkpoint、CSV与传输包不进Git。当前仅准备，未新formal/clock/GPU；官方5849保持隔离，完整selected≥.996才一次评估。
