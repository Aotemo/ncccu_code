# R179/R180：固定联合目标复核与原生 Large 独立比较

当前唯一GPU为R178/order90，313d5ab冻结，实际八rank/345全训/5空Adam/自然首25有限已证。32e3唯一Git/GPU，719a准备root0–11/12核、peer保留12–23/12核，与训练24–119/96核互斥，nice19/隐藏GPU，不派生或恢复Agent。按已读设计/计划流程直接实施用户授权的必要最小修改，无审批等待。

R179/order91仅在完整工程有效R178唯一selected严格超过R107 `.9884405731067525` 与R120 `.9894773092062407`、且actualsafe后，复核同Small/fe6/固定pairlog+BCE各半/原512/G64P8/优化与EMA配方的seed20261027；否则不normalformal/clock/GPU。

独立R180/order92直接原R107/seed29/512/ImageNet输入/原logistic/G64P8/采样增强/AdamW/LR/EMA/age/24000-800-p4，仅将骨干及其匹配公开初始化成对换为原生ConvNeXt Large/自然ImageNet22K→1K固定dc2e53fa来源。完整原生网络、原avgpool/final LN/Dropout.2/Linear1536→1/外层age一次；不是容量的纯因果消融，不叠加R178联合loss或DINOv3/ROI/其他policy，不扫来源/输入/LR/seed。Small/Base已有结果是负先验，Large不预设收益；完整内部AUROC为唯一主判，失败关闭固定组合。

## 实施计划与精确范围

- [x] 必要来源/模型目标先RED，再最小实现：现`tools/pretraining/base_source.py`加显式large224 profile及model/features读取，默认base224/Base224与CLIP入口保持；现`base_strict.py`按profile选原生Large factory/1536宽度及新转换文件名，完整344/public1000逐张量映射和41 LN/36 GELU-SD/40 Conv等检查原算法复用，不复制来源平台。现`test_pretraining_sources.py`扩一个完整Large映射/旧默认目标。
- [x] `runtime/bundle/retrain/models.py`仅将Large接入既有原生ConvNeXt工厂/Dropout及固定provenance分支，`config.py`只扩model白名单；首次模型检查已在真实冷加载发现`inference.py`还有独立架构白名单，现仅加Large一项，不改加载算法。fresh/optimizer/DDP/EMA/保存/启动/监控实现不改。新`tests/test_convnext_large.py`验证同seed原生头/age RNG、512/P8有限梯度、5空Adam全参数覆盖、age一次、EMA与真实CheckpointStore→inference冷图；native随机初始化fixture和真实公开原件strict/fresh分别表述，CPU storage不当DCU峰值上界。原source目标已PASS不重复，修正后只复测受影响模型目标。
- [x] 既有`experiment91.py`两身份/互斥/seed/source/结构与保存选点绑定，现`check_candidate.py`复用相同结构检查；扩现candidate目标和两SCI。真实完整原权重/LFS、适用完整许可、两factory strict/非张量及两public fresh完成后再固定SHA，不将metadata登记当加载验证。
- [x] 现`ROUND-STORAGE-POLICY.json`仅为新精确Large round登记32GiB（默认12/旧例外/free16GiB不改），扩现存储函数目标；原子新blob前prune情景按实际状态大小核，非最大瞬时峰值保证。按已授权清理规则，仅对已完整归档/无活跃依赖的旧轮checkpoint生成真实清单并清理，永久KEEP3/锚点/当前运行/公开资产及数据缓存保护。
- [ ] 受影响CPU检查、Git push/实际回读、两技术包在R178训练期间完成。现27900训练/31500整轮/32400绑定及资源policy保持，CPU不能证明Large DCU吞吐或满额完成；若工程截断不得判科学阴性。收尾前唯一正常CPU/精确部署，actualsafe后按完整分支一次activation/direct launch，真实八rank/自然首更新证明GPU可运行，无额外GPU预演。

15:15实际来源344/full1000两factory strict、许可/LFS及两public fresh、4个受影响目标/11编译通过，[原件汇总](../../records/preparation/large22k-20261102/README.md)保留失败及CPU整树证据。R179 whole91507a65、SCI7d8ed30c；R180 whole3360682e、SCI34e6c689、converted b27d6640。当前Git同步/冻结部署待完成，未新formal/clock/GPU；完整selected≥.996才官方5849一次，不用官方调参。raw .999未达。

15:00已按[逐文件清单](../../records/storage/closed-r170-r172-r174-r176-20261102/README.md)清理35个旧checkpoint，删除前allocated36,507,485,184 B，文件系统可用约57.8GB，保护文件前后一致、无训练信号/锁。R170的临时模型保护在失败完整归档且独立R174完整安全后释放；它仍为工程失败/selected null/双门null，不改原指标或补称科学阴性，所有失败与侧车/index元数据保留。实际R178及KEEP3/锚点/公开资产/数据缓存均未动。

来源及模型目标已原RED2fail；首次GREEN中source目标通过，但真实冷加载发现inference白名单缺Large。只增加一项并复测受影响模型目标PASS，实际512/P8全345有限梯度、5空Adam/EMA/age一次/CheckpointStore冷图exact；CPU saved storage15,313,271,040 B不当DCU峰值上界。完整权重原件的HTTP2/HTTP1及直连代理失败保留，固定同一LFS分段传输进行中；没有完整SHA和公开strict/fresh证据之前不宣称来源或候选通过。
