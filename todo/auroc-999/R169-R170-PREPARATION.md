# R169/R170 固定后续准备

目标仍为单模型raw AUROC .999。R168保持2683a8a冻结；当前训练期间完成下一轮必要代码、公开来源、配置、Git部署和CPU验收。准备dca3根23核0–22、peer1核23与训练96核24–119互斥，不增资源或GPU预演。

R169/order81只在完整R168所选内部AUROC严格超过R107 `.9884405731067525` 与R120 `.9894773092062407` 时复核原beta2=.95配方的固定seed27；其余合同不变，未满足条件不normalformal/activation/GPU。否则独立R170/order82：沿R116的原生ConvNeXt Base/512/原logistic/G64P8/seed29/age/AdamW默认beta2=.999/原LR/EMA/24k-800-p4，只换为公开CLIP LAION-2B→ImageNet12K→1K初始化，不叠R168。

来源固定为[发布者模型卡](https://huggingface.co/timm/convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384)的 `768f3849e9b97ac2f93191ad5fdc458dfafa25f3`，实际Pod registry证明public/non-gated/Apache2，原bin SHA `ee2f8ca014aa7f8362081f4c4098757e76812f5b7a35a3b018bb7a13971bb241`、354495341字节。只有下载及完整344张量映射/两factory strict/41LN/36GELU-SD/40Conv和非张量配置通过后才可生成正式SCIENCE；登记不是已加载或收益。医学专用预训练/N06不使用，已排除NC来源。

R170明确保留R116的比赛ImageNet均值/标准差与512预处理；来源卡384/squash/CLIP均值标准差仅记录来源属性，不默认采用，也不声称原发布者端到端预处理复现。与R116的同架构比较只能检验此新初始化在原输入合同下的表现；与Small主锚点的差异同时涉及架构/来源。新Base尺寸/参数形状须实核一致，不能用ImageNet榜单推断本任务收益。

必要最小范围：`tools/pretraining/base_source.py`、`base_strict.py`增可选固定source profile，旧默认API/CLI保持；`runtime/bundle/retrain/models.py`仅为新自然图像来源增精确provenance分支，不改网络/forward；`experiment91.py`注册R169/R170与精确身份/默认优化器互斥；复用并扩`tests/test_candidate_binding.py`现有目标及两SCIENCE。实际受影响CPU验证/来源回执记入`records/preparation/baseclip-20261102/`。训练、启动、保存、评估、监控、data/normalization、loss、gather、EMA及优化器实现不新建或改写。

完整内部AUROC是主判，AP/难组不救；R170若阴性关闭该固定来源配方，不扫分辨率/归一化/LR/seed。若正向，再另定固定seed27；官方仍完整selected≥.996才一次，不用官方调参。

- [x] 来源真实下载/完整strict与两public fresh，实际初始化及5空Adam组身份；1受影响配置目标/5编译通过，R170 CPU合成P8反传345梯度有限。原件见`records/preparation/baseclip-20261102/`。
- [x] 实际受影响配置/初始化路径窄验；源码Git58bc761同步GitHub回读、精确技术包与R170部署。
- [x] R170独立后备的正常CPU验收在R168训练期间完成18/18，无grant/clock/activation/GPU；R169仍等待完整结果资格。该CPU提前准备不授权选支或并发GPU，若完整结果选择R169则R170保持CPU-only。
- [ ] 完整R168选点决定唯一后续，确认actualsafe后一次activation/direct launch，真实八rank和首自然成功验证。
