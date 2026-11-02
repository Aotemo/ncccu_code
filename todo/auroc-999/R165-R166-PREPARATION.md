# R165/R166：DeiT III 固定接续 Implementation Plan

**Goal:** 在R164冻结训练期间完成可直接接续的最小CPU准备，避免再次因后续源码/来源准备偏晚让DCU空等。

**Architecture:** R165/order77只在R164完整内部所选AUROC严格超过R107与R120且安全结束后，运行同一原生CLS配方的固定seed20261027。否则跳过，独立fresh R166/order78保留R164公开8d59b281/512/原logistic/G64P8/age/优化器与时钟，仅将原生最终LN后的读出从CLS改为排除CLS的全部1024 patch平均，随后原Dropout.2/Linear/age各一次。通过同一backbone特征检验读出聚合差异，不加入模型、参数、forward、ROI、teacher或dev名单，不声称平均池能定位视盘、解决临床机制或必然提高AUROC。

**Tech Stack:** 既有Python3.12/torch2.10/timm1.0.27，原公开资产/训练保存评估监控入口；准备池aa0f根0–22/23核、nice19隐藏GPU，peer23与训练24–119互斥，过期前新实核分配。R164的f061305冻结/clock/grant不改。

原生CLS是公开自然分类配方；另一选择是更换新模型/来源，会再次增加下载和适配成本。此处只选一个固定meanpatch读出对照，保留原最终LN和fc_norm Identity，避免同时改变归一化位置。原R124的Conv内容注意池及R141的DINOv2 CLS+meanpatch不等价，但它们的阴性也不支持这里必然获益；本次只可由完整内部AUROC检验，AP/局部难组不救主判，阴性后不扫pool/权重/温度或补seed。

- [x] 先在`tests/test_deit3_model.py`补固定meanpatch原生/独立均值参考、None兼容、RNG/参数身份、P8梯度/保存冷图目标，并在本人独立prep观看预期RED。
- [x] 精确实现范围仅`runtime/bundle/retrain/config.py`（版本化可选policy/验证/保存）、`models.py`（严格public加载后只置原生pool为avg、记录非张量图）、`experiment91.py`（R165/R166身份/normalization），已有`inference.py`/`check_candidate.py`按原生产factory/图描述消费，不另建链路；有实际失败才补不可缺少的小差异。
- [x] 更新原`tests/test_candidate_binding.py`目标及`configs/R165-SCIENCE.json`/`R166-SCIENCE.json`，先实测公开fresh初态/5空Adam组，默认None与省略保持旧job/图。公共176 strict/原库身份及原loss/WORLD按相同SHA复用，无新来源下载或GPU预演。
- [ ] 仅运行受影响目标/编译，保留RED与失败回执；实际Git提交、push/回读、精确源SHA及既有技术包在R164训练期间完成。完整R164决定唯一分支，收尾期间正常formal/部署，安全后一次activation/direct launch，真实八rank初态/自然首更新证明GPU链。

本文是固定准备范围，不是新GPU启动或已通过证据。均不触及官方5849、原始标签或既有selected数据。

4目标/5compile及两public fresh已真实通过，[精简证据](../../records/preparation/deit3-meanpatch-20261102/README.md)。待实际Git同步/回读与现有技术包，不新normalformal/grant/clock；原R164继续。正文未勾最后一项，避免将CPU通过写成后续GPU已启动。
