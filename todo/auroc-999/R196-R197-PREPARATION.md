# R195 之后固定曝光增强假设施工计划

23:57两[实际technical包](../../records/preparation/gamma-20261102/TECHNICAL-PACKAGES.json)已固定436ae5b/tree3613123a，R196 SITE737d93b7/guard6429n469与R197 SITEd26295ea/guardm026ckp2均0；归档s8jndylw/f25e2958…30c5 exit0。无normalformal/新GPU allocation/clock/GPU；只在既定R195之后接续，不混入当前R193/2a或先行693。下文23:55“尚未Git冻结/technical包”为当时状态。

23:55已完成[5必要目标/5源码编译与两actual public fresh](../../records/preparation/gamma-20261102/README.md)，最终guard gagx9qd0/dbc776c0…f88f exit0。R196 whole9150/SCI93058a7b、R197原whole50c6/SCI6000c2a8、均345全训/5空Adam，尚未Git冻结/technical包/normal/clock/GPU。首两次None身份fixture错误已如实保留并纠正测试，旧格式保持；最后3RED因gamma未实现，随后3GREEN/两候选GREEN。当前R193仍原2a、3200成功/0skip/eval3，暂EMA `.9819986008277032`，8rank活跃；不因本组准备延后R194/R195接续。

使用 brainstorming / writing-plans / TDD / executing-plans，32e3本人在既有20%池完成；用户已明确最小修改无需审批，不派生/恢复Agent。当前R193/2a与既定R194/R195/693d48c不变，本文只为更后续准备。

目标：复用当前模型、训练、保存、评估和启动链，独立检验非线性曝光增强。备选继续换公开架构（近期固定组与整体均未改善）、继续改优化日程（已由R193/R195单独检验）、或在fit中增加有限gamma变化。本次选择第三项；原32图描述中难正例亮度均值较低仅是小样本线索，未证明光度是机制，不追认新临床ROI或逐例权重。用户明确1=青光眼、标签仅据眼底照片；0仅任务阴性。

R196/order109仅在完整有效R195唯一内部selected严格胜R107 `.9884405731067525`与R120 `.9894773092062407`且actualsafe后，以同原Small/fe6/512/sigmoid/固定8000衰减、seed20261027独立fresh复核。否则仅技术准备。R197/order110独立直接R120原Small/fe6/seed20261029/512/原sigmoid/原backbone1e-4、head-age1e-3/default AdamW与原24000余弦，不叠R193半LR或R195短衰减；仅新增`training_photometric='gamma-reciprocal075-v1'`。

固定数学合同：保留原训练mirror/brightness/contrast/color/rotation/scale/translation全部八次sample-key随机抽样及其顺序。原RGB canvas完成后、转FP32及原normalization之前，以独立`random.Random(_seed(augmentation_seed,'training-gamma-reciprocal075-v1')).choice((.75,1.,4/3))`等概率选一档。每通道相同LUT `round(255*(i/255)**gamma)`，i=0..255，Pillow RGB point；0/255端点保持，空间位置不变。新随机流不消耗原八次或任何全局Python/numpy/Torch RNG。只training=True且新policy应用，eval/旧None/旧identity均精确保持。gamma不能恢复此前裁剪或量化丢失的信息，单调映射不证明临床征象保真或收益。不扫描范围/系数/seed，不用dev名单或官方挑参数。

永久范围仅`runtime/bundle/retrain/data.py`现有ImageDataset验证及_image_tensor末端小分支、`config.py`现有字段取值验证、`experiment91.py`两身份/互斥/来源/保存job、`configs/R196-SCIENCE.json`/`R197-SCIENCE.json`、新`tests/test_gamma_policy.py`与原`tests/test_candidate_binding.py`，及必要todo/records小件。无新字段、依赖、图像缓存、模型参数或forward、训练平台/启动/保存/评估实现；两轮默认12GiB/free16GiB保持。

- [x] 先写受影响真实ImageDataset的独立像素oracle/三档完整LUT、sample-key确定性、原八次几何流、全局RNG、None/identity/eval保持与512/P8有限张量目标，Pod内真实RED后最小实现。oracle单独构建原canvas，独立逐像素计算，不读取实现输出当期望。
- [x] 配置load/job_contract保存新字符串、旧省略与显式None的各自身份格式保持、非法取值及与短衰减/半LR等叠加拒绝；两候选job/SCI/来源/seed/原训练合同及旧R194/R195兼容身份目标，真实RED→GREEN。仅编译受影响源码，原模型/optimizer/loss/EMA/WORLD/保存/推理按未变SHA复用；不重跑全套或GPU预演。
- [x] 两public fresh核实际初态/5空组与最终SCI，写窄验/实际依赖/小回执，源码配置实际Git同步/远端回读后技术打包。仅技术包不等于normalformal/activation/GPU或收益。
- [ ] 本组仅在R195之后依完整结果选择唯一分支，正常CPU/精确部署在可行收尾前完成，actualsafe后沿既有入口一次接续，真实八rank初态和自然首更新证明GPU可跑。不得因本准备阻塞当前R193→R194/R195。

当前准备53ab/root0–11、保留peer12–23，与训练24–119互斥；全部解析/测试/哈希/Python在逐次核UID/Ready的既有Pod内，nice19、隐藏GPU。训练24k上限/800评估/p4/EMA1600/固定9:1/唯一内部AUROC selector和官方5849隔离保持；完整内部selected≥.996才官方一次，raw .999未达。
