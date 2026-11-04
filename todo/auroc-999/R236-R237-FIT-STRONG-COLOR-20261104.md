# R236/R237：固定强光度视图的fit-only实例对比

在R234仍运行且尚未查看其任何dev分数或逐例预测时登记。最近已核实监督325/c0、完整1600及真实交接；本备选仅在R234完整未严格胜R107与R120、且唯一actualsafe后考虑R236/order151 seed20261029。R234完整双门过仍优先原R235/order150 seed27。R237/order152 seed27只在R236完整严格双门后。用户已要求凭现有图片/CSV自主决定；不再请求方案选择或例行审批，不改当前R234冻结565/SITE c5ad/clock/token/grant。

唯一问题：同R234 fresh公开Small、均匀fit/G64P8、原两视图几何、NT-Xent .1/128 anchor/1目标/126负视图、临时768→1024→256头和1600预算，仅将适配视图的原±10%亮度/对比度/色彩改为固定强光度组合，是否改善原完整R107监督的排序。新增约束是更大光度改变及偶发灰度下的同图识别，不增加损失类型、更新数、温度或batch，不以低SSL loss、同图检索或局部dev改善代替疾病排序。原代码是mirror→photometric→rotate-expand→整图fit画布，不是random-resized-crop；本轮没有局部裁剪或定位。

依据为[SimCLR原论文](https://proceedings.mlr.press/v119/chen20j.html)关于增强组合的研究，以及[作者data_util.py](https://github.com/google-research/simclr/blob/master/data_util.py)的固定jitter .8/gray .2与strength1幅度。采用明确的Pillow/torchvision RGB实现，不声称逐位复现TensorFlow、完整SimCLR或保持所有疾病征象。每视图独立：p=.8时随机排列亮度/对比度/饱和度/hue四操作，乘性前三因子均匀[.2,1.8]、hue均匀[-.2,.2]；随后独立p=.2转三通道灰度。不加blur、随机裁剪、gamma、mask、第二目标或额外教师，不扫描强度/概率。强光度在原mirror后、rotate与画布前执行，padding和空间几何保持。

仍消费原八个几何/轻光度随机draw以保留原mirror/angle/scale/dx/dy流，新光度只用从每视图augmentation_seed派生的独立本地Random；原三个轻光度因子在新分支不叠加。新seed域固定`ssl-strong-color-v1`，jitter gate→随机操作顺序→四参数→gray gate。采样/loader/学习/公开头及监督factory RNG继续沿R115/R234，强光度只由SSL recipe注入ImageDataset，不进入保存的监督job；监督training_photometric仍None，全部原train/dev/freeze/official/predict像素路径保持。

只传学生encoder/final-LN，原公开head-age保持，监督空Adam/EMA重建；512/prepared512/ImageNet、原G64P8/448 pairlog、24k/800/p4、唯一内部AUROC selector及官方≥.996一次门不改。总适配合同102400主图/204800增强视图/204800 encoder-view forward，与R234相同计数，不把额外像素处理开销当等时耗。主晋级仍严格R107 .9884405731067525与R120 .9894773092062407；完整结果并列R234/R115/R230/R232和既有空间适配，附AP、原固定20/其余147修复/新增错序及真实成本。R122监督photometric identity及既有颜色校正阴性保持关闭，不把新SSL阶段固定组合解释成旧监督单因子或幅度补考。

必要范围：现`data.py`加这一固定SSL光度分支及私有像素helper，原None/identity/gamma路径的像素和原八draw逐位保持；现`ssl_stage.py`登记`fit-ntxent-strongcolor1600-v1`、仅给同一ImageDataset注入政策并记录实际政策值，模型/损失/gather/优化器/预算/保存/derived逻辑保持；现`experiment91.py`补两身份及原监督合同，两SCIENCE；`tests/test_ssl_strong_color.py`及必要候选目标。不新建训练、启动、保存、评估、缓存或监控链。

CPU准备为独立`technical-prep/strong-color-20261104-4e0c`，独占24核0–23/nice19/隐藏GPU，与现训练96核24–119互斥。先缺行为RED，再独立像素oracle、概率/固定参数/隔离RNG、原None和监督像素与八draw、原R115所有学习/采样流、新候选完整监督合同，以及真实fit双视图consumer。未改网络/损失/gather/derived范围按精确源码复用已通过真实局部P8与R234实际八rank证据，不重复模型前向或独立GPU预演；新增训练路径由自然真实八rank初态/首更新证明。

- [x] 两缺行为RED abc29247，最小实现后两新目标/4编译c97bfa75；48原None/identity/gamma像素案例、16原八draw几何RNG、原SSL循环除声明注入外逐字一致、12未改实现文件及真实fit8×2×3×512×512通过a5b11a8e。复用原实际P893ac00ab及R234自然八rank监督f618327b，没有新模型前向。私有metadata仅`check-strong-color.py`、`finish-strong-color.py`、`prepare-strong-color-technical.py`、`export-strong-color-checks.py`，沿既有CPU/prepare及小JSON导出，非新平台；实现步骤见[R236/R237计划](R236-R237-IMPLEMENTATION-20261104.md)。
- [x] 代码/config c57e3ea/tree31f31a21已push/API回读、2792全树bytes/mode914130b6、两既有技术包及R236正式18/18/guard74fdecb0、精确部署24129974/SITE c4cb7dd9通过；当前训练期间原四步入口已填实际身份。私有`prepare-strong-color-committed.py`、`collect-strong-color-packages.py`、`formal-strong-color.py`只沿既有包装参数替换接入prepare/check_candidate/freeze_deployment；不复制训练、保存、评估或监控实现，R237此阶段仅技术包。`compile-post-r234-metadata.py`/`export-post-r234-ready.py`及`export-strong-color-preparation.py`只汇集小metadata/已完成guard和一次原生进度，精确范围限独立strong-color technical-prep；[实际准备](../../records/preparation/strong-color-20261104/README.md)保存正式身份与足额窗口，两分支未激活。
- [x] R234完整阴性、唯一safe94f21ed6后选择本科学分支；R236/R236R1各一次activation在存储门拒绝、无正式clock/派发，原件保留。另存R236R2沿同一SCI/order151和最小存储观测源码c800，真实18/18/精确部署及一次激活/入口后，实际八rank首75通过；[来源、失败与真实2121.741秒接续](../../records/R236R2/README.md)。R237资格仍待本恢复轮完整双门，不以工程失败当科学阴性。
- [ ] 完整内部选择、AP/修复与代价及原严格门，晋级后才固定第二seed；总体.995/AP>.9及raw.999尚未达到。
