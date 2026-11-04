# R240/R241：固定fit标签类配额是否改善同类关系适配

在R238尚未激活、尚无任何R238 dev或逐例预测时登记。最近R236R2监督6550/c8/bad0，原R237/R238优先分支和冻结包不变；这不是根据R238结果追选参数。用户仅有图片/CSV并授权自主选择，不要求其他数据、临床标签复核或新Agent。

唯一问题：在R238同一SupCon损失、强光度/双视图、公开Small/MLP、1600次/G64/每rank8及学习RNG下，将均匀fit图像流换为固定每global batch 8个CSV正例和56个负例，是否改善完整原监督排序。新增的是fit类关系的采样覆盖；不是额外数据、改标、难例加权或新的损失/优化器/预算。fit既有46256图/1490正例，均匀流不能保证每batch同时含两类；这里只检验原监督已用的固定8/64配额，不扫描配额或温度。SupCon公式仍见[原论文](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)及[作者实现](https://github.com/HobbitLong/SupContrast/blob/master/losses.py)，不从论文推断本任务收益。

沿已有`RankBatchSampler`/`BalancedBatchSampler`的fit两类独立排列流及原两reference microbatch→八rank连续分片，reference physical32/effective64/positive8、start0/1600更新。使用R238同一已派生sampler seed；只取该既有sampler输出的indices，再用R238原`_seed(seed,'sample',update,global_position,index)`生成每draw种子，保留其增强种子公式和全部学习/loader RNG域。不同indices导致的像素/随机种子变化是采样处理的组成部分，不称实际图像流或逐draw像素与R238相同。类内按完整permutation流循环，少数类重复暴露如实计成本，不改变CSV、不读取dev/age/official标签，也不为所有rank增加独立采样。

固定每更新64图/128视图，128个anchor、全部127个非自身视图分母；正例anchor各15同类/112异类，负例anchor各111同类/16异类，均有跨类竞争。1600更新固定102400主图、204800视图/encoder forward，其中12800正例主图draw和89600负例主图draw；增加重复正例暴露是实验代价，不声称独立正例数增加。R238均匀流实际正例draw/全同类batch数仅在其完整适配后从既有训练metrics汇总，不用于选择配额。新结果需显式记录采样策略、quota、实际global正例数与标签使用True/ageFalse/label_scope=fit_csv。

仍fresh合规公开自然图像初始化，单模型、固定9:1。只传encoder/final-LN，公开head-age/原监督factory RNG恢复、空Adam/EMA重建；原512/G64P8/448 pairlog监督、最多24000成功/每800 dev/patience4、唯一完整dev AUROC选点与官方selected≥.996后一次保持。没有医学权重/teacher、队列、类别权重、重采样dev、原生裁剪或新增训练/保存/评估/启动/监控链。

R236R2完整过原门仍先R237，否则R238；R238过原门仍先R239。这些原优先分支按合同处理且目标仍未达、唯一actualsafe后，R240/order155 seed20261029才可执行。R240晋级必须严格胜R107 .9884405731067525、R120 .9894773092062407、完整匹配R236R2和R238；完整AUROC主判，AP与固定20/其余147修复、新增错序及真实成本分列。R241/order156 seed20261027只在R240过全部门后，主判严格胜R108/R121，且R237/R239如实际完整存在也须胜；缺失的匹配控制不追加一个已不eligible的旧分支补考，缺R239时不能声称两个seed已隔离复现“quota组件效应”，只能报告整个方法对原监督的配对方向。内部.995/AP>.9及总体raw.999目标均未达到。

必要源码范围仅`runtime/bundle/retrain/ssl_stage.py`增加固定quota recipe/复用原rank sampler的短包装及原循环单分支、`experiment91.py`两身份/原监督合同，两SCI和`tests/test_fit_label_quota.py`。`vicreg.py`/原UniformFitSampler、fit标签表/gather/损失、像素处理、模型/optimizer/derived/原监督/保存/评估/官方均按相同源码身份复用。不要因新实验复制链路。

CPU3独占24核0–23、nice19/隐藏GPU，和96训练核互斥。只做新行为两个必要目标：缺行为RED→八rank索引并集/类quota/原reference顺序与增强seed oracle、重复迭代和跨类排列边界；新recipe/两SCI与原完整监督/学习域一致、旧recipe不变。真实fit双视图/标签对齐与实际配额、128anchor计数做窄验；SupCon损失、原gather、两rank梯度/模型P8无变化则复用已通过证据，不重跑模型或GPU预演。Git同步/API回读/全树与既有正式链、真实自然八rank消费后才能称可运行；技术包可提前，正式窗口在R238监督期间准备。

- [ ] 缺行为RED、最小实现、两必要目标与实际fit消费；未变路径复用。
- [ ] 源码/config同步、技术包，按当前运行进度准备原正式链和具体入口。
- [ ] 完整结果、修复/新增和代价，按预定门接续；不提前声称科学收益。
