# R238/R239：fit CSV同类关系的表征适配

在R236R2仍fit-only无标签适配、尚未读取其任何dev分数或逐例预测时登记；最近封存实际公开八rank首75。用户只有任务图片/CSV并授权自主决定，不请求其他标注或额外数据。R236R2完整双门过仍优先原固定seed27 R237；本后备只有该优先分支按既定条件处理、目标仍未达且唯一actualsafe后才执行。R236R2未过门则可直接接R238/order153 seed20261029。R239/order154 seed20261027仅在完整R238预定门通过后。

唯一问题：同R236强光度视图/G64P8/均匀fit流/公开Small/MLP/1600预算和全部学习RNG，正对从“同图另一视图”改为“fit CSV同标签的所有非自身视图”，是否改善原完整R107监督排序。新增信号是跨图片的既有任务标签关系，属于**有监督表征适配**，不称无标签自监督；不改CSV标签、不假定标签经专家确认。盲审中标签不确定性仍保留。本实验不扫描温度、强度、预算、采样比例或混合损失，不按dev难例分配fit权重。

采用[SupCon原论文](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)的outside-log正对平均形式，参考[作者实现](https://github.com/HobbitLong/SupContrast/blob/master/losses.py)。128个L2归一化视图按原z1全部64、z2全部64排列；每anchor排除自身，分母仍全部127个非自身视图，正对为同CSV标签的非自身视图；对每个正对的log probability先求均值，再对128 anchor均值。固定temperature=base_temperature=.1，因而额外缩放为1。每图两视图保证至少一个正对。全同类batch也保留，既不跳过/重采样也不重加权；其无跨类对比的事实和实际每batch fit正例数、正对count范围记录为成本/限制。粗二元类关系可能压缩同类内差异或放大原标签噪声，论文不构成本任务收益或临床正确性的证据。

复用ImageDataset现有无标签双视图及返回的indices，不改data.py。只在新recipe构建`fit_indices`到CSV label的表，严格仅访问fit行的label键，要求0/1；每个实际loader index必须在表内，age/dev/official标签不用于适配。64个标签通过原rank顺序gather对齐全局图像，重复为两视图标签；标签无梯度。图像采样、几何和强光度stream、网络和head、optimizer/LR/clip/FP32、102400主图/204800视图/204800 encoder-view forward与R236不变，增加一次小标签gather并如实报告。没有额外图像、teacher、医学权重、定位器或原生裁剪。

仍只传encoder/final-LN，公开head-age和原监督factory RNG恢复、空Adam/EMA重建；监督512/G64P8/448 pairlog、24k成功更新/800评估/patience4和唯一dev AUROC selector不改。完整内部AUROC主判，附AP、原固定20/其余147修复/新增错序、R107/R120/R115/R234/R236R2及原空间适配比较和真实成本。R238晋级须严格胜R107 .9884405731067525、R120 .9894773092062407及完整匹配控制R236R2，AP或局部改善不救主判；R239主判胜固定seed27 R108/R121，若原R237实际完整存在还须胜该匹配控制。不追加一个本来不eligible的R237补考；若缺该控制，不能声称两个seed的“标签mask组件效应”已被隔离复现，只报告整个方法对原监督的配对方向。官方仍完整selected≥.996后一次，目标.995/AP>.9及raw.999尚未达。

必要源码仅`vicreg.py`新增稳定SupCon损失、`ssl_stage.py`新增固定recipe/fit标签表及在原循环选择新损失和真实标签使用回执、`experiment91.py`两身份/原监督合同，两SCI及`tests/test_fit_label_contrastive.py`。原gather/NT-Xent/VICReg/数据/模型/优化器/derived/保存/评估/启动代码保持；`ssl_phase_completed`仅在新recipe核对实际fit标签使用及age未用，旧recipe行为按源码复用。沿现有SSL适配阶段封装，旧文件名不等于无标签语义；新记录必须明确labels_used=True、age_used=False、scope=fit_csv，不把该分支写成旧False标志。

CPU在独立`technical-prep/fit-label-contrastive-20261104-4e0c`，CPU3独占24核0–23/nice19/隐藏GPU，与当前96训练核互斥。先缺行为RED；独立手算loss/梯度、unique-label退化到原NT-Xent、view/row变换、全同类和稳定对角mask边界；fit-only标签访问trap/非法index及新recipe/全部原监督与学习/采样stream；真实fit8双视图与标签对齐、两rank CPU Gloo的global顺序/梯度对照。未变模型/P8/derived和原像素按精确源码复用真实证据，不追加CPU整模型前向或独立GPU预演；自然真实八rank首更新证明新分支消费。源码push/API回读/全树核对和既有正式链完成后才执行，不热改当前R236R2冻结c800/SITE65bf/clock/token/grant，不派生Agent/申请资源或等待例行审批。

- [ ] 必要RED、最小实现及上述窄CPU检查；身份一致的原默认/P8证据复用。
- [ ] 代码/配置同步、精确部署及实际入口在当前训练期间备好；保留R237优先。
- [ ] 完整结果、修复与成本，按原门接续/固定第二seed，不提前声称科学收益。
