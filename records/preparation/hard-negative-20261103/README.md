# R204/R205：固定困难负例四分之一混合 CPU 证据

[最终CPU汇总](CPU-CHECKS.json) SHA `1b7cd94133adc0c6c82e227a9d918c8a404cdcf9edd346a6112513cd515b40c0`：6个受影响目标通过、5份源码编译、两次实际公开fresh；[最终guardtvbimxcy](CPU-FINAL-GUARD.json) / `c25f17a6981bf38f866423757fcad6c66976c8373bc605731080fb210a1cceb3` exit0。实核Pod/ec5/root0–11、nice19/隐藏CUDA-HIP-ROCR，与当前R201训练24–119互斥；无新Agent、计算资源、公开模型图像forward或GPU预演。Git冻结和两technical包随后按实际结果登记，此处不冒充已部署或已有收益。

[三个数学目标](MATH-GREEN.json)覆盖真实8×56独立Python标量softplus/逐logit解析梯度、共同平移/共享bias、全同分与第14名边界同分、有限极端值和严格九字段；旧None/logistic/sigmoid/wide值与梯度bitwise保持。实际2rank CPU Gloo可微WORLD gather+DDP，局部正例[0,8]，distinct/alltie/boundary各每rank一次Toy Linear，共3次；真实保留global/local logits.grad，分别对照g和2g，DDP参数梯度对独立X^T*g。零空间不同特征使同分选择能由实际非零参数梯度区分，实际梯度识别的alltie/boundary负例index均0–13，未把参考index当运行观测。共享bias梯度容差2e-7，alltie实值约1.49e-7，不声称FP32数值恰为0。

[三个候选/存储目标](BINDING-GREEN.json)核R204仅原R203的seed27、R205原R107 job仅loss，旧R202/R203兼容；来源、9:1、结构、24k/800/p4/EMA、保存job和非法叠加均检查。历史R107没有现显式official_evaluation，R205采用当前R120的`.996`一次官方门；未封存草稿缺字段在首次候选GREEN前修正，原草稿与guard99biyzjc保留，原R107不改。R204 SCI f079cdd6、R205 SCI7aaa297d。存储只加新精确R204短round16GiB，R205/default12GiB、旧16/32GiB及free16GiB不变；测试调用原guard的size/free fixture，没有实际写大文件。

[两次实际公开fresh](PUBLIC-FRESH-SUMMARY.json)：R204 native ViT-B/DINOv3 publicc6a/seed27，whole6a261e48、165全训，原非持久RoPE单buffer；R205 Small/publicfe6/seed29，whole50c6、345全训/无buffer。均5空原生AdamW、default betas(.9,.999)/eps1e-8；R204骨干1e-5、R205骨干1e-4，head/age均1e-3，WD均`[0,1e-4,0,1e-4,1e-4]`，0图像forward、无训练checkpoint/官方读取。原model/config/engine/data/distributed/EMA/保存/推理按最近精确源码证据复用；新classification_loss移除仅新增kind分支后的两个函数AST与8635原版一致，gather/regularized及output helpers逐字符保持。

最小运行实现只新增PAIR_HARD_NEGATIVE_LOSS独立身份及固定`0.5*mean_all_negative+0.5*mean_top14_negative`，逐负例项为8正例softplus均值，stable降序按WORLD负例原顺序破同分；全部448对保留base梯度，112对多获梯度，无额外参数、状态、forward、teacher或dev名单。原engine/config通过既有PAIRWISE_LOSS_KINDS消费，不另加world因子。有限batch尾部混合仅是独立假设，不是SOPA/LibAUC字面复现或总体无偏估计，可能放大含噪阴性；完整内部AUROC仍为唯一选点，不是pAUC或临床机制证明。

保留全部失败原件：[数学RED](MATH-RED.json)guardck6n26pu为新kind未登记；[候选/存储RED](BINDING-RED.json)guardj8yjecdx为新profile/override未登记且旧候选PASS；对应GREEN guardzpnkrdw5/jy075324均0。最终汇总的前两次失败7um30lni/evnjyiy8分别是把预定新增storage误放未变列表、误写official字段名，均仅私有汇总脚本问题；没有因此修改运行源码或重复数学/fresh测试，原guard/OUTPUT SHA纳入最终汇总核验。私有原件在`exp-v3/technical-prep/hard-negative-20261103-32e3`，全部整树退出/无affinity违规。当前R201/8635与已准备R203/573保持冻结，raw .999未达。
