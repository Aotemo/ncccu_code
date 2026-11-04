# R242/R243：保留 fit 标签0图片的实例差异

2026-11-04 08:08 UTC 登记：唯一 R238 仍为适配阶段，最近实际 950/1600；尚未读取 R238 的任何内部 dev 分数或逐例预测。R239/R240/R241 已登记的分支与优先级保持；本候选只在前述分支按合同处理后、目标仍未达、唯一安全终态时执行。用户仅有图片/CSV、授权自主决定和独占24核准备，不请求新标注、其他任务数据、额外资源或 Agent。

唯一问题：R240 的 fit 同类 SupCon 同时吸引所有标签0图片，是否把标签0中有用的实例差异过度压缩。相同的固定8正/56负、相同像素/RNG与1600预算下，仅改表征 loss 的正对集合：标签1 anchor 保留所有同标签非自身视图，标签0 anchor 仅以同图另一独立视图为正对。所有127个非自身视图仍进入分母，包括其他标签0图片；均值仍按128个anchor等权，不改变类权、采样、温度或预算。这是待检验的解释，不能根据当前loss确认表征已经坍缩。

依据为 Mildenberger 等 [CVPR 2025 原论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Mildenberger_A_Tale_of_Two_Classes_Adapting_Supervised_Contrastive_Learning_to_CVPR_2025_paper.pdf) §4.2 的 Supervised Minority，以及[作者实现](https://github.com/aiforvision/TTC/blob/main/loss.py)中多数类监督比例0的确定性mask语义。论文的其他数据集结果不作为本项目收益证据。只引入该正对规则；不下载其数据/权重/依赖，不复刻其 Lightning/Hydra/双分支训练。原代码在比例0时仍调用randperm；本实验直接构造等价确定性mask，不消费额外 RNG，保留R240已固定的学习与采样stream。不会扫描比例、原型、温度或其他loss权重。

R242/order157 seed20261029；R243/order158 seed20261027。recipe固定 `fit-label-supminquota8-1600-v1`，loss `supervised_minority_out`，`positive_policy=minority_same_label_majority_same_image`、`minority_label=1`；继承R240现有 `fit-class-quota-g64-p8-v1` / global_positive_quota8、公开Small、768→1024→256 projector、.1温度、强光度双512视图、FP32、LR/clip/1600预算及R115原学习/采样域。1是fit CSV中1490/46256的少数标签，0有44766；不以dev/official统计定义少数类。标签仅访问fit行，无age/dev/official/审阅结果参与loss，不改变原标签或假定其经临床确认。

每批64主图/128视图、128×128矩阵/自身排除，标签1有16 anchor，每个15正对/112负视图；标签0有112 anchor，每个1正对/126负视图。相较R240，相同计算/暴露预算与小标签gather，消除标签0跨图片吸引，同时把其他标签0实例用作对比负例；它也可能妨碍正确的类内语义或放大图像偶然特征，须完整报告代价。102400主图draw/204800 encoder-view forward、0 teacher、12800正draw/89600负draw及相同fit覆盖按源码身份复用，不重复生成实际相同的完整采样验收；真实八rank自然首更新验证新mask实际消费。

仍只传encoder/final-LN，原公开head-age/监督factory RNG、345全训/5空Adam、原监督512/G64P8/448 pairlog、最多24000成功更新/每800 dev/连续4轮无提升早停、内部AUROC唯一selector不改。完整AUROC主判，附AP、固定20/其余147及整体修复/新增错序、R107/R120/R115/R236R2/R238/R240配对比较、阶段计算/时间/暴露代价。完整R242严格胜R107、R120、完整R236R2、R238和匹配R240才有R243资格；R243须胜R108/R121，并胜实际上已完整存在的匹配seed27控制（R237/R239/R241）。不为缺失或不eligible控制补跑；缺匹配seed27时只报告整个方法对原监督方向，不能声称mask组件两seed隔离复现。官方仍selected≥.996后一次；目标AUROC≥.995/AP>.9及总体raw.999未达到。

精确源码范围：`runtime/bundle/retrain/vicreg.py`在原稳定SupCon函数增加固定正对policy分支，默认旧路径保持；`ssl_stage.py`新recipe/原循环选择及真实统计/完成回执；`experiment91.py`两身份与原监督合同；两SCI和必要 `tests/test_fit_supervised_minority.py`。不改data/model/optimizer/gather/sampler/derived/保存/评估/启动链，不热改当前R238的3bd冻结包或R240的f22冻结源。私有准备根为 `exp-v3/technical-prep/fit-supmin-20261104-4e0c`，不会另建长期训练或监控平台。

CPU使用当前核验CPU3、nice19/隐藏GPU/0–23核，与GPU训练24–119核互斥。先缺行为RED，再独立标量loss/梯度、二元同类与异实例mask/64全局真实计数、全0退化NT-Xent/全1退化SupCon、旧默认函数数学和view/row变换；新recipe/完成字段及原完整监督身份、fit-only/age边界按受影响路径窄验。标签表/原quota完整1600流/实际fit像素/模型P8按未变源码范围复用；新loss的两rank CPU Gloo须在预导入每rank4线程后比较全局串行loss及gather/DDP梯度缩放，不增加CPU模型前向或独立GPU预演。源push/API回读/精确全树/现有正式CPU部署后才可自然GPU执行。

- [x] 9efea4c1事前登记/API回读；三runtime最小实现后两目标/四编译d56acae9及新两rank Gloo ff404197通过，原quota流/像素/P8按未变源码复用，无模型前向/GPU；[CPU汇总](../../records/preparation/fit-supervised-minority-20261104/README.md)3a8b34b3。实际源同步/technical/正式部署仍待，本提交仅发布准备。
- [ ] 源码同步后精确全树及原链部署，前述分支保持优先。
- [ ] 完整结果及固定第二seed门，不提前声称科学收益。
