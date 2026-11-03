# fit-only 局部梯度重建准备

R226/R227固定检验相同公开Small/fit图/首视图/遮挡与1600预算下，局部归一化梯度目标是否优于直接RGB重建。R224原冻结训练继续；只有其完整所选未过双门时才接R226，否则优先固定第二seed R225。R227也仅R226完整所选严格胜R107/R120之后。暂无新GPU、正式clock、完整指标或收益结论。

[12项必要目标](CPU-CHECKS.json)全部通过，5源码编译；新4目标及候选先实际5FAIL，随后8个空间目标、2个候选和2个原SSL交接/采样目标均GREEN。覆盖平面梯度独立oracle、局部对比度归一化、target detach、仅hidden loss/DDP梯度、无hidden像素泄漏、encoder改变/head-age不动，以及四候选固定监督合同和实际随机流控制。CPU汇总SHA `c65bd8af70d11e9e36b481cb25578fbb94a4c9c631cbdffcf55b27b2ad236859`；RED58722a2b、GREEN234c8fa7。未改模型、保存、原监督、推理等9个文件按实际字节匹配复用，不重跑全套。

[真实公开Small/P8](ACTUAL-P8-CPU.json)79.156秒通过，actual guard a6e1df77：344公开张量strict exact/FP32 finite，原50c6初态、8张真实fit图，8×27×64×64预测/target、154块/2464隐藏cell、344有限梯度、encoder改变、head/age保持，冷derived严格交接与5组空AdamW通过。CPU fixture已删除且从未用于正式训练；没有把CPU结果当GPU实测。P8回执SHA `94c27e91f9a694f5a492f32aa758012bbd7e026ea9c7818ad34f6db9b58c497d`；技术site f0d79462、SCI43964fcf/cdc9822a，均未激活。所有计算在现有Pod、nice19/隐藏GPU/0–23独占24核，与R224训练24–119/96核互斥。

固定目标为逐RGB通道3×3 Sobel、9个无符号方向硬bin、8×8 cell内累加幅值、跨9bin L2归一化（eps1e-6），临时768→432/PixelShuffle4头仅预测被遮挡位置的HOG cell，MSE均值。fit-only均匀采样/增强和监督合同保持；不引入额外任务图、临床标注或旧teacher。依据为[MaskFeat](https://openaccess.thecvf.com/content/CVPR2022/html/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.html)，其大规模Transformer/视频与本CNN方案不同；HOG不是视盘定位，也不能恢复512外细节。

未来源码同时修正SSL回执的sampler_seed字段，使消费与记录使用同一seed对象。R224实际数据流已经按R115 sampling域运行，旧receipt字段误按新version域登记；[真实监督交接及erratum](../../R224/R224-SUPERVISED-RECEIPT.json)保留原字段和实测来源，新源码不热改R224或改变其训练数据。

源码/配置与本批CPU证据已push并API回读Git `21dbb113baf03b8553358b79ba13327accb9cd89` / tree5c975e67；[实际2585文件](TRANSFER-VERIFICATION.json)字节/模式全匹配，12目标按相同源SHA复用，未重跑已过forward。[三技术包](TECHNICAL-PACKAGES.json)R22525915979/R226274beb81/R227efaf7413均已生成，实际guard8d3d5ec491299ca0cadf5ca094e8a18bb88414a05f8d1f75fa1c7eefd01e7bee；全树回执f2b2a97b、技术包回执0fad5bcb。三者均无正式clock或GPU，不把seed27准备登记当作实际新seed实验。后续正式CPU/包部署/时窗及唯一启动需各自回执，现役R224冻结9c82不改。

[R225](R225-FORMAL-READY.json)与[R226](R226-FORMAL-READY.json)均已实际正式18/18 CPU、整树精确部署及完整grant时窗通过：[就绪终核](READY-VERIFICATION.json) SHA e801fbfa，原guard44ee923c/64765316、终核b41e1423。actual SITE分别592d397e/1c4e2d58、真实公开初态91507a65/50c6f55c、345全训/空优化器。新formal clock、activation与dispatch均不存在，未启动第二seed或并发GPU。两候选共享同21db源码；R227仍只有技术包。

当前绑定在保留60秒余量时，R225最迟可于11-03 22:11:20 UTC激活、R226于22:10:01；启动前必须重新核完整grant窗口、唯一actualsafe及完整selected门，不把本次时间快照当以后仍有效。R224完整所选严格胜R107/R120则接R225，否则唯一安全终态后R226；若原未启动grant窗口不够，另做未启动阶段绑定，不改任何活跃/旧grant或clock。ready路径分别为`technical-prep/r225-20261103-4e0c-ready`、`r226-20261103-4e0c-ready`，私有`formal-future.py`只组合原prepare/check_candidate/freeze入口，不创建新运行链路。
