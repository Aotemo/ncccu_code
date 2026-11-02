# R177/R178：固定联合目标 CPU 准备

[真实CPU证据](CPU-CHECKS.json)：本次两项loss目标及一项候选身份目标，共 **3项通过、5源编译通过**。8×56独立标量softplus/BCE及解析逐logit梯度、平移后BCE变化与共享bias梯度、极端logit有限，以及旧None/logistic/sigmoid/hard-positive/margin五分支已核；真实两rank WORLD/DDP使用不同特征和age，一个rank零正例，实际global-logit、分类头weight/bias及age梯度与独立reference一致，无额外world比例。仅CPU小fixture，不是DCU峰值/吞吐、生产八rank或AUROC收益证明。

R178直接R107原公开fe6b/Small/seed29/512/G64P8及原优化/EMA/age合同，目标固定 `.5*mean448 softplus(z_n-z_p)+.5*mean64 BCEWithLogits(z,y)`，只复用一次全局logits/gather，不额外forward/模型/loss参数；系数不扫，未叠加当前DINOv3。逐图项基于采样P8/N56，不等同总体患病率校准。R177仅完整R176同时strict胜R107/R120且safe后的固定DINOv3 seed27复核；[精确合同及范围](../../../todo/auroc-999/R177-R178-PREPARATION.md)另列。

[真实两public fresh](PUBLIC-FRESH-SUMMARY.json)：R177完整342无头公开骨干、whole `5f2abba3302926dc883ab8c32db02cd7d643c5b745428295e0ae006183c276aa`；R178完整344 public1000、whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`与R107一致。两者实际345全训/5空Adam组、betas `.9,.999`，真实来源/SCI/fit-age元数据绑定未mock；0图像forward/0GPU/无旧训练checkpoint或官方读取。models/config/fresh/engine/distributed/EMA/persistence/inference八实现SHA未改，模型运行及冷图证据按相同源码复用。

六份原CPU任务的SHA、exit、OUTPUT SHA、任务树退出及预算/affinity均已核；首次在传输未完成时调用目标，文件缺失exit2且未执行测试，保留原件。随后实际RED为loss未实现，1fail/1error；最小实现后GREEN2/2，候选1/1再完成。未覆盖失败或宣称firsttry。原件位于`exp-v3/technical-prep/pair-bce-mix-20261102-32e3/`，归档guard `c99ff08418a29603b39878c40dad6c04ba4f9c277978eb166cc7bd96fc6af63b` exit0。

准备719a/root0–11/12核、peer保留12–23/12核，与当前R176训练24–119/96核互斥，nice19/隐藏GPU。Git `313d5abeb57240159091b682c3948aec0244b310`/tree `1dd0d3cb97e76a4224bfeb12bd35c9e255cc9bd8`已push/实际远端回读；[R177技术包](R177-TECHNICAL.json)、[R178技术包](R178-TECHNICAL.json)实际源文件/SCI/原CPU任务与输出SHA均匹配，guard0。尚无R177/R178正常formal/activation/GPU；收尾前正常CPU/精确部署后再按完整R176/actualsafe决定唯一接续，raw .999未达。当前R176运行仍冻结e9a7f1d。
