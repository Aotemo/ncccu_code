# 固定SSL强光度组合的CPU准备

[事前合同](../../../todo/auroc-999/R236-R237-FIT-STRONG-COLOR-20261104.md)8f10564已push/API回读，登记时未查看任何R234 dev分数。只在fit适配采用固定jitter .8/gray .2、三个乘性[.2,1.8]及hue[-.2,.2]，原几何和完整监督保持，1600预算不延长，不扫描参数、加入定位/额外图像或医学权重。不是逐位TensorFlow或完整SimCLR复现，不假定全部疾病征象保持；R234完整双门过仍优先R235，否则唯一safe后才考虑R236。

[CPU汇总](CPU-CHECKS.json)d1d46cd1：两必要新目标/4编译c97bfa75；48原None/identity/gamma像素案例、16原八draw几何RNG、39未改函数/类及12整文件一致，原SSL完整循环除声明注入外逐字相同。独立恒定像素oracle核灰度和乘性亮度、全局Python/numpy/torch RNG隔离、同seed确定性；强光度在有标签/eval/共享几何调用中拒绝，两监督job与原R107（仅固定seed）完全相同。真实fit8双视图经原DataLoader为8×2×3×512×512且有限，未返回标签/age，实际政策及tensor SHA2aebe5cd已记录。

原真实公开局部P893ac00ab和R234自然八rank监督f618327b按相同模型/loss/gather/optimizer/derived源码复用，没有重复模型前向、冷derived或GPU预演。首次RED abc29247两目标均明确缺helper/recipe，随后CPU/source复用a5b11a8e通过；[准备回执](PREPARATION-RECEIPT.json)及导出guardc161500d仅小JSON，无图像、逐样本预测或权重出Pod。主Agent自审光度范围、原八draw及监督边界，按用户要求未派生Agent，无独立审查者。

可变technical fixture仅CPU消费，不是正式部署；R236/R237两SCI为266beb7f/372407cb，发布源全树、技术包及R236正式CPU/部署待。当前唯一[R234监督4175](../../R234/R234-PROGRESS-strong-color-checks-ready.json)/c5/bad0、原owner及八rank活跃，临时EMA .9844971228/AP .8149208204，无完整selected。准备24核0–23/nice19/隐藏GPU与训练96核24–119互斥，当前565/SITE c5ad/clock/token/grant不改；.995/AP>.9及raw.999未达。
