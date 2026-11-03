# R228/R229 局部光度一致性准备

在R226完整结果前按[固定计划](../../../todo/auroc-999/R228-R229-FIT-LOCAL-20261103.md)准备。原R115只约束全局池化向量；这条固定算法对同一fit照片的16×16相同位置施加独立原光度增强，再应用原final-LN与同一768→1024→256 MLP、原可微gather/VICReg。全局仍64图/128视图，16384个相关空间token并非独立图像。配对几何是必要变化，不作纯组件因果比较或VICRegL复现。

先在既有Pod独占准备0–23/24核、nice19/隐藏GPU复现两项缺行为，再14必要目标/5源码编译通过：新几何/光度配对、不读label/age、旧默认与首视图像素、8次随机流、局部置换独立oracle、encoder更新/public head-age不变、R228/R229完整R107监督合同、两旧masked候选兼容。RED与GREEN真实guard身份见CPU-CHECKS；测试没有运行在本机或登录节点。

真实公开Small/P8前后向及冷derived已通过：[CPU-CHECKS](CPU-CHECKS.json)828f37f5、[实际P8](ACTUAL-P8-CPU.json)8ea529cf、[guard](CPU-P8-GUARD.json)d3617a1a。8图/16视图、每rank2048×256局部矩阵，346有限梯度、encoder改变/public head-age保持；8/8真实图片的旧21db默认两视图及首视图逐像素一致，冷derived和监督5空组通过。全局16384行的实际分布式gather尚未在GPU证明。109.702秒是P8计算至冷加载前的记录区间，完整task时间见guard；CPU fixture退出删除，不能进入训练。

首P8数学项成功但资源守卫返回2，峰值47 runnable>24（亲和性仍0–23）；随后v2在模型前被原CPU channel的env一致性校验拒绝。两失败回执保留为CPU-FIRST-P8-RESOURCE-FAILED/CPU-RETRY-BOOTSTRAP-FAILED。最终只对CPU数值计算采用12线程池及passive等待，仍在原24核mask/配额内通过原guard，无规则放松、训练资源改动或信号。

Git同步、正式包和实际GPU证据分别记录；准备检查不构成收益。R229仅R228完整selected严格超过R107和R120后才可激活；R226双门通过时优先原R227。当前活跃冻结21db源码不改，没有额外任务图片、teacher或新资源。
