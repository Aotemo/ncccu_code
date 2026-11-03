# R228/R229 局部光度一致性准备

在R226完整结果前按[固定计划](../../../todo/auroc-999/R228-R229-FIT-LOCAL-20261103.md)准备。原R115只约束全局池化向量；这条固定算法对同一fit照片的16×16相同位置施加独立原光度增强，再应用原final-LN与同一768→1024→256 MLP、原可微gather/VICReg。全局仍64图/128视图，16384个相关空间token并非独立图像。配对几何是必要变化，不作纯组件因果比较或VICRegL复现。

先在既有Pod独占准备0–23/24核、nice19/隐藏GPU复现两项缺行为，再14必要目标/5源码编译通过：新几何/光度配对、不读label/age、旧默认与首视图像素、8次随机流、局部置换独立oracle、encoder更新/public head-age不变、R228/R229完整R107监督合同、两旧masked候选兼容。RED与GREEN真实guard身份见CPU-CHECKS；测试没有运行在本机或登录节点。

真实公开Small/P8前后向及冷derived已通过：[CPU-CHECKS](CPU-CHECKS.json)828f37f5、[实际P8](ACTUAL-P8-CPU.json)8ea529cf、[guard](CPU-P8-GUARD.json)d3617a1a。8图/16视图、每rank2048×256局部矩阵，346有限梯度、encoder改变/public head-age保持；8/8真实图片的旧21db默认两视图及首视图逐像素一致，冷derived和监督5空组通过。全局16384行的实际分布式gather尚未在GPU证明。109.702秒是P8计算至冷加载前的记录区间，完整task时间见guard；CPU fixture退出删除，不能进入训练。

首P8数学项成功但资源守卫返回2，峰值47 runnable>24（亲和性仍0–23）；随后v2在模型前被原CPU channel的env一致性校验拒绝。两失败回执保留为CPU-FIRST-P8-RESOURCE-FAILED/CPU-RETRY-BOOTSTRAP-FAILED。最终只对CPU数值计算采用12线程池及passive等待，仍在原24核mask/配额内通过原guard，无规则放松、训练资源改动或信号。

Gitbe5b5fa8/tree60d7432b已push/API回读，Pod[2622文件全树bytes/mode核验](TRANSFER-VERIFICATION.json)及[R228/R229两技术包](TECHNICAL-PACKAGES.json)通过，源码按已通过14目标SHA复用。R228[正式18/18及精确部署](R228-FORMAL-READY.json)通过，actual SITE84b5108e、部署81c0b276，正式guard8fd5045e；ready=`r228-20261103-4e0c-ready`。公开初态50c6、345全训/5空组；无GPU预演或正式clock。实际分布式几何和首成功仍待未来自然更新证明。

同时将原21db HOG的条件seed27 [R227正式18/18](R227-FORMAL-READY.json)预先准备，91507a65/actual SITE9a890d5b/部署106a40e9、guard30d9f4b5；ready=`r227-20261103-4e0c-ready`。两分支均无activation/clock/dispatch，full clock+60秒窗口在[汇总](READY-VERIFICATION.json)时间点均足额，真实启动前重核，旧grant不改。R229仅technical，不能因CPU准备等同于晋级。

准备检查不构成收益。R229仅R228完整selected严格超过R107和R120后才可激活；R226双门通过时优先原R227。当前活跃冻结21db源码不改，没有额外任务图片、teacher或新资源。完整CPU P8 task128.253秒，最终峰值24 runnable/24核、亲和性无越界；首失败与v2拒绝原件保留。

后续实际：R226完整未过双门，R227不eligible/未激活；[R228已实际首25并到675适配更新](../../R228/README.md)，冻结be5b5fa8不热改。一次activation、首次CLI派发前mask拒绝、修正后唯一实际dispatch；完整入口失败/357.314秒到首SSL的间隙保留，旧clock/token/grant保持。21:21 [R229正式18/18/精确部署](R229-FORMAL-READY.json)也通过，公开91507a65、actual SITEfea174de、部署4f1fc7e9、[guard353905d3](R229-FORMAL-CPU-GUARD.json)，无activation/clock/dispatch。此前READY-VERIFICATION只代表20:25原两分支快照，不改历史回执；R229仍须R228完整晋级及启动前新窗口核验。
