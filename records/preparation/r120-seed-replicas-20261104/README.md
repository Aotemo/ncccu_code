# R120独立seed复制准备

R244 seed20261104、R245 seed20261105在当前C4两轮监督期间登记，只改变原R120的seed及相应fresh随机身份，其余完整job/公开源/结构/age/512/FP32/G64P8/损失与LR/EMA/24k/800/patience4不变。原seed27若完整过门仍优先；不加载任何旧训练权重作初始化。

RED b3089119因缺两个新身份明确失败；[两目标与两文件编译](GREEN-CPU.json)/guard3205fd06通过，75个其余runtime文件与C4逐字节相同。旧R120/R121身份仍精确通过。两个新seed已在各自当前Pod使用原production CPU factory实际生成：[R244](R244-INITIAL-CPU.json)/69134c69的fresh tensor55b1ab41，[R245](R245-INITIAL-CPU.json)/5de0e7c2为99a8cb39；没有图像forward/GPU初始化。生成配置中只增加各自当前UID公开资产回执，provisional包绝不作正式准入。

源码范围仅experiment91注册与check_candidate原sigmoid检查扩展，新两SCI与一个窄测试。正式18/18、全Git精确部署、完整时窗及一次activation仍待后续原链；本准备不等于GPU已启动。源/配置发布后按真实文件SHA进入下一阶段，当前R240R1/R242冻结不改。
