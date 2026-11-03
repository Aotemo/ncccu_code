# R232/R233固定EMA历史窗口CPU准备

两正式包已在R230训练期间完成：Git4fea64d/tree dfe968ec已push/API回读，2699文件bytes/mode精确部署；[R232](R232-FORMAL-READY.json)36a9786e、actual SITE b642d573/部署b07d7d19/guard5158fda4；[R233](R233-FORMAL-READY.json)354ead1b、actual SITE36406c3d/部署d7012bf7/guard56d2f29e，均18/18、345全训/空初态50c6或9150。两者均无activation/clock/dispatch，记录时完整clock+60窗口足额，启动前仍新核；[总复核](READY-VERIFICATION.json)guard b0298ca1。技术包沿原prepare.py以同进程exec方式执行，两CPU guard461a377f/0a4f30f3均通过，没有重复模型P8或增加GPU预演。

[24核续建绑定](CPU-ALLOCATION-RENEWAL.json)b3a7bb69实核同一Ready UID、Python3.12.7/依赖、有效120核及原GPU owner/8rank出生身份，独占0–23/24核，probe1a4a1521通过且GPU未初始化；有效终点1793769296.744522。旧7588546c绑定未写，现役GPU的96核/原allocation/clock/token未改；新GPU准备显式引用新CPU验证绑定，仍与训练24–119互斥，无新增Agent或资源。

[事前计划](../../../todo/auroc-999/R232-R233-EMA-HORIZON-20261103.md)以29daea3在R230完整监督结果前同步。固定9600次fit-only，原EMA目标/头/温度/center/动量端点及原监督合同保持；只检验总时长及其必要余弦跨度。不是新增损失或等算力比较，不恢复R230状态，不延长原R115 VICReg。R230完整双门过仍优先R231；否则唯一actualsafe后才走R232，R233仅R232完整双门后。

[10项受影响检查](CPU-CHECKS.json)1bfbb256、8文件编译已通过，guard007625d7/复核6fe53e1d；先[两缺失行为RED](RED.json)b833b70a，再验证两预算LR/教师动量独立算例、真实共享完成计数门拒绝短/错预算/部分/skip/dev记录、候选完整R107合同、旧1600及masked/local候选兼容。原1600教师动量和LR在全部1600整数步逐值精确相同。

62个未改动函数/类的原始源码片段SHA与4099eed一致，包括真实P8使用的模型/归一化头、loss、无梯度教师初态、原型冻结、数据/采样/gather、derived加载和优化器；监督_run在derived元数据门之后的完整函数体也逐字相同。完整未改文件与各片段SHA见回执。原实际Small/P8/冷derived143.844秒、349有限梯度、5空监督组的同一源码范围因而复用，不重复模型前向；新增日程及完成计数只由必要窄目标检验。单模型/最终选点、保存/评估/监控/启动平台未重建。

[Pod代数](EMA-HORIZON-MATH.json)显示1600/9600初始教师直接系数.67026975/.09067713，扣首722冻结后的初始原型系数.91061931/.12989455；它们不测量语义信息、忽略FP32舍入，也不证明收敛或最优预算。无label/dev/官方/预测读取或模型执行。原24核0–23/nice19/隐藏GPU准备与现役96核24–119互斥，当前R230冻结4099eed未改。

此时已完成源码实现、CPU验证、Git新冻结/技术包/正式CPU与精确部署，未activation/clock/dispatch，也未证明收益。既有31500总clock/27900训练hard保持，启动前仍核完整窗口及真实资源；不能以预计耗时替代运行证据。准备完成时[R230只读进度](../../R230/R230-PROGRESS-horizon-formal-ready.json)3625/0skip/c4/bad0、八rank活跃，暂EMA .9864369391/AP .8180667558，无完整selected，尚不选分支。官方仍原门，整体目标尚未达到。
