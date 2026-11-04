# fit CSV同类关系准备与CPU证据

[事前合同](../../../todo/auroc-999/R238-R239-FIT-LABEL-CONTRASTIVE-20261104.md)b489f61已push/API回读，登记在R236R2任何dev读取前。新信号仅为同fit CSV标签的所有非自身视图作为正对；属于有监督表征适配，不声称无标签SSL或标签经专家确认。R236R2完整过门仍优先原R237；新R238还须胜完整匹配R236R2及R107/R120，才资格固定R239，原监督和官方门保持。

[CPU检查](CPU-CHECKS.json)d992863f：两缺行为RED9b6ec1bc；最小3文件实现后两目标/四编译7e07dc7c通过。独立标量手算和双视图梯度、unique-label退化原NT-Xent、view/row/尺度不变及全同类/对角mask稳定；fit/dev/age访问trap、非法indices/标签、新recipe/原监督全job/学习和采样stream、旧recipe及新完成标志验证通过。真实fit8×2×3×512×512与原R236同像素SHA2aebe5cd，46,256 fit标签/1,490正例、实际图片indices到标签一致，age未用；模型/数据/优化器/derived等原函数范围及12整实现文件相同，原实际P893ac00ab按源身份复用，没有新CPU模型前向或GPU预演。

首次整合数值和两rank Gloo均通过，但整树资源guard2f3a8d9b为exit2：runnable峰值48>24，mask仍0–23、无亲和性越界/信号，child exit0。保留该失败，不当整体CPU通过；原因是两个spawn子进程在导入库时继承了完整24线程环境。只为失败集合通信路径提前设置每rank4线程并加60秒初始化超时，复用未变数值/fit/源证据，不重跑这些路径。重验f3fcced0 exit0、runnable峰值8/24、mask无越界；两个rank真实64×256/128×128，标签和embedding rank顺序相同、全局loss等于串行、gather梯度SUM与DDP均值抵消匹配，最大梯度误差0。CPU证据明确world2，真实八rank消费仍待自然训练证明。

R238/R239两SCI为2527b844/1d934cdb，源码/config发布与全树/正式部署待当前提交。可变CPU snapshot不作为正式部署；私有范围为`fit-label-contrastive-20261104-4e0c/{prepare-fit-label-cpu.py,finish-fit-label-cpu.py,recheck-fit-label-gloo.py,export-fit-label-preparation.py}`及小metadata，现有通用测试runner复用，未新建训练、保存、评估、启动或监控平台。主Agent自审loss目标、对角mask、gather、fit-only边界、RNG/原监督和完成证据；按用户要求无子Agent或独立审阅者。

[R237正式包](R237-FORMAL-READY.json)已沿c800原源码完成18/18、公开91507a65/345全训/空Adam，SITEaf2b092b/部署15dc4f71/guarde8b949d4，未激活；31560秒完整窗口满足，最晚activation epoch1793782523.271514。原四步入口已填私有post-r236r2-R237-entry.sh及会话store，完整R236R2双门/唯一safe后才执行；若延续至不足整窗，须为未启动阶段另建ready，不改旧grant。

当时唯一R236R2 [监督2450/c3/bad0](../../R236R2/R236R2-PROGRESS-fit-label-checks-ready.json)，原owner845023及八rank活跃，临时EMA .9799781093/AP .8088639386，无完整selected。完整1600及原监督交接已0d73d0a7/guardec0dee92封存；当前c800/SITE65bf/clock/token/grant不改。准备CPU3独占24核0–23/nice19/隐藏GPU，与训练96核24–119互斥；无额外任务数据、医学权重/旧teacher/病例改标或新计算资源。目标.995/AP>.9及raw.999尚未达到。
