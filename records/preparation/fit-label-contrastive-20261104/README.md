# fit CSV同类关系准备与CPU证据

当前进展：R236R2完整双门false，R237不eligible；用户R120提交与往返SHA已交付。R238已一次激活并[实际公开八rank/自然首50](../../R238/README.md)通过467ae32d/guard3ec3b722；原3bd/SITE4bb2冻结，首unknown保留，完整1600/监督及科学结果仍待。旧R236R2 safe不重查；R239/R240正式CPU/部署按计划在监督期间完成。下文的“未激活”“当前R236R2”为当时准备快照。

[事前合同](../../../todo/auroc-999/R238-R239-FIT-LABEL-CONTRASTIVE-20261104.md)b489f61已push/API回读，登记在R236R2任何dev读取前。新信号仅为同fit CSV标签的所有非自身视图作为正对；属于有监督表征适配，不声称无标签SSL或标签经专家确认。R236R2完整过门仍优先原R237；新R238还须胜完整匹配R236R2及R107/R120，才资格固定R239，原监督和官方门保持。

[CPU检查](CPU-CHECKS.json)d992863f：两缺行为RED9b6ec1bc；最小3文件实现后两目标/四编译7e07dc7c通过。独立标量手算和双视图梯度、unique-label退化原NT-Xent、view/row/尺度不变及全同类/对角mask稳定；fit/dev/age访问trap、非法indices/标签、新recipe/原监督全job/学习和采样stream、旧recipe及新完成标志验证通过。真实fit8×2×3×512×512与原R236同像素SHA2aebe5cd，46,256 fit标签/1,490正例、实际图片indices到标签一致，age未用；模型/数据/优化器/derived等原函数范围及12整实现文件相同，原实际P893ac00ab按源身份复用，没有新CPU模型前向或GPU预演。

首次整合数值和两rank Gloo均通过，但整树资源guard2f3a8d9b为exit2：runnable峰值48>24，mask仍0–23、无亲和性越界/信号，child exit0。保留该失败，不当整体CPU通过；原因是两个spawn子进程在导入库时继承了完整24线程环境。只为失败集合通信路径提前设置每rank4线程并加60秒初始化超时，复用未变数值/fit/源证据，不重跑这些路径。重验f3fcced0 exit0、runnable峰值8/24、mask无越界；两个rank真实64×256/128×128，标签和embedding rank顺序相同、全局loss等于串行、gather梯度SUM与DDP均值抵消匹配，最大梯度误差0。CPU证据明确world2，真实八rank消费仍待自然训练证明。

R238/R239两SCI为2527b844/1d934cdb；源码/config已3bd21524同步并API回读，tree36eb332e。[2845全树核验](TRANSFER-VERIFICATION.json)89e64c42逐项字节与权限一致。[R238正式部署](R238-FORMAL-READY.json)沿现有链完成18/18（guardb2781bcb），公开50c6f55c/345全训/空Adam，SITE4bb2a3dd、部署edefb072，未激活；31560秒整窗满足，最晚activation epoch1793783692.3138187。[R239技术包](R239-TECHNICAL-READY.json)仅由现有prepare生成、未做正式CPU/绑定或启动。[归档](FORMAL-EXPORT-RECEIPT.json)对应私有helper和具体身份，导出guard43e30d3f；可变CPU snapshot不作为部署。

私有范围为`fit-label-contrastive-20261104-4e0c/`中的当前CPU、正式打包和metadata helper；现有通用测试runner、prepare/check_candidate/freeze及observer复用，未新建训练、保存、评估、启动或监控平台。主Agent自审loss目标、对角mask、gather、fit-only边界、RNG/原监督和完成证据；按用户要求无子Agent或独立审阅者。[接续helper](HANDOFF-HELPERS-CPU.json)及guardfa1e4cdc：七文件编译、65种实际类计数公式与五baseline设置通过，不读取未来结果。标签版自然八rank和监督交接观察显式要求fit标签True/ageFalse；旧R2观察不变。

后续字段自审发现上述最初监督观察helper0569a32b错误读取`loss_label_scope`，而已发布训练producer和完成门实际使用`label_scope`；最初helper仅编译过、尚未观察任何R238运行，不将编译说成字段对接通过。原件保留，当前应使用私有`OBSERVE-FIT-LABEL-SUPERVISED-v2.py`/a4c00b39。[实际producer字段对接](OBSERVER-LABEL-FIELDS-CPU.json)/guard03c77e6d复现旧缺字段KeyError、直接提取冻结producer的真实字段，v2接受fit_csv/标签True/ageFalse，拒绝四错标志/范围和旧拼写；新helper编译通过，无模型/GPU或训练源码/冻结包修改。

[R237正式包](R237-FORMAL-READY.json)已沿c800原源码完成18/18、公开91507a65/345全训/空Adam，SITEaf2b092b/部署15dc4f71/guarde8b949d4，未激活；31560秒完整窗口满足，最晚activation epoch1793782523.271514。两分支实际入口已填私有post-r236r2-R237-entry.sh/post-r236r2-R238-entry.sh及会话store，均已Pod内bash -n；在原window/Pod快照/activation/launch之前调用同一已发布verify_storage做真实容量预检，不消耗activation身份。完整R236R2双门/唯一safe后才选择；若延续至不足整窗，须为未启动阶段另建ready，不改旧grant。

R238正式准备时实际free20,510,146,560小于原20 GiB准入，未尝试activation。已按[两个精确清单](../../storage/obsolete-CPU-fixtures-20261104/README.md)处理15个历史CPU-only fixture；当前R2、R237/R238和三KEEP保护，数据/缓存/公开初始化不动。共享文件系统free是瞬时观测，启动前仍须实测；当前R2终态归档及必要闭轮清理helper已在训练期间准备、六文件编译通过，完整selected/唯一safe前不读取未来结果或删除活跃checkpoint。

当时唯一R236R2 [监督2450/c3/bad0](../../R236R2/R236R2-PROGRESS-fit-label-checks-ready.json)，原owner845023及八rank活跃，临时EMA .9799781093/AP .8088639386，无完整selected。完整1600及原监督交接已0d73d0a7/guardec0dee92封存；当前c800/SITE65bf/clock/token/grant不改。准备CPU3独占24核0–23/nice19/隐藏GPU，与训练96核24–119互斥；无额外任务数据、医学权重/旧teacher/病例改标或新计算资源。目标.995/AP>.9及raw.999尚未达到。
