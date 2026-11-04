# fit Supervised Minority 准备

[事前合同](../../../todo/auroc-999/R242-R243-FIT-SUPERVISED-MINORITY-20261104.md)9efea4c1已推送/API回读，登记在R238适配阶段、未读取其dev之前。仅改R240的正对集合：标签1仍同类SupCon，标签0只吸引同图另一视图；全部127个非自身视图仍在分母、128 anchor等权。固定8正/56负、全部像素/增强和学习/采样RNG、1600预算及原完整监督合同保持。目的是检验保留标签0实例差异的信号，不确认当前表征坍缩或临床标签正确。

[CPU汇总](CPU-CHECKS.json)3a8b34b3：两缺行为RED472f4069后，最小三个runtime文件实现；两个目标、四编译GREENd56acae9通过。独立逐anchor标量loss与梯度、旧默认SupCon数学、全0退化NT-Xent/全1退化SupCon、view/row变换、真实64全局的正对1/15及负视图112/126；新recipe/完成实际字段和错误旧mask、两完整原监督job、原结构/公开源/数据与RNG相同均通过。

新loss的两rank CPU Gloo ff404197通过：预导入每rank4线程，64×256、128×128、8个标签1且两rank各4个，标签和embedding rank顺序一致；全局loss匹配串行，gather梯度SUM与DDP均值抵消后最大误差0。不是原生八rank训练证明。实际producer阶段flags为SupMin/label/quota/contrastive/strong-color，唯一loss调用的固定policy对接也已从实际源码核实。

38个原函数/类范围和13个整实现文件相同；quota采样器、标签表/gather、原NT-Xent、模型、数据/增强、优化器、derived、启动/保存/评估等未变。按真实源码复用R240 d587adcb的完整1600×64 fit流及实际8双视图/标签对齐、像素SHA442beeb3和原P893ac00ab，不重复这些前向/图片消费或采样。102400主draw、12800正/89600负、46256唯一fit图及原暴露代价保持；新mask本身的数值/梯度已另验，不以旧Gloo充作新loss通过。

两SCI为R242 `9d17076bf9bbd958c743c2b163e80246345c228e096d55f3ad70014ea79855fb`、R243 `196bb155a5dd130a2e74c38ed0ea08f8240c6827d551b51745621bba7bb4ef5d`。源码/config已 `125ceea8e9693157b5a9e2c2e011268092b141a3` 推送/API回读、tree `49b5d19a363664d5fe9ec0b1e7e05cb4a668e8cb`；[2925全树](TRANSFER-VERIFICATION.json)b1067531及[两technical包](TECHNICAL-PACKAGES.json)由guard0d422b43通过，R242 SITE31efbd37、R243为415041a6。正式CPU/部署/窗口和GPU均未执行，R239/R240/R241原优先与完整门不变，无科学收益声明。CPU汇总中的source_published=false为当时检查快照，不改历史原件；上述真实发布与技术包为后续证据。[原导出](PREPARATION-RECEIPT.json)/guard3443d157仅小证据和两SCI，不含图像、模型或逐例预测。

私有范围为`exp-v3/technical-prep/fit-supmin-20261104-4e0c`及4e的metadata helper。原训练/保存/评估/准入/监控链复用；主Agent自审mask、默认旧数学、全局梯度、源范围和原监督，不派生Agent或申请资源。CPU3独占24核0–23/nice19/隐藏GPU，与当前R238的24–119核互斥；原3bd/f22冻结源不改。目标AUROC≥.995/AP>.9及总体raw.999未达。
