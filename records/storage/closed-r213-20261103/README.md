# R213 已结束权重清理

原完整R213内部AUROC .9830353369271913、双门false/R214不eligible/官方未跑，唯一actualsafe原件保留不重查。[fit子集诊断](../../analysis/20261103-r213-fit-subset/README.md)已实际完成，CPU任务树退出并精简归档，临时checkpoint依赖解除。[固定20组](../../analysis/20261102-persistent-errors/R213-FIXED20-SUMMARY.json)9299/其余4790/总14089，相对R107多2676/1813/4489，两组均未改善；没有新增官方或模型前向。

[只读计划](PLAN-SUMMARY.json)绑定API回读6fb491a及原R213结果6af7ca2d/KEEP清单，plan `f2937adbe334cfc855016ddaf8845213955161557fc047c2527a1b86f32434e0`，guardp_365hdi /2e778fdd exit0。实际9件=8训练checkpoint+1initial，逻辑3,035,218,769 B/allocated3,035,219,968 B，无跳过或进程扫描拒绝；逐路径均限R213闭轮树。

[删除回执](DELETE-RECEIPT.json) SHA `09b062314e97e7e46c08a9fd2a3a183a3fabfbd9fbcf292b5fe33d077b3dfa12`，guardvbzl555j /a783d350 exit0：9删除/0跳过/0错误。三永久KEEP与当前R215依赖前后保护摘要均80bf5128；大payload只核stat/既有sidecar身份，不称逐字节重哈希。当前R215及全部其他round/technical-prep、原始数据、共享缓存和待用公开源保护，未取训练锁或发信号。

[最终核验](CPU-FINAL-GUARD.json) jr0536cd /2f14b5ca exit0：9件实际不存在、18条before/unlinked日志，sidecar及已归档结果SHA保持。空闲空间时点29,224,861,696→32,260,489,216 B，不等同共享文件系统后端物理回收。CSV/precision、索引、锁、fit子集私有预测和小回执保留；R213模型权重已删，不能再称可加载。

原件在`exp-v3/technical-prep/r213-closed-storage-20261103-32e3`；d837/root0–11/nice19/隐藏GPU，与R215训练24–119互斥。原fit前向只发生于清理前已封存的256图诊断，不把这次纯文件清理写成新模型验证。
