# R211 已结束权重清理

R211完整9600/c12/bad4、唯一EMA c8@6400内部AUROC`.987315937198597`，双门false/R212不eligible/官方未跑，原actualsafe不重查。[固定20组](../../analysis/20261102-persistent-errors/R211-FIXED20-SUMMARY.json)为6423、其余147正例4111、总10534错序对质量；相对R107分别-200/+1134/+934。局部组改善但总体恶化，不能作为机制或收益证明。原CSV/precision各读一次，未加载模型/图片或新forward。

[只读计划](PLAN-SUMMARY.json)绑定已回读c9bbde5及真实R211结果/KEEP清单；plan SHA `179ed1fe0c1d0b14db68cb0aaf8fd492aa8e796625b4e33a398ee6ddce51f1ba`，guardfn7qnj4y / e5d96e02 exit0。实际9件=8训练checkpoint+1initial，逻辑6,734,027,331 B、allocated6,734,029,824 B。仅核R211闭轮路径，其他round、全部technical-prep、当前R215、R213所选临时依赖、原始数据、共享缓存和待用公开源保护。

[删除回执](DELETE-RECEIPT.json) SHA `d793465017674f4083576a41863d8beb605185d224b7ad6b03516bfa9de6a6bf`，guard7jg6es0q / 5888eeb7 exit0：9删除/0跳过/0错误。三永久KEEP及当前依赖前后保护摘要均`80bf5128746e6e9c805e911a090932576256797e26ed28fe32acfaf396fce794`；大权重只核stat/既有sidecar身份，未重哈希或加载，不声称逐字节重验。

[最终guard](CPU-FINAL-GUARD.json) bmlbpf8o / d68e919e exit0：9文件实际不存在、18条before/unlinked日志、sidecar及已归档结果SHA保持。空闲空间时点27,410,825,216→34,144,780,288 B，不是共享文件系统后端物理回收保证。预测CSV/precision、索引、锁与小回执保留；R211所选权重已删除，不能再称可加载或可做fit forward。

原件在`exp-v3/technical-prep/r211-closed-storage-20261103-32e3`及`r211-fixed20-20261103-32e3`。d837/root0–11/nice19/隐藏GPU，与R215训练24–119互斥；无训练信号、锁、GPU或官方读取。
