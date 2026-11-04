# R244/R245 已闭轮非 selected 清理

两轮均已完整 actualsafe、结果归档且 selected 严格构造加载，统筹扩展授权清理非 selected，保护 selected EMA 和最小可加载依赖。先封存[逐文件计划](PLAN.json) `393299c2950539fd4a5b017da3dd235d5dcc1139ffac2468d548a1d1518d852f` 及[清单](INVENTORY.json) `ed7e35c508fe7eb8252f094530aa8b82bab93a268b7454ff9d5631c12846d8c3`：15项，预计已分配 11,090,099,712 B；计划时 free 60,505,980,928 B。

[删除回执](DELETE-RECEIPT.json) `11c5f635c3c11d8a3a18d927130bf8982f78a77d6884976adea43a870e7b6ef9`：15/15，0跳过/0错误，已分配释放 11,090,099,712 B；free 点测 60,298,362,880→71,388,102,656 B，受同时运行影响，不是后续容量保证。受保护文件 stat 和小文件 SHA 前后一致；KEEP、其他 selected、数据/共享缓存/公开源、R246/R247 及其依赖、R120 提交依赖未列入候选。GPU 训练未发信号、未取得训练锁，checkpoint 本体未重算或加载。
