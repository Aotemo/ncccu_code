# 存储保留规则

用户最终确认：**“保留这三个,然后删除其他文件,只保留实验结果”**。

| 永久保留模型 | 完整官方AUROC | 用途 |
| --- | --- | --- |
| N06 FLAIR ResNet50，EMA@2133 | 0.9952862924941565 | 研究专用；医学预训练，不能用于后续比赛训练 |
| R17 Tiny1K focal2，EMA4@2844 | 0.9893446521535555 | 保留单模型 |
| R8-C25 Tiny，无年龄，EMA4@2844 | 0.9890383629536272 | 保留单模型 |

各保留一个精确权重对象及最小可加载配置、age、normalization、variant、来源/SHA。路径和身份见[保留清单](../records/storage/KEEP-MODELS.json)，可加载参数入口见 [models/](../models/README.md)。历史划分和precision不同，此排名仅用于存储保留，不声称同条件科学改进。

取消R91/R92教学例外。其他历史checkpoint、CPUfixture、冗余说明、过程副本先提取精简实验结果，再按实际清单删除。未保留实验只留下结果与结论；不永久保存旧index引用的last/previous/protected模型。

活跃运行及其真实依赖临时保留至安全真实终态、最终结果核实后，再删除非结果产物。R92 的安全终态现已核实，其第二阶段非结果产物清理尚未执行。原始数据、当前共用prepared512缓存、实际后续使用的公开预训练是运行依赖。

GitHub只存代码、配置和精简实验结果；数据、模型、缓存及样本级预测留在服务器。清理按具体路径和SHA清单执行，复核目标与占用并保存真实回执。本规则覆盖旧的全部模型/原件永久保留政策。

清理进度及未完成事项维护在 [todo/storage-retention/](../todo/storage-retention/README.md)。环境按 [EXECUTION.md](EXECUTION.md) 动态发现；历史回执保留原始环境值，不作为新执行默认配置。

本次实际清理因原 Pod 移除而中断（exit137 / NotFound），没有整体成功回执；prepared768 最后一项和 R92 第二阶段不得标为已完成。具体事实见 [CLEANUP-STATUS.md](../records/storage/CLEANUP-STATUS.md)。暂停期间不新建资源、不自动改绑续跑。
