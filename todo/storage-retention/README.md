# 三模型保留与历史产物清理

用户已确认：“保留这三个,然后删除其他文件,只保留实验结果”。本任务保留历史清理事实与剩余清单；最新授权的新执行A先核真实占用和依赖，不恢复旧Agent/旧清理进程，不因目录整理另发起重复删除。八卡适配及exp-v3迁移见[当前任务](../exp-v3-8card/README.md)。

**清理已中断：原计算 Pod 被移除，进程真实 exit137，实时查询为 NotFound。** 只读原日志已记录 7380 个单文件 action，最后一项无用 prepared768 目录清理未取得完整终态；无最终成功回执。R92 已完成终态核实，但其后续第二阶段清理尚未执行。事实见 [CLEANUP-STATUS.md](../../records/storage/CLEANUP-STATUS.md)。不新建计算资源或自动改绑来续删。

- [x] 登记 N06、R17、R8-C25 的唯一 EMA 权重路径与 SHA：[权威清单](../../records/storage/KEEP-MODELS.json)。
- [x] 提取精简历史实验结论：[Markdown](../../records/storage/EXPERIMENT-CONCLUSIONS.md)、[JSON](../../records/storage/EXPERIMENT-CONCLUSIONS.json)。
- [x] 已传回三个模型的 15 份既有参数/来源 JSON 原件，见 [models/](../../models/README.md)。
- [ ] checkpoint 内参数新导出与加载核验因 Pod 移除尚未执行；传输后 SHA 复核也未执行。
- [x] 已保存清理中断事实、原计划摘要和硬链接守卫记录；不声称整体清理完成。
- [ ] 在用户授权且存在合适执行环境后，沿既有真实清单/日志核实剩余 prepared768 项和三模型保留状态，取得真实最终回执。
- [ ] 在同样前提下处理 R92 第二阶段非结果产物；本次没有执行。

N06 为研究专用；不用于后续比赛训练。原始数据、当前共用缓存与实际后续公开预训练仍是运行依赖。规则见 [STORAGE.md](../../docs/STORAGE.md)。
