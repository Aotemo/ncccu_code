# 三模型保留与历史产物清理

**2026-11-02窄清理已完成：仅R164/R166/R168的25个checkpoint/initial实删，结果保留。** [实际清单与回执](../../records/storage/20261102-closed-rounds/README.md)记录0跳过/无错误、保护对象前后stat/小SHA一致、三阶段CPU guard0。pre-unlink allocated11,580,709,888字节；共享FS free38,677,774,336→50,258,247,680，点时差11,580,473,344另列。当前R172在独立训练96核池继续，准备根23核nice19隐藏GPU；KEEP三模型、失败R170、比较/中断保护对象、公开来源/原始数据/共用缓存/technical-prep及小型结果/index/锁/one-shot保持，无训练信号或实例锁。旧清理与旧失败原件不改；下文为此前批次。

**最终第三批亦已收口：100,652文件+385目录实删，旧prepared768精确路径已不存在。** [最终实证](../../records/storage/20261101-cleanup/README.md)分列前两批372项与旧缓存100280文件/385目录；全部文件allocated260478643200B，最终sharedFS free22961717248→283262320640B，点时差260300603392B。第三批完整lstat清单100665项/无链接与当前引用冲突，首次低效自有子进程经明确授权精确SIGINT后真实exit-2；旧journal确认18614项、边界无缺after/伪成功，再以新独立217软件CPU入口只接81666项，52.06秒完成/guard0，原d29与旧训练clock/grant不动。保护对象stat/小SHA、prepared512与census依赖均一致；94项保护或未证冗余model/package继续保留。以下“两批已完成/旧cache未重审”文字是前序阶段。

**2026-11-01新授权清理已完成两份明确清单，历史中断原件保留。** 现有Pod动态UID/Ready/A12 guard实核后，先实际归档同步R134R1终态4968edc，再实删372文件（320历史checkpoint/initial、49已完全PASS formalCPUfixture、3过期源码运输包），两批无执行错误/跳过。最终sharedFS free22961717248→249898729472B，实测增加226937012224B；删除前allocated226940564992B与logical另列，不冒充物理释放，也不把同一FS的exp-v2/v3重复相加。[真实回执与路径分类](../../records/storage/20261101-cleanup/README.md)已落盘；两批保护对象stat/小SHA均一致，KEEP18路径可见，R130全部/R120及比较依赖/R134R1 selected与完整census依赖、原始数据/共用cache/公开来源/unknown、小结果/失败成本/锁/one-shot保持。剩94项model/package含保护和未证冗余，R114缺完整终态、4个CPUfixture缺final receipt继续保留；不为收口重扫/重载模型。以下旧Pod清理中断是历史阶段，不能据本次清单完成改写其原exit137。

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
