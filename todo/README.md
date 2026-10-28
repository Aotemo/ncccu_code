# 当前任务

任务状态与接续在本目录维护。已完成结果见 [records/](../records/README.md)，长期规则见 [AGENTS.md](../AGENTS.md)。

**2026-10-28 最新授权：当前实例继续使用，不关闭、不暂停；执行Agent可开子Agent并行加速迁移构造。** 当前1规划+2执行，A/B各已创建1名文本编码子Agent，精确分文件、不再派生、不恢复旧Agent或运行。A保持唯一Git写/GPU owner，负责生产集成；B负责已完成的来源CPU子门及当前exp-v3依赖迁移。子Agent无Pod/Git写/GPU权，父级在现有12/12份额内验收。源码同步`Aotemo/ncccu_code`，非源码产物存`/public/home/acfwvv9ejn/SothisAI/exp-v3`；R93未启动，无新资源/定时任务/heartbeat。

当前阶段：A生产18/18、启动集成29/29、准入合同CPU11/11及部署commit→tree关联3/3通过。全量具体diff v2已固定；规划科学/采样/一次性控制/cache接口文本审阅未发现新阻碍，范围见[审阅记录](exp-v3-8card/REVIEW.md)。B来源与迁移全部完成：120380文件/75373650642B，train51396、valid5849新索引全部通过，RELOCATION与最终CPU退场原件已交；B/B1冻结，不再使用12核份额。A按实时资源另建后续GPU owner总20%绑定，旧allocation不热改。当前最终记录及文档纳入v3，随后**具体diff冻结→成功同步GitHub并核远端commit/实际SHA冻结→唯一A有界GPU准入→正式R93**；不重跑旧通过项，归档润色/历史清理后置。远端同步与GPU尚未发生，详情见[本轮接续](exp-v3-8card/README.md)。

| 任务 | 状态 | 入口 |
| --- | --- | --- |
| 当前8卡适配与exp-v3 | 必要CPU与迁移通过，文本审阅收拢；最终v3及Git同步/冻结待执行，未GPU | [阶段、角色与计划](exp-v3-8card/README.md) |
| 仓库结构与三模型参数归档 | 文本/原件已整理；旧 Pod 消失造成的剩余核验待新环境窄检查 | [整理与核验](repository-layout/README.md) |
| 固定 9:1 单模型，raw AUROC ≥ 0.999 | 后续已授权；R92 已完成且未达目标，R93 待适配验收与同步 | [实验任务](auroc-999/README.md) |
| 只保留三个模型并提取历史结论 | 进行中；以实际清理回执为准 | [存储接续](storage-retention/README.md) |

当前主机、Agent 与计算 Pod 通过[环境发现工具](../docs/EXECUTION.md)查询。不要将任务记录中的环境值当默认参数。完整发现、验收、缓存、冻结部署和运行产物保存在 `exp-v3`；本仓库只接收源码、配置、必要精简结论与小回执。本机不进行解析、哈希或测试。
