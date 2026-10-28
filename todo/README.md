# 当前任务

任务状态与接续在本目录维护。已完成结果见 [records/](../records/README.md)，长期规则见 [AGENTS.md](../AGENTS.md)。

**2026-10-28 最新授权：当前实例继续使用，不关闭、不暂停；执行Agent可开子Agent并行加速迁移构造。** 当前1规划+2执行，A/B各已创建1名文本编码子Agent，精确分文件、不再派生、不恢复旧Agent或运行。A保持唯一Git写/GPU owner，负责生产集成；B负责已完成的来源CPU子门及当前exp-v3依赖迁移。子Agent无Pod/Git写/GPU权，父级在现有12/12份额内验收。源码同步`Aotemo/ncccu_code`，非源码产物存`/public/home/acfwvv9ejn/SothisAI/exp-v3`；R93未启动，无新资源/定时任务/heartbeat。

当前阶段：代码已同步GitHub `20af511b0177c0f1576c8c630baaf1865d8001f9`，实际remote tree与Pod部署SHA已核；必要CPU及75.37 GB迁移通过，B/B1已退场。**首次有界GPU准入失败，R93未启动**。用户最新纠正：执行Agent准备计算合计20%，模型训练及对应GPU准入整树80%实时有效CPU（仍120核则最多96），覆盖此前20%被扩大到GPU的误读。A合并资源语义修正、必要helper适配和A1有界TERM/wait/reap修复，旧20af/attempt1/分配及失败原件不改；helper全128仍超新80%额度。一次具体diff/必要CPU窄验/同步后另建冻结和独立准入，科学合同不变，不新增门或平台。详情见[本轮接续](exp-v3-8card/README.md)。

最终用户要求：训练80%与下一轮计划/CPU验证20%同时运行；两个互斥CPU池从实时环境取得总额、核数及mask，比例集中在版本化配置并由env注入，代码不散写策略值。允许准备池持续活动，不再要求全体prep/peer退场。A完成这次最小修复/必要窄验/同步冻结/准入后直接fresh R93；规划在训练期间统筹20%池的下一轮验证，不新增文档门或例行等候。

| 任务 | 状态 | 入口 |
| --- | --- | --- |
| 当前8卡适配与exp-v3 | Git/部署完成；首次GPU准入因CPU helper越界及退出收集失败，正在最小修复，R93未启 | [阶段、角色与计划](exp-v3-8card/README.md) |
| 仓库结构与三模型参数归档 | 文本/原件已整理；旧 Pod 消失造成的剩余核验待新环境窄检查 | [整理与核验](repository-layout/README.md) |
| 固定 9:1 单模型，raw AUROC ≥ 0.999 | 后续已授权；R92 已完成且未达目标，R93 待适配验收与同步 | [实验任务](auroc-999/README.md) |
| 只保留三个模型并提取历史结论 | 进行中；以实际清理回执为准 | [存储接续](storage-retention/README.md) |

当前主机、Agent 与计算 Pod 通过[环境发现工具](../docs/EXECUTION.md)查询。不要将任务记录中的环境值当默认参数。完整发现、验收、缓存、冻结部署和运行产物保存在 `exp-v3`；本仓库只接收源码、配置、必要精简结论与小回执。本机不进行解析、哈希或测试。
