# 当前任务

**2026-10-29当前：R94已实际训练，R95在20%准备池并行适配。** B唯一只读观测R94冻结`212b41f9d3aa922519e62182701bfbea527c852d`为3200/6400成功更新、0 skip、4轮已评，owner及8rank同身份存活；该时点无终态，不能因Agent会话取消推断训练中断。用户已明确R95最多24000成功更新、连续4轮无提升早停，并选择内部dev AUROC；[R95合同](auroc-999/R95-PLAN.md)已确定，不再标指标待答。A回报受影响CPU检查13/13通过，既有prepare配置生成及新Git/部署仍待实际完成，R95尚未运行。R94冻结6400步运行不热改；[并行准备](auroc-999/NEXT-PREPARATION.md)按既有边界收口，无新增启动门。

任务状态与接续在本目录维护。已完成结果见 [records/](../records/README.md)，长期规则见 [AGENTS.md](../AGENTS.md)。

**2026-10-28 最新授权：当前实例继续使用，不关闭、不暂停；执行Agent可开子Agent并行加速迁移构造。** 当前1规划+2执行，A/B各已创建1名文本编码子Agent，精确分文件、不再派生、不恢复旧Agent或运行。A保持唯一Git写/GPU owner；B来源与exp-v3迁移已完成，现按规划安排在验证池做下一轮最小CPU可行性准备。子Agent无Pod/Git写/GPU权，父级只使用实时核验有效的份额。源码同步`Aotemo/ncccu_code`，非源码产物存`/public/home/acfwvv9ejn/SothisAI/exp-v3`；无新资源/定时任务/heartbeat。

当前阶段：**R93工程终态通过，R94科学决策已冻结并交A最小适配**。R93正式实现`80e06f31c33bc6256f6b622e6ed162273d33f1ba`完成6400成功更新/0 skip、完整16条内部验证，唯一选择c3 EMA@2400；A终态审计25/25、一次官方身份一致、8rank退出、GPU无使用者/idle及原锁释放均已核，owner inactive/encodedexit0但Z保留，不称reaped。终态回执Pod SHA`e1d0e20895f3a8047f2bd4b8477875f85b36f32ab013e6994d4b95d9864cf205`。R94仅将backbone LR由2e-4降为1e-4，同公开fresh/seed及其余合同；依据仅完整内部与工程证据，官方数值不参与选择。见[R94决定](auroc-999/R94-DECISION.md)、[R93原件](../records/R93/README.md)和[本轮接续](exp-v3-8card/README.md)。旧失败及80e冻结包不改、不重启。

2026-10-29最新要求：R94仅改LR，删除重复hardcoded LR断言、停止新增拒绝case/重跑已过检查，取消独立零forward GPU预演并如实记未运行；必要Git实际冻结后直接由真实训练8rank初态/组/空Adam/资源校验接管，保留一次activation/官方一次，不等文档门。R94启动后，训练80%与准备20%按集中policy和实时互斥池并行；准备覆盖不依赖结果的代码适配、配置注入、干净打包及现有CPU验证，最终科学值依完整内部结果定稿，不用官方调参，不新增当前启动工作或平台。

| 任务 | 状态 | 入口 |
| --- | --- | --- |
| 当前8卡适配与exp-v3 | R93工程有效；R94实际训练中，R95配置/早停实现并行准备 | [阶段、角色与计划](exp-v3-8card/README.md) |
| 仓库结构与三模型参数归档 | 文本/原件已整理；旧 Pod 消失造成的剩余核验待新环境窄检查 | [整理与核验](repository-layout/README.md) |
| 固定 9:1 单模型，raw AUROC ≥ 0.999 | R94已开训；R95按用户新预算与早停合同准备，尚未运行；不称总体完成 | [实验任务](auroc-999/README.md) |
| 只保留三个模型并提取历史结论 | 进行中；以实际清理回执为准 | [存储接续](storage-retention/README.md) |

当前主机、Agent 与计算 Pod 通过[环境发现工具](../docs/EXECUTION.md)查询。不要将任务记录中的环境值当默认参数。完整发现、验收、缓存、冻结部署和运行产物保存在 `exp-v3`；本仓库只接收源码、配置、必要精简结论与小回执。本机不进行解析、哈希或测试。
