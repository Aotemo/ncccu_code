# 当前任务

任务状态与接续在本目录维护。已完成结果见 [records/](../records/README.md)，长期规则见 [AGENTS.md](../AGENTS.md)。

**后续实验已按用户要求暂停，先整理 GitHub 仓库。R93 尚未启动；必须等用户新的明确恢复指令才能接续实验。现有规划与执行均已暂停，原 heartbeat 已删除。** 本次仅收尾仓库结构、三模型参数归档及必要 CPU 验收。

| 任务 | 状态 | 入口 |
| --- | --- | --- |
| 仓库结构与三模型参数归档 | 文本/原件已整理；原 Pod 消失，剩余核验未执行 | [整理与核验](repository-layout/README.md) |
| 固定 9:1 单模型，raw AUROC ≥ 0.999 | 暂停；R92 已完成且未达目标，R93 未启动 | [实验任务](auroc-999/README.md) |
| 只保留三个模型并提取历史结论 | 进行中；以实际清理回执为准 | [存储接续](storage-retention/README.md) |

当前主机、Agent 与计算 Pod 通过[环境发现工具](../docs/EXECUTION.md)查询。不要将旧任务文档中的环境值当默认参数。临时工具输出可放在 `.local/`，该目录不进入 Git。
