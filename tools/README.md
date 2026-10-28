# 工具入口

| 目录 | 用途 |
| --- | --- |
| `environment/` | 动态查询计算 Pod、核 UID/Ready、发现解释器并执行 CPU 工具 |
| `models/` | 从权威保留清单中的 checkpoint 导出精确加载参数 |
| `repository/` | 源码导入与目录/参数一致性检查 |
| `storage/` | 既有存储盘点、结论提取和按清单清理 |
| `data-contract/` | 历史固定划分与年龄合同生成、检查源码 |
| `activation/` | 已绑定 R92 的历史正式绑定工具 |
| `observer/` | 已绑定 R92 的只读观察工具 |
| `validation/` | 原 R92 准备检查及已执行的终态审计/导出源码 |

新执行先读 [EXECUTION.md](../docs/EXECUTION.md)。已有按轮绑定的脚本保留其真实源码身份，不能作为其他环境的通用入口。R92 已完成终态核实，原观察/activation/launch 入口不再用于接续。`validation/finalize_receipt.py` 和 `export_receipt.py` 是当时已实际执行的终态审计/导出原源码，其中旧绑定值仅作来源证据。目录整理不重跑这些阶段或清理脚本。
