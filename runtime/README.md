# 训练与推理源码

`bundle/` 保存模型、数据处理、训练、EMA、推理与指标实现。根部配置、`PLAN.md` 和控制入口来自已核实的 R92 部署包；原始来源及 SHA 见 [R92 记录](../records/R92/README.md)。

实际 train → freeze → official-once 调用链、模块职责及历史入口的区别见 [SOURCE-MAP.md](../docs/SOURCE-MAP.md)。

已运行的包有其冻结路径、Pod 与设备合同。`launch_once.py`、`run_training.py`、`run_round.py` 等绑定入口不是新环境的启动示例；不得在另一个目录直接执行，也不因仓库整理改写服务器活跃副本。历史绝对路径留作来源证据，后续新部署应由执行 owner 使用[动态环境发现](../docs/EXECUTION.md)和新的实际清单绑定。

当前任务及其进度只在 [todo/](../todo/README.md) 维护。后续源码仍在本目录演进，以 Git 历史追溯；不为每轮增加一套开发副本。
