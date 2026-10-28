# 训练与推理源码

`bundle/` 保存模型、数据处理、训练、EMA、推理与指标实现。当前科学配置见 [R93-SCIENCE.json](../configs/R93-SCIENCE.json) 和 [PLAN.md](PLAN.md)。[prepare.py](../tools/activation/prepare.py) 在 Pod 按当次显式 site、迁移回执和实际源码 SHA 生成运行配置、fresh/source 清单及 CPU 验收包；旧 R92 生成配置和机器证据由 [R92 记录](../records/R92/README.md) 与 Git 历史追溯。

实际 train → freeze → official-once 调用链、模块职责及历史入口的区别见 [SOURCE-MAP.md](../docs/SOURCE-MAP.md)。

已运行的包有其冻结路径、Pod 与设备合同。`launch_once.py`、`run_training.py`、`run_round.py` 等绑定入口不是新环境的启动示例；不得在另一个目录直接执行，也不因仓库整理改写服务器活跃副本。历史绝对路径留作来源证据，后续新部署应由执行 owner 使用[动态环境发现](../docs/EXECUTION.md)和新的实际清单绑定。

当前任务及其进度只在 [todo/](../todo/README.md) 维护。后续源码仍在本目录演进，以 Git 历史追溯；不为每轮增加一套开发副本。
