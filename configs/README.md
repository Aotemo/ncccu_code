# 配置入口

本目录用于可复用配置与模板。当前 R92 的冻结科学配置仍在 [runtime/TRAIN_CONFIG.json](../runtime/TRAIN_CONFIG.json)，它与已有部署清单绑定；当前任务合同在 [todo/auroc-999/](../todo/auroc-999/README.md)。三个保留模型的参数在 [models/](../models/README.md)。

[PRETRAINING-ASSETS.json](PRETRAINING-ASSETS.json) 是原预训练盘点的回执路径/SHA 索引，不包含新的执行环境配置；保留其路径供正在进行的存储工作接续。

主机、namespace、Pod、UID、容器、Python 与实际工作路径通过工具发现和当次绑定获取，见 [EXECUTION.md](../docs/EXECUTION.md)。发现输出使用忽略的 `todo/.local/` 或服务器 `technical-prep`；正式身份随真实回执归档。
