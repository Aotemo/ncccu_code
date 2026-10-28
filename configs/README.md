# 配置入口

本目录用于可复用配置与模板。[R94-SCIENCE.json](R94-SCIENCE.json) 保存当前 backbone LR 减半的科学配置；[R93-SCIENCE.json](R93-SCIENCE.json) 保留直接对照。[prepare.py](../tools/activation/prepare.py) 必须显式选择 `--science-config configs/R94-SCIENCE.json`，根据当次 site 与真实资产回执，在 Pod 生成 TRAIN_CONFIG、ASSETS、fresh/source 清单和新的 CPU 验收身份。site 固定配置 path/SHA，package 保留相同 bytes 的 SCIENCE.json；消费者读取该配置，不回退旧配置。旧 R92 生成模板与 CPU 身份从当前入口移除，原件保留于 Git 历史及 [records/R92](../records/R92/)。任务合同在 [todo/auroc-999/](../todo/auroc-999/README.md)，三个保留模型参数在 [models/](../models/README.md)。

[PRETRAINING-ASSETS.json](PRETRAINING-ASSETS.json) 是原预训练盘点的回执路径/SHA 索引，不包含新的执行环境配置；保留其路径供正在进行的存储工作接续。

主机、namespace、Pod、UID、容器、Python 与实际工作路径通过工具发现和当次绑定获取，见 [EXECUTION.md](../docs/EXECUTION.md)。发现输出使用忽略的 `todo/.local/` 或服务器 `technical-prep`；正式身份随真实回执归档。
