# 配置入口

本目录用于可复用配置与模板。[R95-SCIENCE.json](R95-SCIENCE.json) 按用户新要求配置最多24000成功更新、每800步内部dev验证、online/EMA较高AUROC连续4轮未刷新最好值即早停。首次验证建立基线，min_delta=0；最多30轮，warmup和EMA warmcopy仍为1600步，cosine衰减按24000步预算。早停后仍从实际完成的全部内部读数中唯一选点，官方仅最终一次比较。[R94-SCIENCE.json](R94-SCIENCE.json) 与 [R93-SCIENCE.json](R93-SCIENCE.json) 保留历史对照，R94活跃冻结包不变。[prepare.py](../tools/activation/prepare.py) 必须显式选择 `--science-config configs/R95-SCIENCE.json`，根据当次 site 与真实资产回执，在 Pod 生成 TRAIN_CONFIG、ASSETS、fresh/source 清单和新的 CPU 验收身份。site 固定配置 path/SHA，package 保留相同 bytes 的 SCIENCE.json；消费者读取该配置，不回退旧配置。旧 R92 生成模板与 CPU 身份从当前入口移除，原件保留于 Git 历史及 [records/R92](../records/R92/)。任务合同在 [todo/auroc-999/](../todo/auroc-999/README.md)，三个保留模型参数在 [models/](../models/README.md)。

[PRETRAINING-ASSETS.json](PRETRAINING-ASSETS.json) 是原预训练盘点的回执路径/SHA 索引，不包含新的执行环境配置；保留其路径供正在进行的存储工作接续。

主机、namespace、Pod、UID、容器、Python 与实际工作路径通过工具发现和当次绑定获取，见 [EXECUTION.md](../docs/EXECUTION.md)。发现输出使用忽略的 `todo/.local/` 或服务器 `technical-prep`；正式身份随真实回执归档。
