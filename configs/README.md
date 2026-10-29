# 配置入口

最新复核为 [R103-SCIENCE.json](R103-SCIENCE.json)：直接对照R102，仅将训练seed20261027改为预先固定的20261029，补齐Tiny/Small与两个预登记seed的匹配比较；不是seed搜索或挑最好seed。Small资产、BCE、LR/SD/dropout、固定splitseed与24k/p4、6h/8h预算不变，新整体初态须由同一生产CPU工厂实测。

当前后续方案为 [R102-SCIENCE.json](R102-SCIENCE.json)：直接对照预登记R95/BCE/seed20261027，成对更换ConvNeXt Small架构与其fe6b358a公开checkpoint，不称纯容量消融。显式SD0.1、dropout0.2、backbone1e-4、head/age各1e-3；其余24000/800/patience4与固定9:1不变。新来源哈希、实测结构与整体fresh初态写入同一pinned SCIENCE；不能复用Tiny控制或训练权重。显式传入 `--science-config configs/R102-SCIENCE.json`，不同轮次使用各自冻结包。

本目录用于可复用配置与模板。[R95-SCIENCE.json](R95-SCIENCE.json) 按用户要求配置最多24000成功更新、每800步内部dev验证、online/EMA较高AUROC连续4轮未刷新最好值即早停。首次验证建立基线，min_delta=0；最多30轮，warmup和EMA warmcopy仍为1600步，cosine衰减按24000步预算。早停后仍从实际完成的全部内部读数中唯一选点，官方仅最终一次比较。R93、R94和R95配置保留历史对照，原冻结包不变。[prepare.py](../tools/activation/prepare.py) 必须显式选择当轮 `--science-config`，根据当次 site 与真实资产回执，在 Pod 生成 TRAIN_CONFIG、ASSETS、fresh/source 清单和新的 CPU 验收身份。site 固定配置 path/SHA，package 保留相同 bytes 的 SCIENCE.json；消费者读取该配置，不回退旧配置。旧 R92 生成模板与 CPU 身份从当前入口移除，原件保留于 Git 历史及 [records/R92](../records/R92/)。任务合同在 [todo/auroc-999/](../todo/auroc-999/README.md)，三个保留模型参数在 [models/](../models/README.md)。

[PRETRAINING-ASSETS.json](PRETRAINING-ASSETS.json) 是原预训练盘点的回执路径/SHA 索引，不包含新的执行环境配置；保留其路径供正在进行的存储工作接续。

主机、namespace、Pod、UID、容器、Python 与实际工作路径通过工具发现和当次绑定获取，见 [EXECUTION.md](../docs/EXECUTION.md)。发现输出使用忽略的 `todo/.local/` 或服务器 `technical-prep`；正式身份随真实回执归档。
