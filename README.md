# ncccu_code

青光眼单模型实验的统一源码仓库。后续代码、配置和必要的小型复现记录同步到本仓库；数据、预训练权重和训练模型保存在计算服务器。

## 当前任务

在固定 9:1 划分上串行比较合规预训练的单模型，目标为 raw AUROC ≥ 0.999。内部 dev 负责选出唯一 checkpoint，再对完整官方 5849 样本评估一次。内部 dev 与官方指标分别报告。

| 模型 | 状态 | 内部 dev AUROC | 官方 AUROC |
| --- | --- | --- | --- |
| R91 ResNet50 ImageNet1K V2 | 已完成，未达目标 | 0.9732399267421321 | 0.9746659041823744 |
| R92 ConvNeXt Tiny ImageNet1K | 按既定流程串行接续 | 待结果 | 待结果 |
| ConvNeXt Tiny ImageNet22K→1K | 后续候选 | 待运行 | 待运行 |

固定 fit 46256、dev 5140；每周期 800 成功更新，共 8 周期；warmup 与 EMA warmcopy 均为 1600 更新。官方集不参与训练、年龄统计、早停、难例挖掘或 checkpoint 选择。

## 目录

- `runtime/`：当前服务器部署包的训练、推理、模型、EMA、指标、配置和独占运行入口。R91 基线导入后在同一路径接续 R92，通过 Git 历史记录差异。
- `tools/data-contract/`：固定划分和年龄合同的生成、验证源码；已冻结的数据不重新生成。
- `tools/activation/`：对应实验的历史正式绑定入口，含固定服务器路径，保留原件供追溯。
- `tools/observe_run.py`：对应实验的只读观察入口。
- `tools/storage/`：指定 Pod 内使用的存储盘点与清理工具。
- `records/`：必要的包清单、来源身份、汇总指标与最终回执；不包含样本级数据和模型大文件。
- `docs/`：源码同步、部署与存储保留规则。

## 执行环境

所有项目代码在指定计算 Pod 内运行。本机只编辑、传输和查看文件。

- Pod：`vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0`
- namespace：`acfwvv9ejn`
- UID：`eec33bd6-b567-40ef-b866-633e272ea019`
- Python：`/usr/local/python3.12/bin/python3.12`，已记录版本 3.12.7
- 已记录依赖：torch 2.10.0、torchvision 0.25.0、numpy 1.26.4、Pillow 12.3.0

`runtime/` 保留已通过服务器检查的源码 bytes 和当时路径合同。正式运行需要执行 A 在指定 Pod 进行路径绑定、资源准入和 SHA 核对；不能在新目录直接启动历史绑定入口，不能将 CPU fixture 当训练模型。依赖使用已有 DCU 环境，不在本机安装或运行项目依赖。

## 每轮为什么会出现 code 目录

训练入口 `runtime/bundle/parallel_fresh_train.py` 在创建新运行目录时，保存 `source-files.json` 并复制本次源码到 `code/`，用于绑定模型与实际源码。这些是冻结运行副本。今后维护的代码集中在本仓库，以 commit 记录版本；运行目录只保留所需的冻结部署和复现资料。

详见 [源码同步规则](docs/CODE-SYNC.md) 和 [存储保留规则](docs/STORAGE.md)。
