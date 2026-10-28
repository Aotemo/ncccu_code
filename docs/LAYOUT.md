# 目录约定

| 位置 | 保存内容 | 更新时机 |
| --- | --- | --- |
| `README.md` | 项目用途、目录导航 | 项目入口变化时 |
| `AGENTS.md` | 长期协作、执行和实验边界 | 用户修改长期约定时 |
| `todo/` | 当前任务的目标、合同、待办、状态依据与接续 | 当前工作推进时 |
| `runtime/` | 模型、训练、推理和运行控制源码 | 代码实现变化时 |
| `configs/` | 可复用配置说明与模板 | 新增可复用配置时 |
| `models/<id>/` | 保留模型的参数与加载身份，权重仅引用服务器对象 | 核实或更新保留模型时 |
| `tools/` | 环境发现、数据合同、检查和存储工具 | 工具行为变化时 |
| `records/<round>/` | 已产出的实验结果、原始小回执及来源清单 | 真实结果产出时 |
| `records/storage/` | 权威模型保留清单、精简历史结论、清理回执 | 存储策略或真实清理完成时 |
| `docs/` | 目录、执行、源码同步、存储使用说明 | 流程变化时 |

当前任务的状态只在 `todo/` 维护。README 通过链接引导读者；AGENTS 不再携带当前运行的主机、UID、Python 路径、Agent ID 或进度。

`runtime/` 在导入的R92源码基础上继续演进，当前`PLAN.md`描述R93；历史R92的实际部署身份由Git历史和`records/R92/`追溯。旧生成件`TRAIN_CONFIG.json`、`ASSETS.json`、`CLOCK_TEMPLATE.json`、`FRESH-SPEC.template.json`、`RUNTIME_BINDING.template.json`、`SOURCE-MANIFEST.reference.json`及`evidence/`内三份旧CPU身份已从当前入口移除，服务器历史冻结副本不改。

当前科学配置来自`configs/R93-SCIENCE.json`；`tools/activation/prepare.py`按当次显式site与真实迁移资产回执在Pod生成新的配置、资产、fresh/source清单和CPU验收身份。`CLOCK_POLICY.round_id`由site提供，不默认R92。生成产物在exp-v3，本仓库维护生成源码、科学配置及必要小回执；CPU准备产物不等于正式运行或GPU准入。流程见[执行说明](EXECUTION.md)和[源码入口](SOURCE-MAP.md)。

`configs/PRETRAINING-ASSETS.json` 是已导入的预训练盘点回执索引，暂保原路径以便正在进行的存储工作接续；它不是运行环境配置。今后的同类回执归入 `records/`。

## 任务生命周期

1. 在 `todo/<task>/README.md` 记录目标、已确认事实、验收条件与待办。
2. 参数和代码分别维护于 `configs/`、`runtime/`、`tools/`，任务文档只链接这些文件。
3. 使用工具即时发现执行环境。完整发现结果、技术准备、缓存、冻结部署和运行产物统一放用户指定的 `/public/home/acfwvv9ejn/SothisAI/exp-v3`；正式执行身份写入该次真实回执。Git只收必要小回执与结论。任务文档可记录当次绑定，但必须标明它不是环境默认值。
4. 结果核实后存入 `records/`，更新任务勾选；完成的任务从当前任务索引移除，保留精简结果链接。模型参数由 `models/` 指向权威保留清单，避免另造一个排名清单。
