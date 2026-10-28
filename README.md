# ncccu_code

青光眼单模型实验的统一源码仓库。代码、配置、模型参数说明与实验结论通过 Git 管理；模型权重、数据和缓存保存在计算服务器。

## 从这里开始

| 要了解的内容 | 入口 |
| --- | --- |
| 项目结构与文件归属 | [目录约定](docs/LAYOUT.md) |
| 训练/评估入口与模块依赖 | [源码地图](docs/SOURCE-MAP.md) |
| Agent 的长期协作与执行规则 | [AGENTS.md](AGENTS.md) |
| 当前目标、进度和下一步 | [todo/](todo/README.md) |
| 保留的最优模型及加载参数 | [models/](models/README.md) |
| 已完成实验的结果与回执 | [records/](records/README.md) |
| 执行环境发现与 CPU 工具入口 | [执行环境](docs/EXECUTION.md) |
| 源码同步、部署与模型保留 | [源码同步](docs/CODE-SYNC.md)、[存储规则](docs/STORAGE.md) |

## 目录

```text
ncccu_code/
├── README.md             # 项目说明与导航
├── AGENTS.md             # 长期协作规则
├── todo/                 # 当前任务、验收条件、待办与接续记录
├── runtime/              # 训练、推理与模型实现
├── configs/              # 可复用配置的入口说明
├── models/               # 三个保留模型的参数、加载身份与权重索引
├── tools/                # 环境发现、数据合同、校验与存储工具
├── records/              # 实验结论、来源清单与真实回执
└── docs/                 # 目录、执行、同步和存储说明
```

执行前通过工具查询当前主机、Pod 与 Python，再绑定实际身份。具体主机名、Pod UID、解释器路径和 Agent ID 不作为项目的固定环境配置。历史回执中的原始路径与 SHA 保持原义，使用方式见[执行环境](docs/EXECUTION.md)。
