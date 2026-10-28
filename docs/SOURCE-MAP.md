# 源码入口与依赖

源码由R92导入后持续演进，当前八卡适配与R93进度见[todo](../todo/README.md)。本页区分新生成流程和历史调用证据；不能将历史绑定或旧CPU模板复制到新环境当作准入。

## 当前准备入口

`configs/R93-SCIENCE.json`保存科学配置与工作根内相对资产路径；`tools/activation/prepare.py`消费当次显式site和真实迁移回执，在Pod生成TRAIN_CONFIG、ASSETS、fresh/source清单及新的CPU验收身份。`runtime/CLOCK_POLICY.json`的round_id由site注入，`runtime/PLAN.md`描述R93。旧R92六个生成配置/模板和`runtime/evidence/`三份旧CPU身份已移出当前入口，原件由Git历史及`records/R92/`追溯，服务器旧部署不改。

新准备结果先完成必要CPU验收，再由唯一A提交同步具体源码/配置并核远端commit，按实际SHA冻结新部署，之后才允许GPU准入及新正式绑定。准备阶段的uncommitted snapshot和CPU初态checkpoint只作技术证据，不成为正式初始化或已同步声明。

## 已验证 R92 的实际路径

```text
已正式绑定的冻结运行目录
  launch_once.py                     原一次性 owner 派发
    run_training.py                  控制 train → freeze → official-once
      component_gate.py              组件进程身份与放行
      run_round.py                   SHA、资源、时限与独占检查
      r92_initial_gate.py            原四 rank fresh 初始化放行
      bundle/parallel_fresh_train.py  实际分布式训练入口
        retrain/distributed_launcher.py  rank 监督与退出
        retrain/distributed_engine.py    更新、online/EMA 与内部 dev
      bundle/single_model91.py freeze
        retrain/single91.py          内部 dev 唯一选点与冻结
      bundle/single_model91.py official-once
        retrain/single91.py          一次性 reservation 与官方评估
```

`single91.select_checkpoint` 要求 8 周期 × online/EMA 共 16 个内部 dev 读数，依次按 `error_twice` 升序、AP 降序、成功更新数升序、EMA 优先选唯一项。R92 最终选中 cycle5 online@4000；历史三模型永久保留 EMA，不改变本轮真实选点。

`single91` 在冻结时核 checkpoint、tensor、源码和内部 dev 身份，再为官方评估写一次性 reservation。已有 R92 完成结果见 [records/R92/](../records/R92/README.md)，不得重复执行这些阶段。

## 模块职责

下表文件均相对 `runtime/bundle/retrain/`：

| 职责 | 主要模块 | 使用边界 |
| --- | --- | --- |
| 配置、合同、公开来源 | `config.py`、`fresh.py`、`experiment91.py`、`models.py` | 科学参数与 fresh 身份；不由环境发现改写 |
| 数据、图像与年龄 | `data.py`、`preprocessing.py`、`verification.py`、`age.py` | 固定划分、只读缓存、fit-only 年龄合同 |
| 模型构造 | `models.py`、`imagenet_resnet.py`、`flair_model.py`、`dinov2_model.py` | 按 job 构建对应骨干；N06 的 FLAIR 仅研究保留 |
| 分布式执行 | `execution_policy.py`、`distributed.py`、`distributed_launcher.py`、`distributed_engine.py` | 拓扑、设备、rank、训练与协调退出 |
| 更新与数值 | `engine.py`、`precision.py`、`finite.py`、`mixup.py`、`ema.py` | 优化器、精度、有限值与显式 EMA variant |
| 模型持久化 | `persistence.py` | 精确 blob、原侧车、SHA 与 checkpoint index；不可凭 orphan 猜选模型 |
| 内部/官方评估 | `parallel_evaluation.py`、`inference.py`、`single91.py`、`n05_metrics.py` | 内部 dev 选点、一次官方评估和精确指标 |

## 其他入口的定位

| 文件 | 定位 |
| --- | --- |
| `runtime/bundle/predict.py` | `retrain.inference.main` 的通用推理命令入口；不能替代当前官方一次性合同 |
| `runtime/bundle/train.py` | 通用训练实现/环境和源码身份辅助；R92 实际由 `parallel_fresh_train.py` 派发 |
| `runtime/bundle/parallel_evaluate.py` | 独立并行评估入口正按显式site适配；不能因文件存在绕过内部选点和官方一次性合同，不是历史R92官方入口 |
| `runtime/bundle/fresh_train.py`、`distributed_train.py`、`stage_train.py`、`pipeline.py` | 历史训练流程入口；保留来源追溯，不因文件名存在就另行启动 |
| `tools/activation/prepare.py`、`activate.py`、`activate_audit.py`、`site_preflight.py` | 新显式准备/绑定链路；以当前CPU回执与后续冻结版本核准，旧生成件不能沿用 |
| `tools/observer/`、`data-contract/`及未适配的按轮工具 | 历史路径和SHA只用于追溯；不得直接派发新环境 |
| `tools/validation/check_candidate.py` | 新生产模型CPU集成检查；不执行图片前向或GPU准入 |
| `tools/pretraining/`、`tools/migration/` | 公开来源严格映射、当前内容证明和exp-v3复制/缓存索引；不提供Git写或GPU派发 |
| `tools/environment/pod_cpu.sh` | 新 CPU 工具通道：实时查 Pod、解释器并核 UID/Ready；没有训练派发能力 |
| `tools/models/export_parameters.py` | 精确读取三模型参数与 EMA 身份；不建模型、不前向、不复制权重 |

历史冻结部署保持原bytes和来源SHA。当前八卡适配在同一源码路径演进，具体文件边界见[适配任务](../todo/exp-v3-8card/README.md)；必要Pod CPU验收后先同步Git，再由唯一owner生成新绑定并核实际引用，之后才GPU准入。通用执行流程见 [EXECUTION.md](EXECUTION.md)，未完成验证见 [todo/](../todo/README.md)。
