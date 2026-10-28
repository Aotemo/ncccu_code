# 源码入口与依赖

本页根据已导入源码说明调用关系。全部后续实验已暂停；这些入口不构成启动授权，也不是可直接复制到任意环境的命令示例。

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
| `runtime/bundle/parallel_evaluate.py` | 历史独立并行评估入口，含旧 N08 `n05_gates`/`aggregate`/`SCIENCE_CONFIG.json` 绑定；相关旧门未在当前仓库作为完整包提供，不是 R92 官方入口 |
| `runtime/bundle/fresh_train.py`、`distributed_train.py`、`stage_train.py`、`pipeline.py` | 历史训练流程入口；保留来源追溯，不因文件名存在就另行启动 |
| `tools/activation/`、`observer/`、`data-contract/`、`validation/` | 原 R92 等按轮准备/绑定工具，含原运行路径；新环境不能直接复用 |
| `tools/environment/pod_cpu.sh` | 新 CPU 工具通道：实时查 Pod、解释器并核 UID/Ready；没有训练派发能力 |
| `tools/models/export_parameters.py` | 精确读取三模型参数与 EMA 身份；不建模型、不前向、不复制权重 |

历史绑定脚本没有在这次文档整理中重写数值实现或移动内部模块，原科学语义与来源 SHA 保持可追溯。未来若经用户恢复而适配新环境，应在同一源码路径修改、先同步 Git、再由唯一 owner 生成新绑定并核实际引用。通用执行流程见 [EXECUTION.md](EXECUTION.md)，未完成验证见 [todo/](../todo/README.md)。
