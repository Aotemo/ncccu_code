# 当前任务与接续

目标是在固定 9:1、单模型和内部 dev 唯一选点的合同下继续实验；raw AUROC 尚未达到 0.999，不称总体完成。官方验证仅用于每轮最终比较，不用于训练、选点或下一轮调参。长期约束见 [AGENTS.md](../AGENTS.md)，执行入口见 [EXECUTION.md](../docs/EXECUTION.md)。

R113 冻结早期特征与 R114R1 最后下采样 stride1 均已完整结束，内部主指标未严格胜 R107 `0.9884405731067525`，各自分支按预定停止。原 R114 1426 步是工程失败，不充当完整科学结论。精简记录见 [R113](../records/R113/README.md)、[R114R1](../records/R114R1/README.md)及对应决定页。

R115 的固定 fit-only SSL1600 联合轮已完整结束，内部所选 EMA c4 AUROC `0.9863815501913928`，未胜 R107，分支按预定停止；原 once official 单报，不参与选参。必要归档见 [R115](../records/R115/README.md)。指定的[全历史复盘](auroc-999/EXPERIMENT-REVIEW-20261030.md)已完成，用户暂停后续训练的等待条件由协调确认解除。

当前唯一候选是 [R116](auroc-999/R116-DECISION.md)：直接从 R107 seed20261029 出发，只把 Small 与其公开权重成对换为普通 ConvNeXt Base 及对应公开 ImageNet22K→1K 权重。512、单视图、P8/448 pairs、FP32、原优化与 24k/800/p4、6h/8h/9h 保持。新来源、Base 初态、存储上限和当前 Pod 资源均须在正式包及唯一 fresh 前按实测回执验收；完整内部 AUROC 严格胜 R107 才考虑固定 seed27。

当前执行 A `e62c5cb0-3516-4798-8b37-ed8da2d958ad` 是唯一 Git 写入者和 GPU owner；规划负责科学决定；执行 B 无 Git 写入、GPU、activation 或 launch 权。原执行者的历史回执保留，运行和 Agent 均不恢复。CPU 准备与训练分别使用实时有效 allocation 的互斥份额，入口须重新核 Pod UID、Ready、TTL、预算及独占状态。

源码与必要精简结论维护在本仓库；数据、模型、图像、样本预测和运行原件保存在 `/public/home/acfwvv9ejn/SothisAI/exp-v3`。当前 Pod、解释器、CPU 和设备状态均由 [动态发现工具](../docs/EXECUTION.md)实时核验，不把本文记录当默认配置。本机仅做文本、Git、传输和展示；解析、哈希、测试及模型执行在受控 Pod 内。
