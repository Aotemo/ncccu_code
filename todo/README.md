# 当前任务与接续

目标是在固定 9:1、单模型和内部 dev 唯一选点的合同下继续实验；raw AUROC 尚未达到 0.999，不称总体完成。自 R117 起，只有完整训练并冻结的所选内部 dev AUROC 达到 `0.996`，才运行一次官方 5849 验证；未达则记 `not_run_by_policy`。官方结果不用于训练、选点或下一轮调参。长期约束见 [AGENTS.md](../AGENTS.md)，执行入口见 [EXECUTION.md](../docs/EXECUTION.md)。

R113 冻结早期特征与 R114R1 最后下采样 stride1 均已完整结束，内部主指标未严格胜 R107 `0.9884405731067525`，各自分支按预定停止。原 R114 1426 步是工程失败，不充当完整科学结论。精简记录见 [R113](../records/R113/README.md)、[R114R1](../records/R114R1/README.md)及对应决定页。

R115 的固定 fit-only SSL1600 联合轮已完整结束，内部所选 EMA c4 AUROC `0.9863815501913928`，未胜 R107，分支按预定停止；原 once official 单报，不参与选参。必要归档见 [R115](../records/R115/README.md)。指定的[全历史复盘](auroc-999/EXPERIMENT-REVIEW-20261030.md)已完成，用户暂停后续训练的等待条件由协调确认解除。

R116 Base 配对完整内部所选 EMA c4 AUROC `0.987677169289011`，未严格胜 R107，分支不做 seed27；原一次官方已自然完成并单报，资源安全退出。精简结论见 [R116](../records/R116/README.md)。

R117 Swin V1 Base 配对完整内部所选 EMA c7 AUROC `0.9850076641408516`，未严格胜 R107，分支不做 seed27；内部低于 `0.996`，官方按政策未运行。round 已完成且资源安全退出。精简结论见 [R117](../records/R117/README.md)。

R118 DINOv1 ResNet50 来源替换已完整结束，内部所选 online c5 AUROC `0.970734180141627`，低于 R111 和预冻 T29，分支不做 seed27 或 A1 替补；低于 `0.996`，官方按政策未运行，round 已完成且资源安全退出。精简结论见 [R118](../records/R118/README.md)及[原决定](auroc-999/R118-DECISION.md)。下一独立科学方案尚未采用，不据此启动新轮。

当前执行 A `e62c5cb0-3516-4798-8b37-ed8da2d958ad` 是唯一 Git 写入者和 GPU owner；规划负责科学决定；执行 B 无 Git 写入、GPU、activation 或 launch 权。原执行者的历史回执保留，运行和 Agent 均不恢复。CPU 准备与训练分别使用实时有效 allocation 的互斥份额，入口须重新核 Pod UID、Ready、TTL、预算及独占状态。

源码与必要精简结论维护在本仓库；数据、模型、图像、样本预测和运行原件保存在 `/public/home/acfwvv9ejn/SothisAI/exp-v3`。当前 Pod、解释器、CPU 和设备状态均由 [动态发现工具](../docs/EXECUTION.md)实时核验，不把本文记录当默认配置。本机仅做文本、Git、传输和展示；解析、哈希、测试及模型执行在受控 Pod 内。
