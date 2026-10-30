# 当前任务与接续

目标是在固定 9:1、单模型和内部 dev 唯一选点的合同下继续实验；raw AUROC 尚未达到 0.999，不称总体完成。官方验证仅用于每轮最终比较，不用于训练、选点或下一轮调参。长期约束见 [AGENTS.md](../AGENTS.md)，执行入口见 [EXECUTION.md](../docs/EXECUTION.md)。

R113 冻结早期特征的实验已完成并安全退场。内部选择 c4 online@3200，AUROC 0.9838264352051979，低于预定 R107 对照 0.9884405731067525；按预定停止该分支，不启动 seed27。权威精简结果见 [R113 记录](../records/R113/README.md)及 [R113 决定](auroc-999/R113-DECISION.md)。其最终五件归档已同步至 GitHub main `8e3101519a401c7f72910d7345a31f6d28b28458`，tree `f6c18d5a16bd7f011d416a468d048eb55fb1ed68`。

当前唯一候选是 [R114](auroc-999/R114-DECISION.md)：直接对照 R107，只把 ConvNeXt Small 最后一次下采样的卷积 stride 从 2 改为 1。科学配置、生产接线和聚焦测试已文本准备，尚需在实际 Pod 完成必要 CPU 验证、同步新 Git tree、正式冻结和唯一 fresh 启动；此页不代表 R114 已开训。首轮内部主指标须严格超过 R107，才按已定计划考虑第二 seed。

执行 A 是唯一 Git 写入者和 GPU owner；规划负责科学决定；新执行 B 负责其独占的 R114 `spatial_stride.py`/聚焦测试文本及独立 CPU 来源核查。旧 B 和旧运行不恢复。2026-10-30 16:03 北京时间，实际 Pod UID `51fe3daf-ba0d-46e8-919c-99caec599c99` 下 CPU13 future 分配有效 120 核：准备池 0–23 中 A 0–11、B 12–23，训练池 24–119，互斥；环境只读回执见 `exp-v3/technical-prep/continuation-20261030-a/ENVIRONMENT-LIVE-2.json`。这项观测只证明当时设备无使用者和原锁释放，后续入口仍须实时核身份、预算和独占状态。

源码与必要精简结论维护在本仓库；数据、模型、图像、样本预测和运行原件保存在 `/public/home/acfwvv9ejn/SothisAI/exp-v3`。当前 Pod、解释器、CPU 和设备状态均由 [动态发现工具](../docs/EXECUTION.md)实时核验，不把本文记录当默认配置。本机仅做文本、Git、传输和展示；解析、哈希、测试及模型执行在受控 Pod 内。
