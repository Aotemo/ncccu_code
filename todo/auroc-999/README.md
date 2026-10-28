# 固定 9:1 单模型实验

文档整理日期：2026-10-28。进度接续已导回的真实最终回执，不重复训练或官方前向。

**用户已明确暂停全部后续实验，先梳理 GitHub 代码结构。现规划与唯一执行 A 已确认暂停，R93 尚未启动；未经用户新的明确恢复指令，不创建新实验文件/作业，不做后续试跑或准备。** 原 heartbeat 已删除；仓库整理与三个保留模型参数归档按已授权范围收尾。

## 目标与合同

- raw AUROC ≥ 0.999；内部 dev 与完整官方 5849 分别报告。未达目标不标完成。
- fit 46256、dev 5140，固定 9:1；单模型、合规公开 fresh 初始化。
- 每周期 800 次成功更新，8 周期共 6400；warmup 与 EMA warmcopy 均为 1600 次更新。
- 内部 dev 按预定规则选择唯一 checkpoint，再做一次完整官方最终评估。官方不用于训练、年龄统计、早停、难例或选点。

## 已确认进度

| 实验 | 最近记录的状态 | 内部 dev AUROC | 官方 AUROC | 依据 |
| --- | --- | --- | --- | --- |
| R91 ResNet50 ImageNet1K V2 | 已完成，未达目标 | 0.9732399267421321 | 0.9746659041823744 | [完成记录](../../records/R91/README.md) |
| R92 ConvNeXt Tiny ImageNet1K | 已完成，6400 更新/8 周期/0 skip，终态 audit 22/22、exit0；未达目标 | 0.9829293755140032 | 0.9833687740474035 | [最终回执](../../records/R92/FINAL-RECEIPT.json) |
| ConvNeXt Tiny ImageNet22K→1K | 后续候选暂停，R93 未启动 | 未运行 | 未运行 | [预训练盘点索引](../../configs/PRETRAINING-ASSETS.json) |

## 待办与验收

- [x] R92 已有真实训练终态、6400 成功更新、8 周期和组件/owner exit0 回执。
- [x] 内部规则选出唯一 cycle5 online@4000；冻结来源与 tensor SHA 已实核。R92 选中项为 online，不能因保留的历史三模型为 EMA 而改写。
- [x] 官方完整 5849 一次最终评估完成，内部与官方指标分列；未达目标。
- [x] 真实最终结果已归入 `records/R92/`，原实例锁释放、owner/ranks 已退场。
- [x] 按用户要求暂停现有规划与执行；R93 未启动，原 heartbeat 已删除。
- [ ] 等用户新的明确恢复指令后，由现规划复盘后续候选，再由唯一执行 owner 接续。
- [ ] 仅在证据齐全且实际达标时标记总体完成。

## 最近运行与既有协作的接续记录

以下值记录已完成 R92 的历史身份和既有协作，只用于对账；不授权再次 activation/launch、官方前向或重置 clock：

- R92 token：`f1d4b2f895a84a15a3d81634b5153cee`。
- 既有正式根：`/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/rounds/R92-TINY1K-9010`。
- 先前授权规划 ID：`f6bd0299-40c6-4928-916b-5c0c7d766ad8`；执行 A ID：`d4fc4135-2309-4712-9ee3-182fb79e4ef0`，后者为唯一 GPU owner。
- 原 heartbeat：`b64409b9`，已因用户暂停删除（协调会话已获工具 success 回执）；未经新的明确授权不恢复。

恢复前先核对用户新的明确授权；仅查询 Agent 或发现环境不构成恢复授权。使用 Paseo `list_hosts`、`list_agents`、`get_agent_status`/`get_agent_activity` 核对上述角色是否仍属于本任务，再通过工具查询实际 Pod 和 UID。对已有绑定执行操作时，发现值须与该次正式绑定相符；不符则停止该次操作，保存事实。禁止新增 Agent、定时训练或 heartbeat。
