# 固定 9:1 单模型实验

文档整理日期：2026-10-28。进度接续已导回的真实最终回执，不重复训练或官方前向。

**最新用户授权覆盖旧暂停，当前实例继续使用；1规划+2执行及A/B各1名文本子Agent，不恢复旧Agent或旧运行。** A是唯一Git写入者/GPU owner，R93工程终态与GPU安全退场已核，直接接续已定稿[R94单变量方案](R94-DECISION.md)的最小实现；B来源/迁移和最小LR函数检查已完成，子Agent由父级统一验收且不再派生。角色、边界和环境见[本轮接续](../exp-v3-8card/README.md)，原heartbeat保持删除。

## 目标与合同

- raw AUROC ≥ 0.999；内部 dev 与完整官方 5849 分别报告。未达目标不标完成。
- fit 46256、dev 5140，固定 9:1；单模型、合规公开 fresh 初始化。
- 全局有效 batch64、全局正例配额8，reference physical32；8卡目标为每rank8、accum1。不强制每rank正例数相同，不改变原全局采样流。
- 每周期 800 次成功更新，8 周期共 6400；warmup 与 EMA warmcopy 均为 1600 次更新。
- 保持512、FP32、EMA decay0.999、head/age LR1e-3、weight_decay1e-4和fit-only age；backbone LR由本轮版本化SCIENCE合同读取：R93为2e-4，已书面决定的R94唯一科学变化为1e-4。不能因卡数修改科学参数或更新时钟。N06禁止用于初始化、teacher、BN或训练参数来源。
- 8周期×online/EMA共16条完整内部dev，按error_twice升序、AP降序、成功更新数升序、EMA优先选择唯一checkpoint，再做一次完整官方最终评估。官方不用于训练、年龄统计、早停、难例、后续调参或选点。

## 已确认进度

| 实验 | 最近记录的状态 | 内部 dev AUROC | 官方 AUROC | 依据 |
| --- | --- | --- | --- | --- |
| R91 ResNet50 ImageNet1K V2 | 已完成，未达目标 | 0.9732399267421321 | 0.9746659041823744 | [完成记录](../../records/R91/README.md) |
| R92 ConvNeXt Tiny ImageNet1K | 已完成，6400 更新/8 周期/0 skip，终态 audit 22/22、exit0；未达目标 | 0.9829293755140032 | 0.9833687740474035 | [最终回执](../../records/R92/FINAL-RECEIPT.json) |
| R93：ConvNeXt Tiny ImageNet22K→1K | 冻结80e06f31c33b完成6400/0 skip，工程终态25/25及GPU安全退场通过 | c3 EMA@2400；AUROC0.9866344126546825、AP0.8167243430095755、error_twice22200 | 原生一次完成，独立分报，不用于下轮决策 | [内部原件](../../records/R93/INTERNAL-DEV-SUMMARY.json)、[运行实证](../../records/R93/README.md) |
| R94：同来源、backbone LR1e-4 | 科学决定已冻结并交A最小适配；尚未开训 | 尚未运行 | 尚未运行 | [单变量决定与实施范围](R94-DECISION.md) |

Tiny22K→1K来源末端LN为1e-6，R92为1e-12；匹配实际来源，不把R93差异说成仅预训练来源。原blob与转换权重SHA、完整来源链、B独立工厂strict-load及A生产工厂集成均已有真实回执，分别保留源码身份。旧[预训练盘点索引](../../configs/PRETRAINING-ASSETS.json)只保留来源，不替代当前核验。R94已依据R93完整16条内部结果、工程终态和LR函数可行性决定为单变量检验，不预定收益，不使用官方指标选择参数。

## 待办与验收

- [x] R92 已有真实训练终态、6400 成功更新、8 周期和组件/owner exit0 回执。
- [x] 内部规则选出唯一 cycle5 online@4000；冻结来源与 tensor SHA 已实核。R92 选中项为 online，不能因保留的历史三模型为 EMA 而改写。
- [x] 官方完整 5849 一次最终评估完成，内部与官方指标分列；未达目标。
- [x] 真实最终结果已归入 `records/R92/`，原实例锁释放、owner/ranks 已退场。
- [x] 历史暂停已有记录，最新用户授权现已覆盖暂停；原heartbeat保持删除。
- [x] 新规划、执行A及追加执行B已接单，旧Agent不恢复。
- [x] 八卡最小适配、来源与资产验收完成，具体差异和Pod CPU真实回执齐全。
- [x] 执行A统一GitHub同步，核远端commit及实际SHA冻结部署，再进行GPU准入和训练。
- [x] 有界八rank准入通过，R93按精确合同以新fresh绑定一次激活并实际成功更新。
- [x] R93完成6400成功更新、完整内部16条选点、一次官方最终比较及真实退场状态核实；按内部/工程证据形成R94书面决策。
- [ ] A完成R94最小配置/合同适配、沿用已有CPU证据、Git同步实际冻结后一次fresh开训；按2026-10-29用户指令取消独立零forward GPU预演，由真实训练初态校验接管，删除重复LR常量断言，不新增拒绝case/重跑已过检查，不触碰R93运行身份。
- [ ] 仅在证据齐全且实际达标时标记总体完成。

## 历史运行身份（不再接续）

以下值记录已完成 R92 的历史身份和既有协作，只用于对账；不授权再次 activation/launch、官方前向或重置 clock：

- R92 token：`f1d4b2f895a84a15a3d81634b5153cee`。
- 既有正式根：`/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/rounds/R92-TINY1K-9010`。
- 原规划 ID：`f6bd0299-40c6-4928-916b-5c0c7d766ad8`；原执行 A ID：`d4fc4135-2309-4712-9ee3-182fb79e4ef0`。原GPU owner身份仅属于当时运行，不是当前权限。
- 原 heartbeat：`b64409b9`，已因用户暂停删除（协调会话已获工具 success 回执）；未经新的明确授权不恢复。

历史值不得成为当前默认绑定。当前角色按工具实时核验和[本轮接续](../exp-v3-8card/README.md)记录操作；已有绑定须与真实身份相符，unknown不当作空闲或成功。不得自动改绑；仅使用本次获准的A/B各1名文本子Agent，不再派生或创建计算资源/定时训练/heartbeat。
