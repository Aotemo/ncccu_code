# R94训练期间的下一轮接续准备

2026-10-29：R94冻结`212b41f9d3aa922519e62182701bfbea527c852d`继续训练，B唯一只读观测为3200/6400、0 skip、4轮已评、owner及8rank同身份存活。本清单不是新启动门。用户随后直接要求R95最多24000成功更新、连续4轮无提升早停，并确认内部dev AUROC；按[R95合同](R95-PLAN.md)和A实际`configs/R95-SCIENCE.json`收口，不再留monitor待答槽，不按R94部分读数或官方结果另选参数。

## 分工与交付

B立即复用A已交的212b权威源码、干净Git归档、实际PREPARE/ACTIVATE参考及现有package manifest。仅准备尚未完成且不依赖R94结果的材料；既有接口够用就不改，不另建脚本平台、框架、证明桥、测试套件或Agent。A保留当前runtime/tools/configs及唯一Git/GPU写入权，B不改活跃包或与A共写文件。

| B独占新增文件 | 完成内容 |
| --- | --- |
| `configs/templates/NEXT-SCIENCE.json.in` | 按A实际R95字段准备24000/800×最多30、AUROC/max(online,EMA)、patience4/min_delta0/min_cycles1、绝对warmup与EMA1600；不重复实现科学配置，不把模板当已冻结运行 |
| `configs/templates/NEXT-IDENTITY.patch.in` | 使用A唯一R95 profile/candidate/order5，原身份替换位置若已由A实现则注明无需再次应用；不另拟名称，不修改现有运行源码，不引入通用身份框架 |
| `configs/templates/NEXT-RUN.env.example` | 按现有真实命令准备输入槽和调用顺序，复用显式`--science-config`及既有启动链；不复制旧clock/token或把旧Pod/资源份额当新默认 |
| `records/preparation/next-after-r94/READY.md` | 已就绪文件、现成源包/实际入口命令、零接口改动或确切剩余缺口；区分已准备材料与只能在最终Git/当前资源确定后生成的运行身份 |
| 同目录必要的`PACKAGE-RECEIPT.json` | 仅在实际需要时记录既有干净归档证据或必要Pod检查结果，不为填额度重验模型/native/18/旧9 |

准备产物位于独立`exp-v3/technical-prep/next-round-ready-20261029-b`。优先原样复用A已验证的干净Git归档，传输沿现成文件清单/`COPYFILE_DISABLE`方式，不夹带`._*`、`.DS_Store`、`__MACOSX`或旧生成包/控制状态。不要为了产生新回执重复打包已证实的同一对象。

本机仅文本/Git只读/传输；确有必要的包检查、解析或CPU调用只在实时核验UID/Ready、未过期validation allocation的B份额内完成，与A观测合计不超20%，和训练80%池互斥。不运行模型、native重验、GPU、activation、正式clock生成或官方前向。B可以准备命令材料，正式资源绑定、Git同步和启动仍由A完成。

## 收口条件与结果后动作

- [ ] B交付上述配置/身份/调用材料，并明确复用的干净源包及实际入口；不存在可提前解决而被留到结果后的接口缺口。
- [ ] 规划一次核对分工内材料；若现有接口无须改动，明确记录零改动而不是编造适配任务。完成后B停止准备，不为占满20%增加计算。
- [x] 用户明确R95最大预算/早停并选内部dev AUROC，具体语义见R95合同；不再等待同一可选澄清。
- [ ] A完成R95受影响实现及既有配置生成，补新Git版本和实际运行身份；R94工程终态与GPU安全退场核实后沿现成材料串行启动。R94完整内部结果独立归档，官方数值不改变本用户指定方案。

新Git确定后的实际文件digest、当前UID/TTL/资源分配、一次性clock/token不可能提前成为真实证据，须到时按现有入口生成；这不应成为重新研究或重复建立整条启动链的理由。科学变量若要求212b现有接口无法表达的真实新行为，先明确该必要差异，不预造框架。官方只作最终独立比较，不能用于选择下一轮参数。
