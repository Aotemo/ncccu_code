# R103：Small在预定seed20261029下的一次匹配复核

2026-10-29协调采纳规划与B一致建议：以[R102](R102-DECISION.md)为直接基线，唯一训练科学变化为`job.seed: 20261027 → 20261029`。补齐Tiny/Small与既有20261027/20261029两seed的固定对照，不搜索新seed、不选best seed、不自动扫参。该选择旨在复核已有表征差异，代码准备不等待官方数值或结果归档。

精确配置=`configs/R103-SCIENCE.json`；profile=`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-v1`；candidate=`r103-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-9010`；order13。此处`p4`表示patience4，采样仍为G64/P8。

## 对照与解释边界

| 公开表征组合 | seed20261027 | seed20261029 |
| --- | --- | --- |
| Tiny22K→1K | R95 | R101 |
| Small22K→1K | R102 | R103（本次） |

R103与R102比较Small自身的训练随机性敏感性，与R101比较同seed下的公开表征组合；两组架构对照分别为R95→R102和R101→R103，不把每个模型挑出的最高seed拼成比较。相同seed跨架构不保证相同随机数消耗或初态，架构与公开checkpoint仍成对变化，不能归因于纯容量。

R95所选内部AUROC0.9866645153288838/AP0.8193841805300731/error_twice22150；R101为0.9843911613732117/AP0.8278718323176861/error25926。两次Tiny结果相差约0.00227，只说明这两次的敏感性，不足以估计可靠方差或显著性。

已文本核R102原生freeze完整14条、selected及TRAIN_RESULT：5600成功更新、7周期、skip0、bad4正常早停，训练组件exit0、hard_deadline_met=true、无signals；唯一c3 EMA@2400内部AUROC0.9867391699609026/AP0.8333424024491386/error22026。其相对R95的AUROC增量仅约0.000075，不能据此宣称稳定容量收益；AP可独立分报。c6、c7 EMA从c5回升，c7为0.98656818677144，但仍低于c3，故c4–c7均未刷新历史最好值；不能写成后段单调下降，也不因此延长patience。A已于10:33:52实核整轮completed/complete、原组件exit0及资源安全交接，owner为Z只称inactive、不称reaped；规划科学依据仅用内部原件，不读取官方数值。

已文本核R94历史原件：c3 EMA@2400内部AUROC0.9867415781748388/AP0.8200017145302917/error22022，仍略高于R102，所以不能称“全历史新纪录”。R94是6400 horizon、固定8周期；R95/R102采用24000 horizon和patience4早停，R94仅用于纠正历史纪录，不能替代本次R95/R102的匹配基线。

本次主要反例是：再运行一次并不直接改善方法，可能只确认波动；但现有资产与接口已就绪，补齐缺失的同seed架构对照比继续添加超参轴更有助于判断当前差异。两seed依然不能证明显著性、稳健性或消除固定dev多轮选择偏差；若只有AP改善而AUROC无改善，按两个指标分别陈述，不改主目标、不挑最好seed补救。结果报告各自预定选点、完整内部轨迹和停止步数，不把不同早停长度称为等计算量实验，不承诺0.999。

## 全部保留的R102合同

复用`convnext_small`及`timm/convnext_small.fb_in22k_ft_in1k@fe6b358aa409414cdb522ae3d43a2c9922f7cc1b`公开来源、转换、许可与已验结构；不下载新模型、不继承任何训练checkpoint，不使用N06或保留EMA。BCE（无focal）、512输入/现有prepared512、FP32、G64/P8、reference32、8rank各8/accum1、显式SD0.1、dropout0.2、backbone LR1e-4、head/age独立各1e-3、LN1e-6、原WD/clip、增强、normalization均不变。

固定9:1 fit46256/dev5140、同manifest/FREEZE与fit-only age统计、`data_split.seed=20261024`不变。训练seed会影响fresh head、采样顺序、增强及模型随机性，不能称仅改head初始化，更不能重划分数据。

最多24000成功更新，每800更新完成一周期；每完整周期max(online,EMA)内部dev AUROC严格刷新历史最好值才清零，首轮baseline，patience4/min_delta0/min_cycles1。warmup与EMA warmcopy绝对1600、首blend1601、cosine horizon24000/尾倍率0.01、EMA decay0.999保持。合法早停为completed，不延长patience或追满预算；按实际已完成周期的error_twice/AP/update/EMA规则唯一选点。完整官方5849仅一次最终独立比较，不用于本决定、早停或后续调参。

工程预算沿R102集中配置：train21600秒、total28800秒、GPU grant32400秒；实际创建/到期身份另绑定未执行R103，不改旧clock、grant或冻结包。这些是6h/8h/9h工程限额，不保证吞吐或云实例租期，不申请/延长资源。准备20%与训练80%按实时有效CPU和有效TTL使用互斥池，不沿过期CPU5。实际超时或资源异常不能伪称科学正常完成。

## 最小实施与来源

A复用现有Small生产工厂、配置消费、准备及启动链，仅更新科学身份/seed及派生初态控制。通过现有CPU工厂真实生成新的整体fresh tensor，记录零age、空Adam、online=EMA后填入SCIENCE；不得复制R102的`91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67`，也不借用Tiny的eba或18313。新hash是seed变化的派生身份，不是新增超参。

只做改动必需的既有CPU接续，不重跑未变的Small/Tiny来源、转换、迁移或native套件，不新增平台、模板、独立GPU预演或测试套件。A统一一次Git同步，按实际commit/tree/源码与配置身份冻结部署；仅在R102完整工程有效且原一次官方/进程/锁/设备实际安全退场后一次fresh启动，不把内部训练结束等同于可用GPU。文档不是新批准门，不等待官方数值、常规确认或归档润色。

Small沿用的实际来源身份（来自既有Pod回执，本机未重哈希）：

- 原始权重：`c912f45cd738512978a900cc8aefade65bc44e507321f0144378a2d98aa5e914`；转换权重：`7aaea5ce6fe6db953874d9792d3f1711a5cc0193a10d740e023108043918c610`。
- provenance：`8ea3e957e2381a40e2f8fc5feae5df31d0d8b33faf3bdd3fa5749e64ae98368b`；唯一handoff：`929562052131ac0a458efbb99315231d3f157e4c1d6058a8463d0d3d66f92957`，见`technical-prep/small22k-20261029-b/SMALL-HANDOFF.json`。
- R102冻结Git：`620baa9d524f4d5978b79cbc3ad3b9c9ea4837b1`；内部原件已普通导回工作区`work/exp-v3-8card-review-20261028/r102-update/r102-terminal-export/`。`SINGLE-MODEL-FREEZE.json`原Pod SHA为`4a9ccb7646de8c63ec872e0ceb71fae0c200d1bd9c59ec6ca32f33690d9f68a4`；`evidence/TRAIN_RESULT.json`原Pod SHA为`59a72f04012fb728a119d0d7f7ce46b76544ca2f537f48a668624829311b0864`。本次只文本核内部两件及既有`source_snapshots`对应SHA，没有重新解析、计算或哈希。

规划本次只新增本页，实际内部证据已核，科学决定冻结交A与最小实现/实际证据统一集成；不追逐过程状态另作提交。
