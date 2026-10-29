# R105：在既有 seed20261027 下复核固定尺度干预

2026-10-29 协调正式采纳一次匹配复核：直接对照 R102 Small/seed20261027，仅新增 `job.training_scale=[0.85,1.0]`，替代 R102 缺省的 `[0.94,1.0]`。与 R104 比较则只改变训练 seed20261029→20261027；不是新增 seed 或继续搜索尺度。本轮科学选择不等待官方指标、B 辅助分解或结果归档，A 可并行最小准备。

- 配置：`configs/R105-SCIENCE.json`。
- profile：`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-v1`。
- candidate：`r105-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-9010`；order15。`p4` 为 patience4，采样仍是 G64/P8。

## 固定比较与解释

| Small 训练 seed | 原尺度 `[0.94,1.0]` | 固定干预 `[0.85,1.0]` |
| --- | --- | --- |
| 20261027 | R102（本轮直接基线） | R105 |
| 20261029 | R103 | R104 |

分别报告 R102→R105、R103→R104 的所选内部整体 AUROC、AP、错序贡献及停止步数；不跨 seed 挑最高结果拼比较，不把不同早停长度称为等计算量。R102 的既有内部基线为 c3 EMA@2400，AUROC `0.9867391699609026`、AP `0.8333424024491386`、error_twice `22026`，5600更新/7周期正常早停。

信息目标是检验同一个已定尺度干预的观察方向是否仅限 seed29，而非寻找最高分 seed。主要反例是 seed27 无益或反向，表明当前结果依赖 seed/优化与选点轨迹；即两个配对方向一致，也只有两个既有 seed，不能证明显著性、可靠方差、普遍收益或尺度病因。

尺度下限变化仍同时影响有效分辨率、填边及相同 dx/dy 比例下的实际位移。现有 rotate-expand 与后续 fit-to-canvas 不变；不新增裁切不等于保留全部临床细节，可能因细节损失或边框依赖而无益。0.85 保持既定有限干预值，不用 80 张描述图反推最优参数。[聚合分析与8例审计](../../records/analysis/20261029-internal-errors/README.md)仅支持描述与同源标签传播一致；独立医学标注/患者关系仍未知，用户仅有图片与 CSV，不改标签、不删样本、不重采或回灌 dev。

## 全部保留的合同与实施边界

沿 R102：`convnext_small` 与公开 `timm/convnext_small.fb_in22k_ft_in1k@fe6b358aa409414cdb522ae3d43a2c9922f7cc1b` fresh 初始化、seed20261027、BCE、512/原 prepared512、FP32、G64/P8、SD0.1、dropout0.2、backbone LR1e-4、head/age 独立各1e-3、LN1e-6、原 WD/clip/normalization 不变。固定9:1 fit46256/dev5140、splitseed20261024、原 manifest/FREEZE 与 fit-only age 统计不变，不使用旧训练 checkpoint 或 N06。

复用 R104 已落地的配置注入；训练尺度上限1，镜像/颜色/旋转/dx/dy 及所有随机调用数量、顺序保持。验证仍 scale1/居中原视图。无新资产、缓存预处理、模型视图、损失或分布式算法；仅必要 SCIENCE/候选身份接线和已有流程接续，不新增平台、套件或独立 GPU 预演。

Small seed27 既有公开 fresh 工厂控制为 `91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67`。由新 Git 正式包的原生 CPU 流程实际匹配该初态、空 Adam 与既有组合同，不加载 R102 checkpoint、不沿用 seed29 的50c初态、不额外建立 control-only 流程。

最多24000成功更新，每800更新一周期；每完整周期以 max(online,EMA) 内部 dev AUROC 严格刷新历史最好值才重置无提升计数，首轮 baseline、patience4/min_delta0/min_cycles1。warmup/EMA warmcopy绝对1600、首blend1601、cosine horizon24000/尾倍率0.01、EMA decay0.999 保留。合法早停为 completed，不延长 patience；按所有实际已完成周期的既有 error_twice/AP/update/EMA 规则选唯一 checkpoint。

沿集中工程配置 train21600秒、total28800秒、GPU grant32400秒，准备20%/训练80%使用实时有效 CPU、TTL 与互斥池；不热改旧 clock/grant，不将工程限额当作测得吞吐或云实例租期保证。官方5849仅一次最终独立比较，不参与科学选择、早停或辅助分组。

## 预定判据

整体5140 dev 的既有选点与早停优先。R105 合法完成后若所选整体 AUROC 未超过匹配 R102，则该 seed 不支持此次主目标收益；若与 seed29 方向不一致，明确报告依赖 seed，不挑最好 seed、再扫尺度或新增 seed 补救。即整体提高，也仅为这次配对观察，不承诺0.999或显著性。AP 独立分报，不能替代主指标或挽救未胜的主结论。

固定现有20难正例与其余正例分别对全部负例的错序贡献、修复/新增可作辅助；负组另报，不与正组相加重复计数，不依据新结果重选难组。B 当前分解未出时不推断 R104 已解决共同难例；之后结果只影响解释，不构成批准门、不改变整体选点或本次科学值。难组改善被其它组退化抵消不算整体进步，整体提高但难组不变也不能宣称解决原难例机制。固定 dev 的自适应选择偏差保留。

## R104 事实来源收口

已文本核 A 普通导回的原生 freeze：c1–c7 各 online/EMA 共14条，唯一 selected 为 c3 EMA@2400，内部 AUROC `0.9850823187728706`、AP `0.8116753198258198`、error_twice `24778`。TRAIN_RESULT 为 completed、5600成功更新、7周期、bad_cycles4/stopped=true，训练组件 exit0、hard_deadline_met=true、signals为空；A 实核 attempted=successful、skip0。c4–c7 的两条轨迹均未刷新 c3 的最好 AUROC，合法早停；c4 EMA 的 AP 高于所选 c3，仍按预定主指标选点，不能改挑 AP 最大项。

相对匹配 R103 的 AUROC `0.9840901346312001`、AP `0.8089351087388285`、error_twice `26426`，R104 所选主指标方向为提高，AUROC 约增加0.000992，AP也提高。它未触发预定“未胜主指标”的否定分支，但只支持 seed29 下的正向观察，不证明稳定收益或解决共同难例。R104 停止5600/选点2400，与 R103 停止10400/选点7200不同；这是同选点/早停政策的比较，不是等步数对照。R94 历史 AUROC `0.9867415781748388` 仍略高于 R102，且 horizon 不同，不称 R104 或 R102 为全历史新纪录，也不拿 R94 替代匹配基线。

R104 冻结 Git 为 `018fc627d1a93915bfec9a85b03441bcd28974af`。内部原件已普通导回工作区 `work/exp-v3-8card-review-20261028/r104-update/r104-terminal-export/`；原件位于 exp-v3 对应 R104 round 的 `SINGLE-MODEL-FREEZE.json`、`evidence/TRAIN_RESULT.json`，其原 Pod SHA 为：

| 原件 | 原 Pod SHA256 |
| --- | --- |
| `SINGLE-MODEL-FREEZE.json` | `e810b994214e64f287a39a02cfa6cc767f878ab35f82dbe42992c5e0398236ae` |
| `evidence/TRAIN_RESULT.json` | `bd250a25db76b88ad0f104dc3f95ca604077d9af23ed6f685a5b0d2abebd3f76` |

本机仅文本读取，未重新解析、计算或哈希。A 于13:35:06实际核验 train/freeze 已 exit0，原一次官方仍活跃、未安全交接；官方数值未参与本决定。B 固定辅助分解尚未用于本文，不等待它作批准门，也不推断难例已改善。

A 统一必要实现、本页与原始小回执一次 Git 同步，按实际 commit/tree/源码与配置身份冻结部署；只有在 R104 整轮工程有效、原一次官方结束及进程/锁/设备实际安全交接后，A 才唯一一次 fresh 启动。规划本次只新增本页，不另改状态、不 Git 写、不新增批准门，原运行不热改。

原件已核，单件决定冻结交 A；不追逐后续过程状态修改本文。
