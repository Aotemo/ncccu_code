# R104：匹配 R103 的一次训练尺度下限对照

2026-10-29 协调明确选择一次有限干预：以预先登记的 R103 Small、训练 seed20261029 为直接基线，唯一科学配置变化为 `job.training_scale=[0.85,1.0]`，替代旧字段缺省行为 `[0.94,1.0]`。0.85 是本轮预定值，不是从 80 张描述样本反推的最优值；不搜索其它下限，不按结果改选 seed 或基线。

- 配置：`configs/R104-SCIENCE.json`。
- profile：`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-v1`。
- candidate：`r104-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-9010`；order14。`p4` 表示 patience4，采样仍为 G64/P8。

## 假设与反例

假设是在 fit 侧增加温和的尺度/填边变化，可能改善在原验证视图上的整体排序；现有分析只提供取景和内容占比的描述性关联，没有证明尺度是错误原因。选择此轴主要因为可复用资产、单视图计算和既有启动链，避免立即增加一致性前向或改变分布式损失。

现有流水线先旋转且 `expand=True`，再缩放并粘贴到 512 画布。降低尺度下限会同时降低有效分辨率、增加填边、扩大相同 dx/dy 比例对应的实际位移；一个配置参数不等于单一作用机制。上限保持 1，不新增对当前预处理图的裁切；保全这一视野不等于保留全部临床细节。主要反例是缩小丢失细节、强化黑边依赖，或关联实际来自长宽比/内容差异，导致整体排序不改善。

既有 80 张描述样本不能证明统一模糊、低对比或黑边更多。固定 8 例标签审计仅确认同源传播一致，用户确认只有图片和 CSV，缺乏独立标注与采集依据；不能由低分推断错标，不能排除未知患者/双眼/近重复关系。本轮不改标签、不删样本、不重采 dev、不回灌 dev、不统一提亮或锐化。详见[聚合分析与审计](../../records/analysis/20261029-internal-errors/README.md)。

## 匹配合同与最小接线

其余全部沿 R103：`convnext_small` 与 `timm/convnext_small.fb_in22k_ft_in1k@fe6b358aa409414cdb522ae3d43a2c9922f7cc1b` 公开 fresh 权重；seed20261029、BCE、512/原 prepared512、FP32、G64/P8、SD0.1、dropout0.2、backbone LR1e-4、head/age 独立各1e-3、LN1e-6、原 WD/clip/normalization 不变。固定9:1 fit46256/dev5140、splitseed20261024、原 manifest/FREEZE 与 fit-only age 统计不变，不使用 N06 或历史训练 checkpoint。

保留镜像、亮度/对比度/颜色、旋转、dx/dy 分布及全部增强随机调用的数量和顺序。仅训练分支读取新增尺度字段；旧字段缺省仍为 `[0.94,1.0]`，验证分支继续 scale1、居中原视图。A 负责 SCIENCE、配置接收、训练构造透传及既有身份接线；B 仅负责 `data.py` 与约定的两个小像素用例文本，A 统一必要 CPU 验收。复用缓存和权重，不重做预处理、不新增视图或模型前向、不改变本地 BCE mean 与分布式聚合语义。

最多24000成功更新、每800更新一周期；每完整周期按 max(online,EMA) 内部 dev AUROC 严格刷新历史最好值才清零，首轮 baseline、patience4/min_delta0/min_cycles1。warmup/EMA warmcopy绝对1600、首blend1601、cosine horizon24000/尾倍率0.01、EMA decay0.999 保留。合法早停为 completed，不延长早停；按实际全部已完成周期的既有 error_twice/AP/update/EMA 规则唯一选点。官方5849仅原有一次最终独立比较，不参与本决定、早停或后续调参。

工厂与 seed 不变，整体 fresh tensor 预期匹配 R103 实测 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`；由新 Git 正式包的既有原生 CPU 流程一次真实确认，不加载旧 checkpoint、不另做 control-only 流程。沿集中工程预算 train21600秒/total28800秒/GPU grant32400秒；准备20%与训练80%按实时有效 CPU/TTL 和互斥池绑定，不热改旧运行。工程限额不是吞吐或云租期保证。

## 预定证伪与辅助报告

完整 dev 的既有选点与早停保持不变。若 R104 合法完成后，所选 checkpoint 的整体 AUROC 未超过工程有效的匹配 R103，则该次干预不支持主目标收益；不能用 AP、top1 或难组改善挽救主结论，不继续扫下限。若仅有微小正差，只报告本 seed 观察，不证明统计显著性、尺度病因或达到0.999；AP 独立分报，不用 AP 替换主指标。

辅助组沿用现成固定难正例组，不按 R104 结果重挑。分别报告“固定难正组 × 全部负例”和“其余正例 × 全部负例”的错序贡献与修复/新增变化；负组分解另报，不与正组贡献相加，避免重复计数。难组改善被其它组退化抵消，不算整体改善；整体提高但难组不变，不能宣称解决了已识别难例机制。辅助组不参与 checkpoint 选择或训练；多轮使用同一 dev 的自适应选择偏差仍存在。

## R103 内部事实与来源

已文本核对 A 普通导回的原生 freeze：c1–c13 各 online/EMA 共26条，唯一 selected 为 c9 EMA@7200，内部 AUROC `0.9840901346312001`、AP `0.8089351087388285`、error_twice `26426`。TRAIN_RESULT 为 completed、10400成功更新、13周期、bad_cycles4/stopped=true，训练组件 exit0、hard_deadline_met=true、signals为空；A 实际观测 attempted=successful、skip0。c10–c13 未超过 c9 历史最好值，不能把 c13 或 online best 当最终选点，也不能把轨迹写成单调下降。

| 匹配训练 seed | Tiny 所选内部 AUROC | Small 所选内部 AUROC | 该 seed 的方向 |
| --- | ---: | ---: | --- |
| 20261027 | R95：0.9866645153288838 | R102：0.9867391699609026 | Small 略高 |
| 20261029 | R101：0.9843911613732117 | R103：0.9840901346312001 | Small 略低 |

不能挑各模型最高 seed 拼接比较；两组方向不一致，不能称稳定 Small 收益，两 seed 也不足以证明方差或差异显著。架构与公开 checkpoint 成对变化，相同 seed 不保证跨架构的随机轨迹相同。本次仍以预登记的 R103 为直接对照，不因其结果较低而回选更高 seed；早停长度不同，不声称等计算量。

R103 冻结 Git 为 `44b883ece72f9428bad9992a60919b48d4ca01ad`。内部原件远程根为 `/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r103-small22k-seed20261029-24k-p4-9010-20261029`，已普通导回工作区 `work/exp-v3-8card-review-20261028/r103-update/r103-terminal-export/`。

| 原件 | 原 Pod SHA256 |
| --- | --- |
| `SINGLE-MODEL-FREEZE.json` | `552f29f35d1f73e5733afb3d264b9bb543a57caf52d4857173d72b3c81d381fa` |
| `evidence/TRAIN_RESULT.json` | `8e01269893a5e22247492056fc47ccb31c42bdb057a663ceb3ddc07a17939ee3` |

本机只文本读取，没有重算、解析或哈希。A 于12:23:57实际核验时 train/freeze 已退出0，原一次官方仍活跃、尚未 safe handoff；这不影响预定科学选择，也不代表新 GPU 可启动。时间自相矛盾的外部快照不作为证据，官方数值未参与本页。

A 统一必要实现与本页、既有聚合分析一次 Git 同步，按实际 commit/tree/文件与配置身份冻结部署。新 GPU 仅在 R103 整轮工程有效、原一次官方结束且进程/锁/设备实际安全交接后由 A 一次 fresh 启动；内部训练结束本身不表示 GPU 可接续。不新增独立 GPU 预演、平台、批准门或过程提交，活跃 R103 冻结包不改。

规划仅新增本页，原件已核、决定冻结交 A；不追逐后续过程状态修改本文。
