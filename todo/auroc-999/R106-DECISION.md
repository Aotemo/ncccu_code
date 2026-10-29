# R106：同图双视图平均 BCE 科学决定与短实施计划

**Goal:** 以预先匹配的 R104 Small/seed20261029 为直接基线，只将每个训练主样本的增强视图数由1改为2，检验所选整体内部 AUROC 是否提高。

**Architecture:** `ImageDataset` 一次读取原图或已有缓存，从未增强图独立生成两个同分布视图；共享 helper 展平视图并同序重复标签与可选年龄。两个训练入口各使用一次拼接 batch forward、现有 mean BCE 和一次 backward，主采样、coverage、optimizer/LR/EMA 时钟仍按原主样本和成功更新计数。

**Tech Stack:** 现有 PyTorch/DDP、Pillow、prepared512、版本化 SCIENCE 与原生 CPU/部署/一次 fresh 启动链；不引入新依赖或资产。

**Spec:** 2026-10-29 协调给定的 R106 精确合同；继承 [R104 决定](R104-DECISION.md)、`configs/R104-SCIENCE.json` 与 [执行规则](../../docs/EXECUTION.md)。本页同时承载科学决定与 writing-plans 短计划。用户已指定 A/B 分工、一次必要验收和唯一集成；覆盖技能默认的另建计划、额外 RED、频繁 commit、新 review Agent 或设计确认等待，不增加批准门。

## 身份与固定合同

- 配置：`configs/R106-SCIENCE.json`；唯一新增科学字段 `job.training_views=2`。
- profile：`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-views2-v1`。
- candidate：`r106-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-views2-9010`；order16。
- round：`r106-small22k-seed20261029-scale085-views2-24k-p4-9010-20261029`；`p4` 表示 patience4，采样仍为 G64/P8。
- 继承 R104：公开 fresh `convnext_small`，`timm/convnext_small.fb_in22k_ft_in1k@fe6b358aa409414cdb522ae3d43a2c9922f7cc1b`；seed20261029，`training_scale=[0.85,1.0]`，BCE（旧 `loss` 缺省语义）、FP32、512/原 prepared512、SD0.1、dropout0.2、backbone LR1e-4、head/age 独立各1e-3、LN1e-6、原 WD/clip/normalization 均保持。
- 固定9:1 fit46256/dev5140、splitseed20261024、原 manifest/FREEZE、fit-only age 统计及主 sampler 流不变。无新 seed/尺度搜索，不用 N06、旧 checkpoint、dev 回灌或重标。
- 每成功更新64个主抽样位置、8个正例位置，共128个增强视图；8rank各8个主样本、16个视图。视图不是新增独立样本；正例视图为16个，主配额仍 P8。coverage 与年龄统计不重复计算，optimizer、LR、EMA 每成功更新各推进一次。
- 每800成功更新验证一次，最多24000；每完整周期取 max(online,EMA) 内部 dev AUROC，严格刷新才重置 bad_cycles；首轮 baseline，patience4/min_delta0/min_cycles1。warmup 与 EMA warmcopy 绝对1600、首 blend1601、cosine horizon24000、原唯一 checkpoint selector 保持。验证始终原单视图，像素处理不变。
- 公开 fresh 工厂初态预期为 R104 已实得的 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`。新 Git 正式包原生 CPU 流程实际匹配，保持空 Adam，不加载 R104 权重，不另建 control-only。
- 集中工程配置 train21600秒、total28800秒、GPU grant32400秒不变。准备20%与训练80%按实时身份、TTL、互斥 CPU 池执行；不延长实例或申请资源。双视图约增加一倍图像前反向工作量，不是等 FLOP 比较，不保证两倍 wall time 或相同显存。一次拼接会提高激活峰值；实际超限如实失败，不偷改单视图或预算。

## 依据、反例与预定判据

R104 是预先选定的 seed29 干预链基线：c3 EMA@2400，整体内部 AUROC `0.9850823187728706`、AP `0.8116753198258198`、error_twice `24778`。不因 R105 更高峰值切换到 seed27；不同合法早停长度/选点仍按同政策比较，不称等更新或等计算量对照。

已文本核 R105 的14条内部记录与 TRAIN：5600更新/7周期正常早停，唯一 c3 **online**@2400，AUROC `0.987161811506687`、AP `0.8058457289282671`、error_twice `21324`。相对 R102 c3 EMA，AUROC 增 `0.000422641545784419`，AP 降约 `0.02749667`。B 固定组聚合为原20难正例净少1612错序对、其余147正例净多1261、整体净少351；pair mass 按 ties 半计，不是 FN/FP 人数。组选择曾使用 R102 结果，只属描述性辅助，不能证明机制、显著性或全面改善。R105 原 freeze/TRAIN Pod SHA 分别为 `82b582ea83446f79fd5f00e864b56a4d5a55f737e2441557ff9457006f8c7081` / `f61af0168808c3a78d16fa8cd213079b807afda7a1e41d4afe5f3e5aaa044bf9`；聚合原件在 `exp-v3/technical-prep/r105-error-decomposition-20261029-b/SUMMARY.json`，Pod SHA `909041fae7d068a068565db28f6cb9689bb63d4cf849dfa77653b1a5bcf4febf`，CPU guard SHA `aa40bb35394cea65c9ada7287fbd43ded1ac76d0dba5ac5f636488abd71c5d5b`。本机仅读取原件文本。

本轮改变视图采样数量，先不增加 JS/一致性系数，以免混合两项干预。两视图来自同一增强分布，在固定模型下平均 BCE 的监督目标期望不变；可能改变梯度估计与优化轨迹，但当前证据没有诊断增强噪声为瓶颈。主要反例是只增加计算而无整体排名收益，或 AP 继续下降；重复视图无法修复未知标签问题、缺失细节或表征不足。第一视图保留像素/RNG 流，不代表模型 dropout/SD 的 Torch 随机消耗或整个训练轨迹与 R104 相同。

主判据仅为合法完成后，按原 selector 所选完整5140 dev AUROC 是否超过 R104。未胜则不支持本次主目标，不以 AP 或难组改善挽救；胜但 AP 下降则如实记录取舍。固定原20难正例与其余正例对全部负例的错序贡献只作辅助，负组另报避免重复计数；不重选难组、不改变 checkpoint 选择。单 seed 不证明稳定、显著或尺度病因，固定 dev 的自适应偏差仍在；无论结果如何，不自动加 JS、调整视图数、扫 seed 或延长早停。官方5849仅一次最终独立分报，不参与本决定或后续选参。

## 精确文件与接口

| 负责人 | 文件范围 | 交付接口与边界 |
| --- | --- | --- |
| B | `runtime/bundle/retrain/data.py`、新增 `tests/test_training_views.py` | dataset 双视图与共享 flatten helper；仅文本实现和三个小型有意义测试，交 A 统一 Pod 验收 |
| A | `runtime/bundle/retrain/config.py`、`experiment91.py`、`engine.py`、`distributed_engine.py`；`configs/R106-SCIENCE.json`；既有 `tests/test_candidate_binding.py` 配置 fixture | 可选字段/身份消费、两训练入口接线；唯一 Git 写入、实际包与部署、唯一 GPU owner |
| 规划 | 仅本页 | 单件科学决定及短实施计划；不改源码、分析 README 或 R105 决定 |

1. `ImageDataset(..., training_views=None)`：缺省/1 保留旧 `[C,H,W]` tensor、全部原 RNG 调用顺序和3/4元素 tuple。仅 `training=True` 且 views2 返回 `[2,C,H,W]`；标签、index、age features 每主样本各返回一次，`age.unpack_batch` schema 不变。原图/缓存只读一次；两个视图都从未增强原始 PIL 对象独立生成，不能把第一幅增强结果再增强。
2. 第一视图使用原 `augmentation_seed`，严格保持原像素及镜像/颜色/旋转/scale/dx/dy 调用顺序；第二视图使用现有 `data._seed(augmentation_seed, 'training-view', 1)`，不扰动主 sampler 流。验证忽略训练视图扩展，保持原像素、shape 和 tuple。
3. `data.flatten_training_views(images, labels, age_features, training_views=None)` 返回 `(images, labels, age_features)`：缺省/1 原对象直接返回；2 将 `[B,2,C,H,W]` 按 sample-major 展平为 `[B*2,C,H,W]`，labels 与非空 age 用 `repeat_interleave(2)` 同序扩展。indices 不进入 helper、不展开。
4. 两训练循环在原 labels/device 解析后、model forward 前调用 helper。正式 DDP 入口照原 age forward 支持重复后的年龄；单卡旧入口传 `age_features=None`，保持其现有 age 支持边界，不顺便扩功能。训练 dataset 接收 `job.get('training_views')`，eval 不进入 flatten。
5. 一次拼接 batch forward、`classification_loss(..., None)` 的 mean BCE、一次 backward；flat mean 等于 `0.5*(BCE(v1,y)+BCE(v2,y))`，不是先平均 logits 再 BCE，也不额外除2。各 rank 视图数相等，沿现有 DDP 平均梯度，不新增跨 rank 正负配对或双 forward 同步。主 indices 的 coverage/采样计数保持在 flatten 之外；R106 mixup=null，不开发多功能组合，必要时明确拒绝 views2 与 mixup 联用。

## 短实施计划与必要验收

关注点限于本次实际变化：旧像素/默认路径、独立第二视图、batch 与 age 对齐、BCE 归一化、主样本计数。它们合并进下列三个测试与 A 的受影响入口核对，不再增加全套或独立 GPU 预演。

- [ ] **B 先写三个测试文本，再实现两个接口。** ①合成小图比较旧单视图、缺省/1、views2 的第一视图与 RNG 调用序列完全一致，eval 仍为旧单视图；②原图只读一次，第二视图等于从原图用指定派生 seed 独立渲染的结果，重复调用确定、tuple 中 label/index/age 不重复；③小 tensor 证明 sample-major 展平、label/age 同序、None/1 对象保持，以及 flat mean BCE 的值和 logits 梯度等于两个视图 BCE 的算术平均。不加载模型/数据或新增机制诊断。
- [ ] **A 完成配置和两入口接线。** SCIENCE 仅新增 views2 及本轮身份，缺省旧 job 不注入新字段；沿既有配置 fixture 窄核字段消费。检查一次 forward/backward、coverage 主 indices、一次 update/LR/EMA 不变；不修改 `age.py`、模型工厂、loss 实现或旧冻结包。B 冻结后不共写其两文件。
- [ ] **A 在实时核验的既有 Pod 一次运行受影响 CPU 验收。** 现有 `pod_cpu.sh` / CPU guard 下使用 `tools/validation/run_checks.py --source-root "$R106_SOURCE" --output "$R106_CHECKS" --pattern test_training_views.py`，两个路径由当次独立 technical-prep 提供；合并既有配置 fixture 的必要项，保留实际输出、exit0/source unchanged 才称通过。不增加单独 RED 轮次、不重跑无关旧像素/来源/模型检查，本机不执行。
- [ ] **A 唯一集成提交并接续。** 必要窄验、此页与实现一次 Git 同步并实核远端 commit/tree；新 Git 原生正式 CPU 包一次匹配50c初态/空Adam，按实际源码/配置 SHA 冻结部署。A 自行确认 R105 完整工程有效、原一次官方自然结束及进程/锁/设备安全交接后，一次 activation/fresh launch；实际训练报告8rank初态、视图资源与首成功更新。无额外 control-only、独立预演、新 Agent 或例行确认。

本页科学字段与接口冻结交 A；checkbox 表示待执行交付，不冒称已经通过。A/B 可并行准备，不等待文档批准。本页不查询 R105 官方或安全状态，不追逐过程更新，不改已冻结决定或另建报告。
