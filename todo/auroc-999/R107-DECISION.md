# R107：单视图全局批内 pairwise logistic 决定与短实施计划

**Goal:** 直接匹配预定 R104 Small/seed20261029，只将 BCE 换为纯批内 pairwise logistic，检验原 selector 所选整体内部 AUROC 是否提高。

**Architecture:** 64个主样本各一次单视图 forward；默认全局 WORLD 可微汇集 logits、普通 tensor 汇集 labels，各 rank 同算全局8正56负的448对均值损失。Gather 反向 SUM 与原 DDP 参数梯度平均配合，一次 backward/optimizer/LR/EMA，不增加图像 forward、配对库或辅助优化器。

**Tech Stack / Spec:** 现有 PyTorch/DDP、公开 Small、prepared512 与 SCIENCE/原生正式包/部署链。遵循2026-10-29协调给定的 R107 合同、[R104 决定](R104-DECISION.md)、`configs/R104-SCIENCE.json` 及[执行规则](../../docs/EXECUTION.md)。科学合同与 writing-plans 短计划仅在本页；既有 A/B 分工和一次必要集成覆盖额外 RED、全套检查、频繁提交、新 review Agent 与设计批准等待。

## 精确身份和科学合同

- 配置：`configs/R107-SCIENCE.json`。
- profile：`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-v1`。
- candidate：`r107-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-9010`；order17。
- round：`r107-small22k-seed20261029-scale085-pairlog-24k-p4-9010-20261029`。
- 直接基线是 **R104，不是 R106**。`training_views` 保持 R104 缺省单视图，不复制 R106 的2；`training_scale=[0.85,1.0]`，seed20261029。唯一科学变化为 `job.loss` 的下列六字段，不新增或搜索 margin、温度、BCE 混合权重、hard mining。

```json
{
  "version": "binary-pairwise-logistic-v1",
  "kind": "binary_pairwise_logistic",
  "temperature": 1.0,
  "reduction": "mean",
  "dtype": "float32",
  "scope": "global_batch"
}
```

以完整全局 batch 的原始 logits 计算 `L = mean_{p,n} softplus(z_n - z_p)`，P=8、N=56、分母448；不先 sigmoid、不平均 logits、不再乘类比例或 world size。配对仅来自本次 batch，不枚举全数据对、不使用历史预测或病例库。448个共享64样本的对不是448个独立样本。

全部其余科学字段继承 R104：公开 fresh `convnext_small` / `timm/convnext_small.fb_in22k_ft_in1k@fe6b358aa409414cdb522ae3d43a2c9922f7cc1b`，512/原 prepared512、FP32、SD0.1、dropout0.2、LN1e-6、backbone LR1e-4、head/age 独立各1e-3、原 Adam/clip/weight_decay/normalization；固定9:1 fit46256/dev5140、splitseed20261024、manifest/FREEZE、fit-only age 统计、原增强与主 sampler 流不改。无旧 checkpoint/N06 初始化、dev 回灌或重标。

正式执行限定8rank、每 rank8主样本/8视图、accumulation1，全局64主样本/64视图/G64P8。每步一次 model forward/backward/optimizer/LR/EMA；每800成功更新验证，最多24000。每完整周期 max(online,EMA) 内部 dev AUROC 严格刷新历史 best 才重置计数，首轮 baseline，patience4/min_delta0/min_cycles1；warmup/EMA warmcopy绝对1600、首blend1601、cosine horizon24000、原唯一 checkpoint selector 与单视图验证保持。`p4` 是早停耐心，采样正例仍 P8。

公开 fresh 工厂初态沿 R104 的 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`；新 Git 原生正式 CPU 包实际匹配、空 Adam，不加载 R104 权重、不另做 control-only。工程配置 train21600秒/total28800秒/GPU grant32400秒保持，准备20%与训练80%按实时有效 CPU/TTL/互斥池执行，不延长实例或申请资源。额外开销是小矩阵运算与跨卡通信，不能将其称为零成本或未经测量承诺吞吐。

## 当前依据与反例

R104 基线为 c3 EMA@2400，内部 AUROC `0.9850823187728706`、AP `0.8116753198258198`、error_twice `24778`。R106 已文本核22条内部结果与 TRAIN：8800更新/11周期/bad4，训练组件 exit0、hard_deadline_met=true、signals为空；唯一 c7 EMA@5600，AUROC `0.9842033206861964`、AP `0.8140529507628876`、error_twice `26238`。因此 R106 本次双视图未达到预定主目标，AP 略升不能挽救；不证明所有多视图方法无效或过拟合/噪声机制。

A 的 Pod 计时汇总：R106 training_seconds/update 约0.735654秒，对 R104 约0.422017秒，比值1.743187；这是各自运行平均，停止步数/优化轨迹不同，不是受控等步数吞吐实验。TRAIN 组件墙钟为7103.695秒对2828.015秒，不含冻结/官方，不当整轮总 wall。B 原固定20组/其余147组错序质量分别净增358/372、整体净增730，tie半计；只是不同 selected 点的描述分解，不是 FN/FP 人数、显著性或病因证据。

内部原件源于 `exp-v3/rounds/r106-small22k-seed20261029-scale085-views2-24k-p4-9010-20261029/`：freeze Pod SHA `09500603403acc77f76b0010f08019b06468846768d149899e870895e204cb1a`，TRAIN Pod SHA `b89de22738e316158338be4ceae0565f30463685d18334e58a0400c833c6bb87`。辅助原件 `exp-v3/technical-prep/r106-error-decomposition-20261029-b/SUMMARY.json`，Pod SHA `f0303c66542e757d32339e920a7eb7bd2914e3ef1b245dcb2c440bdcdae6507b`；本机只读原件文本，无新计算或官方指标。

R107 检验与正负排序直接相关的目标，依据是 [RankNet 的 pairwise logistic 形式](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)，不是该论文证明本眼底数据有效。此法不需要 AUC minmax 的 a/b/alpha、PESG、CE 旧权重或新训练阶段。反例包括异常标签影响许多配对、AP下降、全局/少数正例梯度变化而无泛化收益；替换 BCE 同时改变梯度权重与尺度，即使 LR/Adam/clip 数值不变，也不是相同优化动力学。共同 logit offset 不受监督，head bias 梯度可为0，但应存在且有限，不能当作漏参数；不承诺概率校准、阈值表现或 AP 改善。

主判据保持完整5140 dev 与原 selector/p4：合法完成后未超过 R104 AUROC，就不支持此次主目标，不以 AP 或难组改善挽救。AP另报；原固定20与其余正例对全部负例的错序贡献仅辅助解释，不重选分组或改选点，负组不与正组重复相加。共享固定 dev 的自适应偏差、单 seed 与非独立难组边界保留；不凭一次结果宣称稳定、显著或0.999。无论结果如何，不自动混 BCE、调温度、换 seed/尺度或延长早停。官方5849仅一次最终分报，不用于本决定与后续选参。

## 接口与梯度归一化

**B 仅拥有** `runtime/bundle/retrain/classification_loss.py` 与新增 `tests/test_pairwise_loss.py`。`validate_loss(specification)` / `classification_loss(logits, targets, specification=None)` 新增上述显式分支，None/BCE 与 focal 原分支保持；纯本地 loss 消费完整正负 batch，可作 CPU 数学参考。新接口 `gather_pairwise_inputs(logits, targets) -> (global_logits, global_targets)` 只用默认全局 WORLD：logits 通过 `torch.distributed.nn.functional.all_gather` 保留反向图，labels 通过普通 nongrad tensor all_gather。不得传 `ctx.group` 半组，不得使用日志 object gather/detach gather 或只替换本 rank tensor 的技巧。

**A 拥有** 新 SCIENCE、`runtime/bundle/retrain/config.py`、`experiment91.py`、`distributed_engine.py`、`engine.py` 的必要配置/身份/执行限制接线与既有 `tests/test_candidate_binding.py` 配置 fixture；唯一 Git/GPU 集成。仅正式 DDP 训练的 pairwise 分支在 loss 前调用 gather；BCE/focal 不进入此 collective。保留原 local logits/labels 给已有 local `sampled_loss` 等统计，不能重复统计每 rank 拿到的全局 batch。

数学归一化依据 [PyTorch 上游 v2.10.0 `_AllGather.backward`](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/torch/distributed/nn/functional.py) 的 SUM 与[标准 DDP 梯度平均](https://docs.pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html)：令全局 mean loss 对 rank r logits 的梯度为 g_r，其模型 Jacobian 为 J_r，W个 rank 各算同一个 L；Gather backward 将本 rank 所需梯度累加为 W·g_r，DDP 平均后为 `(1/W) Σ_r J_rᵀ(W·g_r) = Σ_r J_rᵀg_r`，正好等于完整 batch 参考梯度。故不能额外乘/除 W。所有 rank 都调用 forward/backward collective，缺正例 rank 也有全局 loss 与本地负例梯度；不能局部正负平均或只在有正例的 rank backward。

当前厂商2.10接口尚待本次 CPU 窄验；上游实现不是实装成功回执。CPU2进程 Gloo 验证梯度语义，不能冒称8rank NCCL/GPU已通过；正式 NCCL 沿实际训练首更新报告。新增 pairwise 如 accumulation!=1、mixup 非空、views2 或非正式8rank路径，明确拒绝不兼容执行。单卡旧 engine 的 accum2 会漏跨 microbatch 对，本轮不支持该 pairwise 路径、不扩缓存或多功能平台；纯 loss 的单进程完整 batch 参考不受此生产入口限制。

正常首更新日志从实际 global_targets 记录8正/56负/448对及 global loss；记录64primary/64view和各rank8/8。全局正负/对数不能再把8份复制值相加；local indices 的 coverage、年龄统计仍按主样本计一次。无需新独立证明阶段，模型/source/age/EMA 工厂不改。

## 短实施计划

- [ ] **B 先写3个有意义的新损失测试文本并实现接口，冻结两文件交 A。** 范围合并为：①小 logits 的 pairwise mean 数值/梯度与明确全对参考一致；②极端有限 logits 的稳定性、共同 offset 不变性及 bias 零但有限/存在梯度；③真实 CPU2进程 Gloo + DDP 的参数梯度，对照单进程完整64样本/8P56N参考，包含一个 rank 无正例的划分，验证448对与无额外 W 缩放。使用小线性模型/合成 tensor，无公开模型加载、图片或 mock collective。
- [ ] **A 完成 SCIENCE/身份与 DDP 条件接线。** 保留 R104 缺省单视图与其它字段、旧 loss 分支；生产限制集中明确，沿已有 fixture 只核真正受影响配置。保留本地统计与全局日志含义，不扩旧拒绝矩阵、单卡功能或新框架。
- [ ] **A 在实时核验的 `CPU-ALLOCATION-8` 准备绑定内一次运行受影响验收。** 这里的8为分配版本，不是另授8核。现有 CPU guard / `tools/validation/run_checks.py` 选择 `test_pairwise_loss.py` 并合并必要配置项，真实两个 Gloo 子进程仍计入 A 已分配 mask/线程总额。保留输出、exit code、source unchanged 与子进程退出事实；不另做 RED、旧全套、额外模型 control、GPU预演或新 Agent。当前 checkbox 不代表已执行通过。
- [ ] **A 一次必要集成并接续。** 窄验后统一实现、本页与精简原件一次 Git 同步、实读远端 commit/tree；新 Git 原生正式 CPU 包匹配50c初态/空Adam，按实际文件/配置 SHA 冻结部署。A 实核 R106 完整工程有效、原一次官方自然结束和进程/锁/设备安全交接后，唯一一次 activation/fresh launch；由真实首成功更新报告8rank资源、8P56N/448对、64primary/64view和全局 loss。科学决定不等待官方数值或文档批准。

规划本次只新增本页，不改 R106/旧决定、分析 README、过程状态或源码，不进行 Pod/Git 写入。字段与接口冻结交 A，后续不追逐过程润色；未冻结任何后续扫描。
