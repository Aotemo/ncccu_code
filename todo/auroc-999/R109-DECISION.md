# R109：固定 P4 曝光／梯度采样对照与短实施计划

**状态：科学合同已冻结，尚非执行通过回执。** R108 完整16条内部 freeze／TRAIN 已文本核实；后续 GPU 安全交接由 A 实核，不等待官方数值作科学选择。

**Goal:** 以既定 seed29 干预链中的 R107 为直接基线，只将每个全局 batch 的正例配额8改为4，检验同一学习时钟下的曝光和随机梯度采样影响。

**Architecture:** 复用现有连续类流采样、单视图、WORLD gather及 pairwise loss；仅正式注册 fresh 路径通过一个共享 resolver 消费已有 SCIENCE 的800步验证周期。自然负例 sweep 与科学验证周期分开，既有旧/非fresh行为保留。

**Tech Stack / Spec:** 现有 Small／PyTorch DDP／prepared512／SCIENCE 与正式部署链；依据本次 R109 精确授权及 [R107 合同](R107-DECISION.md)。writing-plans 短实施计划仅在本页，用户既定分工覆盖额外 RED、全套、频繁提交、新 Agent 与设计批准等待。

## 精确身份与科学合同

- 配置：`configs/R109-SCIENCE.json`。
- profile：`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-pos4-v1`。
- candidate：`r109-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-pos4-9010`；order19。
- round：`r109-small22k-seed20261029-scale085-pairlog-pos4-24k-p4-9010-20261029`。
- 直接基线 **R107**，不是因某个 seed 峰值更高临时换基线。唯一科学 delta 为 `job.positive_per_batch: 8 → 4`。名称 `pos4` 明确正例配额，原 `p4` 仍指早停耐心。

全局 G64、8rank各8主图/8单视图、参考 microbatch32、accumulation1保持；两个参考 microbatch各2正30负，按既有连续切片重组，rank0–7正例数应为 `[2,0,0,0,2,0,0,0]`。全局4正60负、240对，每步一次 forward/backward/optimizer/LR/EMA。无正例 rank 仍参与完整 WORLD 梯度通信，不改 loss/gather 归一化或局部统计。

`job.loss` 原六字段完全不变：

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

仍为全批 `mean_{p,n} softplus(z_n-z_p)`，不混 BCE、不加 margin/类权重/hard mining/配对库/辅助优化器。全部其余 R107 科学字段保持：公开 fresh `convnext_small`／revision `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b` 及既有 Small 资产、seed20261029、`training_scale=[0.85,1.0]`、`training_views` 缺省单视图、512/原缓存/FP32、SD0.1/dropout0.2/LN1e-6、backbone LR1e-4、head/age各1e-3、原 Adam/clip/WD/AGE/增强、mixup关闭。固定9:1、splitseed20261024、fit/dev/manifest/FREEZE、fit-only年龄统计与验证原视图均不变。

**科学时钟固定为800／24000／1600。** 每800成功更新验证、最多24000，cycles30、warmup_cycles2；warmup和EMA warmcopy绝对1600、首blend1601、cosine24000。每完整周期 max(online,EMA) 内部 dev AUROC 严格刷新历史 best 才重置，首轮baseline、patience4/min_delta0/min_cycles1；唯一 selector不变。不得按新 P4 的自然负例 sweep 缩短周期、预算或 warmup，也不在800边界重置类内采样流。

fresh整体初态保持 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，新 Git 正式原生 CPU 包一次实匹配、空Adam；不加载旧 checkpoint、不另做 control-only。工程 train21600秒/total28800秒/GPU grant32400秒不动；准备20%与训练80%按实时有效CPU/TTL/互斥池执行，不申请或延长实例资源。

## 依据、可证伪判据与反例

已核 R108 内部 freeze16／TRAIN：6400尝试=成功、skip0、8周期/bad4正常早停、TRAIN exit0/hard_deadline_met=true/signals为空；selected为c4 EMA@3200，AUROC .9872328538178018、AP .8354670944168833、error_twice21206。与 R105 c3 **online**配对，AUROC仅增约 .0000710423；seed29的R104→R107所选AUROC增约 .0033582543。两个固定seed方向同向但幅度异质、AP均升，不证明全轨迹改善、稳定性或机制。B固定组分解为20难正例错序净增1070、其余147净减1129、整体仅少59对；是组间抵消而非普遍改善，AP升幅不能解释该抵消原因。分组含早期结果定义、选点variant/cycle不同与固定dev自适应偏差继续保留。

R108原件远程根 `exp-v3/rounds/r108-small22k-seed20261027-scale085-pairlog-24k-p4-9010-20261029/`：freeze原Pod SHA `5a57b0931f1f988a0e47590ba33205feefa69c5f9b9a83fc291c76c6e15e8a7b`；TRAIN原Pod SHA `fe199888262071d3ba8474ce8b7d3f95897f5964fdd9b82f41b7f486e94f2dc3`。辅助聚合源 `exp-v3/technical-prep/r108-error-decomposition-20261029-b/SUMMARY.json`，原Pod SHA `7c9b5f2b05b40eefde9af6d92fd06847e92f405154b9b7a2246042a648f04e5c`。仅用内部文本/聚合，不导入病例表或官方数值，不重新计算或哈希。

P4是有限曝光／梯度采样假设，不是已证明的纠错或自然类别先验修复。在类内均匀、正负对等权假设下，期望排序风险仍为同一正负对损失；本次改变实际曝光频率、批内组成和梯度相关性。每800步正例曝光6400→3200，负例44800→48000；每个入批正例聚合等权从1/8→1/4，批内正例多样性更低，异常正例单步可能更主导，但出现频次下降，不能推断总噪声影响增减。样本组成会改变图像与augmentation seed的位置对应，不能称同图同增强下的纯频率变化。240对共享64张图，不是240独立观测；64图前反向不减少，配对数降低不等于GPU快一半，也不承诺收益。

主比较为 **R107完整内部dev原selector** 所选 AUROC .9884405731067525（c4 EMA@3200，AP .8274326888154083）。R109按原selector/p4完成后未超过此整体AUROC，就不支持此次主目标；AP独立分报、原固定20/其余正例错序分解仅辅助，不用难组/AP救结果或重选checkpoint。改善也仅是单seed/固定split下的描述结果，不证明病因、显著性或0.999。无论结果如何，不自动扫P2/P16、温度、seed或改早停。官方仅原一次最终分报，不参与决定。

## 精确接口与文件边界

**B只改** `runtime/bundle/retrain/fresh.py`，新增 `tests/test_sampling_clock.py`。共享接口：

`resolve_steps_per_cycle(job, negative_count, *, fresh_spec=None) -> int`

对注册 `experiment91` 且带 fresh_spec 的路径，读取 `experiment91.schedule()['steps_per_cycle']`；真实SCIENCE/source SHA pin仍由现有ex绑定及 `validate_fresh` 核验。其它旧/非fresh路径沿原 `ceil(negative_count / (effective_batch-positive_per_batch))`。不新增配置键或独立schedule平台；`validate_fresh` 原有schedule/cycles/warmup/identity检查保留，正式spec必须等于既有pinned SCIENCE.schedule，总量仍30×800、warmup2×800。

**A拥有** `configs/R109-SCIENCE.json`、`runtime/bundle/retrain/experiment91.py`、`runtime/bundle/parallel_fresh_train.py`、`runtime/bundle/retrain/distributed_engine.py`、`tools/activation/prepare.py`、`tools/activation/activate.py`及 `tests/test_candidate_binding.py` 原配置fixture：

- ex91的正例配额期望读pinned SCIENCE；prepare的运行名中`p8`由实际job配额生成。
- activate原拓扑 `(8,8,1,32,64,8)` 只将末尾quota改为已核job/SCIENCE的4，其余五项保持。
- parallel fresh与DDP engine在原steps推导处共同调用B接口并传fresh_spec；之后既有总量/early-stop/LR/EMA/selector沿800步科学周期，不另推一套值。

当前路径使用 `retrain.execution_policy.resolve_topology`。**不改** 仅旧测试使用的 `runtime/n08_topology.py`、已取消执行的 `tools/validation/gpu_admission.py`、legacy单卡 `engine.py`、stage、旧data-contract/SCHEDULE/FREEZE；单卡pairwise仍不支持，不扩功能。`data.py`采样算法、loss/gather、模型及公开资产完全复用。sampler自身自然sweep不是dev cycle；历史source metadata SCHEDULE中的P8保持原证据，新quota/clock由新SCIENCE pin和job绑定明确表达，不伪称历史已改成P4。

## 短实施计划

- [ ] **B冻结两文件交A。** 两项必要测试文本/实现：①注册fresh按pinned schedule保持800/24000/1600，旧/非fresh仍兼容原ceil；②实际P4八rank参考重组，检查64图/4P60N/240对、rank正数 `[2,0,0,0,2,0,0,0]` 及跨attempt799/800的类流连续，不在科学周期边界重置。使用小型合成rows，无模型、图片或新Gloo测试。
- [ ] **A接配置与正式调用链，统一一次窄验。** 沿现有CPU9准备绑定实时核UID/Ready/TTL/额度后，仅执行B两项与必要SCIENCE/prepare/activate配置接线验收；若绑定已失效，使用既有入口为未执行阶段取得新有效绑定。不重Gloo/math/像素/来源/control或旧全套，不新增GPU预演、平台/Agent。
- [ ] **A唯一集成与接续。** 必要窄验后一次Git同步并实读实际tree；新原生正式CPU包一次匹配50c初态/空Adam，按实际清单冻结部署。A实核R108整轮工程有效、原一次官方自然结束及进程/锁/设备安全交接后，唯一activation/fresh启动。真实首更新报告64主图/64视图、4P60N240对、rank配额与800/24000/1600；不设额外独立证明阶段或文档批准门。

规划本次仅新增本页并冻结交A/B；不修改旧决定、分析README或过程状态，无Pod/测试/Git写入。准备不等归档或常规批准，未执行结果不预填通过。
