# R110：关闭 stochastic depth 的单次对照与短实施计划

**状态：科学合同冻结；不代表新实验已执行或通过。** R109 完整16条内部 freeze 与 TRAIN 已一次文本核实。科学选择不使用官方数值，GPU安全交接由A实核。

**Goal:** 直接匹配既定seed29干预链的R107，仅将训练 stochastic depth 最大概率由0.1改为0，检验移除随机残差分支屏蔽这一独立轴。

**Architecture:** 复用Small公开fresh工厂、512缓存、单视图G64P8及原pairwise WORLD梯度路径；只把生产候选/job的SD期望正确绑定到SCIENCE。36个SD模块保留，实际概率全部为0，残差架构不变。

**Tech Stack / Spec:** 现有SCIENCE、Torchvision 0.25、PyTorch DDP及正式包/部署链；依据本次R110精确授权与[R107合同](R107-DECISION.md)。writing-plans短计划仅在本页；用户既定A执行方式覆盖额外RED、全套、频繁提交、新Agent及设计批准等待。

## 精确身份与唯一变化

- 配置：`configs/R110-SCIENCE.json`。
- profile：`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-sd0-v1`。
- candidate：`r110-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-sd0-9010`；order20。
- round：`r110-small22k-seed20261029-scale085-pairlog-sd0-24k-p4-9010-20261029`。

**直接基线为R107，不是R109。** 唯一科学delta为 `job.stochastic_depth: 0.1 → 0.0`；候选身份中的SD值同步反映该设置，不构成第二科学轴。以R107为底稿，正例配额保持8，不能继承R109的4；相对R109恢复P8并关闭SD并非单变量比较。`p4`仍指早停patience4。

其余全部R107合同保持：`convnext_small`及公开revision `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b`、既有Small权重/来源SHA、seed20261029、`training_scale=[0.85,1.0]`、`training_views`缺省单视图、512/原prepared512缓存/FP32、dropout0.2、LN1e-6、backbone LR1e-4、head/age各1e-3、原Adam/clip/WD/AGE/增强、mixup关闭。固定9:1、splitseed20261024、原fit/dev清单、fit-only年龄统计、验证原视图及唯一selector均不动。

全局64主图/64视图，8rank各8主图/8视图、参考microbatch32、accumulation1；全局8正56负448对，沿原采样布局rank正数 `[4,0,0,0,4,0,0,0]`。一次model forward/backward/optimizer/LR/EMA；loss/gather/采样算法不改。原loss六字段完全保持：

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

仍为全局 `mean softplus(z_negative-z_positive)`，默认WORLD可微logits gather，不加world-size缩放、BCE混合、margin或hard mining。每800成功更新验证、最多24000，warmup及EMA warmcopy绝对1600、首blend1601、cosine24000；cycles30/warmup_cycles2。每完整周期max(online,EMA)内部AUROC严格刷新历史best才重置，首轮baseline、patience4/min_delta0/min_cycles1及完整dev唯一选点政策不变。

同seed/public-fresh整体tensor预期为 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，须由新Git正式原生CPU流程一次实际匹配及空Adam证明；不把预期写成已通过，不加载旧checkpoint或另做control-only。36个SD模块及结构计数保留，生产非tensor proof应记录36个 `mode=row,p=0`，不能仅凭相同tensor摘要声称运行配置相同。

工程train21600秒/total28800秒/GPU grant32400秒保持；准备20%/训练80%依实时有效CPU及互斥池，不新增或延长实例资源。

## 已核证据、理由与反例

R109内部原件为6400成功更新、8周期、连续4周期未刷新，TRAIN completed/exit0、hard_deadline_met=true、signals为空；完整16条的唯一选点是c4 EMA@3200，AUROC .9855326547789199、AP .8333775688375845、error_twice24030。A原生观测确认attempt=success/skip0、实际quota4/4P60N240对与800/24000/1600时钟匹配。相对R107 c4 EMA@3200的AUROC .9884405731067525/AP .8274326888154083，主差为−.0029079183、AP差为+.00594488；按预定判据，本次P4未获支持。c2一度较好不能改变选点，也不能把退化归因于噪声、自然先验或某个采样机制。A报告training/update为 .42455997秒，仅本轮描述；TRAIN组件wall3222.9628秒不等整轮时长。

原件远程根：`exp-v3/rounds/r109-small22k-seed20261029-scale085-pairlog-pos4-24k-p4-9010-20261029/`。`SINGLE-MODEL-FREEZE.json`原Pod SHA `6f3071726fe6d86855054b4151cf1556cb460fd02bed52e331dc2bc4c827ec52`；`evidence/TRAIN_RESULT.json`原Pod SHA `ae3f53e7f00a7da8ecdb1d8ba198792bc1f99994520f1cac8e3a6e6b61b6adcc`。仅文本读内部两件，不本机重算/哈希。

B已交一次预定聚合：固定20难正例错序质量6334→9120、净增2786；其余147为3266→2895、净减371；整体9600→12015、净增2415（tie半计，非FN/FP人数）。这只描述selected结果的退化集中位置，原组含早期结果定义、固定dev/单seed偏差，不是因果或医学标签证据，AP及其余组改善不能挽救主目标。聚合远程 `exp-v3/technical-prep/r109-error-decomposition-20261029-b/SUMMARY.json`，原Pod SHA `fb1fe0528db16f64749019c3bfaa6129e8468901d00921a070c02f2574b20fd4`；CPU回执原Pod SHA `c1580f4242c1c914e598b189ef87043dac2ed7290b70ca087f6d4bab35b083e1`。不引入逐例表/图片，不以该分解新增门。

选择SD关闭是有限的训练干预，优先理由是资产与接口可复用、能检验不同于配额/损失数值扫描的轴。它不证明现有SD过强、容量不足、错标噪声或过拟合。降低正则可能反而伤害泛化；也可能没有可辨收益。当前实现先计算残差分支，再应用SD，因此关闭不增加模型参数或名义层数，不等于“提高容量”，也不应宣称明显省掉主干计算或固定吞吐提升。[Torchvision ConvNeXt实现](https://docs.pytorch.org/vision/0.25/_modules/torchvision/models/convnext.html)

Torchvision 0.25在p=0时直接返回输入，不进行Bernoulli抽样；这会改变后续Torch dropout随机数消耗。采样及PIL增强仍沿独立sample-key流，不能声称整个训练保持与R107相同随机轨迹。eval下原SD已不执行随机屏蔽，验证输入/推理规则保持；训练后权重当然可能不同。[SD实现](https://docs.pytorch.org/vision/0.25/_modules/torchvision/ops/stochastic_depth.html) 原始SD论文在其它任务的收益只是其正则价值的反例背景，不是本眼底数据保留或关闭SD的有效性证明。[原论文](https://arxiv.org/abs/1603.09382)

主判据仍是完整dev按原selector/p4所选整体AUROC与R107比较；未超过 .9884405731067525，则不支持本次主目标。AP独立分报、既定固定20组仅辅助，不用它们救结果或重新选checkpoint。改善也不能凭单seed/同dev自适应选择证明显著、稳定、病因或.999。无论结果如何，不自动扫SD .05/.2、seed、温度或混BCE。官方只作原一次最终分报，不用于科学取值。

## 最短A接线计划

**A独占本次实现：** 新增 `configs/R110-SCIENCE.json`，修改 `runtime/bundle/retrain/experiment91.py` 的身份/SD期望及 `tests/test_candidate_binding.py` 原fixture；B无新增实现任务。

- [ ] **配置与绑定。** 从R107复制科学配置，仅job SD改0及同步candidate SD/新身份；不继承R109 P4。ex91的固定 `CANDIDATE` 不再存SD常量，`candidate_from_configuration(value)` 在函数内将 `value['job']['stochastic_depth']` 合入本地expected并核candidate同值；不在module import时读site。`validate_job` 的SD期望同读绑定SCIENCE的configured值，不能散写新0。工厂 `models.py` 已直接消费job，`describe_model/validate_normalization` 已按SCIENCE验36个线性p列表，零上限自然全0，不改算法。
- [ ] **仅必要窄配置验收。** A在当前有效准备池按原入口核TTL/live资源，运行原fixture中配置消费与实际结构期望相关项，确认candidate/job一致且36模块p期望全0。旧Small source strict/handoff中的SD0.1完整保留，不重跑或改写来源、转换、provenance、SHA：`structure`仅计数等字段，`verify_public_asset`继续消费原结构证据。生产SD0由现有正式包非tensor proof承担；不新增来源/Gloo/math/像素/模型forward/control-only/独立GPU预演或全套检查。
- [ ] **一次集成与真实接续。** A唯一Git同步并实读新tree，新正式原生CPU18一次核36个SD p0、50c整体tensor/空Adam及既有包身份，按实际SHA部署。A实核R109原流程整轮有效、一次官方自然结束及进程/设备/锁安全后才唯一fresh启动；真实初态证明消费新非tensor期望，首更新仍64图/8P56N448对。准备不等文档/归档或新批准，未执行步骤不预填成功。

规划仅新增本页并冻结交A，不改旧决定、分析README或入口状态，不执行Pod/测试/Git写入。21:54的A观测仍为官方活跃、尚未safe；其后安全状态由A实核，不由本科学决定推断。
