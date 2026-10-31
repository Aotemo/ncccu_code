# R124：固定内容加权汇聚的匹配对照

状态：R124 原工程 attempt 在唯一 activation audit 失败后收口，未启动 GPU/训练，无科学结果；不得在原 prep/root 重试。原科学合同保留，协调方已另授权一次独立未来 [R124R1 工程续接](R124R1-DECISION.md)。以下保留预登记内容，实际失败与已通过证据见末段。

- order：`34`；配置：`configs/R124-SCIENCE.json`（A 生成）。
- profile：`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-contentpool-v1`。
- candidate：`r124-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-contentpool-9010`。
- round：`r124-small22k-seed20261029-scale085-pairlog-contentpool-24k-p4-9010-20261031`。
- candidate.readout：`native_content_attention_pool_final_LN_dropout_linear_plus_age`。

## 问题与已有反证

直接匹配 R107 / seed20261029，唯一科学变化是最终特征图的固定内容加权汇聚；年龄恢复原 enabled=true（可缺省），resolved AGE 原 contract不变。与无年龄 R123 比较会同时改变年龄和汇聚，不能作单变量归因。假设是 GAP 可能稀释最终特征图中已有的稀疏判别信息；收益未知，不声称已定位病征、噪声或瓶颈，也不能补回512缓存与下采样已丢失的细节。

这不是从未尝试的空间方法。旧 R20 是 Tiny/BCE/8:2 的固定2×2位置残差读出，相对 R15 共同前8周期 AUROC差 `-.000578387554`，未获支持。旧 R26 的同一内容汇聚规则内部 AUROC `.9906309160216713`，对 R15 差 `+.000838707785`，条件95%区间 `[-.00015588751566933122,.0021090920395942427]` 略跨0，仅探索信号；N11 的旧 Tiny attention 低骨干LR配方亦未胜 R26。当前 Small公开22K→1K/pairlog/9:1 合同未见等价实测，不能把旧8:2分数直接排名当前dev。伪影、黑边或高激活噪声可能被新权重放大；旧反证不因更换合同而消失。依据为[历史汇总](../../records/storage/EXPERIMENT-CONCLUSIONS.md)、现有 `runtime/bundle/retrain/attention_pool.py` 与已读 R20/R26 冻结源码文本。

## 固定读出与初始化

`job.attention_pool` 和 `candidate.attention_pool` 均采用原 `ATTENTION_POOL_RULE` 全字典，同版本、无调值：

```python
{
    'version': 'content-attention-pool-v1',
    'kind': 'centered_softmax_weighted_mean',
    'channels': 768,
    'score_normalization': 'per_location_nonaffine_layer_norm',
    'norm_eps': 1e-6,
    'score_scale': 'inverse_sqrt_channels',
    'temperature': 1.0,
    'parameter_name': 'attention_pool_weight',
    'parameter_shape': [1, 768],
    'weight_init': 'zeros',
    'bias': False,
    'lr': .001,
    'weight_decay': .0001,
    'dropout': 0.0,
    'dtype': 'float32',
    'pooling_implementation': 'original_avg_plus_centered_weighted_sum',
    'position_encoding': False
}
```

单次 backbone 得到 `F=[B,768,H,W]`，位置数 `S=H*W`。令原位置向量为 `h_s`，`score_s=q·LN_nonaffine(h_s,eps=1e-6)/sqrt(768)`，`a=softmax(score)`，输出 `u=原avgpool(F)+Σ_s[(a_s-1/S)h_s]`。**score使用LN特征，value仍为raw h_s**；不改成先归一化所有value再聚合。512/原stride2预期16×16特征网格不冒称新增实际前向证明。u依次经过原final LN1e-6、Dropout .2、原Linear及年龄残差一次，保留一个模型、一个输出、一个图像级原pairwise loss。

q为FP32 `[1,768]` zeros，分配不消耗RNG；初始均匀权重/零修正应由必要toy证明与原GAP函数一致，不能在q=0时绕过分支。q须留在计算图、梯度存在且finite；非均匀toy可验证非零梯度，不要求每元素都非零。原classifier完整执行，avgpool无参数，不遗留未用head，不用放宽DDP未用参数处理掩盖接线错误。q进入现有head组LR `.001`/WD `.0001`，final LN仍在原backbone组；q参加clip1改变更新范数与后续梯度轨迹，初始函数一致不等于优化轨迹一致。EMA按原浮点混合更新q，不fixed-copy。

增加q使whole state SHA必不同原R107 `50c…`。正常正式单工厂实测新whole、q全0、排除q后的共同state与原初态exact、实际optimizer唯一成员/空Adam及非tensor读出身份；不另建第二控制模型。原public1000严格来源证据不改，不加载旧训练checkpoint。正式8 rank与真实first batch须核enabled age、四元素tuple和唯一前向/更新；未取得回执不记通过。

## 保持的合同与最短实施

R107其余不变：Small公开fe6b/seed20261029/fulltrain/stride2、单图prepared512、scale `[.85,1]`、原None光度、G64/P8（64图、8P56N/448对）、FP32/noTF32、SD .1/dropout .2、原LR/WD/AdamW/clip1/EMA/age、固定9:1和原selector。损失为原六字段 `version=binary-pairwise-logistic-v1, kind=binary_pairwise_logistic, temperature=1.0, reduction=mean, dtype=float32, scope=global_batch`；原WORLD gather与DDP归一化不变。每800成功步验证、最多24000步、patience4、绝对warmup/EMA1600不变。不新增source/cache/ROI、双编码、辅助loss或N06/FLAIR参数。

- [ ] B 仅 `runtime/bundle/retrain/attention_pool.py` 窄准 Tiny/Small FP32 与新增 `tests/test_content_attention_pool_small.py` 两toy：①None兼容、q0函数/RNG/共有state与q在图；②非零q对独立weighted-mean值/梯度参考、真实optimizer唯一成员/LRWD、EMA与strict重载。原规则数学不改，不扩Base或汇聚平台。
- [ ] A 独占其余SCIENCE/config/prepare/experiment91/正式身份接线及第三项必要绑定fixture。解除当前候选attention=None/describe禁q/`tools/validation/check_candidate.py`仅最终Linear为head的硬期望，改为精确消费本轮合同，保留旧缺省行为；fresh/save/load/eval/inference同factory复建，实际head包括q而finalLN不迁移。
- [ ] A 动态核有效准备份额与前驱safe，统一受影响窄验、唯一必要Git、实际清单部署、正常正式原生包及一次activation/fresh，回报真实8 rank初态和首成功。不重旧loss/Gloo/source/cache套件，无独立GPU预演、第二控制模型或样本审计。

单编码、64图原输入链；新增最终小特征图LN/softmax/归约和768可训标量不是免费计算。满额24000步含验证/保存约3–3.5小时仅为旧Small量级估计，非本轮benchmark或保证，不依赖早停救预算。训练21600秒/整轮28800秒/grant32400秒、默认round12GiB、启动free20GiB/运行floor16GiB、300秒保存及保护/atomic/prune/failfast均不变；不新建或延长资源。最终包≤500MB及指定硬件端到端100ms仍未实证，不能由参数增量小或训练卡吞吐推定通过。

## 预冻结主判与分支停止

完整工程有效selected内部AUROC **strict > R107 `.9884405731067525`** 才科学正向；还须 **strict > 单次best R120 `.9894773092062407`**，才由协调方另定固定seed27对R108 `.9872328538178018`复核。本页仅授权R124，不自动启动第二seed。只胜R107未破R120则有限正向、无新best，本阶段结束；未胜R107关闭该固定读出×当前合同。不继续q温度、维数、公式、pool架构、LR或seed网格。AP独立报告，partial/AP/official不救主指标；收益与`.999`可达均未知。

完整训练及唯一checkpoint冻结后，selected内部 `>=.996` 才完整官方5849一次；不足则 `not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，不当运行失败。此门与科学/实用/条件seed门独立，无平台test。总体`.999`目标持续；完整结果后依预定分支收口再由协调规划决策下一问题，当前不列新候选队列。本页freeze后停写，执行实证由唯一A交回。

## 启动前工程失败收口（2026-10-31）

以下引用协调方转交 A 实际回执，规划未另查 Pod、运行测试或重算SHA。原实际 Git main `8bd6abf3dff915e5459ea7899a3033999f715530`、tree `f1b7c757c8f6b05890f64a76f57e83ae0bf861b3`，精确10文件回读/source615核实。B两toy已PASS；A绑定fixture首次因错误mock site失败，只复未过项后PASS，原失败保留。正式CPU20/20 GREEN，新whole `1ea8442bb4b9c41009eded6863d7f4c2906144e0f15e6726fe34b594545d643c`，原345张量exact、新q全0、age enabled和空Adam已有实证；这些是本attempt证据，不冒充未来包的新实跑。

site preflight exit0，随后唯一 activation AUDIT exit1：`verify_preflight: fresh CPU site preflight differs`。原 observed_at=`1793420919.712`，activation exit=`1793421045.579`，协调方文本核确认120秒freshness已过且两处检查位于第一次正式mutation前；SSH/wrapper/回执往返延时与此次过期相关。只读失败后核实 `FORMAL-CLOCK`、`FORMAL_BINDING_RECEIPT`、round_root、dispatch均不存在。未进入GPU/训练，没有内部选点、AUROC或可报告训练进度；官方属于工程启动前未执行，**不是 `threshold_not_met`**，因为尚无selected。

原activation已attempted，按不可逆约束禁止重试同prep/root/audit；不删除marker、改时间戳、伪造observed_at/Ready、重写receipt或放宽120秒。旧失败原件和已通过证据全部保留；不存在的旧clock不补造。此结果不构成内容汇聚科学阴性，不关闭pool研究问题。

协调方仅授权R124R1一个独立未来工程attempt，科学profile/candidate/order34与完整resolved job保持，使用新prep/round/root/未来绑定及授权引用。工程时间与失败attempt如实累计，不改称原attempt首次成功；R124R1再次失败不得自动第三次尝试。具体连续preflight→audit修复和停止边界见[R124R1合同](R124R1-DECISION.md)。本页到此收口并冻结，无新科学候选。
