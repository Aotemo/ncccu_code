# R125：第三阶段 GAP 残差读出

状态：协调方已采用，现有 A/B 已获必要实施分工；本页是固定科学合同，不是启动或验收回执。SCIENCE、Git、正式包、activation、实际8rank初态及首success均待A实证。仅授权本轮seed29一次fresh，必要准入齐直接执行，不等文档批准。原R124/R124R1已收口，冻结页与失败审计不改。

## 身份与问题

- order：`35`；配置：`configs/R125-SCIENCE.json`，由A生成。
- profile：`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-stage3residual-v1`。
- candidate：`r125-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-stage3residual-9010`。
- round：`r125-small22k-seed20261029-scale085-pairlog-stage3residual-24k-p4-9010-20261031`。
- candidate.readout：`native_final_LN_dropout_linear_plus_stage3_gap_residual_plus_age`。

直接对照R107/seed20261029，只增加末次下采样前第三阶段的固定残差读出，检验该表征能否补充原最终层分类。不是新增原生像素或定位器，也不是已证实的信息瓶颈。相对R124R1同时撤去content pool并增加本支路，不作单变量归因；不沿用R120 sigmoid、R123 noage或R122光度identity。

旧R20在Tiny/BCE/8:2上用最终特征图2×2空间残差，未获支持；R26最终层内容池化仅有探索性正向、条件区间跨0，N11与当前R124R1亦有阴性，stride1完整试验未胜。已读N04代码为冻结最终特征融合，不是本支路，但综述已有旧中层探索，不能声称此前从未试过中层。最强反例是较早表征会带入采集、曝光或纹理伪影，直接路径可能削弱最终阶段的不变性；GAP也仍可能稀释小病征。收益未知，不承诺`.999`，不以C=0、FIT256 unknown或dev难组证明机制。

## 唯一读出干预

`job.stage3_residual`与`candidate.stage3_residual`使用同一完整固定 `STAGE3_RESIDUAL_RULE`；缺省为None，旧路径不变。规则语义固定为：

| 项目 | 固定值 |
| --- | --- |
| version / kind | `stage3-gap-residual-v1` / `intermediate_gap_linear_residual` |
| feature_index / channels | `5` / `384` |
| normalization / norm_eps | `nonaffine_layer_norm_after_gap` / `1e-6` |
| parameter_name / shape | `stage3_residual_weight` / `[1,384]` |
| 初始化 / bias / dropout | zeros / 无bias / `0` |
| 参数LR / weight_decay | `.001` / `.0001` |
| dtype / detach / backbone_passes | `float32` / `false` / `1` |

按原顺序执行每个 `network.features` 模块一次，保留 `features[5]` 的F3，其后 `features[6:]` 顺序不变。令F4为最终特征，`g=原完整classifier(avgpool(F4))`，`u=nonaffine_layer_norm(GAP(F3),384,eps=1e-6)`，图像logit为 `g+sum(q*u)`，之后仅一次原age residual。512输入下F3预计为384×32×32，须由实际工厂结构绑定核实；没有真实forward不写成已测输出。

q在原public/head/age初始化完成后用zeros附加，不耗RNG；预期原345共有tensor逐键一致，新增q使whole SHA必然不同，不能抄R107的50c或R124的1ea。q0仍留在计算图；原classifier含dropout的执行顺序保留，新支路无随机算子。无hook、第二次骨干、auxiliary loss、detach/no_grad、门控、softmax、可调混合系数或新dropout。原头及所有骨干参数仍使用，不通过未用参数开关掩盖接线问题。

q按head LR/WD进入optimizer恰一次，final LN保持原backbone分组；EMA对q按原浮点mix更新。q0初始输出一致不等于训练轨迹相同：新增q梯度进入全局clip1，随后影响骨干更新。age只消费一次，保持原tuple4/age true；这些生产事实均待A实际验收。

## 不变合同与资源

恢复R107原final GAP、`attention_pool=None`、`training_photometric=None`及age enabled true，原AGE contract、固定age分层split不变。Small公开fe6b/seed20261029/fulltrain/stride2、prepared512单图、scale[.85,1]、G64/P8/448对、原pairlog六字段（`binary-pairwise-logistic-v1`、`binary_pairwise_logistic`、temperature1、mean、float32、global_batch）、SD.1/dropout.2、FP32/noTF32、原LR/WD/AdamW/clip1/EMA全部保留；固定9:1、800成功更新一cycle、最大24000、patience4、绝对warmup/EMA1600不改。合规公开fresh，不加载历史训练checkpoint或N06/FLAIR参数、teacher、BN、统计。

CPU18已过期；A每入口动态核当前Pod UID/Ready/Python、effective CPU、本人future entry/TTL及互斥20%准备/80%训练份额，整树线程不超本人额度。旧allocation/grant/clock不改，不新增资源或续租。保持训练21600s/整轮28800s/grant32400s、默认12GiB round上限、启动free20GiB/运行floor16GiB、全部保存保护、300秒保存和atomic门。不能降batch/精度/尺寸/步数、扩时或删保护以救限制，也不能依赖早停才满足预算。

一张既有缓存图、骨干一次，新增第三阶段GAP+384维LN及点积不是免费。满24000及30次内部eval约3–3.5h仅为历史Small量级估计，非本轮benchmark；实际时耗和内存由正式运行如实记录。新增384个FP32参数不等于已验交付包，最终500MB及指定硬件端到端100ms仍未实证。

## 最短实施计划与文件边界

- [ ] B仅新增 `runtime/bundle/retrain/stage3_residual.py` 与 `tests/test_stage3_residual.py`，提供 `STAGE3_RESIDUAL_RULE` / `attach` / `validate_model` / `image_logits`。两toy覆盖：缺省与q0共同state/RNG/模式/输出保持、各层单次调用和q在图finite grad；非零q独立值/梯度、真实optimizer中q恰一次且原共有组不变、EMA与strict重载。B无Pod/Git/GPU权，未运行不称GREEN。
- [ ] A独占其余生产、config/ex91/prepare/engine/models/check_candidate/SCIENCE及第3项最小SCIENCE→factory/save-load-inference绑定窄验；生产工厂各入口复建同字段，参数分组显式包含q。正常正式包实核新whole、共有345 tensor exact、q0、空Adam与非tensor读出身份，不把state哈希当输入或接线证明。
- [ ] A统一必要CPU验收、唯一Git精确同步部署、当次资源与安全准入，通过后唯一activation/fresh；真实8rank、首batch tuple4/age true及首success按回执记录。仅3项新窄验及正常formal必要资源，不重source/loss/Gloo/cache/pixel大套件或拒绝矩阵，不加独立GPU预演、第二控制模型或文档批准门。

## 预冻结判据与接续

完整工程有效selected内部AUROC **strict > R107 `.9884405731067525`** 才为同seed正向；同时 **strict > R120 singlebest `.9894773092062407`** 才由协调方另定固定seed20261027，对照R108 `.9872328538178018`。仅胜R107未破R120则有限正向后本阶段结束；未胜关闭此固定第三阶段残差，不扫层号/层组合/归一化/维数/系数/LR/seed。只授权R125首seed，不自动启动27；AP、partial、难组、official或平台分数不救主判。R120仍是单次best，不升级方法基线。

完整训练并冻结唯一checkpoint后，selected内部`>=.996`才完整官方5849一次；未达记 `not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，整轮可正常completed，safe仍须A实际核实。该门独立于科学与实用门，不生成平台test CSV。总体`.999`未达、持续授权不暂停；本轮结果后协调方依据完整内部证据决定下一问题，不自动扩候选队列。本页冻结后规划停写，不作为A执行等待门。
