# R122：训练光度增强整组identity消融

**最终状态：完整工程有效，但selected内部AUROC低于R107及单次best R120；本整组关闭消融结束，不做seed27或拆分扫描。** 原None光度增强继续作为比较基线。末尾记录实际终态，下文保留原预登记。2026-10-31。

| 身份 | 唯一值 |
|---|---|
| order | 32 |
| 配置（A独占） | `configs/R122-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-photoidentity-v1` |
| candidate | `r122-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-photoidentity-9010` |
| round | `r122-small22k-seed20261029-scale085-pairlog-photoidentity-24k-p4-9010-20261031` |

## 唯一问题与输入合同

直接科学对照为R107 seed20261029，唯一新增`job.training_photometric="identity-v1"`。检验训练时是否需要原亮度、对比度、色彩三项不变性；三项整组关闭，不能归因其中某一项，不搜索强度或概率。

仅训练分支跳过`ImageEnhance.Brightness/Contrast/Color`三次PIL应用；镜像之后、rotate之前仍按原位置和顺序消费三个`uniform(.9,1.1)`。总8draw及后续旋转角度、scale[.85,1]、dx/dy取值保持；几何操作、插值、画布、归一化和sample-key不变。None/缺省沿旧pixel exact路径；eval即使带新字段也保持旧像素、零增强RNG。单图prepared512/原cache、label/index/age tuple不变，无新原图扫描、读取probe、cache、色彩空间、CLAHE、锐化、裁框或双视图。

原扰动只有±10%，没有证据证明它破坏病征或是当前瓶颈。关闭可能保留可用光度信号，也可能丢失曝光/跨设备鲁棒性、强化采集捷径或过拟合；既有80图描述不支持统一光度病因，不能推医学标签或收益。三项一起关闭只检验该整组policy，不得阴性后拆分三因子扫描。[Pillow官方文档](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html)只支持API语义；网页stable当前13不代表Pod冻结12.3，本轮不升级依赖。

## 固定可比较合同

回到R107原loss，不沿R120/121 sigmoid：

```json
{"version":"binary-pairwise-logistic-v1","kind":"binary_pairwise_logistic","temperature":1.0,"reduction":"mean","dtype":"float32","scope":"global_batch"}
```

相对R120还包含loss变化，不能单变量归因；R120仅作已记录最佳实用参考。其余保持R107：已验公开Small22K→1K、seed20261029、stride2/fulltrain、SD.1/drop.2、单图512/scale[.85,1]、G64/P8/64主图64视图/8rank各8/448对、FP32/noTF32、原age与fit-only统计、原LR/WD/clip1/AdamW/EMA、固定9:1清单及splitseed、800/24000/patience4/绝对warmup及EMA1600。保留全局WORLD可微gather及原梯度归一化，无mixup、SSL、freeze或局部输入分支。

公开fresh初态预期仍`50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，由正常正式CPU实际匹配；同tensor不证明input policy，实际首batch须记录消费的Dataset policy。正常参数组/空Adam/EMA身份保留，不加载旧训练checkpoint，不另control-only。

## 最短实施与工程边界

- [ ] **B独占** `runtime/bundle/retrain/data.py`与`tests/test_photometric_policy.py`。`ImageDataset.__init__`在原`include_labels=True`后追加`training_photometric=None`，self保存None或`identity-v1`；A不共写data.py。
- [ ] **A独占其余** config/experiment91/engine/distributed参数透传、SCIENCE及必要生产绑定；A统一实际guard、唯一Git与GPU。新policy不能只登记而未传到实际Dataset。
- [ ] **三项必要synthetic窄验**：独立legacy像素oracle与8draw状态兼容；新identity的geometry-only参考、三draw照耗及angle/scale/offset后续RNG一致；eval同像素/零Random与真实job字段注入。仅新窄验及正常formal初态/参数组/空Adam，不重loss/Gloo/math/source/cache旧套件，不做独立GPU预演。
- [ ] 必要证据齐后，A按实际Git/清单冻结部署、动态身份/互斥CPU预算与前驱safe/资源准入，执行新一次activation/fresh。旧包/clock不改，不等页面批准或例行回复。

成本只沿既有单图组件估算：R1076400步training2718.788秒、eval146.153秒、checkpoint136.526秒、processwall3066.314秒。满24k含30次eval约3–3.5小时是非benchmark估计，不是保证，不以少三次PIL应用许诺加速或依靠早停救预算。保留训练6h/整轮8h/grant9h、默认Small12GiB、健康9C+I、启动free20GiB/运行floor16GiB、保存保护/300秒/atomic/prune/failfast；不增资源。实际≤500MB及指定硬件端到端100ms仍未由本轮证明。

## 预冻判据与接续

完整工程有效、原selector所选内部AUROC **严格胜R107 .9884405731067525**才称科学对照正向；还须**同时严格胜已记录单次best R120 .9894773092062407**，才投入固定seed20261027对R108 **.9872328538178018**的复核。R120不是单变量对照，该额外门只限制后续预算。仅胜R107未超R120，记有限正向/无新最佳并停止本阶段；未胜则本消融不支持。均不拆三因子、range、LR或seed扫描，AP不救主判，两seed亦不称显著或消除dev自适应偏差。

完整训练且唯一checkpoint冻结后，selected内部AUROC≥.996才完整official5849一次；未达正常`not_run_by_policy/threshold_not_met`、evaluated=false、metrics=null。此门与复核双门独立，中途partial不触发；无平台test授权。R120/121一胜一负已停止主线升级，R120仅单次best保留，R107/R108继续作比较锚点。总体.999目标未达且不保证可达；本阶段结束由规划据完整结果主动接下一独立问题，不在本页预立其它候选。

## 完整终态与有限结论

2026-10-31 10:02:02北京A实际终态：completed/early_stopping，5600success=attempt/24000、skip0、eval7、bad4；完整freeze14条唯一selected **EMA c3@2400，AUROC .9874796957462513、AP .8211659433777129、error_twice20796**，tensor SHA `4aa597b03788019483a0b348f108876435f264c19b3b0b1f9be302176ecfbb2f`。一次文本核`work/r122-20261031-a/progress-1000/SINGLE-MODEL-FREEZE.json`的selected及同根`evidence/TRAIN_RESULT.json`；TRAIN completed/exit0/hard_deadline_met、signals[]。此根无`round/`前缀。

主指标低于直接R107 **.9884405731067525**，也未超过单次best R120 **.9894773092062407**；两门均未通过，关闭本整组photoidentity消融，不做seed27、不拆Brightness/Contrast/Color或扫描range/LR/seed。只否定本合同下整组关闭的收益，不归因某个因子、光度病因或标签问题；原None增强路径保留。R120仅singlebest，不因此升级方法基线。

以下官方/安全事实引用A实际回执经root转交，规划未另读official或查询Pod：official=`not_run_by_policy/threshold_not_met`、evaluated=false、metrics=null，ROUND/done completed；owner262727Z不称reaped、8rank消失、lockholders[]、deviceusers[]、denied0、8busy0、safe=true。progress原Pod SHA `6fd08ef4aceced9a24821c49c3abf1a310c894640425ce11521cc6691971f853`，guard SHA `fabd140512817a78c21200250c94d85b4de91017c93bf7c4addbcb879fef843e`，仅记录交接值，未重算。最新actual main由A/root记录为`c70494f6b06941a6c606868c0efa3413e1834606`，R122必要tokenfree归档由A另做，旧包/clock不改。

连续阴性未说明512缓存必然缺失所需信息，也未证明其充分；现有8例审计缺独立标注定义/诊断依据，不能把模型错误当错标。下一步优先补一个可判别的有限fit-only证据缺口，另以文本提案交root，不机械排下一个增强/系数/模型tag。总体.999未达但不宣称不可达，本页至此收口停写，不预立新轮次。
