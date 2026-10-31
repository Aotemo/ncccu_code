# R126：按优化器角色分别进行 L2 norm1 梯度裁剪

状态：R126已完整工程有效，A于2026-10-31 17:34:48.499北京时间实核整轮completed/safe。唯一内部selected低于R107与R120两门，关闭固定三role norm1分支/no seed27。以下保留预登记合同，实际各阶段与解释边界见末段；R125与更早决定页保持冻结。

## 身份与可证伪问题

- order：`36`；配置：`configs/R126-SCIENCE.json`，由A生成。
- profile：`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-roleclip1-v1`。
- candidate：`r126-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-roleclip1-9010`。
- round：`r126-small22k-seed20261029-scale085-pairlog-roleclip1-24k-p4-9010-20261031`。
- readout：`native_avgpool_final_LN_dropout_linear_plus_age`。

直接对照R107/seed20261029，仅改变裁剪作用范围：现有全参数共用一个裁剪系数，改为backbone/head/age各自norm1。假设是隔离共同缩放可能改变各角色Adam矩历史及优化路径，收益未知；没有证据证明梯度爆炸、head或age主导、噪声或欠拟合，也不增加先验norm审计门。

已核历史R96/97/98改角色LR、R19改余弦跨度、R109改P4曝光、R120/121改损失，均非此合同；相邻优化改动的阴性与两seed不一致是反例，不能因未做过就期待收益。当前已读历史未见这个精确三role规则的完成实验，不宣称全历史从无。最强反例是原全局裁剪本来限制困难batch的大更新，按角色裁剪可能加重后期退化；Adam对恒定梯度缩放近似不敏感，若很少超过cap也可能近乎没有干预。不把FIT256 unknown、首cache缺项或C=0当瓶颈证据。

## 固定policy与执行顺序

可选 `job.gradient_clip_policy` 与candidate同名字段使用同一完整 `ROLEWISE_CLIP_RULE`：

```json
{
  "version": "optimizer-rolewise-l2-norm1-v1",
  "kind": "optimizer_rolewise_l2",
  "roles": ["backbone", "head", "age"],
  "max_norm": 1.0,
  "norm_type": 2.0,
  "epsilon": 1e-6,
  "merge_decay_groups": true,
  "scope": "ddp_averaged_unscaled_gradients",
  "nonfinite": "world_consensus_before_mutation",
  "global_second_clip": false,
  "parameter_membership": "unique_optimizer_parameters"
}
```

缺省None保持原全局clip1路径。新policy限R107的FP32合同，不扩AMP或原单卡age功能。按真实optimizer的role合并decay/no-decay组，形成恰3个集合而不是5个param_group：backbone包含原final LN，head是原末端Linear的weight/bias，age为原两标量；所有optimizer参数恰归一个集合。

DDP平均梯度与unscale完成后，先只读取得全部3个L2 norm；在任何grad修改前做**一次**现有默认WORLD finite合成判断，任一rank非有限则一致拒绝。通过后各role执行 `g *= min(1, 1 / (norm + 1e-6))`，随后一次optimizer；无rank局部裁剪，无额外global第二次裁剪。原loss/参数/Adam有限性检查、fail-fast、cursor/skip、LR、EMA、gather语义保持。

这是科学数值路径变化，不能称“clip1完全不变”：3个集合分别cap1时，拼接梯度总界可到`sqrt(3)`，同时改变尺度分配和总界，不是纯粹只解除耦合。时变缩放会影响Adam历史，恒定缩放近似不敏感也不能保证轨迹相同。coef=1如实记未触cap，不造已裁剪或已证明瓶颈。

## 直接对照与资源合同

其余全部沿R107：已验合规公开Small fe6b fresh、seed20261029、原GAP/age true、`attention_pool=None`、`stage3_residual=None`、光度None、training_views缺省单视图、stride2/fulltrain、prepared512/scale[.85,1]、G64/P8/448对、原pairlog六字段T1/mean/float32/global_batch与默认WORLD可微gather、SD.1/dropout.2、FP32/noTF32。backbone LR1e-4、head/age1e-3、原WD/AdamW betas/eps/EMA、固定9:1及AGE/split、800成功步一cycle/最多24000/patience4/绝对warmup及EMA1600不变。相对R125不是单变量比较，不加载旧训练checkpoint、医学参数，不重新下载已验来源。

预期原345state及50c初态须由正常formal实际匹配，空Adam也须实核；policy是非tensor身份，state相同不能证明裁剪已执行。恢复原图像/age输入与模型，不启用已关闭扩展。

每入口动态核UID/Ready/Python、effective CPU、本人entry/TTL及20%准备/80%训练互斥。CPU19历史TTL为 `1793436437.850184`，只作历史记录，不假定当前有效；过期则由A建立必要future软件份额，整树线程不超本人，不改旧allocation/grant/clock，不新增资源或续租。保留实际Git冻结与精确部署、实时safe、原单链120秒freshness、唯一activation/fresh；必要齐直接执行。

训练21600s/整轮28800s/grant32400s、默认12GiB round限额、启动free20GiB/运行floor16GiB、全部保存保护/300秒保存/atomic门不扩。不得靠早停、降batch/精度/尺寸/步数、扩时或删保护救限。单编码与数据I/O不增加，三个role reduction及控制同步并非免费；满24000+30次内部eval约3–3.5h仅旧Small量级估计，不是本轮benchmark。模型参数与checkpoint结构无新增，不等于已验实际交付500MB或指定硬件端到端100ms，这两项仍未知。

## 最短实施与必要窄验

- [ ] B仅新增 `runtime/bundle/retrain/gradient_clip.py` 与 `tests/test_gradient_clip_policy.py`（任务 `c31d336f-b20b-458a-a0d1-4b9cdb0331bd`）。公开 `ROLEWISE_CLIP_RULE`、`build_role_groups(optimizer,policy)`、`measure_role_norms(role_groups)`、`apply_role_clip_(role_groups,norms)`。build(None)返回None且无副作用；measure只读全部norm、不通信不裁剪；apply先本地检查全部finite再统一缩放，只由A在WORLD gate通过后调用。B不执行Pod/测试、无Git/GPU权。
- [ ] 两toy方法：手算role向量与独立参考，覆盖一role大、below-cap、同role跨decay组、late-nonfinite无部分修改、None无副作用；真实原 `_optimizer` 每参数唯一角色、原LR/WD/空Adam、两步AdamW与独立role参考及矩一致、strict state不扩。仅必要新数学/成员边界，未运行不称GREEN。
- [ ] A独占两个engine入口、config/ex91/prepare/check_candidate/SCIENCE/非tensor绑定（任务 `2a9b16c6-f79d-460f-801e-e93ebffcecef`）及第3fixture：缺省global精确兼容，新3norm→一次WORLD finite→统一改grad→一次optimizer，两个入口/保存加载/正式policy与空Adam绑定。A在有效本人prep guard统一验收，仅未过项窄修，不重Gloo/source/loss/cache/pixel/EMA大全套或拒绝矩阵，不加控制模型、独立GPU预演。

正式包核原345/50c及新非tensor规则；真实8rank与首batch原tuple4/age true/光度None、首success待A回执。只用该真实首成功步已有norm/coef记一次小聚合：三role名称、实际参数tensor数/元素数、preclip norm、coef、是否触cap及WORLD finite消费证据。不新增forward、梯度probe、benchmark或监控，不以注册或toy代替真实执行。

## 预冻结主判与停止

完整工程有效selected内部AUROC **strict > R107 `.9884405731067525`** 才同seed正向；同时 **strict > R120 singlebest `.9894773092062407`** 才由协调方另定固定seed20261027，对照R108 `.9872328538178018`。只过第一门则有限正向后本阶段结束；未胜关闭此固定三role norm1规则，不global cap/role阈值/自适应clip/LR/seed网格。只授权R126首seed，不自动27或下一科学候选；AP、partial、official、平台分数均不救主判。

完整训练并冻结唯一checkpoint后，selected内部`>=.996`才完整官方5849一次；未达为 `not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，可正常completed，safe仍由A实际核。该门独立于晋级，不生成平台CSV，不保证`.999`。R120仅singlebest、R107/R108锚点不动；总体授权持续，本阶段结果后由协调方按完整内部证据决定下一问题。两页冻结后规划停写，不成为启动等待门。

## 完整内部结果与最终收口（2026-10-31）

规划仅一次文本核 `work/r126-20261031-a/progress-1730/SINGLE-MODEL-FREEZE.json` selected及 `evidence/TRAIN_RESULT.json` 必要字段：唯一 **EMA c4@3200，内部AUROC `.9867283329981902`、AP `.8252218982626471`、error_twice22044**；tensor `d4294587db120b143898e07aaa6916b0bda097d5e60e28945a9a7b8e053b7351`、blob `0e25aeff2e17470ea0340de9568ba4c99c8161deeb030cded1341e3c21ee0dbc`。TRAIN completed/exit0、successful_updates6400、cycles8、bad4/stopped=true、hard_deadline_met=true、signals及live_owned_ranks为空。success=attempt6400/max24000/skip0/eval8/early_stopping及以下官方/安全状态引用协调方转交A回执 `f1d8c1d3-ce5a-4cb2-8e36-ed5e22fe9793`，不另查Pod、官方、预测或重算。

完整selected低于直接R107 `.9884405731067525`及singlebest R120 `.9894773092062407`，双门false，关闭**此固定三role norm1 × 当前R107合同**，no seed27，不global cap/role阈值/adaptive clip/LR/seed补考。AP不救；不推裁剪普遍无效或全局1普遍最优，原None globalclip1继续R107比较路径。原update1默认WORLD finite与三role实际消费已有直接证据，保留为首成功步证据，不推广为长期裁剪频率、某角色主导或梯度爆炸；不增加norm审计/forward补证。

工程阶段分别保留：原汇总脚本沿旧 `backbone=27820128` 错断言exit1；随后按实际formal19项生成成功摘要exit0，没有formal重跑。正式19/19、唯一activation/launch、真实首75及本次完整终态分列，不删旧失败或把摘要修复记成重新训练。运行Git `c03c11738178929fedfbcda78c171911cf1ec3a9`、tree `79367a4527f770fa0de2cbe9aca2fe86398612f9`；必要最终归档待唯一A完成，不假填新archive Git。

selected不足`.996`，official为 `not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，正常completed而非评估失败，TRAIN/ROUND/done完整。owner294462、原start346304405匹配现Z，只称Z不称reaped；原8rank294494–294501退出，lockholders/deviceusers为空、denied0/8busy0，17:34 actualsafe=true。以上引A实证。新CPU20软件allocation SHA `a0fb90eb98615680367dd3591d1ff3e68596a138e17522d64427cc3bed734a08`、valid_until `1793442754.9876158`，当次A0..11/12与train24..119互斥，动态UID51fe…Ready/Python3.12.7/probe0；仅为已发生入口证据，不作为下一入口默认，CPU19过期禁止新入口。不新资源续租或修改旧grant/clock。

原件pins均转引A、不本机重算：progress `2b30be736d5bab157c53cdf606289491e4ae133a95a1b8664283d69a87f4492b`，CPU `c88981b943be19388e00d15c4752029eb041a3bc2fe1f817d8014cbb6d56840d`；freeze `5bc10e7d94d54029e6ed763ba2f3da488c0f48b6c05bee99ddd1ee7e2a17d8b2`，TRAIN `2111c6be7ab859dd860761b2c7e9f081a87a5afc04812cd5819f4f6e52cba69e`，ROUND `1695351f684e4fbcdcbbde77143df67c668a154b7e18635f0ab433e72f01db1f`，done `3f22fa49cfdf669bf02f3887bfd0da8e7059a380672bc34c00db52daedc722bf`。

R107/R108科学锚点、R120仅singlebest、`.996`官方门、`.999`总体目标及6h8h9h/default12GiB/floors/保存合同不变。本页收口冻结交A必要归档；持续授权未暂停，下一问题仅作一次有界既有文本评议交协调方，不注册新轮次、派工、扩裁剪网格或重开FIT/cache/C0审计。
