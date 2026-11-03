# R218/R219：条件 EMA 复核与 Small12k 公开初始化

13:20源码已b8b8bbe6 push/GitHub API回读，完整源树传Pod、受影响SHA与7目标5compile精确一致，[R218/R219两technical](../../records/preparation/small12k-20261103/TECHNICAL-PACKAGES.json)均exit0，无新formal clock/GPU。后备normal在当前R217接近收尾时绑定，完整源树逐文件部署校验由原入口完成；不提前消耗授权窗口。R217最近13:19实观3125 success=attempt/0skip/c3/bad0、八rank活跃，暂online .9832508720744716 / EMA .9812412175448018，未完整selected。当前6fb/SCI92c/SITE4c4冻结保持。下方为此前时点。

13:10实际[7受影响目标/5compile/两完整public fresh](../../records/preparation/small12k-20261103/README.md)已通过，final98ae50cd/guardtt0qtfwv/b890d1c3 exit0。R218 whole91507a65/SCIcbdd1629，R219 whole568263ab/SCIca352c2e；实际provenance和源完整strict已通过，两候选及受影响旧绑定GREEN，原缺orig SHA校验的失败保留并最小补齐。源码/SCI尚待Git同步回读和两technical、后备normal，无新clock/GPU。当前R217/6fb/SCI92c/SITE4c4已真实8rank及自然首25，最近1525 success=attempt/0skip/eval1，暂内部 .9790918866068387，未完整selected；当前冻结不改。下面为12:37设计及当时状态。

沿已启用brainstorming/writing-plans/TDD/executing-plans，由32e3唯一Git/GPU在20%准备池推进；用户明确最小修改无需审批，不恢复/派生/唤醒Agent。当前R215/c9冻结、后备R217/6fb/SCI92c087c8和已复制lease2 technical均不改；R217原normal18/18已通过，临近接续再绑定完整窗口。R218/R219尚无SCI、候选注册、formal/clock/GPU或收益声称。

**固定设计。** R218/order131仅完整R217唯一selected严格胜R107 .9884405731067525与R120 .9894773092062407且actualsafe后，同fe6/原512/long EMA .9999/1600原配方独立seed20261027；否则technical。R219/order132独立直接原R107 Small/seed29/512/ImageNet/G64P8/448 logistic/原AdamW-LR-WD/增强/age/globalclip/24k-800-p4及原EMA1600/.999，只换合法自然图像公开初始化：`timm/convnext_small.in12k_ft_in1k_384`，固定revision `e9f96b3e3e8be16706d721c30d85112a87b82a8b`。完整源先strict加载1000类再沿旧转换/二分类factory丢原1000头，原原生Linear+age初始化规则保持；不叠R217 EMA或其他loss/optimizer/ROI。

**选择依据与限制。** 保留已选R217；SAM/Tiny/L2-SP等既定未采用方向和dropout/SD/LR/WD邻近扫描不重开。采用同Small结构的另一固定公开初始化，可复用当前训练函数，避免新增架构链；这是预训练来源/配方的组合比较，不能纯归因12k或384像素。R213的256图fit子集高于其dev仅是描述性泛化线索，不证明本来源更好，也不改变冻结参数。发布者[模型卡](https://huggingface.co/timm/convnext_small.in12k_ft_in1k_384)说明ImageNet-12k子集预训练→1k及Apache2；实核registry public/non-gated、fixed API/config/card Git blob一致。native静态核344形状/36gamma/41LN/36精确GELU/40Conv/avgpool/无前置LN或MLP/无buffer通过，0forward，尚未加载公开权重。原件`exp-v3/technical-prep/small12k-source-screen-20261103-32e3`；原bin201017809 B/LFS SHA `0a127fb2d1e4634a1ac0a33a999fb18203d2eb190362f8af63bed917a4c71ccf`。公开384/squash只是来源属性，比赛仍原512预处理；不声称原生来源前向数值一致、临床机制、DCU峰值或AUROC收益。

**精确范围。** `tools/pretraining/small_source.py`仅加显式`small12k384` profile/固定revision/tag/training_data、384来源及对应card校验；原两个位置参数/默认small224/旧常量不改。`small_strict.py`复用完整映射及所有现有检查，仅新profile额外核norm_pre/head.pre_logits Identity。`runtime/bundle/retrain/models.py`仅加精确Small12k provenance/license接入，不改factory/forward。`experiment91.py`只新增R218条件别名、精确R219来源和互斥绑定；`tests/test_candidate_binding.py`必要配置/身份目标，`configs/R218-SCIENCE.json`/`R219-SCIENCE.json`。默认12GiB与free16GiB、所有已有存储例外不变。无新依赖或训练/启动/保存/评估/监控平台，不改既有6个隔离草稿。

- [ ] 实际生产download以新显式profile先RED（未登记），再最小profile GREEN；使用真实fixed metadata/完整bin/Apache全文，Pod核SHA和字节，不用mock证明下载通过。
- [ ] 原small_strict按真实来源回执做344全1000/342+2逐张量exact/FP32/41LN/36GELU-SD/40Conv/无BNbuffer检查和冷读取，0forward；固定license SHA，生成原provenance/HANDOFF，不弱化旧检查。
- [ ] 两SCI按既有metadata生成，仅上述科学差异；候选绑定先RED再最小GREEN，拒错源/EMA/seed/LR或叠加策略，旧默认行为按相同源码证据复用。实际Small12k provenance消费先RED再最小接入，不把不支持来源伪装成ImageNet1k。
- [ ] 两actual public fresh/5空组/345全训/新初态SHA、必要受改源码compile，保留真实失败与最终CPU回执；仅必要路径，不重全suite或GPU预演。
- [ ] 精简证据Git同步/API回读、完整源树、两technical及后备正式CPU/精确部署提前完成；唯一GPU owner只在前轮完整安全终态后沿原四调用接续。原R215/R217包、clock/token/grant及官方隔离保持。

12:37设计固定，来源bin仅原始传输已结束，尚待Pod完整SHA及上述实际接入；未将metadata/随机结构等同公开加载或可启动。d837/root0–11/nice19/隐藏GPU与训练24–119互斥，peer closed且12–23保留。raw .999未达。
