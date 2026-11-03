# R212/R213 原生 EfficientNetV2 与公开固定 BN Implementation Plan

> 沿已启用brainstorming/writing-plans/TDD，由32e3本人在独立technical-prep实施。用户授权最小必要修改无需审批、不恢复/派生Agent，故不追加设计批准、外部reviewer、全套重复测试或GPU预演。R211/158冻结不改。

**Goal:** 在R211训练期间准备固定NAdam seed27条件复核，以及不同原生卷积架构/匹配自然图像公开初始化/固定BN的独立fresh对照。raw .999仍未达。

**Spec:** R212/order125仅完整有效R211所选严格胜R107 .9884405731067525及R120 .9894773092062407且actualsafe后，原NAdam配方独立seed20261027，否则仅technical。R213/order126直接原R107的512/FP32/ImageNet/G64P8/448pairs/原logistic/原AdamW/LR-WD/增强/age/globalclip/24k-800-p4/EMA1600，改为timm原生efficientnetv2_rw_s.ra2_in1k和匹配公开ImageNet-1K fresh；112个BN模块使用公开running statistics且保持eval、affine可训练，336个buffer在online及EMA中不可变。它是架构、公开初始化与固定BN的组合假设，不声称纯架构因果、临床机制或收益，不扫变体/BN方案/LR/seed。完整所选>=.996才官方一次。

备选为同步训练BN、新优化器或原生固定BN架构；同步BN需改变微批通信及统计路径，当前选择固定BN来复用现有微批协议。NAdam结果尚待，不据局部结果先判阴性；已关闭的池化/ROI/配额/光度/SSL配方不重开。固定源[完整证据](../../records/preparation/efficientnetv2-screen-20261103/PUBLIC-SOURCE-STRICT.json)：ee38fe1c4e8ca084db9b6ec22c3d6a7d02990704、bin8bb25557/96,674,637 B、794state/458参数张量、23,941,296参数/1792features/112BN/336buffers、BN eps1e-5、自然ImageNet-1K与Apache声明、四个timm1.0.27实现文件pin。不是TV的110BN/1280模型，不做跨框架转换，来源288/test384不替代比赛512输入。

**Architecture:** 完整原生1000类先strict/exact加载，再将原classifier替换为Dropout.2+原生Linear1792→1，并令native.drop_rate=0避免双dropout。原forward_features、SE/卷积/SD.1和avgpool保留，外层age一次。wrapper.train只让BN保持eval，不冻结参数。新模型名/固定来源/frozen_bn进入保存job与SCI，冷factory按job重建并strict载state；EMA沿原1600/.999，只将新336个公开BN buffer固定copy。原R50的159和旧无BN路径保持。

预计binary459参数张量/22,150,091参数、795state（含336buffers）；这是由已测原生源和头替换推导，实际fresh仍须另核whole tensor及带network前缀的buffer digest，不抄原生裸名cc6b摘要。默认12GiB/free16GiB，不新增存储例外。代码消费只在当前新模型/注册分支进入，不改旧模型图或旧运行。

## Files 与任务

1. 新增`runtime/bundle/retrain/efficientnet_model.py`、`tests/test_efficientnet_model.py`；最小修改models/config/inference/ema/distributed_engine，复用已有工厂、optimizer、CheckpointStore与cold加载。先两个原生结构/RNG/固定BN和P8梯度/EMA/冷图目标RED，再实现GREEN。旧159 buffer兼容放在实际受影响测试内，不重跑无关模型/WORLD全套。
   09:55实际调用点复核发现`distributed.py`的`RankContext.prepare_model`及`ReferenceBatchNorm`仅豁免旧R50，须为新固定BN类型加入同一豁免。精确增加该文件两处类型tuple及一个CPU方法目标：WORLD8 context不初始化process group，不转换SyncBN、不合并固定统计。先RED后GREEN，不新增GPU预演或通信平台；原初版6目标/11编译回执保留，最终应为7目标/12唯一源码编译。
2. 来源仅把既有0e88完整strict/许可/原件校验记录生成小型provenance/handoff，复用已有byte fetch/JSON写入，不新建下载或转换平台；必要新增`tools/pretraining/efficientnet_source.py`，实际CLI/compile窄验，来源权重不入Git。
3. 修改`experiment91.py`、`tools/validation/check_candidate.py`、`tests/test_candidate_binding.py`及新增`configs/R212-SCIENCE.json`/`R213-SCIENCE.json`：精确候选/来源/BN-EMA/保存身份、显式互斥及官方隔离，候选身份先RED后GREEN。r92_initial_gate与single91现已动态消费candidate BN计数/不可变buffer，复用不改；原loss/gather/采样/LR/age数学保持。
4. 两actual public fresh、必要源码compile/最终身份回执、Git同步回读、两个technical包；后备normal在当前训练可行收尾前完成。真实八rank/自然首更新接管GPU可运行证明。

## 接续时间

上轮末训练日志至新首25记录315.833秒，超过300。下次仅在工具编排层把既有窗口、实时snapshot、一次activation、显式设备direct launch预拼成同一SSH传输内串行调用，每阶段非0立即停止；先在已核Pod做shell语法窄验。保留全部原拒绝、校验与独立回执，不新增启动器/平台、不改当前冻结、GPU日志频率或时钟，不提前创建正式clock。实际间隔仍按原三时点分别报告，不承诺连续利用率。

## Ledger

- 10:25原normal完整窗口实拒绝bbydqhcu/e04d6649 exit1，无clock/activation/GPU，不是科学失败。新未启动`r213-20261103-32e3-ready-lease2`只字节复用technical包/936材料、保留原normal19及旧grant；guardbyyubyrz/e37039b9 exit0。lease2无新normal/GPUallocation/clock，临近当前收尾再按既有入口创建；不可再用旧f2d或旧四调用。R211最新7725/0skip/eval9/bad1、暂best EMA c8`.987315937198597`，当前r211 env/start4不变。独立诊断7件已06e980f统一push，9测试复用，源码/test与Git blob在Pod逐字节pin，不新增GPU门。

- 10:08原R213正常CPU19/19、精确部署已完成，actual SITEf2d45c09/guardfm3w09dh0；whole cb874611/459全训/五空组/112BN336buffer、公开512保存冷图exact，源码936冻结，7/12复用，无clock/activation/GPU。10:09实观R211 c7@5600新best`.9865332676693667`、bad0，最早终态后移；实际接续若不再覆盖完整clock，保留旧normal和回执，仅为新未启动阶段另绑资源，禁止改旧grant或clock。现役R211原158不动；R209已封存难组7408/3632并完成9旧权重清理/保护核验。

- 10:07源码936dde0/tree06cc0c7实际push/API Pod回读与完整source SHA匹配，两technical包e7fb2277/guardjg7qzezu、fiwf8ds0及archivevzbsyy9t均0，默认12GiB、无normal/clock/GPU。既有四调用同SSH模板bash-n通过h63cwzxl，非新启动器；真实SITE及actualsafe后才调用一次。R211最新4925/0skip/eval6/bad2，后备normal按可行收尾前准备，保持当前158冻结。

- 09:58最终7目标/12唯一源码compile及两实际public fresh通过，CPU-CHECKS 63d5d662/gveot90f exit0，详见[精简回执](../../records/preparation/efficientnetv2-20261103/README.md)。初版6/11、绑定缺门槛和WORLD8固定BN未豁免的失败均保留；必要两类型豁免已RED/GREEN，历史Tiny-only审计不在实际入口故不改。R213 whole cb874611/buffer eb77f82b；R212仍91507a65。现在待Git同步回读、两technical包与后备normal，不把准备登记当启动。09:58另一个既有会话由用户唤起做独立诊断（CPU12–23），本人未派生/恢复/发消息，唯一Git/GPU不变；其新DIAGNOSTIC-EVIDENCE文本不混入本次提交。

- 09:34设计固定；源已完整strict但二分类适配、测试与SCI均未实现。当前R211实核850/0skip/eval1/八rank活跃；准备d837/root0–11/nice19/隐藏GPU，与训练24–119互斥。范围以本页列项为限；当前R211与原EC5/GPU allocation/clock/token不变，R209封存难组/冗余权重清理在下一包准备不受阻时完成。
