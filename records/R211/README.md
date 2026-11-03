# R211：原生解耦 NAdam 独立 fresh

已[完整结束](R211-RESULT.json)：9600 success=attempt、12次内部dev、连续4轮无提升、零跳过；唯一EMA c8@6400 AUROC`.987315937198597`、AP`.8143068719077303`、error_twice21068。严格R107/R120双门均false，R212不eligible，官方按阈值未跑，raw .999未达。所选tensor168f1e56/blob9d427920/CSV70f65bb8/precisiona623bb91；原freeze.status=null保留。[唯一终态](R211-COMPLETION.json)与[CPU回执](R211-CPU-COMPLETION.json)已证owner inactive、八rank退出、无设备holder/user，旧safe不重查。归档guard50ut1gx3/96f8031a exit0，独立文件[R211-CPU-COMPLETION-ARCHIVE.json](R211-CPU-COMPLETION-ARCHIVE.json)保留原首成功归档回执。随后原入口串行接续[R213](../R213/README.md)，本轮冻结/clock/token未改。以下启动阶段原记录保留；所选权重暂保护，难组与清理尚未完成。

09:23[真实八rank初态与自然首25](R211-START.json)通过，09:26实观400 success=attempt/0skip/eval0、八rank活跃。345全训/5组/各rank空优化器state/whole50c6与正式CPU一致；首25 loss`.6543780565261841`有限，WORLD8/G64P8/448pairs、每rank8primary/8view均相符。真实训练链可运行，完整AUROC收益尚待；raw .999未达。

直接R107 Small/publicfe6/seed29/RGB512/FP32/ImageNet/原logistic/G64P8，仅`nadam_policy='nadam-decoupled-native-v1'`改为原生`torch.optim.NAdam(groups,decoupled_weight_decay=True)`。原五组成员/顺序/LR/WD及age保持，原生betas(.9,.999)/eps1e-8/momentum_decay.004/foreach=None；不叠AMSGrad/beta095/Lion/hinge或新读出。最多24000成功/每800双variant内部dev/patience4/EMA1600及唯一选点保持，完整selected>=.996才官方5849一次。

[8受影响目标/6compile/两actual public fresh](../preparation/nadam-20261103/README.md)与[normal19/19及精确部署](R211-PREPARATION.json)均在R209训练期间完成，[正式CPU实际NAdam五组kwargs](../preparation/nadam-20261103/NORMAL-R211-NATIVE-OPTIMIZER.json)相符。[源码身份复用](R211-CPU-REUSE.json)不重复测试。现有非Lion启动观测读取八rank空state、345张量初态和自然更新；字段all_rank_optimizer_identity_matches_formal仅其generic非Lion条件，不能称逐rank逐组NAdam kwargs或mu_product/m/v遥测。首条LR列表来自rank0自然日志，不外推独立全rankLR测量。

冻结source `158bfb0c9e174d9a71991651a1c7e33d77c3eaf8` / tree `2e993be9c5838c20bec55a1addb18e29cc047f77`，SCI `1ef87681e57bed8ab057bf100a81486ebf9b7f4e116d63434ea5c483952533b0`，actual SITE `c8d7435c5b2835bd648abd4662dbeef1a48169788a3079e63a2a4fdf7b60fa0e`。manifest4b182302/cpu-prep6573ca60/deployment703a8456；default12GiB/free16GiB/storage93f8a86f保持。[窗口guardjnwey0zc](R211-CPU-WINDOW.json)/0d4ea84d、[一次activation guardc123p991](R211-CPU-ACTIVATION.json)/cfc014e4、[first4 guardx54_k_78](R211-CPU-FIRST-SUCCESS.json)/273d25fc均0，显式八设备direct launch一次0，无独立GPU预演。初期start1/2/3的starting_or_unknown原件保留，不当失败或重启理由。

R209[完整8800/c11/bad4所选](../R209/R209-RESULT.json)EMA c7@5600内部AUROC`.9867066590727653`，双门false/官方未跑/actualsafe，R210不eligible，旧safe不重查。[实际交接](R211-HANDOFF-GAP.json)整轮结束至dispatch102.804秒/最新设备169.122秒/首25记录249.651秒，均<=300；末训练日志至三项168.986/235.304/315.833秒，最后一项超300如实保留。连续DCU利用率未测，不把三时点或设备上下文等同持续计算。后续只在工具编排层预拼并串接已有窗口/snapshot/activation/direct调用，减少独立SSH往返与采样等待；各原有拒绝/退出码及安全终态仍保留，不新建启动器/平台、不改当前冻结链或日志频率。

owner721112/start372507013；rank721131–721138/start372511594,596,597,598,599,601,602,603。训练24–119/96核，rank24–111每11核、control112–119；准备d837/root0–11/12核与保留peer12–23/12核互斥，当前r211-pod-context.env/start4。永久三KEEP和所有运行依赖保护。后续准备在20%池继续，完整结果前不作科学收益判断。
