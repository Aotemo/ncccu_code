# R232 固定9600 EMA历史窗口完整阴性

后续[8 checkpoint清理](../storage/closed-r232-20261104/README.md)已完成并终检，plan af24732d/delete62b9d079/最终CPU248c6655：5.941GB逻辑字节，三KEEP及当前R234保护stat/小SHA保持，free点观测28.616GB。只删除实际清单内PT，derived在清单外；历史结果、预测CSV、precision及sidecar保持。下方归档时点“待清理”已由本条更新。

[完整结果](R232-RESULT.json)64420323：9600 fit适配后原监督6400成功=attempt/0skip/c8/bad4正常早停，唯一selected为EMA c4@3200，AUROC **0.9829462330115558**、AP **0.8046405473793247**、error_twice28326。未严格胜R107或R120，R233不eligible且未激活；官方未达.996，按政策未运行，.995/AP>.9及raw.999未达。已确认[唯一actualsafe](R232-COMPLETION.json)6e329f85/guard53a425d1：owner退出、原八rank缺失、锁/设备使用者空、denied0/八卡busy0，不再重查旧GPU。随后[实际接续R234](../R234/README.md)，当前训练冻结保持565。

[配对报告](PAIRED-REPORT.json)只复算原selected CSV。对R107，全dev错序9600→14163，修复1728/新增6291/净增4563；原固定20为6623→10251，修复961/新增4589/净增3628；其余147为2977→3912，修复767/新增1702/净增935。图片配对bootstrap2000/PCG64 seed20261103，ΔAUROC vs R107 −.005494340，95%区间[−.009501728,−.002096380]；vs R120 −.006531076、[−.012526475,−.001932531]；vs R115和同目标短窗R230区间也均低于0，vs R224/R226/R228跨0。未校正患者聚类、seed、选择或多重比较，不推出临床因果或模型上限。

实际fit适配614400主图/1228800增强视图/2457600 encoder-view forward，其中1228800教师forward；首至末SSL日志8004.983秒，联合训练组件11232.067秒，启动至SSL完成8098.022秒，随后监督至组件结束3134.045秒。本固定长窗及其必要日程伸展整体验证阴性，不重开预算/动量扫描，不复用本轮状态作初始化或teacher。结果归档guarda122b88a、[回执](R232-RESULT-RECEIPT.json)验证原safe和CSV，没有新前向/权重加载。所选tensor98c500e6、预测afd06ae6、precision a55fad69保留身份；完整结果与新首成功同步后，旧checkpoint按真实清单清理，当前尚不声称已删除。

以下是本轮实际启动和交接的历史记录；其当时“待”事项已由上方完整结果更新。

02:18 [监督1500快照](R232-PROGRESS-formal-next-ready.json)e27394d0/guardd4388182仍为原owner及八rank活跃，1500成功=attempt/0skip/c1/bad0；首个dev online/EMA均为AUROC .9661043888494878/AP .7346687961376994，仅临时结果，完整selected待。当前训练期间已完成[未启动R233新足额绑定及后备R234正式CPU/精确部署](../preparation/instance-contrastive-20261104/README.md)，actual SITE f5c09d84/c5ad926f，各18/18，均未激活；原四步入口命令已填实际身份，完整双门与唯一actualsafe前不执行。

02:07 [完整适配](R232-SSL-RESULT.json)f4ecae75与[监督交接](R232-SUPERVISED-RECEIPT.json)e781325d已核验：9600成功=attempt、0skip、dev0，仅fit、无标签/age用于适配loss；八rank学生derived64971ebf、教师772df690及center bef7afb9最终各自一致，教师无梯度且teacher/projector/center不持久化。原head-age与监督factory RNG保持，实际监督完整初态8996b216/345全训/5组空AdamW通过原门47bc841c。

[自然监督首25](R232-SUPERVISED-START.json)40bc9926/guard8f2aa998：loss .61780423有限，原G64/P8、448配对及实际采样流=登记保持，owner825876/start377725472和八rank真实活跃、资源mask匹配。元数据归档guard500a4991通过，只读小JSON，没有模型加载、新前向或冻结修改；私有archive helper 7b0f8b5f仅编译回执0182dda7先行。当前私有r232 env的first指针已从SSL首成功改为这一监督proof，历史原件保留。完整内部选点与配对修复/代价仍待；首个dev后完成从未激活R233和后备R234的新正式绑定，未选择或启动分支。

01:12 [实际SSL5725](R232-PROGRESS-postcommit-1.json)5e8d0672/guard737fc963仍为原owner和八rank活跃，loss/gradient有限；9600结束、监督交接和完整排序仍待。后备NT-Xent源码5654574已发布/2744精确整树/两技术包准备，尚未正式绑定或激活；当前冻结4fea不改，完整双门过仍优先原9600第二seed R233。当前监督首个dev后才为未启动两分支建立足额新绑定，预写入口仅语法检查，未提前执行。

00:03后续[SSL750快照](R232-PROGRESS-after-r230-cleanup.json)c0aec43d/guardd2370b0f仍为原owner和八rank活跃，loss/gradient有限，原型已按722冻结边界自然解冻。旧R230[9 checkpoint清理及终检](../storage/closed-r230-20261103/README.md)完成，当前R232及其冻结/clock/token/grant保持，完整9600与监督收益仍待。

唯一问题仍是[事前登记的时间尺度](../../todo/auroc-999/R232-R233-EMA-HORIZON-20261103.md)：在同一EMA跨视图目标上固定9600步，随后保持原监督合同。只用fit图片和CSV，fresh合规公开初始化，不继承R230状态。完整内部AUROC、AP和固定20/其余147修复及新增错序尚待；当前fit诊断不证明收益。

[真实首200](R232-START.json)431b2df7/guardd925f9e4核实owner825876/start377725472和八rank活跃；[公开初态门](R232-SSL-INITIAL-GATE.json)、[八rank初态](R232-SSL-INITIAL-RANKS.json)均通过。完整公开模型50c6f55c及344公开张量、空初始优化器与正式CPU一致；教师1a1d51c8与学生及原P8精确一致，无梯度或参数共享。实际G64/P8、128唯一视图、64×65536输出、学生/教师合计256 encoder-view forward；首1 loss10.446009、首200 loss10.861097，梯度和损失有限，722步内原型冻结符合合同。9600完成、教师最终一致/丢弃和监督冷交接仍待。

[准备及启动来源](R232-PREPARATION.json)：冻结Git4fea64d/tree dfe968ec，SCI776a37be，actual SITE b642d573，ready=`r232-20261103-4e0c-ready`。2699整树、10必要目标/8编译/62相同源码范围复用、18/18正式CPU及精确部署均在R230训练期间完成；未追加P8前向或独立GPU预演。原1600全部LR/教师动量逐步精确兼容，监督数学保持。启动前[新核窗口](R232-WINDOW.json)通过；一次activation guard5e10d5a9、一次入口CLI/实际dispatch，无调用方预设taskset，由原入口按实核绑定96核。准备回执中的未启动标志是历史时点，实际运行由独立启动证据证明。

[接续实测](R232-HANDOFF-GAP.json)：R230整轮23:47:08.651 UTC结束后243.238秒dispatch、339.998秒首SSL、359.991秒自然首25；后两项超过300秒。从前轮最后训练更新至首SSL为438.536秒。终态观测延迟和临时编写入口命令消耗了接续时间，后续将现有入口命令文本在训练期间准备好；必要activation和真实初态校验仍保留。连续GPU利用率未测，不将这些事件间隔当作设备空闲实测。旧safe33c2226b只复用，未重复查旧GPU或重置任何身份。

启动时私有`r232-pod-context.env`使用ACTUAL-START-1（SSL）；现已按上方真实监督证据更新first指针。4e0c唯一Git写入者/串行GPU owner，独占24准备核0–23与96训练核24–119互斥；CPU新b3a7bb69至epoch1793769296.744522，旧allocation和各轮clock/token/grant保持。R233仅在本轮完整严格胜R107和R120后，且须新核足额窗口；R231因R230双门失败不eligible/未激活。目标.995/AP>.9及raw.999未达。
