# R232 固定9600 EMA历史窗口实际启动

00:03后续[SSL750快照](R232-PROGRESS-after-r230-cleanup.json)c0aec43d/guardd2370b0f仍为原owner和八rank活跃，loss/gradient有限，原型已按722冻结边界自然解冻。旧R230[9 checkpoint清理及终检](../storage/closed-r230-20261103/README.md)完成，当前R232及其冻结/clock/token/grant保持，完整9600与监督收益仍待。

唯一问题仍是[事前登记的时间尺度](../../todo/auroc-999/R232-R233-EMA-HORIZON-20261103.md)：在同一EMA跨视图目标上固定9600步，随后保持原监督合同。只用fit图片和CSV，fresh合规公开初始化，不继承R230状态。完整内部AUROC、AP和固定20/其余147修复及新增错序尚待；当前fit诊断不证明收益。

[真实首200](R232-START.json)431b2df7/guardd925f9e4核实owner825876/start377725472和八rank活跃；[公开初态门](R232-SSL-INITIAL-GATE.json)、[八rank初态](R232-SSL-INITIAL-RANKS.json)均通过。完整公开模型50c6f55c及344公开张量、空初始优化器与正式CPU一致；教师1a1d51c8与学生及原P8精确一致，无梯度或参数共享。实际G64/P8、128唯一视图、64×65536输出、学生/教师合计256 encoder-view forward；首1 loss10.446009、首200 loss10.861097，梯度和损失有限，722步内原型冻结符合合同。9600完成、教师最终一致/丢弃和监督冷交接仍待。

[准备及启动来源](R232-PREPARATION.json)：冻结Git4fea64d/tree dfe968ec，SCI776a37be，actual SITE b642d573，ready=`r232-20261103-4e0c-ready`。2699整树、10必要目标/8编译/62相同源码范围复用、18/18正式CPU及精确部署均在R230训练期间完成；未追加P8前向或独立GPU预演。原1600全部LR/教师动量逐步精确兼容，监督数学保持。启动前[新核窗口](R232-WINDOW.json)通过；一次activation guard5e10d5a9、一次入口CLI/实际dispatch，无调用方预设taskset，由原入口按实核绑定96核。准备回执中的未启动标志是历史时点，实际运行由独立启动证据证明。

[接续实测](R232-HANDOFF-GAP.json)：R230整轮23:47:08.651 UTC结束后243.238秒dispatch、339.998秒首SSL、359.991秒自然首25；后两项超过300秒。从前轮最后训练更新至首SSL为438.536秒。终态观测延迟和临时编写入口命令消耗了接续时间，后续将现有入口命令文本在训练期间准备好；必要activation和真实初态校验仍保留。连续GPU利用率未测，不将这些事件间隔当作设备空闲实测。旧safe33c2226b只复用，未重复查旧GPU或重置任何身份。

当前私有`r232-pod-context.env`以ACTUAL-START-1（SSL）为first receipt，监督首成功后才更新。4e0c唯一Git写入者/串行GPU owner，独占24准备核0–23与96训练核24–119互斥；CPU新b3a7bb69至epoch1793769296.744522，旧allocation和各轮clock/token/grant保持。R233仅在本轮完整严格胜R107和R120后，且须新核足额窗口；R231因R230双门失败不eligible/未激活。目标.995/AP>.9及raw.999未达。
