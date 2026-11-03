# R228 fit-only 局部光度一致性实际启动

21:32 [1450次适配观测](R228-PROGRESS-SSL-2.json)：owner808714及8rank同出生身份活跃，原64图/128视图/16384×256矩阵及finite loss/gradient保持，完整1600→监督交接仍待。同期R226已按Git提交及精确清单清理9个checkpoint/6.733GB逻辑字节，free实观32.320GB，三KEEP及当前R228保护不变；没有新GPU预演、训练信号、锁或clock重置。

21:23 [最新真实观测](R228-PROGRESS-SSL-1.json)：675次fit适配，原owner和8rank同出生身份活跃；global64图/128视图、16384×256相关token矩阵、loss/梯度有限，监督交接尚未证明。条件[R229正式18/18/精确部署](../preparation/local-invariance-20261103/R229-FORMAL-READY.json)也已在准备池完成，actual SITEfea174de/公开91507a65，guard353905d3；未激活/无clock/dispatch，不能等同于晋级。

11-03 21:15:50 UTC实际首SSL成功，首25也已实证：8个rank、公开模型50c6/encoder/public张量及空初态匹配正式CPU；全局64图/128视图、实际矩阵16384×256、每图256个相关空间token，loss/梯度有限。首1 loss21.812551、首25 loss18.627014；token不是独立图像。owner808714/start376783424及八rank活跃，[实际首成功](R228-START.json)、[公开初态门](R228-SSL-INITIAL-GATE.json)、[正式准备与guard](R228-PREPARATION.json)保留。完整1600到监督交接及完整dev结果尚未证明，不能据SSL loss宣称收益。

冻结Gitbe5b5fa8/tree60d7432b、SCI087564fa、actual SITE84b5108e，ready=`r228-20261103-4e0c-ready`。14必要目标/5编译、真实公开Small/P8双视图与冷derived、18/18正式CPU及2622全树bytes/mode核验均在R226训练期间完成；[准备](../preparation/local-invariance-20261103/README.md)828f37f5/8ea529cf，实际公开初态50c6、345全训/5空监督组。启动前新核full clock+60秒窗口，一次activation exit0，无独立GPU预演。当前唯一Git/串行GPU监督4e0c，准备0–23/24核与训练24–119/96核互斥；私有实际入口`goal-20261103-4e0c/r228-pod-context.env`，旧r226 env仅历史，当前first receipt为ACTUAL-START-2，监督交接后再换成真实监督首成功证据。

[固定问题](../../todo/auroc-999/R228-R229-FIT-LOCAL-20261103.md)：相同fit照片的同一几何位置，在独立原brightness/contrast/color下学习16×16局部一致性，区别于R115只约束全局GAP向量。相同公开Small、R115的1600/G64/P8、两视图/512、MLP768→1024→256/optimizer/LR和随机域；仅必要配对几何及空间目标改变，不称纯组件因果比较或VICRegL复现。第一视图与原默认双视图已有8/8真实fit图逐像素兼容证明。只用fit图，不用label/age、旧teacher或额外任务图；临时projector结束丢弃，encoder-only derived后原R107监督24k/800/patience4、公开head-age、空AdamW/EMA保持。

首次入口被调用方额外`taskset -c24-119`提前收紧，标准入口要求先核原始uncontracted CPU资源，因此在prepare_native_entry内拒绝，尚未写admission/dispatch/controller/owner。四标记均不存在后，仅移除外部taskset，用fresh launch-only Pod快照调用原入口；由原入口核原资源并绑定96核，返回0并产生唯一实际dispatch。两次入口调用、一回activation；原clock/token/grant及冻结源码均未改，也没有重启活跃运行。错误来自调用方，原stdout-only日志没捕获stderr，原工具栈和四标记检查回应保留为失败来源，不伪称另有原始stderr文件。

[实测接续](R228-HANDOFF-GAP.json)：R226整轮结束至新dispatch258.174秒、首SSL357.314秒、自然首25为374.255秒。后两项超过300秒，主要可避免原因为上述入口亲和性错误及纠正/观测传输时间；准备并未欠缺代码/CPU/正式包，没有文档或审批等待。连续GPU利用率未测，不能把整个间隔等同设备空闲。旧R226唯一safe ac9cbc91仅复用，不重查。

[R226完整11200/c14/bad4](../R226/README.md)的唯一EMA AUROC.9858902745/AP.8162271394未过R107/R120双门，R227不eligible且未激活；R228按事前分支接续。完整内部AUROC继续主判，附AP、相对R107/R120/R115/R224/R226的原固定20/其余147修复与新增错序及实际成本；严格双门后才激活固定seed27的R229。官方仍原≥.996完整selected后一次，当前未运行。目标尚未达到。
