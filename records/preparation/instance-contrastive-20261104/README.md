# 固定fit-only实例负对照的CPU准备

02:18 [正式两分支回执](FORMAL-READY-VERIFICATION.json)1b8b76e8/guardd4388182通过：从未激活R233的新独立ready为`r233-20261104-4e0c-ready`，沿4fea/tree dfe968ec、2699源、18/18 CPU、公开91507a65/345全训/空优化器；[actual SITE](R233-FORMAL-READY.json)f5c09d84、部署9526b793、guard5d5d5a3e。旧R233/36406c3d及其grant原件不改。R234沿已发布5654574/tree131b36bb、2744源、18/18 CPU、公开50c6f55c/345全训/空优化器；[actual SITE](R234-FORMAL-READY.json)c5ad926f、部署4dfc6395、guard35407385。均无activation/clock/dispatch/独立GPU预演；正式检查没有重跑真实P8或冷derived。

两新grant均核完整31500秒clock+60秒启动余量，latest activation分别epoch1793770160.0080159/1793770284.7596166；临近窗口前仍须检查，过期只能为从未启动阶段另建独立输出，不能改旧grant。两条原四步入口已用这些实际READY/SITE填好，私有本机`post-r232-R233-entry.sh`/`post-r232-R234-entry.sh`及会话命令store保留，未执行；沿用原模板语法证据，不声称另有实际activation。唯一分支仍等完整R232严格双门和唯一actualsafe；R233资格成立时优先，否则R234。

[当前R232快照](../../R232/R232-PROGRESS-formal-next-ready.json)e27394d0为监督1500成功=attempt/0skip/c1/bad0，原owner及八rank活跃；c1临时online/EMA均.9661043888/AP .7346687961，尚无完整selected。24核b3a7准备与96核训练保持互斥，同UID实核120；02:12唯一4e已核。小JSON导出沿既有元数据模式，私有helper a150e964/guardd4388182只校已完成两CPU树、部署SHA及窗口并调用原progress observer；[导出清单](FORMAL-EXPORT-RECEIPT.json)保留Pod侧SHA，没有图像、样本预测、模型加载或新前向。

此前已将源码/config及CPU证据随5654574/tree131b36bb推送并API回读，[2744整树bytes/mode核验](TRANSFER-VERIFICATION.json)b7c2951e通过。两[技术包](TECHNICAL-PACKAGES.json)c2f3e291用同一发布源码直接exec生成，R234/R235技术SITE c0cb7540/dcbe5dab、guardbf5fda9c/21a67aa1；它们不是GPU正式绑定。6个私有元数据helper编译及固定五baseline设置核对通过f15add86，[准备回执](NEXT-METADATA-PREPARATION.json)不包含未来R232结果或预测。R232结果分析已明确9600预算并追加同目标短窗R230比较，但尚未执行未来完整分析。首个dev后两条件分支正式CPU/部署已按上方完成，旧R233和当前R232原grant/clock/token不改。

现有四步接续命令在当前训练期间预写，保留window→新Pod快照→唯一activation→唯一launch顺序，GPU入口不加外部taskset且显式八设备可见性。[命令语法回执](ENTRY-TEMPLATE-SYNTAX.json)b1ad7d7f/guard76be1542只运行bash -n、没有执行入口；实际新READY/SITE身份已填，仍等R232完整selected与唯一actualsafe。当前原source4fea保持，完整9600及真实监督交接见[R232](../../R232/README.md)，完整监督收益待证明。

[事前问题](../../../todo/auroc-999/R234-R235-FIT-NTXENT-20261104.md)仍为同R115配方仅换NT-Xent损失，不延长旧VICReg、不加入定位或疾病标签。R234/R235两SCI分别3ee3cbc3/7328a2f1；只有R232完整未过原双门且唯一actualsafe后才考虑R234，完整双门过仍优先R233。正式新绑定已完成，activation、dispatch及GPU收益仍未发生；R235仍只有技术包。

[CPU汇总](CPU-CHECKS.json)08730037：3必要目标、4受影响文件编译通过，原gather/VICReg、SSL模型/采样/RNG/优化器/完成门/derived加载及原默认预处理共22个函数或类范围精确相同；11个未改实现文件字节相同。独立Python标量oracle、双分支double梯度检查、同图目标索引、N64的128×128几何/126负视图、自身排除、换序及逐行尺度不变，以及新两候选完整R107监督、原R115全部RNG流和既有受支持配方均验证。原实际冷derived及5空组证据026afc26按相同加载范围复用，不新增冷加载前向。

[真实公开Small局部P8](ACTUAL-P8-CPU.json)93ac00ab/guarda4dab8d7：公开50c6f55c、344张量严格匹配；8个fit/16独立视图输入8×2×3×512×512，实际输出8×2×256和局部16×16损失矩阵/14负视图。346梯度均有限，clip前范数53.2212、loss2.451242，按首步LR因子.00625完成更新，encoder确实改变、公开head-age保持。CPU实际110.889秒/12数值线程/0–23 mask，无权重fixture文件或GPU初始化；这不是实际八rank128×128证明，后者留给自然首成功。

缺行为RED df3d57d7原件保持。首轮两个损失/梯度目标通过，但第三目标把历史R107原配置错误地当作当前工厂支持项，整体guardf17c7981 exit1不作通过。97k9v39t/d4670285逐例对照冻结4fea：R107在旧/新工厂同样拒绝，其余R115/R230/R231/R232/R233同样接受；未修改历史配置、源码兼容范围或指标。R107完整监督字段比较仍保留，正向兼容目标限受支持配方；仅重跑失败目标并复用两个通过目标的相同方法AST和原运行代码字节，最终guard2e9e11da通过。原gather及其DDP缩放按相同函数范围复用，没有重复GPU预演。

主Agent按代码审查技能核过索引、自身mask、归一化、全局gather/DDP缩放、recipe/RNG、头及监督隔离和日志几何；遵照用户不派生Agent要求，没有独立审查者。准备为原24核独占池/nice19/隐藏GPU，当前R232原96核和冻结4fea/SITE b642不改。导出guard8bf5b4a6通过，只有源码/config和这些小汇总入Git；图片/样本级预测/凭据未导出。00:39封存的[当前SSL3350快照](../../R232/R232-PROGRESS-ntxent-CPU-ready.json)08558c4d表明原owner及八rank活跃、损失/梯度有限，完整9600和监督选点待；.995/AP>.9及raw.999未达。
