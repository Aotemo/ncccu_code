# R230 fit-only EMA跨视图分布实际启动

实际八rank公开初态、教师初态及自然首成功已经核实：[首75观测](R230-START.json)60a79c5a/guard316efefc；公开模型50c6f55c、全部344公开张量及空初态匹配18/18正式CPU。[教师/学生初态](R230-SSL-INITIAL-RANKS.json)八rank一致、无梯度/无参数共享，完整教师摘要1a1d51c8与真实P8 CPU一致；[原公开初态门](R230-SSL-INITIAL-GATE.json)已释放。实际全局64图/128唯一视图、输出64×65536，学生与教师合计256 encoder-view forward；首1 loss10.446009、首75 loss10.965660、loss/梯度有限，原型在722步内冻结。目标熵与平均目标熵只作fit诊断，不用来选SSL checkpoint或宣称收益。

完整1600适配、最终教师/center跨rank一致及丢弃、监督交接和完整dev收益仍待证明。当前核实点owner817444/start377243020及八rank活跃；冻结Git4099eed/tree4888dc5c、SCI8cb98a2b、actual SITE d0385fcd、ready=`r230-20261103-4e0c-ready`。当前私有r230 env的first receipt为ACTUAL-START-2（SSL），全程保持该轮原clock/token/grant；监督后再更新为真实监督first receipt。4e0c唯一Git/串行GPU，准备0–23/24核与训练24–119/96核互斥，无新Agent或资源。

[准备及启动来源](R230-PREPARATION.json)：19必要目标/5编译、真实公开Small/P8/冷derived及两正式18/18/2662文件精确部署均在R228运行时完成，source4099eed已push/API回读。prep中的false标志是22:16未启动快照；其后一次activation/入口CLI/实际dispatch及真实首成功由同文件的独立回执与上述生产证据证明。第一次元数据观测仅确认owner和8个rank已出生、公开/首成功文件尚未写齐，记unknown；后续观测才确认首75，没有重启或再次派发。启动前新核42370秒剩余≥完整clock+60的31560秒，window guard28598461，activation guard2df2fcc7；未加外部taskset，由原入口实核未收紧资源后绑定96核。

[接续实测](R230-HANDOFF-GAP.json)：R228整轮结束22:27:41.696 UTC后185.675秒实际dispatch、278.461秒首SSL成功、298.386秒自然首25，三个整轮终态代理均≤300秒。若从前轮最后训练更新计，则至新首SSL为385.818秒；连续GPU利用率未测，不能把任一完整间隔当作实际设备空闲。旧safe6b3e3a47仅复用，未重查旧GPU或重置任何旧身份，未等待文档或审批。

[固定问题](../../todo/auroc-999/R230-R231-FIT-EMA-ASSIGNMENT-20261103.md)在R228完整结果前已登记：以fresh学生自身的EMA教师、居中/锐化stop-gradient跨视图分布，补充R115同时刻全局统计之外的历史目标。仅fit图，原R115均匀采样/独立两视图/512/1600/优化器/LR/随机域保持，三层2048/256及65536单位scale原型为临时头；结束只传学生encoder/final-LN，监督仍原R107。它不是完整DINO或眼底论文结果的复现，计算成本也不同。完整内部AUROC主判，AP、原固定20/其余147修复/新增错序及成本并列；完整严格双门后才固定seed27 R231，官方仍原门。

前轮[R228完整阴性](../R228/README.md)：6400/c8/bad4/0skip、唯一EMA c4 .9845260214/AP .8101778592未过原R107/R120，R229不eligible/未激活/官方未跑；固定20和其余147错序分别净增2173/1078。目标尚未达到。
