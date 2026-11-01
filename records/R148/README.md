# R148 模型随机正则联合置零：八卡首成功

20:16:05.118783 UTC实际running50success=attempt/0skip/eval0；真实八rank345全训/5组空AdamW/whole50c与formal一致，首自然25 finite loss`.6842917203903198`、WORLD8/G64P8/448pairs。owner449447/start359140571及八rank449466–449473的start/argv/mask/8PCI匹配。[首成功](R148-START.json)与[实际随机正则元数据](R148-ACTUAL-STOCHASTICITY.json)分别证明：八rank唯一head dropout为0、各36个SD均p0；仅读初始化小件，无新模型、forward或checkpoint加载。

冻结runGit `d546d3024c8e11bcc5165c050af26dd34f1c8223`、tree `7154b12f77916f36901c37f0e568ffe10e119f30`，SCI SHA `ca60e78edbdb62f234ab7c9976b52303d59099324320b35fa93ab10b012b9121`、SITE SHA `fa03b3a7f2365c88c0f7db583ac7b628aa1ab01be43a8bbeee62d68c7e60a33a`。必要两CPU/2源compile按精确SHA复用，唯一normalformal18/18（64.945秒）与R146冻结收尾重叠，精确Git部署9.683秒；一次activation/direct launch0，未重formal/clock/token/grant。[准备](R148-PREPARATION.json)、[复用](R148-CPU-REUSE.json)、[部署](R148-DEPLOYMENT.json)和各CPU阶段原件分列。

[接续](R148-HANDOFF-GAP.json)实际前轮完成至dispatch **170.030秒**、至最晚八卡设备初始化 **228.462秒**、至首自然25记录 **295.074秒**；这三个事件间隔均≤300秒。没有连续DCU利用率测量，不把事件间隔冒作连续空闲证明。安全退出后立即只执行既有activation/direct launch，归档未阻塞GPU。

[固定SCIENCE](../../configs/R148-SCIENCE.json)直接R107/public fresh Small/seed29/global512，唯一方法是模型head dropout `.2→0`与stochastic depth `.1→0`联合消融；不拆分归因、不扫rate/LR/seed。原图像增强、G64P8、FP32/noTF32、AdamW/LR/WD/age/EMA/24k800p4/唯一selector及官方隔离保持。完整selected同时strict>R107 `.9884405731067525`与R120 `.9894773092062407`且actualsafe，才固定seed27；否则关闭精确联合配方。当前无完整selector或official，raw .999未达。

唯一Git/Pod/GPU owner32e3；训练24–119/96核，准备00202581本人0–22/23核、槽23/1核保留不用，两个池互斥，总prep cap24。当前入口`work/r137-20261101-a/r148-pod-context.env`绑定START2。R146完整安全终态已固定，不重新观察；其selected/冻结包、R130全部、KEEP与所有实际依赖继续保护。
