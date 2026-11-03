# R206/R207：固定逐配对平方 hinge CPU 证据

07:00 R205实观4850/0skip/eval6/bad3且八rank活跃时，后备R207的[normal18/18和精确部署](NORMAL-R207.json)已完成；[guard6to2lglz](NORMAL-R207-CPU.json) / `e3cb8c8d13cc7a4d38221ff7758f181e5b3f4c864c450c96a97f49dd99d7d957` exit0。[原5目标4compile按SHA复用](NORMAL-R207-REUSE.json)，345全训/5空原生AdamW、whole50c6一致。actual SITE `1718c7f3c2114d1209a2e2c09b5a184805525fcd073c30a4567958f9b068e000`、manifest4ad852b1/cpu-prepc0ce027c/deployment859dc9f9；原78e80aa/SCIa146cfe2及默认12GiB/free16GiB保持。无clock/activation/GPU；完整R205双门与actualsafe后决定唯一接续，R206仍仅technical。ec5准备0–11与训练24–119互斥，不重复normal或重查旧safe。下文为早期时点。

05:43已按GitHub实际回读commit `78e80aa25701888b482e96de3066511de63f2def` / tree `f6bd57161068be6d658ca23af4471f4ebd2cf2aa`构建[两technical包](TECHNICAL-PACKAGES.json)，真实source/SCI与CPU证据逐项相符。R206 SITEee3210f8、R207 SITEdd296d6d，实际run_name均100字符、默认12GiB；[R206 guardfk2ca9_y](CPU-TECHNICAL-R206.json) / [R207 guardyvd374om](CPU-TECHNICAL-R207.json)均0。无formal-cpu/gpu-inputs/真实round，无新allocation/clock/GPU。[归档guardk9rxoipf](CPU-TECHNICAL-ARCHIVE.json) / `99b9859643693af5f6e6fdf4029e692250ce8646b840313c6f7a06398f43632e` exit0，汇总SHA be989789。原R203/573和既定R204/R205/7ead不改。下文待Git/technical包为CPU完成时点。

[CPU汇总](CPU-CHECKS.json) SHA `4799663278c0b1ecfda413241f48ea9a29903b9fdf247e874ed57171e8007ee0`：5个受影响目标通过、4份源码实际编译、两次实际公开fresh；[最终guardkksy1dne](CPU-FINAL-GUARD.json) / `b8b655dab855c5792375bf48474c3846a79a96addab3f34d62bb841469fa6cbd` exit0。Pod实核ec5/root0–11/nice19/隐藏CUDA-HIP-ROCR，与R203训练24–119互斥。尚待本次Git冻结/实际回读及两technical包，没有新normal、allocation、clock或GPU。

[数学GREEN](MATH-GREEN.json) guard1pi0tzpl exit0，3目标/8.297秒：原8×56完整配对，独立Python双循环标量与逐logit解析梯度、共同平移/共享bias容差2e-7、混合间隔/全过/恰边界/全同分、有限极端值及严格六字段；旧None/logistic/sigmoid值和梯度bitwise保持。真实2rank CPU Gloo/WORLD+DDP，本地正例[0,8]；四case分别实核全部global梯度g、local梯度2g和DDP参数梯度X^T*g，不加world因子。不当作新八rank/DCU或真实图像证据。

[候选GREEN](BINDING-GREEN.json) guard16llskka exit0，2目标/10.474秒，包含新R206/R207和受影响旧R204/R205。R206仅原R205 seed27条件复核；R207直接R107原Small/fe6/seed29/512/原AdamW/LR/WD/24k-800-p4/EMA，仅loss改固定`mean(relu(1+n-p)^2)/4`。来源、9:1/结构/日程/早停、保存job和非法叠加均检查。原R107无显式官方字段，R207明确采用现R120的selected>=.996才一次官方门，历史不改。SCI分别 `11e6f001985b1dd2c5f3c1e2a25a806c52666c6bd067c5c13ec86a151def5670` / `a146cfe2855eebd6aa77bb9d572159df059c220f1cffa51ba076a6ce883bc1b9`。

[实际公开fresh](PUBLIC-FRESH-SUMMARY.json) guardk6kcnp5n exit0：R206 whole91507a65、R207 whole50c6，均345全训/无buffer/5空原生AdamW，骨干1e-4、head/age1e-3，原betas(.9,.999)/eps1e-8及WD保持，0图像forward、无训练checkpoint/官方读取。R206/R207原件SHA d4b82e9a/f56c0e48。原model/config/engine/data/distributed/EMA/fresh/保存推理路径按精确不变SHA复用；去掉仅新增两行分支后classification_loss函数与7ead原版逐字符相同，validate_loss/WORLD gather/regularized及output helper也相同。不重跑旧套件/模型P8或GPU预演。默认12GiB、旧16/32例外和free16GiB不变。

固定margin1和1/4归一化，只让已过间隔的pair无目标梯度；与R160类内方差目标不同。1/4只匹配d=0处softplus导数0.5，不保持全部梯度或优化轨迹。错序尾部可放大离群pair；无临床机制或AUROC收益证明。完整内部AUROC仍唯一选点，不按难组代替总体或读官方调参。[设计与范围](../../../todo/auroc-999/R206-R207-PREPARATION.md)保留理论来源和限定。

原失败保留：[数学RED](MATH-RED.json) lwpn5gco/5818a920为新kind未登记；[候选RED](BINDING-RED.json) blh632qd/aff2df33为新profile未登记。[编译guardcn46zi13](CPU-COMPILE.json) / 82430223和[两fresh guard](CPU-FRESH.json) / d256f121均exit0。新测试初次传输早于目录完成失败，顺序补传后执行；R203第9次观测提前请求被原owner flock拒绝，未启动观察/无新CPU回执，该次unknown，此后串行。原件均在`exp-v3/technical-prep/squared-hinge-20261103-32e3`。

私有既有prepare helper只增两身份白名单及证据路径/5目标4编译消费，SHA `89857348cd3289b4fe2d192f2318ea194d180f513e046ee24e32db7845d8e496`；逆向移除四处增补后与原8a1727e9逐字符相同，原helper和storage消费段保持。[helper guard193ixbuw](CPU-TRANSPORT-HELPER.json) / 4b297cd5 exit0并compile。没有新训练/保存/监控链，不改现役R203或已准备R204/R205。
