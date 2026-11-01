# R135：首层固定 RGB+X+Y

R135/order47已完整结束。2026-11-01北京时间17:47:30.314352本轮唯一completion记录done/TRAIN/ROUND completed，11200success=attempt/0skip/eval14，early-stop bad4/stoppedtrue，train component exit0/signals[]/errornull。唯一EMA c10@8000 AUROC `.9884598388182413`、AP `.8245256334018152`、error_twice19168，tensor `cfe343da157e330aa6aba024317a09826674ba63d441fd658402bac0f8126a39`；strict>R107为true、strict>R120为false，固定stemXY整体关闭/no seed27，不扫radius/site/init/scale/LR/seed，也不推为所有坐标方法无效。单seed结果不声称显著、稳定或等成本收益，本轮11200步成本独立保留。freeze.status原值null；ROUND official为not_run_by_policy/threshold_not_met/evaluatedfalse/metricsnull，observer official_readfalse，没有追加官方前向。

同次actualsafe=true/terminal_idle：owner382423原start354807447匹配现Z非reaped，八rank382435–382442 absent；global同inode锁文件仍在、holders[]，deviceusers[]/scanDenied0，8expectedPCI全部seen/busy0。仅读取小件完成收口，未为观察或归档hash/load大checkpoint、重freeze/forward或查旧safe。保留本轮selected及全部实际依赖和成本，当前无活跃训练、无新候选授权。[完整结果](../../records/R135/README.md)与原启动实证分阶段保留。

原用户“开启训练”授权reference `4df7d17c-b82a-4542-a774-f0513685fe98`；16:14:58.385977启动观测running200success=attempt/0skip/eval0，首自然记录update25 finite/WORLD8/64P8/448pairs只作历史。A为唯一Git/Pod/GPU owner；冻结runGit `dbbe65531d511b91c89f63cd8a4cbba2e68e7303`，源码tree `218462c423c12a9f9ec03f3110bf2e1c26a7c213`，归档提交另列。fit诊断工程unknown不作本轮根因或前置门，不修复或重跑该诊断；旧五件及未知analysis保持隔离。

两个必要窄目标实际RED 2FAIL/0ERROR后GREEN 2/2（7.735秒），五个受改Python已编译；唯一正常formal 20/20（86.866秒）通过，精确Git blob/mode部署已核。13项新入口absence后，唯一activation exit0/CPU guard0与live launch exit0；SITE SHA `9fcebc3395f33ddffb554ba9918034735c1b120ed236802131d3ae3d4f30340b`。真实八rank初态completed，345全训/实际5组/初始Adam0、新whole与stem `[96,5,4,4]`及无state的坐标descriptor均匹配formal。启动时owner382423/start354807447及8rank382435–382442 argv/start/mask与8PCI匹配，control112–119/ranks24–111；首记录25的loss/global_loss均`.6514426469802856`。未独立观察update1或逐rank Adam step。[启动与CPU实证](../../records/R135/README.md)保留16:13 waiting-start-barrier及后续running原件来源，均不当当前进度。

配置为`configs/R135-SCIENCE.json`；profile `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-stem-rgbxy-v1`，candidate `r135-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-stem-rgbxy-9010`，round `r135-small22k-seed20261029-scale085-pairlog-stem-rgbxy-24k-p4-9010-20261101`，独立prep `technical-prep/r135-20261101-7c17`。

直接继承R107 seed20261029的全部科学值，唯一方法变化是原`network.features[0][0]`从RGB3输入扩为RGBXY5。原public fe6b native1000 strict加载与binary head/age初始化完成后，原Conv2d/weight对象和名称保留，RGB三片原样、XY两片零、bias不变，不额外消耗初始化RNG；增加3072个元素，全部沿原backbone LR/decay组训练。在既有BinaryModel.forward对最终增强和归一化后的512画布附加X列/Y行，固定`2*k/(512-1)-1`，训练/dev/官方完全同图，无radius、额外site或scale菜单，无新模块/class/hook/buffer。公开fresh CPU构造实测新whole为`75482be7c755c1792739c90dfc4947b8b45ada3cc2d78e855e01b7c7975dfad2`，同一模型取回RGB三片及其余原张量的345项SHA为原R107 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`。实际345全训/49,458,531元素/5组/Adam0/buffer0，XY全零；该测量无图像forward、GPU未初始化。不承诺RGB3与RGBXY5初始输出逐位相同，也不保证AUROC收益。

新whole已替换SCI准备占位，最终SCI SHA为`b0e57ced9cd6c6040fe60b0e6d5acd191de274414770d97a643d18a492407dda`。`PUBLIC-INITIAL.json` SHA `bedf55f742500d9a9416909596085c2e0f418096d14154a725ab16df5a7f4671`保存测量输入阶段身份，最终SCI已由本轮唯一正常formal实际消费；首次准备调用漏传`_seed`设备参数在模型构造前失败，原件保留，修正调用后一次构造成功，未为此改源码或重跑已过目标。

保留固定9:1/名单seed20261024、512单视图scale[.85,1]、原旋转光度、G64/P8、FP32/noTF32、pairlog六字段、原AdamW/LR/WD/一次clip/SD/dropout/age/EMA及1600warmup，最多24000成功步、每800双variant dev、连续4轮无提升早停、唯一selector。Tri3、MixStyle、logit-square、LoRA、layerwise、旧checkpoint等不继承；public fresh/step0/空Adam。

仅原位修改models/config/experiment91、check_candidate、test_candidate_binding五个既有文件，加本配置与决定页；不改训练/launch/保存/EMA/DDP/推理/监控链，不新增执行器或GPU预演。两个受影响既有测试目标先真实RED后GREEN，验证坐标与解析梯度、零扩展/名称/对象/RNG/None身份和默认factory保存strict重建。独立测得公开fresh初态whole后pin最终SCI；实际Git push/readback、精确部署、正常formal positive链仅一次，正式链已承接public strict/RGB和其余参数/初态空Adam/实际组/真实结构与默认factory strict重建。真实八rank初态与自然首记录update接管GPU证据，不补step1或另forward。repo外`work/r135-20261101-a/pod-context.env`及本轮SITE/真实START2仅保留历史定位，已收口后不重复轻检或completion，不另建监控/timer。

新软件CPU绑定SHA `74bd5221ade55e74b334e45e03a8d649ab023d33964ebf7328e19ba1d98dd9fe`实际确认同UID Ready Pod、有效120核，准备cap24，A12核0–11/root12–23保留；有效至1793529273.03317，旧217不改或复用。新GPU阶段仍须实时身份/资源准入和独立9h grant，训练整树floor80%与准备floor20%互斥；6h训练/8h整轮、Small12GiB/free20GiB/floor16GiB及保存保护不扩。上级Job lease不可读沿既有unknown语义，不作无限租期，不扩权限或资源。所有R130/R120/KEEP/R134R1 selected、原public/prepared512及历史失败成本保护。

完整工程有效selected必须同时strict>R107 `.9884405731067525`与R120 singlebest `.9894773092062407`才固定seed20261027对照R108 `.9872328538178018`；任一不过关闭这个stemXY整体，不扫radius/site/init/scale/LR/seed。工程失败独立记录，不判科学阴性；完整freeze>=`.996`才官方5849唯一一次，raw`.999`未证不称总体完成。
