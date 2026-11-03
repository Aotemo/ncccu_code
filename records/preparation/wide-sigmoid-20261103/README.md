# 固定较宽 sigmoid 目标 CPU 准备

03:43原R201未启动stage的完整clock窗口实际被拒绝：[guardg8lc2hm_](R201-ORIGINAL-WINDOW-CPU.json) / `d1700c08fe4cbdc38cff8b91adca706963a255346efa7ede45ab99f3e46f77d4` exit1，原helper明确剩余准入时间不足覆盖完整clock+60秒。这是资源时窗条件，不是代码/模型失败或科学阴性；原normal18/18、grant及回执不重写，不创建clock/activation/真实round。window2仅技术包，后续可行收尾前按既有入口为新未启动stage生成绑定；原数学/源码/初始化证据保持。

03:34 R199 c7微升后早停归零，接续后移；原R201 normal18/18、其GPU grant及全部回执保持，不重置。另备未启动window2的[纯技术包](R201-WINDOW2-TECHNICAL.json)，[guardnr5hysvh](R201-WINDOW2-CPU.json) / `5ccabc9d93c07239248e06a395b3d03d96e1c95f010320d809db25670a78afa4` exit0，原source8635/SCIbb449eb8/技术SITEf2ce4173与原包相同，5目标4编译证据按SHA复用。window2当前无formal、GPU allocation、clock或activation；后续按真实headroom为未启动阶段建立可覆盖完整clock的新绑定，不热改旧grant/当前R199或科学值。

11-03 03:28后备R201已在R199/bad3仍训练时完成[正常CPU18/18与精确部署](NORMAL-R201.json)，[guardk3w5kpql](NORMAL-R201-CPU.json) / `7bdb18d0566a78c5c3fdd6ad4f7203f6bc2c5e82f357d8ac0a3ef1275e136ab5` exit0。原5目标/4编译按SHA复用，新完整producer实际通过；345全训/5空AdamW/whole50c6、actual SITE `d50c8471e7a1f64bb509d4d50d8d30b1f5de8e24eebae6fd39fbc60637f40788`，manifest2a440a9e、CPU-prep2a077d5c、deployment74f6e00b。source8635b53/tree844d586、SCIbb449eb8、默认12GiB/free16GiB不变，无新clock/activation/GPU。使用新ec5准备0–11，与R199训练24–119互斥；完整R199双门及actualsafe决定唯一分支，下文待normal状态为历史。

01:48两[actual technical包](TECHNICAL-PACKAGES.json)已按GitHub实际回读commit `8635b53728f3b52829e44d9faa24434ec0c2197f` / tree `844d58670225afab2aca011223a0ba46b99909cc`生成。R200 SITE07639d30 / [guardaydxy8l5](CPU-TECHNICAL-R200.json)、R201 SITEf2ce4173 / [guard14i5jk_w](CPU-TECHNICAL-R201.json)均0；归档v88xsue3 / `bee0963906fc762a62cf7be4e325529a440858a18e614e8ddc06cb522bb1e06e` exit0，实核无formal-cpu/gpu-inputs/真实round目录，无新clock/GPU。当前R197/436与先行R198/R199/82add93保持冻结，本组正常CPU/精确部署仍在前轮可行收尾前执行一次。

[CPU汇总](CPU-CHECKS.json) SHA `34706b04cdfda1197191e44dd3d35b7ad8dbb07a794b9c7dd0f276d91450111b`：5必要目标通过、4份受影响源码编译、两actual public fresh；[最终guard](CPU-FINAL-GUARD.json)m7dh84qz / `b6c78d314553926f6acfa79ec9823d18caa5760fb2baa36ed966060b5e4e6a92` exit0。全部在当次实核UID/Ready的既有Pod、53ab/root0–11、nice19/隐藏GPU，与R197训练24–119互斥；无新Agent或计算资源。

[三个数学目标](CHECKS-GREEN.json)覆盖完整8×56独立标量目标/逐logit解析梯度、平移/bias0、零点与原sigmoid同峰值导数、镜像、有限极端值、严格独立版本/温度2/字段，以及旧None/logistic/sigmoid数值和梯度bitwise保持。实际两rank CPU Gloo/WORLD可微gather+DDP，局部正例[0,8]，保留实际global/local logit梯度，与独立解析值及world倍数比较，参数梯度与独立`X^T*g`比较；原gather/regularized/helper函数源码逐字符相同。这里有每rank一次Toy Linear前向，没有生产图像/公开模型前向或八DCU证明。

[两个身份目标](BINDING-GREEN.json)逐项核R200原R199 WD=.05仅seed27、R201原R120仅六字段新loss；来源/9:1/结构/日程/早停/官方门保持，job_contract保存loss与WD，非法叠加拒绝，旧R198/R199兼容。源代码只在原classification_loss新增独立kind/版本/固定温度和两行返回、原experiment91登记/互斥。config/data/optimizer/engine/distributed_engine/model/EMA/保存/推理源码未改；新kind复用既有PAIRWISE_LOSS_KINDS消费链。原模型/P8/EMA/cold按精确未变SHA复用，不重跑全套。check_candidate与上轮WD源码相同；新完整producer尚待normal一次执行。

[实际公开fresh](PUBLIC-FRESH-SUMMARY.json)：R200 whole91507a65 / SCI644d7946，R201原whole50c6 / SCIbb449eb8，均345全训/5空原生AdamW/0图像forward。R200五组WD`[0,.05,0,.05,.0001]`，R201原`[0,.0001,0,.0001,.0001]`；原LR/default betas/eps与public fe6完整344张量来源保持，不叠gamma/8k/半LR。新loss通过原job身份保存，不增加模型参数或state字段。

保留全部RED：mjgdhxxf / a2649d0e的两个数学/版本目标因新kind尚未登记失败，随后pv0uh4bv / e4e1823b三个GREEN；候选e01yr_81 / 93a14ae4为新profile未登记、旧身份PASS，随后3_d0ufdw / 8cc0c4f3两目标GREEN。原guard/OUTPUT SHA由最终汇总逐件核验，整树退出、无affinity违规；注册修改后数学源码SHA未变，未重复Gloo。私有原件`exp-v3/technical-prep/wide-sigmoid-20261103-32e3`。

固定`mean(4*sigmoid((n-p)/2))`仅扩大同分数下的梯度范围，峰值保持不代表所有梯度/轨迹相同，极端尾部仍饱和，学习的logit尺度也可适应；训练loss单位不同，不跨目标比较。没有临床、固定难组或AUROC收益证明。现仅CPU证据，Git实际回读及技术包随后登记，无normal/新clock/GPU；R197/436及先行R198/R199/82add93冻结不改。24k/800/p4/EMA1600、默认12GiB/free16GiB与官方隔离保持，raw .999未达。
