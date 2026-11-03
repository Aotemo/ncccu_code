# 固定较宽 sigmoid 目标 CPU 准备

[CPU汇总](CPU-CHECKS.json) SHA `34706b04cdfda1197191e44dd3d35b7ad8dbb07a794b9c7dd0f276d91450111b`：5必要目标通过、4份受影响源码编译、两actual public fresh；[最终guard](CPU-FINAL-GUARD.json)m7dh84qz / `b6c78d314553926f6acfa79ec9823d18caa5760fb2baa36ed966060b5e4e6a92` exit0。全部在当次实核UID/Ready的既有Pod、53ab/root0–11、nice19/隐藏GPU，与R197训练24–119互斥；无新Agent或计算资源。

[三个数学目标](CHECKS-GREEN.json)覆盖完整8×56独立标量目标/逐logit解析梯度、平移/bias0、零点与原sigmoid同峰值导数、镜像、有限极端值、严格独立版本/温度2/字段，以及旧None/logistic/sigmoid数值和梯度bitwise保持。实际两rank CPU Gloo/WORLD可微gather+DDP，局部正例[0,8]，保留实际global/local logit梯度，与独立解析值及world倍数比较，参数梯度与独立`X^T*g`比较；原gather/regularized/helper函数源码逐字符相同。这里有每rank一次Toy Linear前向，没有生产图像/公开模型前向或八DCU证明。

[两个身份目标](BINDING-GREEN.json)逐项核R200原R199 WD=.05仅seed27、R201原R120仅六字段新loss；来源/9:1/结构/日程/早停/官方门保持，job_contract保存loss与WD，非法叠加拒绝，旧R198/R199兼容。源代码只在原classification_loss新增独立kind/版本/固定温度和两行返回、原experiment91登记/互斥。config/data/optimizer/engine/distributed_engine/model/EMA/保存/推理源码未改；新kind复用既有PAIRWISE_LOSS_KINDS消费链。原模型/P8/EMA/cold按精确未变SHA复用，不重跑全套。check_candidate与上轮WD源码相同；新完整producer尚待normal一次执行。

[实际公开fresh](PUBLIC-FRESH-SUMMARY.json)：R200 whole91507a65 / SCI644d7946，R201原whole50c6 / SCIbb449eb8，均345全训/5空原生AdamW/0图像forward。R200五组WD`[0,.05,0,.05,.0001]`，R201原`[0,.0001,0,.0001,.0001]`；原LR/default betas/eps与public fe6完整344张量来源保持，不叠gamma/8k/半LR。新loss通过原job身份保存，不增加模型参数或state字段。

保留全部RED：mjgdhxxf / a2649d0e的两个数学/版本目标因新kind尚未登记失败，随后pv0uh4bv / e4e1823b三个GREEN；候选e01yr_81 / 93a14ae4为新profile未登记、旧身份PASS，随后3_d0ufdw / 8cc0c4f3两目标GREEN。原guard/OUTPUT SHA由最终汇总逐件核验，整树退出、无affinity违规；注册修改后数学源码SHA未变，未重复Gloo。私有原件`exp-v3/technical-prep/wide-sigmoid-20261103-32e3`。

固定`mean(4*sigmoid((n-p)/2))`仅扩大同分数下的梯度范围，峰值保持不代表所有梯度/轨迹相同，极端尾部仍饱和，学习的logit尺度也可适应；训练loss单位不同，不跨目标比较。没有临床、固定难组或AUROC收益证明。现仅CPU证据，Git实际回读及技术包随后登记，无normal/新clock/GPU；R197/436及先行R198/R199/82add93冻结不改。24k/800/p4/EMA1600、默认12GiB/free16GiB与官方隔离保持，raw .999未达。
