# R199 之后固定较宽 sigmoid 梯度范围计划

11-03 04:17[R201真实八rank及首自然25](../../records/R201/README.md)已通过，whole50c6/345全训/5空Adam/有限loss/WORLD8-G64P8-448，当前50成功0跳过。R199完整内部selected`.9858601718742286`低于两基线，R200不eligible；同次actualsafe后，新window2 normal18/18、窗口/一次activation/direct launch串行接续，source8635/SCIbb449eb8/actual SITE4f412fe0冻结。原stage窗口不足拒绝与grant原件保留，不重置；下文仅技术/待normal为历史。完整收益仍待R201终态，raw .999未达。

01:48已按实际GitHub回读commit `8635b53728f3b52829e44d9faa24434ec0c2197f` / tree `844d58670225afab2aca011223a0ba46b99909cc`生成[两actual technical包](../../records/preparation/wide-sigmoid-20261103/TECHNICAL-PACKAGES.json)：R200 SITE07639d30与R201 SITEf2ce4173均guard0，归档v88xsue3 exit0。无normal/新allocation/clock/GPU；现役R197/436及先行82add93不改。下文“尚未Git冻结/技术包”为CPU完成时点。

11-03已完成[5必要目标/4源码编译及两actual public fresh](../../records/preparation/wide-sigmoid-20261103/README.md)，最终guardm7dh84qz / b6c78d31 exit0。R200 whole9150/SCI644d7946，R201原whole50c6/SCIbb449eb8，均345全训/5空Adam/0图像forward；尚未Git冻结/技术包/normal/clock/GPU。现役R197/436与先行R198/R199/82add93冻结不改，R195已完整归档。32e3唯一Git/GPU，不恢复/派生Agent；用户已授权最小修改，无例行审批，既有brainstorming/writing-plans/TDD/executing-plans流程仅20%准备池实施。

备选是再换架构、加模型视野、或只改已有448对上的平滑目标。前两项已有多个独立阴性且涉及更多路径；本次只选择第三项。原封存难20正例在多轮承担大部分错序，但没有统一图像外观或临床机制证据。原hard-positive logistic固定top2混合R156未改善总体或固定组，不能据此声称所有难例方法已失败，也不再按dev名单赋权。R120原sigmoid是本组直接比较基线。

R200/order113仅在完整有效R199唯一内部selected严格胜R107 `.9884405731067525`与R120 `.9894773092062407`且actualsafe后，以同WD=.05、seed20261027独立公开fresh复核；否则只技术准备。R201/order114独立直接R120原Small/public fe6/seed20261029/512/FP32/ImageNet/G64P8/default AdamW、原LR/WD/24k余弦/EMA，只换下述固定pair目标，不叠WD=.05、gamma、8k、半LR、hard-positive或其他policy。现有R*-SCIENCE文本中的pair temperature均1.0；这不是所有历史任务从未试过的证明。

对完整8×56差值`d=n-p`，原目标为`mean(2*sigmoid(d))`，新目标固定`mean(4*sigmoid(d/2))`。相同448对、相同采样/forward/模型参数/WORLD归一化；新梯度每对为`2*s*(1-s)`，`s=sigmoid(d/2)`，在d=0与原目标同为.5，在非零有限d更宽。共同平移不变、共享bias梯度0。固定乘4保持峰值导数，不能说整个优化轨迹或梯度范数相同；新loss数值尺度不同，不能跨目标直接比训练loss。它仍是有界非凸目标，极端错误仍会饱和，不能保证挽救固定难组；学习的logit尺度也可适应，非新的模型表达能力或临床机制证明。不扫描温度/系数，不用官方或逐例dev权重。

拟最小范围：`classification_loss.py`新增精确六字段独立版本/kind（temperature固定2），进入现PAIRWISE_LOSS_KINDS和版本表，仅新分支消费`2*T*sigmoid(d/T)`；旧分支/旧温度1合同原样。`experiment91.py`登记R200条件WD seed27及R201新kind的来源/固定身份/保存job/互斥，禁止其他profile携带新loss。两SCI与必要数学/候选测试、todo/records。原config/engine/DDP/gather/model/data/EMA/保存/推理不改，复用现六字段loss的持久化身份；无新平台/依赖/forward、默认12GiB/free16GiB保持。

- [x] 独立标量解析目标/逐logit梯度覆盖真实8×56、平移/bias0、零点同峰值导数、极端有限性；旧None/logistic/sigmoid原数值和梯度bitwise保持，严格新版本/温度/字段拒绝，实际RED→GREEN。
- [x] 实际两rank CPU Gloo/WORLD可微gather+DDP、局部正例[0,8]，保留实际global/local logit梯度与独立解析值比较，参数梯度核`X^T*g`、无多余world因子。只有Toy Linear前向，不冒充八DCU或生产模型前向。
- [x] 新旧候选身份/仅单项job差异/source/9:1/日程/早停/官方门/保存job及非法叠加实际RED→最小实现→GREEN，4份源码编译。两公开fresh核345全训/5空组/SCI，模型/P8/EMA/cold按未变SHA复用；gather/helper函数源码逐字符保持。
- [x] 精简真实回执、Git8635b53同步实际回读及两technical包已完成；正常CPU/精确部署仍安排在前轮可行收尾前，actualsafe后唯一owner沿原入口串行接续，不重复GPU预演、不改旧clock/token/grant。当前尚无normal/clock/GPU。

标签1=青光眼且仅按眼底照片，0仅任务阴性；不改标签/拆分/难组名单。最多24000成功更新、每800内部评估、p4早停/EMA1600/唯一内部AUROC选点保持，完整内部selected≥.996才官方一次；raw .999未达。
