# R124R1：同一内容汇聚科学合同的独立工程续接

状态：R124R1已完整工程有效，A于2026-10-31 14:33:14.994031北京时间实核整轮completed/safe；内部AUROC未过两门，固定内容汇聚分支关闭/no seed27。以下保留授权消息 `66085325-cd0c-47bd-80f3-9d5f32a7f66b` 所定工程合同；本次不是原R124重激活，原启动前失败与本次科学阴性分开记录，最终结果见末段。

## 未来身份与科学零变更

- 新prep：`technical-prep/r124r1-20261031-e62a`。
- 新round：`r124r1-small22k-seed20261029-scale085-pairlog-contentpool-24k-p4-9010-20261031`，相应新root、新future binding、新authorization reference。
- 保持 order `34`、`configs/R124-SCIENCE.json`。
- 保持 profile：`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-contentpool-v1`。
- 保持 candidate：`r124-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-contentpool-9010`。

完整resolved science job与R124一致，仅site/clock round/output等工程身份变化。如果现有绑定要求alias，由A作最小显式工程alias并证科学delta=0，不占R125或新科学order。保持Small公开fe6b/seed29/fulltrain/stride2、单图prepared512/scale[.85,1]/原光度、enabled age、原固定ATTENTION_POOL_RULE和readout、q初态/头LRWD、pairlog六字段/G64P8/448对、FP32noTF32/SD.1/drop.2、原LRWDclip1AdamWEMA/9:1/80024000p4/绝对1600。不得加载旧训练checkpoint或改读出、采样、损失、初始化、selector。

原attempt已实测formalCPU20/20、whole `1ea8442bb4b9c41009eded6863d7f4c2906144e0f15e6726fe34b594545d643c`、原345共有张量exact/q0/age真/空Adam。新包必须正常实际核其必要绑定与证据，明确哪些模型科学PASS为原件复用、哪些确在新包执行；不能把旧20/20写成新实跑。新8rank/first batch age tuple4/首成功仍待新attempt真实证据。

## 唯一工程修复与验收

先完成所有长准备、必要Git同步、精确部署、脚本参数及传输，再由工具取得真实新的live Pod/owner观测。在同一次Pod受控链中连续完成该snapshot的 **site_preflight→activate_audit**，直接传新receipt SHA，中间不SSH导回、不等待例行回复。保留真实snapshot的observed_at与Ready，不能在链内伪造为now；原两处120秒freshness、不成功也不可重用的activation attempt语义均不改。

- [ ] A独占薄orchestration、fixture及其余生产/Git/GPU；B无新文件任务。仅一项无GPU mock窄验验证一次preflight→一次audit使用同context、原observed_at及新receipt SHA，不扩拒绝矩阵。
- [ ] 新包必要绑定实际核实，复用已过模型科学证据但标明出处；不重toy/source/loss/Gloo/cache大套件，无独立GPU预演或第二控制模型。
- [ ] 当前动态live/安全准入及必要精确Git部署通过后，新attempt只一次activation和唯一fresh；正式clock、部署和首success如实回报。再失败不自动第三次尝试、不回原R124重试。

原prep/root/audit、marker与所有失败回执不动；失败后确认不存在的旧FORMAL-CLOCK/FORMAL_BINDING_RECEIPT/round_root/dispatch不补造。不得删除marker、修改时间戳、重写旧receipt、重置旧clock/token、重放activation或绕过freshness。新attempt及已有失败工程成本分别记账并累计，不伪称仍为首attempt。

CPU17或后续必要future allocation每个入口均动态核UID/Ready/Python/effectiveCPU/本人entry及TTL，准备20%与训练80%互斥；不沿过期预算、不新建或延长云资源。默认round12GiB、启动free20GiB/运行floor16GiB、300秒保存及保护/atomic/prune/failfast、训练21600秒/整轮28800秒/grant32400秒均保持；不放宽freshness、时限或资源以救启动。

## 科学与评估边界

R124原attempt没有内部selected或科学指标，不能当阴性，也不能标官方`threshold_not_met`。R124R1完整工程有效结果才按原合同判断：selected内部AUROC **strict > R107 `.9884405731067525`** 为科学正向，且 **strict > 单次best R120 `.9894773092062407`** 才由协调方另定固定seed27对R108 `.9872328538178018`。只胜R107未破R120则有限正向后阶段结束；未胜关闭该固定读出合同，不扫q温度/维数/公式/pool架构/LR/seed。

AP/partial/official不救主判；完整训练并冻结唯一checkpoint后，内部`>=.996`才官方5849一次，未达才记`not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`。无平台test，`.999`总体目标持续。此次仅工程续接，不展开下一候选；本页freeze后停写，A无需等待再次批准。

## 完整内部结果与最终收口（2026-10-31）

本机原件已到，规划仅文本核 `work/r124r1-20261031-a/progress-1430/round/SINGLE-MODEL-FREEZE.json` 的selected及 `round/evidence/TRAIN_RESULT.json` 必要字段：TRAIN completed/exit0、7200成功更新、9周期、bad4/stopped=true、hard_deadline_met=true、signals与live_owned_ranks为空。唯一selected为 **EMA c5@4000，AUROC `.9861082179096462`、AP `.8290710437486213`、error_twice23074**，tensor `fc83bc5df278a4fc3f8449696c24aaf2d02622f6e47234269cff40561c47ff74`。完整7200success=attempt/max24000/skip0/eval9/early_stopping及下述资源、官方状态引用协调方转交A实证，不另查Pod/预测/官方或重算。

AUROC低于直接R107 `.9884405731067525`，也低于R120单次best `.9894773092062407`，两门均false。关闭**此固定content-attention-pool × 当前Small/pairlog合同**，no seed27，不继续q温度/维数/公式/架构/LR/seed补考。AP不救AUROC；不泛化所有空间信息无效，原 `attention_pool=None` 的读出继续作为比较路径。R120仍仅singlebest，不升级方法基线。

终态q参与更新已有A直接证据：online768非零、L2 `3.7132256`、Adam step7200且exp_avg/exp_avg_sq存在；EMA768非零、L2 `3.5872138`。这些只证明q实际更新，不挽救主判、不追加forward或把原初始化合同里的待验文字当当前未知。原准备阶段 `13>12` 失败记录与随后合规guard0分别保留，失败审计不改、不改写成全程无失败；本页未另核该旧失败细节。

完整selected不足`.996`，official为 `not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，整轮正常completed，TRAIN/ROUND/done完整。原owner start匹配、现Z不称reaped，8rank退出、锁/设备使用者为空、denied0/8busy0、actualsafe=true。Progress Pod SHA `756ce87b07f140676f7e4960f91613270afd9f1ffbfc5e9c085515389d655a42`，CPU SHA `7e706f678e266962339b9d9f5df7f8c3a47694b747f20c5a3248e42d59b0b99e`，均引转交实证，未重算。原R124没有selected、官方因启动前失败未执行的事实不变，不能倒填为本轮threshold_not_met。

实际本轮training `3026.2428818391636s`、evaluation `162.35906827962026s`、checkpoint `161.58141998760402s`/30次、process wall `3428.396689357236s`；TRAIN component wall `3505.6715223789215s`单列，不混为吞吐benchmark。原包/clock/token和6h8h9h/12GiB合同不改。本页收口freeze交A必要归档；总体`.999`未达，下一问题仅由规划作有界文本建议交协调方，不自动新轮。
