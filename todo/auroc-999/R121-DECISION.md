# R121：固定seed27复核有界对称成对排序目标

**最终状态：完整工程有效，但seed27未严格胜预冻R108，固定pairsigmoid分支停止主线升级。** 两seed一胜一负，不追加seed或近邻补考；整体目标未达，不等于项目暂停。末尾记录实际终态，下文保留原预登记。2026-10-31。

| 身份 | 唯一值 |
|---|---|
| order | 31 |
| 配置（A独占） | `configs/R121-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261027-scale085-pairsigmoid-v1` |
| candidate | `r121-convnext-small-imagenet22k-to-1k-24k-p4-seed20261027-scale085-pairsigmoid-9010` |
| round | `r121-small22k-seed20261027-scale085-pairsigmoid-24k-p4-9010-20261031` |

## 配对问题与唯一变化

[R120](R120-DECISION.md)完整selected EMA c6@4800 AUROC **.9894773092062407**、AP **.8313831149805229**，严格胜预冻R107 **.9884405731067525**，故执行已定的seed27复核。相对R120只改`job.seed=20261027`、自然factory fresh初态与运行身份；相对直接配对的R108 seed27，唯一科学干预仍为logistic换成相同有界sigmoid目标。训练seed影响初始化、样本/增强及模型随机流，不称只改变head或相同训练轨迹；split seed与固定9:1不变。

损失仍`L=2.*mean(sigmoid(z_negative-z_positive))`，六字段不变：

```json
{"version":"binary-pairwise-sigmoid-v1","kind":"binary_pairwise_sigmoid","temperature":1.0,"reduction":"mean","dtype":"float32","scope":"global_batch"}
```

2锁在版本内，零点斜率匹配不等于全程梯度尺度相同。两尾饱和、FP32零梯度、忽略真实难例、非凸优化及校准限制均保留；R120正差不证明fit标签噪声或尾部抑制机制。

其余完整复用R120/R107：已验公开Small22K→1K源、stride2/fulltrain、SD.1/drop.2、单图prepared512/scale[.85,1]/原增强与ImageNet归一化、G64/P8/64主图64视图/8P56N448对、默认WORLD可微gather及原DDP归一化、FP32/noTF32、同fit-only age、原LR/WD/clip1/AdamW/EMA、800/24000/patience4/绝对1600。R119输入扩展、R50/BCE、SSL或freeze分支不启用。

新正式factory的seed27初态应实际匹配既有R108-SCIENCE控制`91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67`；不得沿用seed29的50c、加载R120/R108训练checkpoint或另做control-only。原生产流程实核全state、新loss非tensor身份、参数组与空Adam，EMA从本轮fresh初始化。

## 最短接续与预算

- [ ] A只做seed配置/身份及受影响绑定，正常正式原生包匹配91507、参数组/空Adam；loss/math/Gloo/source/cache已有通过证据，不重跑或另设GPU预演，B无新文件任务。
- [ ] A统一必要tokenfree结果归档及本次唯一Git同步/精确部署，实时核当前身份、资源与前驱安全后新一次activation/fresh；旧包/clock/token不改，R120 selected保护实际依赖。本页不阻塞并行准备、不要求再次用户确认。

训练6h/整轮8h/grant9h、默认Small12GiB round限额、健康9C+I、启动free20GiB/运行floor16GiB、保存保护/300秒/atomic/prune/failfast均不变。单编码成本沿已验实现，不新模型/缓存/资源；实际包体≤500MB及指定设备100ms不由本页宣称通过。

## 固定判据

完整工程有效且原selector唯一selected AUROC **严格大于R108 .9872328538178018**，才算第二seed方向通过；AP独立分报，不救主目标。若未胜，停止本固定surrogate的主线升级，不挑最好seed、不再补seed或扫描温度/scale/混合。只有两个既定seed均胜，才可考虑作为方法候选/升级基线；仍不证明显著性、泛化稳定、消除dev自适应偏差或.999可达，不依据partial移动阈值。

完整训练及唯一checkpoint冻结后，selected内部AUROC≥.996才完整official5849一次；否则正常`not_run_by_policy/threshold_not_met`、evaluated=false、metrics=null。该测试门与第二seed胜负独立，中途partial不触发；不授权平台test CSV或提交。整体.999目标不变。复核收口后规划主动据完整内部证据决定下一问题，本文不预立其它候选。

## 完整终态与分支停止

2026-10-31 08:31:36.467785北京A实际终态：completed/early_stopping，6400success=attempt/24000、skip0、eval8、bad4；完整freeze16条唯一selected **EMA c4@3200，AUROC .9869089490433972、AP .8355471078657948、error_twice21744**，tensor SHA `273cb362219e0c2569648a410aa0c9d9359af9d07f4f3f16676c9ae8db65c71f`。一次文本核`work/r121-20261031-a/progress-0830/round/SINGLE-MODEL-FREEZE.json`的selected及`round/evidence/TRAIN_RESULT.json`；TRAIN completed/exit0/hard_deadline_met、signals[]。

AUROC严格低于预冻R108 **.9872328538178018**，故第二seed方向未通过。AP略高于R108的.8354670944168833不挽救主目标。R120 seed29胜出保留为单次内部best，不挑好seed救本次、不将两seed平均或挑选代替原判据；固定surrogate停止主线升级，不追加第三seed、温度/scale/混合或近邻surrogate补考。比较基线仍R107/R108，不推整类排序目标无效，也不诊断噪声或饱和为原因。

按A已实核、root转交：official=`not_run_by_policy/threshold_not_met`、evaluated=false、metrics=null，ROUND/done completed；owner254956Z不称reaped、原8rank退出、lockholders[]、deviceusers[]、scanDenied0、8idle、safe=true。未另读官方或查询Pod。progress原Pod SHA `5644d80536516a028b892658569b4be5a1302e0b83d0d89944a40b3770934820`，guard SHA `4ca63351bb025355d067557a2701c8942ba8f5c1c9df26ece86cfa1548627a`，仅原样记录、未重算。A负责必要归档，旧包/clock不改；R120 selected及真实依赖继续保护，尚未成为正式新基线。下一独立问题另作有限文本建议交root，不在此页注册新轮次。本文收口停写。
