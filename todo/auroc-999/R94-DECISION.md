# R94 单变量方案：backbone LR 减半

**2026-10-29执行精简：用户要求去除无效防御性检查。** 本页科学决定不变；删除重复hardcoded LR断言，不新增拒绝case/重跑已过9项或其他检查，R94取消独立零forward GPU预演前置并明确记未运行。真实训练的8rank初态/参数组/空Adam/实际资源校验接管。Git实际冻结、实时身份/独占资源、一次activation/官方一次保留。A不等本页修订即可继续开训。

状态：**2026-10-28科学决策已冻结：执行R94，唯一科学变量为backbone基准LR由`2e-4`降至`1e-4`。** 规划已逐行读取A导回的完整16条内部CSV/JSON，并核对唯一选点及原生训练完成记录。A已确认R93工程终态25/25、CPU guard exit0、原锁释放、8rank inactive/exit0、GPU device users空且8卡idle；原owner inactive、OS encodedexit0但为Z，保留该事实，不称已reaped。终态回执Pod SHA为`e1d0e20895f3a8047f2bd4b8477875f85b36f32ab013e6994d4b95d9864cf205`。A直接沿下述最小适配/CPU验收/Git同步/独立冻结与准入串行接续，无需用户例行确认；本决策未读取或使用R93官方指标。

精确新身份：profile/binding version=`single-tiny22k-bc48a87f-fixed-9010-lr1e4-v1`；candidate id=`r94-convnext-tiny-imagenet22k-to-1k-lr1e4-9010`；experiment_order=`4`；科学配置=`configs/R94-SCIENCE.json`。实际round root/Pod/Git/资源由A当次核验后显式生成，不把本页历史资源数值当默认。

## 科学问题与已有依据

检验在R93同一公开预训练、结构与训练预算下，将backbone基准LR从`2e-4`减到`1e-4`，能否改善按既定内部规则选出的单模型表现。唯一科学配置变量是`job.backbone_lr`；新轮名称、candidate/profile、Git/source/config digest、clock/token是随新实验生成的身份，不是额外训练干预。

直接对照是R93冻结实现`80e06f31c33bc6256f6b622e6ed162273d33f1ba`及其`configs/R93-SCIENCE.json`，不是来源和末端LN均不同的R92。完整内部原件为[16行CSV](../../records/R93/DEV-METRICS-INTERNAL.csv)及[内部JSON](../../records/R93/INTERNAL-DEV-SUMMARY.json)，Pod SHA分别为`a010f562784e742f7678d60ab090ff7ea965ce8f76614c7adf96ea9d105fc131`和`937e43133f645fb96d0b3ecb5a4b62797c2ff611ea6888960f4a10f7fc368549`；规划仅文本核对，不在本机重算。JSON明确`scope=internal_dev_only`、`official_metrics_included=false`，关联[原生freeze](../../records/R93/SINGLE-MODEL-FREEZE.json)的Pod SHA为`4459db88d353628864a9ea9a369c03ee7894f29c4afc0a1972927be260b3e6b0`。

8周期每个online/EMA均存在，成功更新依次800至6400。c1/c2两variant相同，符合warmcopy1600；唯一选择c3 EMA@2400：AUROC `0.9866344126546825`、AP `0.8167243430095755`、error_twice `22200`。c4 EMA的AP `0.8205987880588593`更高，但error_twice `22854`较差，按既定首要规则不替代c3。c4至c8两variant均未改善c3的首要选点指标；c8 online为AUROC `0.9663223322107043`/AP `0.7976385189308282`/error_twice `55938`，c8 EMA为AUROC `0.9689388566522695`/AP `0.8040199182720055`/error_twice `51592`。完整轨迹并非所有指标逐轮单调下降，不用这种过强描述支持方案。

上述完整内部轨迹支持把“减小预训练backbone每步参数变化”选作下一项受控检验，不能据此诊断过拟合、证明较小LR必然更好，或归因于单一训练机制。末端LN属于backbone组；AdamW的参数收缩也随LR缩放，因此不将本方案解释成只改变卷积层或只改变梯度项。R93原生训练completed/exit0且6400成功更新、最终工程审计通过，当前可作为本轮对照；若后续出现足以否定工程有效性的新证据，须如实记录可比性限制。

## 不变合同

| 项目 | R94要求，与R93相同 |
| --- | --- |
| 公开初始化 | `timm/convnext_tiny.fb_in22k_ft_in1k`，revision `bc48a87f119bae1b06418f5340bfa97ef75d4609`；转换权重SHA `1ec1dd0b7045237eafe1c8f1becd58456cc811326c97ec416ae8b2e5ede860ed`，provenance SHA `40c14c8bd780d4feb88ae96d7e9def1ef78499daf93ae4412eff78502780233f` |
| 模型与fresh | native avgpool/768特征、internal/final LN `1e-6`、dropout `.2`、stochastic depth `.1`、原生新binary linear、两维零age、无BN/attention；seed `20261027`，空Adam、online=EMA；不加载R93或任何历史checkpoint，不用N06 |
| 数据与预处理 | fixed fit46256/dev5140、原stage1/FREEZE/fit-only age、512/prepared512、ImageNet normalization、原增强/采样顺序；不改图像bytes或重新迁移数据 |
| 计算与批次 | FP32、8rank各8/accum1、G64/P8、reference physical32；不强制每rank正例相同 |
| LR之外的优化 | head/age基准LR `1e-3`，原参数分组与WD：矩阵`1e-4`、bias/LN向量0、age`1e-4`；BCE、clip1、无mixup |
| 预算与时钟 | 800成功更新/周期×8=6400；warmup1600，原cosine尾倍率`.01`；EMA FP32、warmcopy1600、decay`.999`，成功更新计数驱动；不早停/加周期/恢复旧clock |
| 选点与官方 | 8周期×online/EMA完整16条内部dev；error_twice升序、AP降序、成功update升序、EMA优先；仅唯一checkpoint做完整官方5849一次最终比较，官方不参与本决策或下轮调参 |

新fresh初始化应保留R93初态tensor控制值`eba53feeafdf20bc86855a40648bc453358928d148078c6ae78b65f059846cc7`。利用原生CPU初态回执和实际8rank初态门核对即可；若值不同，先定位并说明额外差异，不把未说明的初始化变化称作单变量对照。

## 已完成的最小CPU可行性

B只在实时核验的验证池、独立`exp-v3/technical-prep/next-round-lr-feasibility-20261028-51fe3daf-b`执行纯函数分析，未修改仓库或R93。`LR-FEASIBILITY.json` Pod SHA为`a121872b9595aa76b73f2f7aec3f6fc3c1f4f43cc8494859a549d728f1ee8e9f`；实际import的80e `retrain/engine.py` SHA为`08a6644b2d21bf6e3bbb0d8c80cb09a71b160c4bcae17eadc1625256aa7c628b`，前后不变。

cursor `0/1599/1600/1601/6399/6400`下，baseline与候选factor相同、候选backbone LR恰为一半、head/age LR不变；同cursor重复调用与initial_lr均不变。factor依次为`.000625/1/1/.9999998939346734/.01/.01`。cursor6399对应第6400次实际更新，6400仅终态，不产生第6401次更新。无模型、权重、forward、optimizer step、EMA循环或官方数据读取；GPU前后未初始化。guard回执SHA `e6229bdace750fd537468aa677388ae147e39440ac4a7759bd756603c9749a67`，exit0、5.526895秒、任务树退出、affinity违规空。这是函数可行性证据，不是训练收益或完整分布式loop证明，不为下一轮重复该分析。

## 执行A的最小实现边界

本决策现交既有执行A直接实现、验收及唯一Git/GPU执行；不另开Agent或平台。规划仅维护本合同。B保持CPU验证角色，不接管训练或Git。

1. 新建`configs/R94-SCIENCE.json`，保留R93配置作历史对照；仅更改backbone LR及上述精确实验身份/比较说明。版本化SCIENCE配置是LR值的唯一来源，不能将R93 profile与新LR混用。
2. 最小更新`runtime/bundle/retrain/experiment91.py`的显式candidate/profile及严格科学合同，`tools/activation/prepare.py`的新配置选择/绑定，以及`tools/validation/check_candidate.py`的预期分组LR。prepare从经实际SHA固定的R94 SCIENCE生成包内合同，相关site/package身份保留配置SHA；validate_job和CPU检查读取该绑定合同的LR，拒绝job/参数组与合同值不符。不得只把多个消费者中的`2e-4`散改成`1e-4`，也不得放宽成任意正数或缺配置时fallback。不修改`engine.py`、优化器算法、调度函数、采样或EMA算法；若现有site producer需携带该明确配置身份，仅做直接调用链的最小改动。
3. A同步本轮合同文字，删除`r92_initial_gate.py`中重复hardcoded `2e-4`断言；新配置及CPU实际参数组已有证据，不新增拒绝case、不重9项或其他已通过检查，不把删重复断言扩成新套件。
4. 使用本轮已取得的实际CPU配置/分组/初始化证据完成必要打包，不重复原生18或LR纯函数分析，不开发继承桥，不重来源/迁移。若某必要身份确因改动需更新，只处理该直接差异，不扩验证范围。
5. A汇集具体diff及真实CPU回执，按实际远端parent同步GitHub并回读commit/tree，再按新Git和实际文件SHA生成独立冻结部署。已有资源policy、native修复、缓存和迁移资产沿经核验的明确身份使用，不为LR变量重写资源设施。
6. R93工程有效且GPU安全退场已核。R94不跑独立零forward GPU预演，状态明确为未运行；新Git实际冻结后直接fresh一次activation/launch，由真实训练8rank初态/组/空Adam/实际资源校验接管，不把未预演伪记成功。CPU准备20%与训练80%按实时资源分池，旧R93不改，无新资源/定时任务/heartbeat。

R94启动后，20%池准备所有不依赖结果的下一轮代码适配、配置注入、干净打包及现有CPU验证，不再仅纯函数分析；最终科学值依完整内部结果决定，不用官方调参。此项不增加R94当前启动工作或平台。

## 比较与停止范围

R94完成后先按同一规则选出其唯一内部checkpoint，再与R93唯一内部选点比较error_twice/AUROC、AP和成功更新位置，并报告完整内部轨迹；不只挑某个训练loss或单一中途值。相同seed/预算提供受控单次对照，不能声称多seed稳定性或统计显著性。较小LR若未改善内部选点，照实报告，不据官方样本或分数追加调参。官方仅作各轮最终对外比较，内部与官方分报；raw AUROC未达`.999`不称总体完成。

## 完成后的内部对照（2026-10-29）

规划仅文本读取R93既有内部CSV与R94原生freeze中的全部16条`dev_entries`及`selected`，没有读取官方指标、运行脚本或重做模型验证。R94原生TRAIN_RESULT为completed、6400成功更新、8周期、组件exit0；A另已核安全退场并串行启动R95。以下是固定seed、固定9:1、同公开fresh控制初态与6400预算下，backbone LR2e-4→1e-4的一次内部对照。

| 唯一内部选点 | R93：2e-4 | R94：1e-4 |
| --- | --- | --- |
| 周期/variant/成功更新 | c3 / EMA / 2400 | c3 / EMA / 2400 |
| AUROC | 0.9866344126546825 | 0.9867415781748388 |
| AP | 0.8167243430095755 | 0.8200017145302917 |
| error_twice（越低越好） | 22200 | 22022 |

各周期内部AUROC如下，展示保留6位小数，原精度及AP/error_twice保留在来源原件；选点未使用表内舍入值。

| 周期 | R93 online | R93 EMA | R94 online | R94 EMA |
| --- | --- | --- | --- | --- |
| 1 | 0.974351 | 0.974351 | 0.973097 | 0.973097 |
| 2 | 0.981000 | 0.981000 | 0.985323 | 0.985323 |
| 3 | 0.981093 | 0.986634 | 0.982686 | 0.986742 |
| 4 | 0.973801 | 0.986241 | 0.977327 | 0.985427 |
| 5 | 0.971117 | 0.981588 | 0.976758 | 0.982403 |
| 6 | 0.966750 | 0.975566 | 0.970528 | 0.978899 |
| 7 | 0.968222 | 0.972679 | 0.973012 | 0.975809 |
| 8 | 0.966322 | 0.968939 | 0.971064 | 0.973381 |

本次唯一选点的主指标与AP均小幅改善，但最佳位置仍为c3 EMA@2400，没有后移。R94 c1两variant均低于R93；c2/c3均更高；c4 online更高而EMA更低；c5–c8两variant均更高，因此不能描述为所有周期全面提升。R94 c4–c8仍没有刷新自身c3的主指标；其c5 EMA AP为0.8289342242964929，高于所选c3 AP，但error_twice为29228，不能据AP另选checkpoint。

这是一个seed、一个固定内部划分上的受控观察，不能推断多seed稳定收益、统计显著性或诊断过拟合。末端LN仍属backbone，AdamW收缩也随LR变化，不能将差异单独归因于卷积梯度。已按用户要求运行的R95同时改变预算上限、cosine曲线和早停，不是另一次LR-only对照；不因本表热改R95，下一轮科学决定等其实际完整内部/早停终态，不用官方调参。

R94权威原件根为`/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r94-tiny22k-lr1e4-9010-20261029`；A已普通字节导回工作区`work/exp-v3-8card-review-20261028/r95-update/r94-terminal-export/`，本机未重算SHA：

- `SINGLE-MODEL-FREEZE.json` Pod SHA `9baf741474085c098170a7dcff60996f9246d60d73e123366fe408dd0ba5a2fb`，包含全部16条及唯一选择；选中tensor SHA `a3ddf7a133216ed86bef28e8b173b2b2f327b28ed1991fdd41ee892fbe9faa2b`。
- `evidence/TRAIN_RESULT.json` Pod SHA `6ab8146f6c8216f7dd1ed12a3e14ee2e1bb9221166dafdbbda2f58c42b5db614`；导出小清单`R94-METADATA-SHA.json` Pod SHA `1462efe9326a00422ec8410acc5da7e937e23cb4ff7a7c345b41da04de7429f7`。
- 安全交接小摘要`HANDOFF-R94-1.json` Pod SHA `025ef7e5d9511d9e239d06b788f69ac13409dee011d9a81b497fdcb6eb0413ac`。本补记仅作结果后统一归档，不增加运行前置或新实现提交。
