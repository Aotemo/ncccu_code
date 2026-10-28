# R99：对照R95，仅提高末端dropout

2026-10-29科学决定已直接交A实施。直接对照为R95：从`configs/R95-SCIENCE.json`创建`configs/R99-SCIENCE.json`，唯一科学delta为`job.dropout: 0.2 → 0.4`。backbone LR保持1e-4，head LR保持1e-3，age独立保持1e-3；不在R98的head2e-3上叠加dropout变化。相对紧邻R98包含head LR返回基线，不能把R98→R99称为单变量比较。

精确profile=`single-tiny22k-bc48a87f-fixed-9010-24k-earlystop-drop04-v1`；candidate=`r99-convnext-tiny-imagenet22k-to-1k-24k-p4-drop04-9010`；order9。此页收口时R99尚未运行，字段已交A并行准备，不设文档批准门。

## R98完整内部事实

规划已文本读原生freeze全部14条dev_entries、selected及TRAIN_RESULT；未读取官方指标选参、执行本机解析/哈希或增加模型评估。R98实际完成5600成功更新、7周期、skip0，c3达到内部峰值，c4–c7连续4周期无新高，训练组件exit0、正常early_stopping。R98原实际参数组已确认head initial_lr为2e-3、age为1e-3，外来age误标不构成代码修复任务。

| 唯一内部选择 | R95：head1e-3 | R98：head2e-3 |
| --- | --- | --- |
| 实际成功更新／完整周期 | 5600／7 | 5600／7 |
| 选择位置 | c3 EMA@2400 | c3 EMA@2400 |
| AUROC | 0.9866645153288838 | 0.9853363853431284 |
| AP | 0.8193841805300731 | 0.8194004616388242 |
| error_twice（越低越好） | 22150 | 24356 |

各周期AUROC显示至6位小数，实际选点及早停仍使用原精度。

| 周期 | R95 online | R95 EMA | R98 online | R98 EMA |
| --- | --- | --- | --- | --- |
| 1 | 0.973097 | 0.973097 | 0.973059 | 0.973059 |
| 2 | 0.985323 | 0.985323 | 0.980893 | 0.980893 |
| 3 | 0.982684 | 0.986665 | 0.984362 | 0.985336 |
| 4 | 0.977746 | 0.985536 | 0.975966 | 0.985267 |
| 5 | 0.977276 | 0.983334 | 0.970156 | 0.982569 |
| 6 | 0.970660 | 0.981346 | 0.965625 | 0.979292 |
| 7 | 0.968865 | 0.979318 | 0.964596 | 0.977376 |

R98的c3 online高于R95同期，但所选EMA AUROC/error_twice较差，所选AP仅略高；不能说全部指标均退步。c4 EMA AUROC 0.9852665471389817未超过c3，尽管AP 0.8252946607526948较高，仍按既定主次顺序选择c3。

在同24k/p4政策下，R96降低backbone LR、R97降低head LR、R98提高head LR的所选内部AUROC分别为0.983341180097075、0.9854519796120608、0.9853363853431284，均未超过R95的0.9866645153288838。此结果支持保留R95作为本次直接参考并暂停局部LR点的机械扩展；不证明R95是全局最优，也不证明这些差异具有统计显著性。各轮只有同一seed与固定划分，且实际停止步数不全相同。

## 单一假设与可实现范围

R99检验较强的末端特征随机失活是否改善本次内部选择结果。dropout从0.2提高到0.4是明确、有限的正则化扰动；依据是已有LR局部探索未提高内部峰值，改检一个已有实现支持的独立科学轴。早期峰值及后段不刷新不能单独证明过拟合，当前汇总也未定位head的具体病因；更高dropout可能降低有效训练信号并使结果变差，不承诺收益或0.999。

文本核现有`models.create_model`：ConvNeXt在原final LN/Flatten之后、末端native Linear之前插入`nn.Dropout(job.get('dropout', .2))`，已支持配置值。只改该处训练随机失活概率，SD仍0.1，age在图像logit之外独立相加；不增加预测时随机采样或模型集成。Dropout无参数，预计fresh tensor仍为既有eba53fee控制值，实际以现有正式包及训练初态证据为准；不能只凭tensor SHA相同就忽略非张量配置变化。

A仅将`experiment91.validate_job`、`describe_model`、`validate_normalization`的既有固定dropout期望改为读取同一pinned SCIENCE `job.dropout`，既有fixture按配置消费。保留实际模型模块类型及p值的对应检查，不在消费者散写0.4，不改其他模型默认、优化器、来源或结构算法，不新增接口、诊断或测试套件。调整dropout会改变训练轨迹，不能声称backbone参数更新相同；唯一变化是科学配置中的概率。

除dropout与新身份外完整沿用R95：固定fit46256/dev5140、同公开Tiny22K→1K revision/权重、seed20261027、fresh native binary+零age、空Adam、online=EMA；512/FP32、G64/P8、8rank各8/accum1、同采样/增强/WD/clip/BCE、内部/末端LN1e-6、SD0.1。最大24000成功更新、每800一周期，完整周期max(online,EMA)内部dev AUROC严格刷新才清零，首轮baseline，patience4/min_delta0/min_cycles1；warmup与EMA warmcopy绝对1600、首blend1601、cosine horizon24000/尾倍率0.01、EMA decay0.999。不得从R95/R98或其他训练checkpoint初始化。

结束后在实际完整周期的全部online/EMA中按error_twice/AP/update/EMA规则选唯一checkpoint，与R95直接比较，同时分报AP及实际停止步数。官方5849仅一次最终独立比较，不参与早停、选点或后续调参；raw AUROC未达0.999不称总体完成。

## 执行及来源

沿现有配置注入、改动必需窄验证、一次Git同步、实际tree/SHA冻结部署、正式包及一次fresh启动链推进；不新增独立GPU预演，不重复来源、迁移、native或旧套件。04:53观测时R98训练/freeze已exit0，但原一次official进程仍活跃、整轮终态未出；A须核其自然安全退场后才能启动R99，不重发官方或改旧运行。准备20%与训练80%按现有集中policy、实时资源和有效TTL互斥并行，旧绑定不热改。

R98冻结Git=`481e5903bd569de3784f5d16b45121551cae9a9c`。原件根`/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r98-tiny22k-head2e3-24k-p4-9010-20261029`，已普通导回工作区`work/exp-v3-8card-review-20261028/r98-update/r98-terminal-export/`。以下为既有Pod SHA，本机未重哈希：

- `SINGLE-MODEL-FREEZE.json`：`a545f16f4b9444c2a7598317635e1944cdc27bea60466381d064b7a399c96841`；`evidence/TRAIN_RESULT.json`：`2b738cfdb939ea630e156091d48838e99b6123b855f92bf71df9d2fcf9fa4c32`。
- R98唯一所选tensor：`2e472e71890c7281ebac406caa72548c55e2a2714666ef5c7f902f08947e3c7f`，仅结果身份，不作R99初始化。
- R95 freeze：`13aa989198ca9b4e64ed5e021f6da9b8733eda748878bd5750d19e2ebccc217b`；TRAIN_RESULT：`dd253b3aa0fdd33bb6a539f01ba596d7f5520dbea1547d543787fdae69f306c1`。基线为`configs/R95-SCIENCE.json`，既有比较见[R97决定](R97-DECISION.md)、[R98决定](R98-DECISION.md)。

规划本次仅冻结本文件，交A与必要实现/结果统一集成，不为过程状态追加提交或等待。
