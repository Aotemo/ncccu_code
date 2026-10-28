# 历史实验结论

历史8:2与当前9:1不同；官方与内部dev分列。缺失或技术失败不视为达标。融合仅保留历史结论，不恢复训练。

下文是实验当时的结果摘要；其中关于原件、路径和保留状态的历史描述，不表示这些产物现在仍存在。当前保留对象以[三模型清单](KEEP-MODELS.json)为准，实际删除与中断状态见[清理记录](CLEANUP-STATUS.md)。R91/R92 的新9:1最终结果分别见[结果目录](../README.md)。

## R2

状态：completed；模型：见指标/原实验定义。

- /R2T/best/metrics：n=10277，AUROC=0.9823576518870772，AP=0.802250129970274。
- /R2T/checkpoints/1/best_metrics：n=10277，AUROC=0.9823576518870774，AP=0.802250129970274。
- /R2T/checkpoints/2/best_metrics：n=10277，AUROC=0.9823576518870774，AP=0.802250129970274。

## R4F

状态：completed；模型：见指标/原实验定义。

- /R4F/checkpoints/1/best_metrics：n=10277，AUROC=0.9863609920522889，AP=0.8395417792519538。
- /R4F/checkpoints/2/best_metrics：n=10277，AUROC=0.9863609920522889，AP=0.8395417792519538。
- /R4F/best/metrics：n=10277，AUROC=0.9863609920522889，AP=0.8395417792519542。

## R4

状态：partial；模型：见指标/原实验定义。

- 没有可归入本表的完整官方/内部选点指标；保留真实状态，不补造数值。

## RESULT.json

状态：recorded；模型：见指标/原实验定义。

- 没有可归入本表的完整官方/内部选点指标；保留真实状态，不补造数值。

## OFFICIAL-RESULT.json

状态：completed；模型：见指标/原实验定义。

- /metrics：n=5849，AUROC=0.9746659041823744，AP=0.7372105728953852。

## R1

状态：completed；模型：见指标/原实验定义。

- /runs/r1a/best/metrics：n=10277，AUROC=0.9702811764030099，AP=0.7315312009106578。
- /runs/r1a/checkpoints/0/best_metrics：n=10277，AUROC=0.9702811764030099，AP=0.7315312009106583。
- /runs/r1a/checkpoints/1/best_metrics：n=10277，AUROC=0.9702811764030099，AP=0.7315312009106583。
- /runs/r1b/best/metrics：n=10277，AUROC=0.9654044496427013，AP=0.7354199017351621。
- /runs/r1b/checkpoints/0/best_metrics：n=10277，AUROC=0.9654044496427012，AP=0.7354199017351627。
- /runs/r1b/checkpoints/1/best_metrics：n=10277，AUROC=0.9654044496427012，AP=0.7354199017351627。
- /probes/results/0/best_metrics：n=10277，AUROC=0.9702811764030099，AP=0.7315312009106583。
- /probes/results/1/best_metrics：n=10277，AUROC=0.945211452134687，AP=0.6170674544109032。

# R1 — 同父低学习率384/512配对实验结果

两支正式训练均完成，每支5,688次成功更新、8次完整开发评估；各有独立stage0预测。本轮尚未达到AUROC 0.998。

| 分支 | stage0 AUROC / AP | 最佳cycle | 最佳AUROC / AP | 最终AUROC / AP | 成功 / 尝试 / skip |
| --- | --- | --- | --- | --- | --- |
| r1a-low-lr-384 | 0.970281176 / 0.731531201 | 0 | 0.970281176 / 0.731531201 | 0.937761313 / 0.731087543 | 5688 / 5692 / 4 |
| r1b-low-lr-512 | 0.945211452 / 0.617067454 | 1 | 0.965404450 / 0.735419902 | 0.955311235 / 0.743976953 | 5688 / 5691 / 3 |

父best19：AUROC 0.970281176，AP 0.731531201。

| 配对比较 | ΔAUROC | AUROC 95%区间 | ΔAP | AP 95%区间 | 修复 / 新增 / 净修复误序对 |
| --- | --- | --- | --- | --- | --- |
| r1a_minus_parent | +0.000000000 | [0.0, 0.0] | +0.000000000 | [0.0, 0.0] | 0.0 / 0.0 / 0.0 |
| r1b_minus_r1a | -0.004876727 | [-0.012437877161276117, 0.0024496896337811236] | +0.003888701 | [-0.02578405698685212, 0.035037331339187516] | 42528.0 / 58489.0 / -15961.0 |
| r1b_minus_parent | -0.004876727 | [-0.012437877161276117, 0.0024496896337811236] | +0.003888701 | [-0.02578405698685212, 0.035037331339187516] | 42528.0 / 58489.0 / -15961.0 |

## r14-w4-tiny-mixup01-fp32-20261026

状态：complete；模型：convnext_tiny。

- /variants/online/best/metrics：n=10277，AUROC=0.9840578302003243，AP=0.8302402519753382。
- /variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9840578302003243，AP=0.8302402519753382。
- /variants/ema/best/metrics：n=10277，AUROC=0.9889333347999262，AP=0.8533188134650791。
- /variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9889333347999262，AP=0.8533188134650791。
- /dev5k/metrics：n=5849，AUROC=0.9887070894382971，AP=0.8360685136139。

R14 Tiny+Mixup FP32 正常早停于8周期，5688成功/5688尝试/0跳步（scaler禁用），四rank exit0。

冻结EMA cycle4：开发集 AUROC 0.988933334800、AP 0.853318813465；同一检查点官方valid AUROC 0.988707089438、AP 0.836068513614。

开发集相对R8 A25 AUROC -0.000511474256，条件95%CI [-0.0022691643001067074, 0.001329964473003797]；AP -0.005055665975。本轮探索支持=False，未达0.998，不自动替换R8模型或提交。

## r25-w4-tiny-native768-s20261026-20261026

状态：complete；模型：convnext_tiny。

- /variants/online/best/metrics：n=10277，AUROC=0.985478897562156，AP=0.8402583865449725。
- /variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.985478897562156，AP=0.8402583865449725。
- /variants/ema/best/metrics：n=10277，AUROC=0.9890185805092255，AP=0.8779228311144733。
- /variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9890185805092255，AP=0.8779228311144733。
- /dev5k/metrics：n=5849，AUROC=0.9857987300565015，AP=0.8265089213814537。

R25 seed26 native768 world4 FP32 completed: 8 cycles, 5688 successful updates, zero skipped updates.

Frozen EMA cycle5: dev AUROC 0.989018580509, AP 0.877922831114; same-checkpoint official valid AUROC 0.985798730057, AP 0.826508921381.

Common first8 versus R23 P8 seed26: AUROC +0.000062941276, AP +0.026239063304; primary method support=False. Conditional AUC interval [-0.0027204886743766145, 0.002598826040608006].

Two-seed primary support=False; secondary repeated AP signal=False. The secondary result does not replace the primary AUC criterion, select checkpoints by AP, or justify a submission change.

All 17 CSV/170281 rows independently recomputed. New formal/finalize allocated card-hours: 7.671412; pre-formal binding/repair retention is listed separately. Original R24 preparation is not charged again. Original cache proof and repair1/1 evidence are retained; resource owner released. Exposed-dev conditional intervals are not final three-fold acceptance.

## r23-w4-tiny-p4-confirm-s20261026-20261026

状态：complete；模型：见指标/原实验定义。

- /runs/r23-4c-tiny-s20261026-b64-p8-fp32-ema/variants/online/best/metrics：n=10277，AUROC=0.9855931695882418，AP=0.8346458182335226。
- /runs/r23-4c-tiny-s20261026-b64-p8-fp32-ema/variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9855931695882418，AP=0.8346458182335226。
- /runs/r23-4c-tiny-s20261026-b64-p8-fp32-ema/variants/ema/best/metrics：n=10277，AUROC=0.9889556392328253，AP=0.8516837678106162。
- /runs/r23-4c-tiny-s20261026-b64-p8-fp32-ema/variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9889556392328253，AP=0.8516837678106162。
- /runs/r23-4c-tiny-s20261026-b64-p8-fp32-ema/dev5k/metrics：n=5849，AUROC=0.9868610261335939，AP=0.826884794541398。
- /runs/r23-4c-tiny-s20261026-b64-p8-fp32-ema/individual_freeze/terminal/training_status/best_metrics：n=10277，AUROC=0.9855931695882418，AP=0.834645818233522。
- /runs/r23-4c-tiny-s20261026-b64-p8-fp32-ema/individual_freeze/terminal/training_status/ema_best_metrics：n=10277，AUROC=0.9889556392328255，AP=0.8516837678106154。
- /runs/r23-4c-tiny-s20261026-b64-p8-fp32-ema/individual_freeze/training_status/best_metrics：n=10277，AUROC=0.9855931695882418，AP=0.834645818233522。
- /runs/r23-4c-tiny-s20261026-b64-p8-fp32-ema/individual_freeze/training_status/ema_best_metrics：n=10277，AUROC=0.9889556392328255，AP=0.8516837678106154。
- /runs/r23-4c-tiny-s20261026-b64-p8-fp32-ema/formal_terminal/training_status/best_metrics：n=10277，AUROC=0.9855931695882418，AP=0.834645818233522。
- /runs/r23-4c-tiny-s20261026-b64-p8-fp32-ema/formal_terminal/training_status/ema_best_metrics：n=10277，AUROC=0.9889556392328255，AP=0.8516837678106154。
- /runs/r23-4c-tiny-s20261026-b64-p8-fp32-ema/official_valid_terminal/candidate/dev_metrics：n=10277，AUROC=0.9889556392328255，AP=0.8516837678106154。

R23 新seed20261026配对轮次完成，按P8→P4各自公共fresh四卡FP32训练。

P8：8周期/5688成功/0跳步，冻结EMA3，dev AUROC 0.988955639233/AP 0.851683767811；同CP valid AUROC 0.986861026134。

P4：9周期/6399成功/0跳步，冻结EMA5，dev AUROC 0.989321981905/AP 0.857234343133；同CP valid AUROC 0.988158359844。

共同前8周期主比较P4−P8：AUROC +0.000366342672、AP +0.005550575322，条件95% AUC区间[-0.0023103187300894112, 0.003343357296090419]；新seed探索支持=False，两seed均过同方法门槛=False。旧seed统计直接复用，不合并预测或样本。

新P4对R8的独立晋级屏幕通过=False；官方valid及年龄/尺寸护栏按各参考、各数据集分别报告。未自动替换R8提交。两模型冻结后才各做一次同CP官方valid，first8不同CP不补GPU评估。

## r24-w4-tiny-native768-fp32-20261026

状态：recorded；模型：见指标/原实验定义。

- 没有可归入本表的完整官方/内部选点指标；保留真实状态，不补造数值。

## r12-w4-convnext-small-capacity-20261026

状态：complete；模型：convnext_small。

- /variants/online/best/metrics：n=10277，AUROC=0.9832230944375799，AP=0.8100496993957974。
- /variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9832230944375799，AP=0.8100496993957974。
- /variants/ema/best/metrics：n=10277，AUROC=0.9872537804486063，AP=0.8389176247553157。
- /variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9872537804486063，AP=0.8389176247553157。
- /dev5k/metrics：n=5849，AUROC=0.985990739147091，AP=0.8119755538701031。

R12 ConvNeXt-Small 正常早停于12周期，8532成功/8539尝试/7AMP跳步，四rank exit0。

冻结EMA cycle3：开发集 AUROC 0.987253780449、AP 0.838917624755；同一检查点官方valid AUROC 0.985990739147、AP 0.811975553870。

开发集相对R8 A25 AUROC -0.002191028607，条件95%CI [-0.0050057614632092184, 0.0004424132900414843]；AP -0.019456854685。本轮探索支持=False，未达0.998，不自动替换R8模型或提交。

## r9-hz6-b192-20261025

状态：complete；模型：见指标/原实验定义。

- /variants/online/best/metrics：n=10277，AUROC=0.9864175169849784，AP=0.8257459638270913。
- /variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9846803683103506，AP=0.8238280568744363。
- /variants/ema/best/metrics：n=10277，AUROC=0.9883425728682768，AP=0.8507366543933486。
- /variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9883425728682768，AP=0.8507366543933486。
- /dev5k/metrics：n=5849，AUROC=0.9864039056962388，AP=0.8179862652893095。

# R9 complete result

One six-rank model completed 15 cycles, 3555 successful / 3559 attempted updates; all six exits zero.

Frozen EMA cycle 4: dev AUROC 0.988342572868, AP 0.850736654393; same checkpoint dev5k AUROC 0.986403905696, AP 0.817986265289.

R9 minus same-seed R8 A25: dev AUC -0.001102236187, conditional content-group bootstrap 95% CI [-0.0033895746989492438, 0.0008651911971544734]; dev5k AUC -0.002035296360.

Full raw dev cycles, fixed-first8 comparison, frozen-threshold age/group guards, pair errors, EMA/AMP/Adam/rank/source audits and original budget are in RESULT-R9.json. Development and monitoring alerts are reported separately. No final three-fold acceptance or automatic replacement is claimed.

## r10-w4-a25-lr1e4-20261025

状态：complete；模型：见指标/原实验定义。

- /variants/online/best/metrics：n=10277，AUROC=0.9848004455997937，AP=0.8247471043424021。
- /variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9848004455997937，AP=0.8247471043424021。
- /variants/ema/best/metrics：n=10277，AUROC=0.9878523336547615，AP=0.8437911786587023。
- /variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9878523336547615，AP=0.8437911786587023。
- /dev5k/metrics：n=5849，AUROC=0.9866195110847079，AP=0.811308170719906。

R10正常完成10 cycles，7110成功/7116尝试，四rank exit0。

冻结EMA cycle4：开发集AUROC 0.987852333655、AP 0.843791178659；同一检查点官方监测集AUROC 0.986619511085、AP 0.811308170720。

开发集相对R8A25 AUROC -0.001592475401，条件95%CI [-0.0035766096279214454, 0.0002490751745699571]。完整逐轮原始预测、首次8轮比较、分层/年龄护栏、成本和状态见RESULT-R10.json。

## r21-w4-tiny-pair025-fp32-20261026

状态：complete；模型：convnext_tiny。

- /variants/online/best/metrics：n=10277，AUROC=0.9867468892954611，AP=0.8198062429790414。
- /variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9867468892954611，AP=0.8198062429790414。
- /variants/ema/best/metrics：n=10277，AUROC=0.9895218051802503，AP=0.8509681832590235。
- /variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9895218051802503，AP=0.8509681832590235。
- /dev5k/metrics：n=5849，AUROC=0.9884452167990836，AP=0.8296006400253195。

R21 Tiny FP32（BCE+.25全局成对排序，无Mixup）正常早停于10周期，7110成功/7110尝试/0跳步（scaler禁用），四rank exit0。

冻结EMA cycle4：开发集 AUROC 0.989521805180、AP 0.850968183259；同一检查点官方valid AUROC 0.988445216799、AP 0.829600640025。

开发集相对R8 A25 AUROC +0.000076996125，条件95%CI [-0.0014660401845453309, 0.0019024354616453433]；AP -0.007406296181。开发集相对R8注册探索性门槛通过=False，未达0.998，不自动替换R8模型或提交。

共同前8周期方法对照R21−R15：AUROC -0.000270403056、AP -0.004795633888，条件95% AUC区间[-0.002079785151912236, 0.0015425520339207507]。全局成对排序筛选=unsupported_or_inconclusive_global_pairwise025_screen；点门槛与条件区间分开。完整训练窗口R21=10周期、R15=9周期另列；相同选中预测字节复用区间，无重复bootstrap。

官方valid相对R8：AUROC +0.000006014743，条件95%CI [-0.002382842288913786, 0.0023609783785575]；AP -0.009379402775，条件95%CI [-0.02276204313888184, 0.005298229446835497]。AP条件区间全部低于0=False。官方valid为已暴露监测，不重新选模或调整阈值。主方法点门槛通过=False；各集、各参考年龄/尺寸告警独立列于JSON，不自动替换R8提交。

## r16-w4-tiny-in22k1k-fp32-20261026

状态：complete；模型：convnext_tiny。

- /variants/online/best/metrics：n=10277，AUROC=0.9854037346786878，AP=0.8184257061706419。
- /variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9854037346786878，AP=0.8184257061706419。
- /variants/ema/best/metrics：n=10277，AUROC=0.987327415631191，AP=0.8535590369927496。
- /variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.987327415631191，AP=0.8535590369927496。
- /dev5k/metrics：n=5849，AUROC=0.9887690875542946，AP=0.8471768230575615。

R16 Tiny FP32（公开22K→1K初始化，无Mixup）正常早停于9周期，6399成功/6399尝试/0跳步（scaler禁用），四rank exit0。

冻结EMA cycle4：开发集 AUROC 0.987327415631、AP 0.853559036993；同一检查点官方valid AUROC 0.988769087554、AP 0.847176823058。

开发集相对R8 A25 AUROC -0.002117393425，条件95%CI [-0.00764730350103858, 0.0016743403111816174]；AP -0.004815442447。相对R8晋级筛选通过=False，未达0.998，不自动替换R8模型或提交。

共同前8周期方法对照R16−R15：AUROC -0.002464792605、AP -0.002204780155，条件95% AUC区间[-0.008163528044540755, 0.0014201851807149589]。初始化筛选=unsupported_or_inconclusive_initialization_screen；点门槛与条件区间分开。完整训练窗口R16=9周期、R15=9周期另列；相同选中预测字节复用区间，无重复bootstrap。

开发集相对R15及R8的注册年龄/尺寸护栏通过，但official valid相对R8的4096plus层（37阳性）AUROC下降0.0140011，触发尺寸监测告警；不能声称两个数据集护栏都通过。监测改善不改变开发集选模或晋级结论。新初始化筛选与R8晋级均未通过，条件AUC区间均跨0；不能把本轮结论单独归因于预训练数据规模。R15保留最高已核验开发集单点，R8保留提交基线。

## r22-w4-tiny-p4-fp32-20261026

状态：complete；模型：convnext_tiny。

- /variants/online/best/metrics：n=10277，AUROC=0.9868956873615139，AP=0.8239726616916401。
- /variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9868956873615139，AP=0.8239726616916401。
- /variants/ema/best/metrics：n=10277，AUROC=0.9905850849951664，AP=0.8587600139473524。
- /variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9905850849951664，AP=0.8587600139473524。
- /dev5k/metrics：n=5849，AUROC=0.9877141942373214，AP=0.8427957011417484。

R22 Tiny FP32（BCE、P4/N60采样及固定711，无Mixup）正常早停于10周期，7110成功/7110尝试/0跳步（scaler禁用），四rank exit0。

冻结EMA cycle5：开发集 AUROC 0.990585084995、AP 0.858760013947；同一检查点官方valid AUROC 0.987714194237、AP 0.842795701142。

开发集相对R8 A25 AUROC +0.001140275939，条件95%CI [-0.0009512823662287728, 0.0034898637453502496]；AP +0.000385534507。开发集相对R8注册探索性门槛通过=False，未达0.998，不自动替换R8模型或提交。

共同前8周期方法对照R22−R15：AUROC +0.000792876759、AP +0.002996196800，条件95% AUC区间[-0.0009954299490836343, 0.0029296574474367396]。P4采样筛选=P4_exploratory_support；点门槛与条件区间分开。完整训练窗口R22=10周期、R15=9周期另列；相同选中预测字节复用区间，无重复bootstrap。

官方valid相对R8：AUROC -0.000725007819，条件95%CI [-0.0039063282908963996, 0.0018676267513638126]；AP +0.003815658341，条件95%CI [-0.015403562284405503, 0.022634971967523437]。AP条件区间全部低于0=False。官方valid为已暴露监测，不重新选模或调整阈值。主方法点门槛通过=True；各集、各参考年龄/尺寸告警独立列于JSON，不自动替换R8提交。

## r15-w4-tiny-nomix-fp32-20261026

状态：complete；模型：convnext_tiny。

- /variants/online/best/metrics：n=10277，AUROC=0.9855366446555524，AP=0.817362599429968。
- /variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9855366446555524，AP=0.817362599429968。
- /variants/ema/best/metrics：n=10277，AUROC=0.9897922082366298，AP=0.8557638171475221。
- /variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9897922082366298，AP=0.8557638171475221。
- /dev5k/metrics：n=5849，AUROC=0.9886682249476717，AP=0.8294849773829656。

R15 Tiny FP32（无Mixup）正常早停于9周期，6399成功/6399尝试/0跳步（scaler禁用），四rank exit0。

冻结EMA cycle4：开发集 AUROC 0.989792208237、AP 0.855763817148；同一检查点官方valid AUROC 0.988668224948、AP 0.829484977383。

开发集相对R8 A25 AUROC +0.000347399181，条件95%CI [-0.0006100794769853174, 0.0015034352360653199]；AP -0.002610662292。相对R8晋级筛选通过=False，未达0.998，不自动替换R8模型或提交。

共同前8周期方法对照R15−R14：AUROC +0.000858873437、AP +0.002445003682，条件95% AUC区间[-0.0006394313370919558, 0.0024017592134552867]。方向筛选=inconclusive_directional_screen；点门槛与条件区间分开，单种子不作普遍结论。R15完整训练9周期、R14为8周期，完整预算结果另列；两者fullbest均为前8所选的同一份预测字节，区间直接复用。

方法门槛需区分：R15相对R14的AUC/AP点阈值通过，但under40按各自全dev冻结阈值新增FN1，触发预注册年龄告警，因此不能标记关闭Mixup有探索支持；反向Mixup也无支持。该小样本工作点告警不是显著伤害证明。R8晋级参考另算：AUC仅+0.000347399，小于+0.0005，未晋级。R15为目前最高已核验开发集单点，但提交及保留的晋级基线不自动改变。

## r8-hz6-age-20261025

状态：complete；模型：见指标/原实验定义。

- /runs/r8-6c-a25-image-age-ema/variants/online/best/metrics：n=10277，AUROC=0.9865067347165748，AP=0.8379512949016218。
- /runs/r8-6c-a25-image-age-ema/variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9865067347165748，AP=0.8379512949016218。
- /runs/r8-6c-a25-image-age-ema/variants/ema/best/metrics：n=10277，AUROC=0.989444809055722，AP=0.8583744794398583。
- /runs/r8-6c-a25-image-age-ema/variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.989444809055722，AP=0.8583744794398583。
- /runs/r8-6c-a25-image-age-ema/dev5k/metrics：n=5849，AUROC=0.9884392020564867，AP=0.8389800428004476。
- /runs/r8-6c-a26-image-age-ema/variants/online/best/metrics：n=10277，AUROC=0.9865282752990322，AP=0.8257491443697049。
- /runs/r8-6c-a26-image-age-ema/variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9865282752990322，AP=0.8257491443697049。
- /runs/r8-6c-a26-image-age-ema/variants/ema/best/metrics：n=10277，AUROC=0.9893583411857159，AP=0.8616940150052851。
- /runs/r8-6c-a26-image-age-ema/variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9893583411857159，AP=0.8616940150052851。
- /runs/r8-6c-a26-image-age-ema/dev5k/metrics：n=5849，AUROC=0.9879492318711032，AP=0.8467857241789106。
- /runs/r8-6c-c25-image-ema/variants/online/best/metrics：n=10277，AUROC=0.9848595676239852，AP=0.8150050857312315。
- /runs/r8-6c-c25-image-ema/variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9848595676239852，AP=0.8150050857312315。

# R8完整结果

|模型|终态cycle|EMA最佳cycle|dev AUROC|dev AP|dev5k AUROC|dev5k AP|
|---|---:|---:|---:|---:|---:|---:|
|C25|8|4|0.988658959721|0.848523624542|0.989038362954|0.839900140042|
|A25|10|4|0.989444809056|0.858374479440|0.988439202056|0.838980042800|
|A26|12|4|0.989358341186|0.861694015005|0.987949231871|0.846785724179|

A25相对C25：dev AUC差+0.000785849334，95%配对内容组bootstrap区间[-0.0003295017292113483, 0.002129649514311051]；dev5k AUC差-0.000599160897。

全部候选先按dev冻结，再一次性评dev5k；未用监测分数更换checkpoint。真实UID510896完整读取三份冻结模型并逐SHA通过，模型保留在原持久路径。

详见RESULT-R8.json的全部曲线、分层护栏、年龄偏倚、误序对、2000次bootstrap、双rank终态及来源。dev5k是已暴露监测集；缺同期C26，不能宣称完整双seed年龄消融支持或最终三折验收。

## r17-w4-tiny-focal2-fp32-20261026

状态：complete；模型：convnext_tiny。

- /variants/online/best/metrics：n=10277，AUROC=0.9867875261389621，AP=0.835671006662209。
- /variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9867875261389621，AP=0.835671006662209。
- /variants/ema/best/metrics：n=10277，AUROC=0.9899571999320479，AP=0.8553826253940112。
- /variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9899571999320479，AP=0.8553826253940112。
- /dev5k/metrics：n=5849，AUROC=0.9893446521535555，AP=0.8246380571087745。

R17 Tiny FP32（Focal γ2，无Mixup）正常早停于10周期，7110成功/7110尝试/0跳步（scaler禁用），四rank exit0。

冻结EMA cycle4：开发集 AUROC 0.989957199932、AP 0.855382625394；同一检查点官方valid AUROC 0.989344652154、AP 0.824638057109。

开发集相对R8 A25 AUROC +0.000512390876，条件95%CI [-0.001054937344931367, 0.002395200709980496]；AP -0.002991854046。开发集相对R8注册探索性门槛通过=True，未达0.998，不自动替换R8模型或提交。

共同前8周期方法对照R17−R15：AUROC +0.000164991695、AP -0.000381191754，条件95% AUC区间[-0.0012559047672556056, 0.0020403445649096754]。Focal筛选=unsupported_or_inconclusive_focal_screen；点门槛与条件区间分开。完整训练窗口R17=10周期、R15=9周期另列；相同选中预测字节复用区间，无重复bootstrap。

官方valid相对R8：AUROC +0.000905450097，条件95%CI [-0.0010497476817905777, 0.0035045131259369695]；AP -0.014341985692，条件95%CI [-0.02777184144238185, -0.0010588804049023529]全部低于0。这是已暴露监测集上的条件告警，不能因AUC点数提高而省略，也不是独立确认或阈值调整依据。本轮两参考的开发集与valid注册年龄/尺寸护栏均无新增告警；主方法点门槛未过，R8提交基线保留。

## r19-w4-tiny-cosine8-fp32-20261026

状态：complete；模型：convnext_tiny。

- /variants/online/best/metrics：n=10277，AUROC=0.986478013939965，AP=0.8389467786436304。
- /variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.986478013939965，AP=0.8389467786436304。
- /variants/ema/best/metrics：n=10277，AUROC=0.9895691639076388，AP=0.853804125838658。
- /variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9895691639076388，AP=0.853804125838658。
- /dev5k/metrics：n=5849，AUROC=0.9887172682334608，AP=0.8299532339240748。

R19 Tiny FP32（BCE、8周期余弦衰减跨度，无Mixup）正常早停于10周期，7110成功/7110尝试/0跳步（scaler禁用），四rank exit0。

冻结EMA cycle4：开发集 AUROC 0.989569163908、AP 0.853804125839；同一检查点官方valid AUROC 0.988717268233、AP 0.829953233924。

开发集相对R8 A25 AUROC +0.000124354852，条件95%CI [-0.0011127711189465317, 0.0014385350648578239]；AP -0.004570353601。开发集相对R8注册探索性门槛通过=False，未达0.998，不自动替换R8模型或提交。

共同前8周期方法对照R19−R15：AUROC -0.000223044329、AP -0.001959691309，条件95% AUC区间[-0.0008632548301613946, 0.0003155438289892342]。提前衰减筛选=unsupported_or_inconclusive_early_decay_screen；点门槛与条件区间分开。完整训练窗口R19=10周期、R15=9周期另列；相同选中预测字节复用区间，无重复bootstrap。

官方valid相对R8：AUROC +0.000278066177，条件95%CI [-0.0029630415397762814, 0.003609085493902963]；AP -0.009026808876，条件95%CI [-0.02324341215064589, 0.004797691076536521]。AP条件区间全部低于0=False。官方valid为已暴露监测，不重新选模或调整阈值。主方法点门槛通过=False；各集、各参考年龄/尺寸告警独立列于JSON，不自动替换R8提交。

## r26-w4-tiny-attnpool-fp32-20261027

状态：complete；模型：convnext_tiny。

- /variants/online/best/metrics：n=10277，AUROC=0.9878321680030994，AP=0.8266080641078781。
- /variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9878321680030994，AP=0.8266080641078781。
- /variants/ema/best/metrics：n=10277，AUROC=0.9906309160216713，AP=0.8562439843105945。
- /variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9906309160216713，AP=0.8562439843105945。
- /dev5k/metrics：n=5849，AUROC=0.9881713146746764，AP=0.8353932227472679。

R26 Tiny FP32（BCE、内容加权汇聚，无Mixup）正常早停于10周期，7110成功/7110尝试/0跳步（scaler禁用），四rank exit0。

冻结EMA cycle4：开发集 AUROC 0.990630916022、AP 0.856243984311；同一检查点官方valid AUROC 0.988171314675、AP 0.835393222747。

开发集相对R8 A25 AUROC +0.001186106966，条件95%CI [-0.00013548514906761843, 0.0027504145941439876]；AP -0.002130495129。开发集相对R8注册探索性门槛通过=True，未达0.998，不自动替换R8模型或提交。

共同前8周期方法对照R26−R15：AUROC +0.000838707785、AP +0.000480167163，条件95% AUC区间[-0.00015588751566933122, 0.0021090920395942427]。内容加权汇聚筛选=attention_pool_exploratory_support；点门槛与条件区间分开。完整训练窗口R26=10周期、R15=9周期另列；相同选中预测字节复用区间，无重复bootstrap。

官方valid相对R8：AUROC -0.000267887382，条件95%CI [-0.0036344473398199186, 0.0028255352437914012]；AP -0.003586820053，条件95%CI [-0.0156504045818534, 0.00889853754793825]。AP条件区间全部低于0=False。官方valid为已暴露监测，不重新选模或调整阈值。主方法点门槛通过=True；各集、各参考年龄/尺寸告警独立列于JSON，不自动替换R8提交。

## r20-w4-tiny-spatial2-fp32-20261026

状态：complete；模型：convnext_tiny。

- /variants/online/best/metrics：n=10277，AUROC=0.9859900662777751，AP=0.8391268022833444。
- /variants/online/fixed_first8_best/metrics：n=10277，AUROC=0.9859900662777751，AP=0.8391268022833444。
- /variants/ema/best/metrics：n=10277，AUROC=0.9892138206821368，AP=0.8541620830144078。
- /variants/ema/fixed_first8_best/metrics：n=10277，AUROC=0.9892138206821368，AP=0.8541620830144078。
- /dev5k/metrics：n=5849，AUROC=0.9878705775448376，AP=0.823700049573868。

R20 Tiny FP32（BCE、2x2空间残差读出，无Mixup）正常早停于9周期，6399成功/6399尝试/0跳步（scaler禁用），四rank exit0。

冻结EMA cycle4：开发集 AUROC 0.989213820682、AP 0.854162083014；同一检查点官方valid AUROC 0.987870577545、AP 0.823700049574。

开发集相对R8 A25 AUROC -0.000230988374，条件95%CI [-0.0015315392024821794, 0.0010621689944154767]；AP -0.004212396425。开发集相对R8注册探索性门槛通过=False，未达0.998，不自动替换R8模型或提交。

共同前8周期方法对照R20−R15：AUROC -0.000578387554、AP -0.001601734133，条件95% AUC区间[-0.0018829825521962744, 0.0007143948533139667]。空间残差读出筛选=unsupported_or_inconclusive_spatial_readout_screen；点门槛与条件区间分开。完整训练窗口R20=9周期、R15=9周期另列；相同选中预测字节复用区间，无重复bootstrap。

官方valid相对R8：AUROC -0.000568624512，条件95%CI [-0.0022457828878436835, 0.0010725945039820254]；AP -0.015279993227，条件95%CI [-0.030683282791430014, 0.00023188272770167077]。AP条件区间全部低于0=False。官方valid为已暴露监测，不重新选模或调整阈值。主方法点门槛通过=False；各集、各参考年龄/尺寸告警独立列于JSON，不自动替换R8提交。

## RESULT-FOLD-1.json

状态：complete；模型：见指标/原实验定义。

- /diagnostics/official_valid/full/metrics：n=5849，AUROC=0.9883128924619544，AP=0.828758172008879。
- /official_valid/metrics：n=5849，AUROC=0.9883128924619544，AP=0.828758172008879。

## RESULT-FOLD-0.json

状态：complete；模型：见指标/原实验定义。

- /diagnostics/official_valid/full/metrics：n=5849，AUROC=0.9790876653360205，AP=0.8024034307466258。
- /official_valid/metrics：n=5849，AUROC=0.9790876653360205，AP=0.8024034307466258。

## r11-w4-a25-groupcv3-20261026

状态：complete；模型：见指标/原实验定义。

- /folds/0/diagnostics/official_valid/full/metrics：n=5849，AUROC=0.9790876653360205，AP=0.8024034307466258。
- /folds/0/official_valid/metrics：n=5849，AUROC=0.9790876653360205，AP=0.8024034307466258。
- /folds/1/diagnostics/official_valid/full/metrics：n=5849，AUROC=0.9883128924619544，AP=0.828758172008879。
- /folds/1/official_valid/metrics：n=5849，AUROC=0.9883128924619544，AP=0.828758172008879。
- /folds/2/diagnostics/official_valid/full/metrics：n=5849，AUROC=0.985354564449355，AP=0.8240530299298858。
- /folds/2/official_valid/metrics：n=5849，AUROC=0.985354564449355，AP=0.8240530299298858。

R11完整三折结果。

fold0: inner-selected EMA cycle11, inner AUC 0.9788490631119513, outer AUC 0.9762919354556738, official valid AUC 0.9790876653360205。
fold1: inner-selected EMA cycle6, inner AUC 0.9879395401101763, outer AUC 0.9833897024018567, official valid AUC 0.9883128924619544。
fold2: inner-selected EMA cycle4, inner AUC 0.9825104178089703, outer AUC 0.9806572790268114, official valid AUC 0.985354564449355。

三折外层均值/范围/最差折、完整内层曲线、2000次外层内容组bootstrap和固定inner阈值分层计数见RESULT-R11.json。
OOF覆盖51396行各一次，仅作补充。未自动改提交或晋级。
逐折outer及同CP official valid均达到0.998：False。

## RESULT-FOLD-2.json

状态：complete；模型：见指标/原实验定义。

- /diagnostics/official_valid/full/metrics：n=5849，AUROC=0.985354564449355，AP=0.8240530299298858。
- /official_valid/metrics：n=5849，AUROC=0.985354564449355，AP=0.8240530299298858。

## RESULT-FOLD-0-OUTER.json

状态：fold_training_and_outer_complete；模型：见指标/原实验定义。

- 没有可归入本表的完整官方/内部选点指标；保留真实状态，不补造数值。

## RESULT-FOLD-1-OUTER.json

状态：fold_training_and_outer_complete；模型：见指标/原实验定义。

- 没有可归入本表的完整官方/内部选点指标；保留真实状态，不补造数值。

## RESULT-FOLD-2-OUTER.json

状态：fold_training_and_outer_complete；模型：见指标/原实验定义。

- 没有可归入本表的完整官方/内部选点指标；保留真实状态，不补造数值。

## R6

状态：completed；模型：见指标/原实验定义。

- /variants/online/best/metrics：n=10277，AUROC=0.9848595676239852，AP=0.8150050857312315。
- /variants/ema/best/metrics：n=10277，AUROC=0.9886589597212496，AP=0.8485236245422765。
- /all_best_metrics/online：n=10277，AUROC=0.9848595676239852，AP=0.8150050857312315。
- /all_best_metrics/ema：n=10277，AUROC=0.9886589597212496，AP=0.8485236245422765。
- /all_best_metrics/R5：n=10277，AUROC=0.9848595676239852，AP=0.8150050857312315。
- /all_best_metrics/R2：n=10277，AUROC=0.9823576518870772，AP=0.802250129970274。
- /all_best_metrics/R4F：n=10277，AUROC=0.9863609920522889，AP=0.8395417792519542。
- /checkpoints/r6-convnext-seed25-512-ema/1/best_metrics：n=10277，AUROC=0.9848595676239851，AP=0.8150050857312311。
- /checkpoints/r6-convnext-seed25-512-ema/1/ema_best_metrics：n=10277，AUROC=0.9886589597212495，AP=0.8485236245422765。
- /checkpoints/r6-convnext-seed25-512-ema/2/best_metrics：n=10277，AUROC=0.9848595676239851，AP=0.8150050857312311。
- /checkpoints/r6-convnext-seed25-512-ema/2/ema_best_metrics：n=10277，AUROC=0.9886589597212495，AP=0.8485236245422765。
- /checkpoints/r6-convnext-seed25-512-ema/3/best_metrics：n=10277，AUROC=0.9846857152634428，AP=0.8278196128309269。

# R6 固定 EMA 结果

正式训练完成 5688 成功 / 5692 尝试 / 4 AMP跳步；仅online推进注册早停。开发筛选目标状态见下表与JSON。

|读出/角色|best cycle|best AUROC / AP|last AUROC / AP|
|---|---|---|---|
|online / online_control|7|0.984859568 / 0.815005086|0.972753149 / 0.823590674|
|ema / method_candidate|4|0.988658960 / 0.848523625|0.986099602 / 0.839742345|

固定规则warmcopy1422，之后FP32 .999/.001，跳步不更新；online和EMA独立best，不挑较高读出包装EMA收益。

|配对比较|ΔAUROC|95%CI|ΔAP|95%CI|
|---|---|---|---|---|
|ema_minus_online|+0.003799392|[0.0013053540291131094, 0.006748939061583382]|+0.033518539|[0.0168940397046285, 0.05044524689270553]|
|ema_minus_R5|+0.003799392|[0.0013053540291131094, 0.006748939061583382]|+0.033518539|[0.0168940397046285, 0.05044524689270553]|
|ema_minus_R2|+0.006301308|[0.00254761179917318, 0.010525696771831591]|+0.046273495|[0.02499590964519264, 0.06870558109073083]|
|ema_minus_R4F|+0.002297968|[-0.000791566061284138, 0.00593825116072681]|+0.008981845|[-0.007382091400588408, 0.024729857959363975]|
|online_minus_R5|+0.000000000|[0.0, 0.0]|+0.000000000|[0.0, 0.0]|

## R3

状态：completed；模型：见指标/原实验定义。

- /R3A/checkpoints/1/best_metrics：n=10277，AUROC=0.9821706612989369，AP=0.8111183328178004。
- /R3A/checkpoints/2/best_metrics：n=10277，AUROC=0.9821706612989369，AP=0.8111183328178004。
- /R3A/best/metrics：n=10277，AUROC=0.9821706612989368，AP=0.8111183328178011。
- /R3B/checkpoints/1/best_metrics：n=10277，AUROC=0.9771151935352588，AP=0.7871486512818584。
- /R3B/checkpoints/2/best_metrics：n=10277，AUROC=0.9771151935352588，AP=0.7871486512818584。
- /R3B/best/metrics：n=10277，AUROC=0.9771151935352588，AP=0.7871486512818584。
- /probes/a/best/metrics：n=10277，AUROC=0.9196195597043838，AP=0.463112714520532。
- /probes/b/best/metrics：n=10277，AUROC=0.7997025260839649，AP=0.13072772522631723。

# R3 — seed26配对复验与较低骨干LR

两支正式实验均完整结束。最佳AUROC：A 0.982170661，B 0.977115194；AP：A 0.811118333，B 0.787148651。
距0.998：0.015642348；开发筛选达标：False，尚未进行独立确认。

| 分支 | best cycle | best AUROC | best AP | last AUROC | last AP | 成功/尝试/skip | 结束原因 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R3A | 6 | 0.982170661 | 0.811118333 | 0.978231943 | 0.800867553 | 8532/8537/5 | registered early stop |
| R3B | 3 | 0.977115194 | 0.787148651 | 0.969084223 | 0.768427781 | 6399/6403/4 | registered early stop |

主比较 B−A：ΔAUROC -0.005055468，95%配对区间[-0.012147956177772062, 0.001038508770536904]；ΔAP -0.023969682，区间[-0.05603851524698146, 0.008786431887590915]。

| 比较 | ΔAUROC | AUROC 95% CI | ΔAP | AP 95% CI |
| --- | --- | --- | --- | --- |
| b_minus_a | -0.005055468 | [-0.012147956177772062, 0.001038508770536904] | -0.023969682 | [-0.05603851524698146, 0.008786431887590915] |
| a_minus_r2 | -0.000186991 | [-0.005800641293064288, 0.005766423296183112] | +0.008868203 | [-0.020455030016157998, 0.03652397706634935] |
| b_minus_r2 | -0.005242458 | [-0.011595805543199713, -8.143071444606553e-05] | -0.015101479 | [-0.03972469270240884, 0.009665340593355854] |
| a_minus_effnet | +0.011889485 | [0.0030719189647491927, 0.021636165863228092] | +0.079587132 | [0.03978410078929399, 0.11989635826908657] |

## R7

状态：completed；模型：见指标/原实验定义。

- /variants/online/best/metrics：n=10277，AUROC=0.9863609920522889，AP=0.8395417792519542。
- /variants/ema/best/metrics：n=10277，AUROC=0.9880935576242662，AP=0.8594669229052662。
- /all_best_metrics/online：n=10277，AUROC=0.9863609920522889，AP=0.8395417792519542。
- /all_best_metrics/ema：n=10277，AUROC=0.9880935576242662，AP=0.8594669229052662。
- /all_best_metrics/R4F：n=10277，AUROC=0.9863609920522889，AP=0.8395417792519542。
- /all_best_metrics/R2：n=10277，AUROC=0.9823576518870772，AP=0.802250129970274。
- /all_best_metrics/R6_EMA：n=10277，AUROC=0.9886589597212496，AP=0.8485236245422765。
- /all_best_metrics/R6_online：n=10277，AUROC=0.9848595676239852，AP=0.8150050857312315。
- /checkpoints/r7-convnext-seed26-512-ema/1/best_metrics：n=10277，AUROC=0.9863609920522889，AP=0.8395417792519538。
- /checkpoints/r7-convnext-seed26-512-ema/1/ema_best_metrics：n=10277，AUROC=0.9880935576242663，AP=0.8594669229052655。
- /checkpoints/r7-convnext-seed26-512-ema/2/best_metrics：n=10277，AUROC=0.9863609920522889，AP=0.8395417792519538。
- /checkpoints/r7-convnext-seed26-512-ema/2/ema_best_metrics：n=10277，AUROC=0.9872247541318198，AP=0.8506031227390112。

# R7 固定 EMA 第二 seed 结果

正式训练完成 6399 成功 / 6404 尝试 / 5 AMP跳步；仅online推进注册早停。新增GPU probe=0。

|读出/角色|best cycle|best AUROC / AP|last AUROC / AP|
|---|---|---|---|
|online / online_control|3|0.986360992 / 0.839541779|0.979721298 / 0.813835812|
|ema / method_candidate|4|0.988093558 / 0.859466923|0.986444863 / 0.843421508|

固定warmcopy1422，之后FP32 .999/.001，skip不更新。每variant按同一AUROC/AP规则选best；online是控制，不择高包装EMA收益。

|比较|ΔAUROC|95%CI|ΔAP|95%CI|
|---|---|---|---|---|
|ema_minus_online|+0.001732566|[-0.0013112818830785023, 0.004478271860965819]|+0.019925144|[0.008520886418409462, 0.031795673300484925]|
|ema_minus_R4F|+0.001732566|[-0.0013112818830785023, 0.004478271860965819]|+0.019925144|[0.008520886418409462, 0.031795673300484925]|
|ema_minus_R2|+0.005735906|[0.0008738550018103447, 0.01069416064475394]|+0.057216793|[0.035130901829090656, 0.079381226900374]|
|ema_minus_R6_EMA|-0.000565402|[-0.00338496652158046, 0.0020754449365946582]|+0.010943298|[-0.0005212095192811016, 0.024704531704512186]|
|online_minus_R4F|+0.000000000|[0.0, 0.0]|+0.000000000|[0.0, 0.0]|
|online_minus_R6_online|+0.001501424|[-0.0007578350086620556, 0.0038095621069962797]|+0.024536694|[0.004305807761735425, 0.04522430734545846]|

## R5

状态：completed；模型：见指标/原实验定义。

- /R5/checkpoints/1/best_metrics：n=10277，AUROC=0.9848595676239851，AP=0.8150050857312311。
- /R5/checkpoints/2/best_metrics：n=10277，AUROC=0.9848595676239851，AP=0.8150050857312311。
- /R5/best/metrics：n=10277，AUROC=0.9848595676239852，AP=0.8150050857312315。
- /factorial_2seed_2resolution/cells/r5/best/metrics：n=10277，AUROC=0.9848595676239852，AP=0.8150050857312315。
- /factorial_2seed_2resolution/cells/r2/best/metrics：n=10277，AUROC=0.9823576518870772，AP=0.802250129970274。
- /factorial_2seed_2resolution/cells/r4f/best/metrics：n=10277，AUROC=0.9863609920522889，AP=0.8395417792519542。
- /factorial_2seed_2resolution/cells/r3a/best/metrics：n=10277，AUROC=0.9821706612989368，AP=0.8111183328178011。

# R5 — 固定 seed25 公共 ConvNeXt-Tiny 全程512

正式完整结束：registered early stop；成功/尝试/AMP skip 5688/5692/4。
best cycle7 AUROC 0.984859568 / AP 0.815005086；last cycle8 AUROC 0.972753149 / AP 0.823590674。
开发筛选≥.998：False；最高已观察距目标 0.011639008，未作独立确认。

| seed | 384实验 | 384 best AUROC/AP | 512实验 | 512 best AUROC/AP |
| --- | --- | --- | --- | --- |
| 20261025 | R2 | 0.982357652/0.802250130 | R5 | 0.984859568/0.815005086 |
| 20261026 | R3A | 0.982170661/0.811118333 | R4F | 0.986360992/0.839541779 |

| 比较 | ΔAUROC | AUROC 95% CI | ΔAP | AP 95% CI |
| --- | --- | --- | --- | --- |
| r5_minus_r2 | +0.002501916 | [-0.002883965174289496, 0.0078112097881953375] | +0.012754956 | [-0.010315445159485024, 0.036168211447679066] |
| r5_minus_r4f | -0.001501424 | [-0.003809562106996281, 0.000757835008662055] | -0.024536694 | [-0.04522430734545847, -0.004305807761735427] |
| archived_r4f_minus_r3a | +0.004190331 | [-0.00016336556284980429, 0.009039693437823309] | +0.028423446 | [0.002986487957374437, 0.05822495138519065] |

固定seed25分辨率支持规则：False；该seed较强配对开发证据：False。
替换最高探索候选规则：False；两比较AUROC区间均正的更强开发证据：False。
分组告警：相对R2 2项，相对R4F 3项；完整年龄/原长边P/N及90/95操作点、告警差值见JSON。
不要求R5赢R4F才算第二seed支持；不以均值掩盖某seed未过门槛，跨seed分组损害限制稳定性表述。两个seed仍共享已暴露Dev，CI仅条件开发证据，晋级/后继由规划者审查。
原早停状态：reference_AUROC=0.9846857152634428，bad_cycles=6，min_delta=.0005；严格AUROC best与早停参考各按原合同维护。
两seed分辨率差值均值/范围：{'auroc': {'mean': 0.003346123245130006, 'range': [0.0025019157369079448, 0.004190330753352067]}, 'ap': {'mean': 0.0205892010975553, 'range': [0.012754955760957531, 0.028423446434153066]}, 'pr_auc': {'mean': 0.020664294449266152, 'range': [0.012812943144371425, 0.02851564575416088]}}；不由两点估计精确seed方差。

## n02-budget-race-recovery-20261027

状态：completed；模型：见指标/原实验定义。

- /official_valid/baseline_R26_valid_metrics：n=5849，AUROC=0.9881713146746764，AP=0.8353932227472683。
- /official_valid/metrics：n=5849，AUROC=0.9898119513860743，AP=0.8475180205539281。

## n06-flair-resnet50-finetune-20261027

状态：completed；模型：见指标/原实验定义。

- /phases/valid/independent_metrics：n=5849，AUROC=0.9952862924941565，AP=0.9106003212618431。
- /phases/valid/metrics：n=5849，AUROC=0.9952862924941565，AP=0.9106003212618431。
- /valid_metrics：n=5849，AUROC=0.9952862924941565，AP=0.9106003212618431。

## VALID_RESULT.json

状态：completed；模型：见指标/原实验定义。

- /independent_metrics：n=5849，AUROC=0.9952862924941565，AP=0.9106003212618431。
- /metrics：n=5849，AUROC=0.9952862924941565，AP=0.9106003212618431。

## n03-dinov2s14-finetune-20261027

状态：completed；模型：见指标/原实验定义。

- /phases/valid/independent_metrics：n=5849，AUROC=0.9898119513860743，AP=0.8475180205539281。
- /phases/valid/metrics：n=5849，AUROC=0.9898119513860743，AP=0.8475180205539282。
- /valid_metrics：n=5849，AUROC=0.9898119513860743，AP=0.8475180205539281。

## n04-frozen-feature-fusion-20261027

状态：completed；模型：见指标/原实验定义。

- /phases/valid/metrics：n=5849，AUROC=0.9898119513860743，AP=0.8475180205539281。
- /valid_metrics：n=5849，AUROC=0.9898119513860743，AP=0.8475180205539281。

## n08-imagenet-resnet50-finetune-20261027

状态：completed；模型：见指标/原实验定义。

- /phases/valid/independent_metrics：n=5849，AUROC=0.9898119513860743，AP=0.8475180205539281。
- /phases/valid/metrics：n=5849，AUROC=0.9898119513860743，AP=0.8475180205539282。
- /valid_metrics：n=5849，AUROC=0.9898119513860743，AP=0.8475180205539281。

## auroc998-20261025/outputs/r1b-low-lr-512-probe

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/outputs/r1b-low-lr-512-probe/effnet-v2-s

状态：completed；成功更新：250；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/outputs/r1a-low-lr-384

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/outputs/r1a-low-lr-384/effnet-v2-s

状态：completed；成功更新：5688；skip：4。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/outputs/r1a-low-lr-384-probe

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/outputs/r1a-low-lr-384-probe/effnet-v2-s

状态：completed；成功更新：250；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/outputs/r1b-low-lr-512

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/outputs/r1b-low-lr-512/effnet-v2-s

状态：completed；成功更新：5688；skip：3。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r3/outputs/r3b-convnext-seed26-backbone5e5

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r3/outputs/r3b-convnext-seed26-backbone5e5/convnext-tiny

状态：completed；成功更新：6399；skip：4。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r3/outputs/r3b-convnext-seed26-backbone5e5-probe

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r3/outputs/r3b-convnext-seed26-backbone5e5-probe/convnext-tiny

状态：completed；成功更新：250；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r3/outputs/r3a-convnext-seed26-reference-probe

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r3/outputs/r3a-convnext-seed26-reference-probe/convnext-tiny

状态：completed；成功更新：250；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r3/outputs/r3a-convnext-seed26-reference

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r3/outputs/r3a-convnext-seed26-reference/convnext-tiny

状态：completed；成功更新：8532；skip：5。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r4/outputs/r4-convnext-seed26-512-probe

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r4/outputs/r4-convnext-seed26-512-probe/convnext-tiny

状态：completed；成功更新：250；skip：1。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r6/outputs/r6-convnext-seed25-512-ema-probe

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r6/outputs/r6-convnext-seed25-512-ema-probe/convnext-tiny

状态：completed；成功更新：250；skip：1。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r6/outputs/r6-convnext-seed25-512-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r6/outputs/r6-convnext-seed25-512-ema/convnext-tiny

状态：completed；成功更新：5688；skip：4。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r2/outputs/r2t-convnext-imagenet-384-probe

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r2/outputs/r2t-convnext-imagenet-384-probe/convnext-tiny

状态：completed；成功更新：250；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r2/outputs/r2t-convnext-imagenet-384

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r2/outputs/r2t-convnext-imagenet-384/convnext-tiny

状态：completed；成功更新：6399；skip：5。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r4f/prior-evidence/probe

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r4f/prior-evidence/probe/convnext-tiny

状态：completed；成功更新：250；skip：1。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r4f/outputs/r4f-convnext-seed26-512

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r4f/outputs/r4f-convnext-seed26-512/convnext-tiny

状态：completed；成功更新：6399；skip：5。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r5/outputs/r5-convnext-seed25-512

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r5/outputs/r5-convnext-seed25-512/convnext-tiny

状态：completed；成功更新：5688；skip：4。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r7/outputs/r7-convnext-seed26-512-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/r7/outputs/r7-convnext-seed26-512-ema/convnext-tiny

状态：completed；成功更新：6399；skip：5。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r9/outputs/r9-6c-a25-b192-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r9/outputs/r9-6c-a25-b192-ema/convnext-tiny

状态：completed；成功更新：3555；skip：4。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r9/outputs/r9-6c-a25-b192-ema-smoke

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r9/outputs/r9-6c-a25-b192-ema-smoke/convnext-tiny

状态：completed；成功更新：150；skip：2。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r9/optimization/native-outputs/engineering-w1-g192-m24-once

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r9/optimization/native-outputs/engineering-w1-g192-m24-once/convnext-tiny

状态：completed；成功更新：12；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r9/optimization/native-outputs/engineering-w4-g192-m24-once

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r9/optimization/native-outputs/engineering-w4-g192-m24-once/convnext-tiny

状态：completed；成功更新：12；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r9/optimization/native-outputs/engineering-w2-g192-m24-once

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r9/optimization/native-outputs/engineering-w2-g192-m24-once/convnext-tiny

状态：completed；成功更新：12；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r11/outputs/r11-4c-a25-cv1-b64-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r11/outputs/r11-4c-a25-cv1-b64-ema/convnext-tiny

状态：completed；成功更新：4797；skip：5。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r11/outputs/r11-4c-a25-cv0-b64-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r11/outputs/r11-4c-a25-cv0-b64-ema/convnext-tiny

状态：completed；成功更新：9061；skip：6。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r11/outputs/r11-4c-a25-cv2-b64-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r11/outputs/r11-4c-a25-cv2-b64-ema/convnext-tiny

状态：completed；成功更新：4264；skip：4。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r11/profile/outputs/r11-4c-a25-cv0-b64-ema-utilization

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r11/profile/outputs/r11-4c-a25-cv0-b64-ema-utilization/convnext-tiny

状态：completed；成功更新：150；skip：1。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r15/outputs/r15-4c-tiny-a25-b64-nomix-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r15/outputs/r15-4c-tiny-a25-b64-nomix-fp32-ema/convnext-tiny

状态：completed；成功更新：6399；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r17/outputs/r17-4c-tiny-a25-b64-focal2-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r17/outputs/r17-4c-tiny-a25-b64-focal2-fp32-ema/convnext-tiny

状态：completed；成功更新：7110；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r19/outputs/r19-4c-tiny-a25-b64-cosine8-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r19/outputs/r19-4c-tiny-a25-b64-cosine8-fp32-ema/convnext-tiny

状态：completed；成功更新：7110；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r21/outputs/r21-4c-tiny-a25-b64-pair025-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r21/outputs/r21-4c-tiny-a25-b64-pair025-fp32-ema/convnext-tiny

状态：completed；成功更新：7110；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r22/outputs/r22-4c-tiny-a25-b64-p4-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r22/outputs/r22-4c-tiny-a25-b64-p4-fp32-ema/convnext-tiny

状态：completed；成功更新：7110；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r24/outputs/r24-4c-tiny-a25-b64-native768-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r24/outputs/r24-4c-tiny-a25-b64-native768-fp32-ema/convnext-tiny

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r26/outputs/r26-4c-tiny-a25-b64-attnpool-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r26/outputs/r26-4c-tiny-a25-b64-attnpool-fp32-ema/convnext-tiny

状态：completed；成功更新：7110；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r8/control/cache-v1

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r8/outputs/r8-6c-a26-image-age-ema-smoke

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r8/outputs/r8-6c-a26-image-age-ema-smoke/convnext-tiny

状态：completed；成功更新：250；skip：1。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r8/outputs/r8-6c-a25-image-age-ema-smoke

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r8/outputs/r8-6c-a25-image-age-ema-smoke/convnext-tiny

状态：completed；成功更新：250；skip：1。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r8/outputs/r8-6c-c25-image-ema-smoke

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r8/outputs/r8-6c-c25-image-ema-smoke/convnext-tiny

状态：completed；成功更新：250；skip：1。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r8/outputs/r8-6c-c25-image-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r8/outputs/r8-6c-c25-image-ema/convnext-tiny

状态：completed；成功更新：5688；skip：4。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r8/outputs/r8-6c-a26-image-age-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r8/outputs/r8-6c-a26-image-age-ema/convnext-tiny

状态：completed；成功更新：8532；skip：5。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r8/outputs/r8-6c-a25-image-age-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r8/outputs/r8-6c-a25-image-age-ema/convnext-tiny

状态：completed；成功更新：7110；skip：4。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r10/outputs/r10-4c-a25-b64-lr1e4-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r10/outputs/r10-4c-a25-b64-lr1e4-ema/convnext-tiny

状态：completed；成功更新：7110；skip：6。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r12/outputs/r12-4c-small-a25-b64-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r12/outputs/r12-4c-small-a25-b64-ema/convnext-small

状态：completed；成功更新：8532；skip：7。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r14/outputs/r14-4c-tiny-a25-b64-mixup01-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r14/outputs/r14-4c-tiny-a25-b64-mixup01-fp32-ema/convnext-tiny

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r16/outputs/r16-4c-tiny-in22k1k-a25-b64-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r16/outputs/r16-4c-tiny-in22k1k-a25-b64-fp32-ema/convnext-tiny

状态：completed；成功更新：6399；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r20/outputs/r20-4c-tiny-a25-b64-spatial2-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r20/outputs/r20-4c-tiny-a25-b64-spatial2-fp32-ema/convnext-tiny

状态：completed；成功更新：6399；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r23/outputs/r23-4c-tiny-s20261026-b64-p8-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r23/outputs/r23-4c-tiny-s20261026-b64-p8-fp32-ema/convnext-tiny

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r23/outputs/r23-4c-tiny-s20261026-b64-p4-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r23/outputs/r23-4c-tiny-s20261026-b64-p4-fp32-ema/convnext-tiny

状态：completed；成功更新：6399；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r25/outputs/r25-4c-tiny-s20261026-b64-native768-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc998-20261025/coordination-huazhong8-20261025/r25/outputs/r25-4c-tiny-s20261026-b64-native768-fp32-ema/convnext-tiny

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/A-n13-20261028/N13-terminal-1793131608/snapshot/outputs/n13-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-rgb01-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/A-n13-20261028/N13-terminal-1793131608/snapshot/outputs/n13-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-rgb01-ema/resnet50-imagenet

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/analysis-n12-ready-20261028/result

状态：technical_failure；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/A-n14r1-20261028/SNAPSHOT-1793145256086730482/snapshot/outputs/n14r1-4c-tiny-a25-b64-attnpool-native768-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/A-n14r1-20261028/SNAPSHOT-1793145256086730482/snapshot/outputs/n14r1-4c-tiny-a25-b64-attnpool-native768-fp32-ema/convnext-tiny

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/analysis-n14r1-dev-20261028-final1/result/verified-view/n14r1-4c-tiny-a25-b64-attnpool-native768-fp32-ema/convnext-tiny

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/A-n15-20261028/TERMINAL-1793153207529363662/snapshot/outputs/n15-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-rgb01-spatial-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/A-n15-20261028/TERMINAL-1793153207529363662/snapshot/outputs/n15-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-rgb01-spatial-ema/resnet50-imagenet

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/analysis-n15-dev-20261028-attempt1/result

状态：technical_failure；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/resume9010-A-stage2-20261028-attempt1/CPU-FIXTURE-NOT-FOR-TRAINING

状态：completed；成功更新：6400；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/A-next-20261028-exec/N11-terminal-1793125984/snapshot/outputs/n11r1-4c-tiny-a25-b64-attnpool-bblr2e5-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/A-next-20261028-exec/N11-terminal-1793125984/snapshot/outputs/n11r1-4c-tiny-a25-b64-attnpool-bblr2e5-fp32-ema/convnext-tiny

状态：completed；成功更新：7110；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/A-next-20261028-exec/N12-terminal-1793128304/snapshot/outputs/n12-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-bblr2e6-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/A-next-20261028-exec/N12-terminal-1793128304/snapshot/outputs/n12-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-bblr2e6-ema/resnet50-imagenet

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/analysis-n11r1-fullstop-20261028T023039/result

状态：technical_failure；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/analysis-n14r1-dev-20261028-partial1/result

状态：technical_failure；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/resume9010-A-stage4-observe-20261028-attempt2/snapshots/progress-16/files/outputs/r91-4c-r50v2-s20261027-b64-p8-fp32-fixedbn-9010

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/resume9010-A-stage4-observe-20261028-attempt2/snapshots/progress-16/files/outputs/r91-4c-r50v2-s20261027-b64-p8-fp32-fixedbn-9010/resnet50-imagenet

状态：completed；成功更新：6400；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/resume9010-A-stage4-observe-20261028-attempt2/snapshots/progress-15/files/outputs/r91-4c-r50v2-s20261027-b64-p8-fp32-fixedbn-9010

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/resume9010-A-stage4-observe-20261028-attempt2/snapshots/progress-15/files/outputs/r91-4c-r50v2-s20261027-b64-p8-fp32-fixedbn-9010/resnet50-imagenet

状态：completed；成功更新：6400；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/resume9010-A-stage4-observe-20261028-attempt2/snapshots/progress-17/files/outputs/r91-4c-r50v2-s20261027-b64-p8-fp32-fixedbn-9010

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/technical-prep/resume9010-A-stage4-observe-20261028-attempt2/snapshots/progress-17/files/outputs/r91-4c-r50v2-s20261027-b64-p8-fp32-fixedbn-9010/resnet50-imagenet

状态：completed；成功更新：6400；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N06/outputs/n06-4c-flairr50-s20261027-b64-p8-fp32-fixedbn-ema-probe

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N06/outputs/n06-4c-flairr50-s20261027-b64-p8-fp32-fixedbn-ema-probe/flair-resnet50

状态：completed；成功更新：100；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N06/outputs/n06-4c-flairr50-s20261027-b64-p8-fp32-fixedbn-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N06/outputs/n06-4c-flairr50-s20261027-b64-p8-fp32-fixedbn-ema/flair-resnet50

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N10/outputs/n10-4c-flairr50-s20261027-b64-p8-bblr2e6-headlr2e4-fp32-fixedbn-ema-probe

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N10/outputs/n10-4c-flairr50-s20261027-b64-p8-bblr2e6-headlr2e4-fp32-fixedbn-ema-probe/flair-resnet50

状态：completed；成功更新：100；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N10/outputs/n10-4c-flairr50-s20261027-b64-p8-bblr2e6-headlr2e4-fp32-fixedbn-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N10/outputs/n10-4c-flairr50-s20261027-b64-p8-bblr2e6-headlr2e4-fp32-fixedbn-ema/flair-resnet50

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N11R1/outputs/n11r1-4c-tiny-a25-b64-attnpool-bblr2e5-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N11R1/outputs/n11r1-4c-tiny-a25-b64-attnpool-bblr2e5-fp32-ema/convnext-tiny

状态：completed；成功更新：7110；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N13-R50-RGB01/outputs/n13-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-rgb01-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N13-R50-RGB01/outputs/n13-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-rgb01-ema/resnet50-imagenet

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/R91-R50-IMAGENETV2-9010/outputs/r91-4c-r50v2-s20261027-b64-p8-fp32-fixedbn-9010

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/R91-R50-IMAGENETV2-9010/outputs/r91-4c-r50v2-s20261027-b64-p8-fp32-fixedbn-9010/resnet50-imagenet

状态：completed；成功更新：6400；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N03/outputs/n03-4c-dinov2s14-s20261027-b64-p8-fp32-ema-probe

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N03/outputs/n03-4c-dinov2s14-s20261027-b64-p8-fp32-ema-probe/dinov2-s14

状态：completed；成功更新：100；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N03/outputs/n03-4c-dinov2s14-s20261027-b64-p8-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N03/outputs/n03-4c-dinov2s14-s20261027-b64-p8-fp32-ema/dinov2-s14

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N07/outputs/n07-4c-flairr50-s20261027-b64-p8-bblr2e6-fp32-fixedbn-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N07/outputs/n07-4c-flairr50-s20261027-b64-p8-bblr2e6-fp32-fixedbn-ema/flair-resnet50

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N07/outputs/n07-4c-flairr50-s20261027-b64-p8-bblr2e6-fp32-fixedbn-ema-probe

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N07/outputs/n07-4c-flairr50-s20261027-b64-p8-bblr2e6-fp32-fixedbn-ema-probe/flair-resnet50

状态：completed；成功更新：100；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N08/outputs/n08-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N08/outputs/n08-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-ema/resnet50-imagenet

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N08/outputs/n08-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-ema-probe

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N08/outputs/n08-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-ema-probe/resnet50-imagenet

状态：completed；成功更新：100；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N12-R50-LR/outputs/n12-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-bblr2e6-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N12-R50-LR/outputs/n12-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-bblr2e6-ema/resnet50-imagenet

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N14R1-TINY-768-ATTN/outputs/n14r1-4c-tiny-a25-b64-attnpool-native768-fp32-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N14R1-TINY-768-ATTN/outputs/n14r1-4c-tiny-a25-b64-attnpool-native768-fp32-ema/convnext-tiny

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N15-R50-RGB01-SPATIAL/outputs/n15-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-rgb01-spatial-ema

状态：completed；成功更新：None；skip：None。官方指标未在该终态中确认，不补造数值。

## auroc999-20261027/rounds/N15-R50-RGB01-SPATIAL/outputs/n15-4c-r50imagenetv2-s20261027-b64-p8-fp32-fixedbn-rgb01-spatial-ema/resnet50-imagenet

状态：completed；成功更新：5688；skip：0。官方指标未在该终态中确认，不补造数值。

# N09–N15 精简结论

依据规划Agent已读取的历史文本整理；未知值保持未知，不补造官方单模成绩。

| 实验 | 留下的结论 |
| --- | --- |
| N09 | 只拟对既有N07 H1做观察性官方评估，后因train-first延期；没有已知完成官方结果，target未知，不能记成模型失败。 |
| N10 | FLAIR head LR从1e-3降至2e-4，仅形成研究内部dev证据；医学初始化规则禁止其后续比赛valid和参数/BN/teacher继承，可靠最终单模数值未知。 |
| N11R1 | Tiny attention骨干LR2e-5，10周期/7110更新/0skip；dev选EMA6，AUROC0.9811356439503656/AP0.8062537151284033，未胜R26 dev0.9906309160216713，不能推定官方改善。 |
| N12 | 普通ImageNet R50降低骨干LR，16条dev选online6：AUROC0.9538890986931435/AP0.6459986313584758；未胜N08新J1，未评官方5849。 |
| N13 | 普通ImageNet R50仅改RGB/255，dev最佳EMA5 AUROC0.9792046300336217/AP0.785362737183347；高于N08新J1点估计但低于R26门，未评本模型官方5849。 |
| N14/N14R1 | 初次技术失败在cycle2，R1完成5688更新/8周期；M14单模dev EMA3 AUROC0.9913501575976231。官方获评的是融合E14（0.9892178798865157，未达目标），M14官方metrics:null，不能冒称单模官方成绩。 |
| N15 | 在N13 RGB01 R50上尝试内容加权空间池化，固定8×711及原LR/BN/EMA；可靠最终单模数值未知，不能套用C1/E14官方分数；N16/旧融合派发已撤销。 |
