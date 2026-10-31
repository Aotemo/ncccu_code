# R127：训练旋转 identity 单项消融

**状态：协调方已采用，仅授权 seed20261029 一次新 fresh，A/B 必要准备中。** SCIENCE落地、Git同步、正式验收、部署、activation、真实8rank与首成功均待A实际回执；本文不是开训实证或新增批准门。2026-10-31。采用及本次activation授权reference为协调消息 `15069bfb-e9c6-4227-b113-bf9530a6aecb`。

## 唯一身份

| 字段 | 固定值 |
| --- | --- |
| order / 配置 | `37` / `configs/R127-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-rotationidentity-v1` |
| candidate | `r127-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-rotationidentity-9010` |
| round | `r127-small22k-seed20261029-scale085-pairlog-rotationidentity-24k-p4-9010-20261031` |
| readout | `native_avgpool_final_LN_dropout_linear_plus_age` |
| newprep | `technical-prep/r127-20261031-e62a` |

新round/root/future binding/clock仅用于本次新attempt，不恢复或重置旧轮。配置和生产接线归A，规划不共写。

## 科学问题与反例

直接对照为R107同seed29。唯一变化是 `job.training_rotation` 与 `candidate.training_rotation` 同为字符串 `identity-v1`，缺省None严格沿原数值路径。检验原训练旋转策略的净效果：移除rotate-expand及其重采样，是否在保留其他增强时改善原验证视图上的整体排序。收益未知，不把未见精确完成当作收益证据。

用户2026-10-31约17:55说明标签按“单张图可见征象”判定；这是科学输入，非独立赛方原文、标注验证或临床确认，不推出无噪声、患者关系已知、应去age或`.999`可达。实际旧路径是mirror→photometric→rotate-expand→整图缩放/放入512画布，不是random-resized-crop；不能以裁剪丢失征象立论。去旋转同时改变方向增强、expand后宽高/黑边和后续有效内容尺寸，不能称纯插值因果或已证像素/临床瓶颈，也不能恢复prepared512之前的信息。

最强反例是R104/R105降低scale下限曾两seed内部AUROC同向改善，不能假设更大内容尺寸必然更好；R122整组光度identity阴性也不支持普遍去增强。去旋转可能损失拍摄方向鲁棒性、强化姿态/设备捷径或过拟合，原±15度本来温和。对既有汇总、决定页和旧research的定向检索未见同R107合同仅rotation identity的完整结果，不等于全历史从无；R104/R105保留旋转且没有新增裁切，R122只关光度三因子，本项不是其分拆或R119局部读取筛查补考。[Pillow官方rotate文档](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate)仅支持BILINEAR/expand的API语义，不证明本数据收益；stable网页版本不替代Pod冻结版本，不升级依赖。

## 精确输入与保留合同

训练mirror及原Brightness/Contrast/Color照常执行；原位置仍消费第5个draw `uniform(-15,15)`。identity忽略该抽样角度，有效角度0，**完全跳过PIL rotate**；之后沿原scale `[.85,1]`、dx/dy三个draw，总8draw、顺序、sample-key和RNG后态不变。保持的是抽样值和随机状态，不是假定画布位置、像素或后续梯度轨迹相同。eval原像素/零增强RNG，不建立augmentation Random。

R107其余全部保持：公开Small `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b` fresh、seed20261029、fulltrain/stride2、原GAP/finalLN/dropout/linear、age enabledtrue及原AGEcontract；512/原prepared缓存/单图、G64P8/每rank8主图8视图/全局64主图64视图和448对；原pairlog六字段 `binary-pairwise-logistic-v1` / `binary_pairwise_logistic` / temperature1 / mean / float32 / global_batch及原WORLD可微gather。SD.1/drop.2、FP32/noTF32、backbone LR1e-4/head与age各1e-3、原WD/AdamW/globalclip1/EMA、固定9:1与原selector、800/max24000/patience4、绝对warmup/EMA1600不变。

明确 `training_photometric=None`、`gradient_clip_policy=None`、`attention_pool=None`、`stage3_residual=None`，不继承R122光度identity、R126角色裁剪、旧q读出或R120 sigmoid。training_views保持原缺省单视图；不改model/loss/sampler/gather/EMA，不加载训练checkpoint或N06/FLAIR。无新源、模型、原图扫描、cache、ROI、多视图或辅助loss。

## 最短实施与证据边界

- **B独占** `runtime/bundle/retrain/data.py` 与新增 `tests/test_training_rotation_policy.py`，只做上述默认兼容分支及两项synthetic像素/RNG窄验；不运行Pod/GPU/Git。
- **A独占其余** production/config/experiment91/两个engine/SCIENCE/tools必要绑定及第三窄fixture、统一动态guard验收、唯一Git与GPU。训练Dataset实收该field，原first_batch_dataset_policy记录实际消费，不能只登记SCIENCE。规划仅本页与todo入口；不同执行者不共写文件，不增Agent/子Agent。
- **三项必要窄证据**：①None对独立legacy像素oracle、总8draw和RNG后态精确兼容；②非对称、非正方形synthetic RGB对独立identity参考，原三光度仍应用、angle draw照耗但不调用rotate，后续scale/dx/dy与原seed一致；③eval同像素且零增强Random，加SCIENCE→实际Dataset/保存重载与eval-inference绑定。只做受影响项，不新拒绝矩阵或平台。
- 正常新正式包实际核原345张量/预期50c初态、参数组和空Adam；同state不证明输入policy，不能抄旧19/19为本轮通过。实际8rank/首成功记录原age tuple4、64primary/64view及真实Dataset的rotation identity/原None光度消费；尚未取得不填完成。不增图像/梯度probe、forward、控制模型、独立GPU预演或重跑已过loss/source/Gloo/cache大套件。

## 完整预算与新attempt

模型、输入batch形状、读取次数和编码次数不增，少一次PIL旋转也不保证端到端更快。满24000步及30次验证约3–3.5h仅旧Small日志量级估计，不是新benchmark；最终500MB与指定硬件100ms仍未知。保持6h训练/8h整轮/9hgrant、Small12GiB round cap、启动free20GiB/运行floor16GiB及全部warm/best online与EMA/last/previous/initial保护、300s保存、atomic/prune/monitor/failfast。不靠早停、降batch/精度/尺寸/步数、扩时或删保护救限。

A每个计算入口工具动态核原Pod UID/Ready/Python/effectiveCPU/本人entryTTL及互斥20%prep与80%训练整树预算，不默认120核。CPU20 allocation `a0fb90eb98615680367dd3591d1ff3e68596a138e17522d64427cc3bed734a08` / TTL `1793442754.9876158`仅历史证据；若过期，仅建必要future软件下一号，不改旧allocation/grant/clock，不新资源或续租。R126已safe归档不重查；R127正常实时核锁、设备、存储与身份，不能以agent idle代替remote状态。

必要Git实际同步及精确部署后，所有长准备/传输先完成，再以真实live观测执行原同链preflight→activation薄链，保持两处120s时效与attempt不可重复语义，不伪造时间戳或放宽门。新attempt只一次activation/唯一fresh；未执行不冒称正式clock或首成功。A必要证据齐即可按已授权链执行，无新增用户确认或文档等待门；不发信号、热改或重启旧包。

## 预冻判据与停止

完整工程有效selected内部5140 AUROC严格大于R107 **.9884405731067525**才是同seed科学正向；同时严格大于singlebest R120 **.9894773092062407**才由协调方另派固定seed20261027对R108 **.9872328538178018**。只过第一门为有限正向、无新best，阶段结束；未胜R107关闭本固定rotation identity。仅本次首seed授权，不自动seed27，不角度/scale/插值/其他几何或seed补考；AP/partial/难组/official/platform不救主判。

完整训练并按原selector冻结唯一checkpoint后，selected内部AUROC **>=.996**才完整official5849一次；不足正常completed并记 `not_run_by_policy/threshold_not_met/evaluated=false/metrics=null`，safe另实核。中途partial达阈不触发；工程未启动没有selected不能写threshold_not_met。此门独立于两科学门，无平台test CSV/提交。`.999`未达，阶段结束仍由协调方接续科学判断，不称总体完成或自动派下一方向。
