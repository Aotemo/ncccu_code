# R141 DINOv2-S/14 当前排序合同接入计划

> **For agentic workers:** 使用 superpowers:executing-plans 在本会话执行；用户已明确必要最小修改直接实施，无需审批，不派生Agent。

**Goal:** 将已保存的官方自然图像DINOv2-S/14公开骨干接入现有R107排序训练链，以一个固定fresh实验检验不同公开表征。

**Architecture:** 复用既有`dinov2_model.py`、训练/保存/评估/监控/正式验收；补当前八rank科学身份及外部官方源码asset绑定。原N03冻结头和CLS+mean-patch读出保持，不开发新模型框架。

**Tech Stack:** 当前已核Pod Python3.12/PyTorch2.10/官方DINOv2源码7764ea0，FP32/SDPA MATH/XFORMERS_DISABLED=1，不安装依赖。

**Spec:** 本页科学值及`AGENTS.md`。采用用户2026-11-01持续推进授权；当前R139优先自然结束，任何新GPU等待其完整终态与actualsafe。

**已实际启动唯一R141，真实八rank初态与首自然成功完成。** R139完整双门false，R140未activation/未训练。R141用同步0c704d5、修复后250c CPU证据与新技术包，normalformal唯一18/18、精确部署、唯一activation/direct launch0；16:19:04.145001 UTC实查running50success=attempt/0skip/eval0，原生178总/177训/5组空AdamW/whole9b4、WORLD8/G64P8/首log25finite及实际资源匹配。见[启动实证](../../records/R141/README.md)。原3eef技术包未部署，字段修复/CPU重验/传输与formal导致本次接续超300秒如实保留；当前冻结0c不热改。下方“未formal/启动”为历史阶段，完整selector/双门/official仍未知。

**实际CPU准备已完成。** [三项窄验及9源compile](../../records/preparation/dinov2-current-20261101/README.md)保留各批真实失败和精确复用。生产公开fresh whole9b4、175公开/178总/177训/5组空Adam及504合成finite、八rank配置与真实online/EMA保存严格重建均已证；未GPU预演或新轮activation，normalformal与实际八rank仍待采用分支后原入口完成。

## 固定科学范围

- R140/order52仅为R139完整selected同时strict>R107`.9884405731067525`及>R120`.9894773092062407`后的局部对比度固定seed27；身份/初态自然变化外全部沿R139，匹配R108`.9872328538178018`。未合格不activation或训练。
- R141/order53为独立DINOv2-S/14、job seed20261029、504输入/512 prepared（patch14要求）、官方无register骨干；原N03 CLS384+mean patch384→dropout.2→固定head seed20261027 Normal(.01)/bias0及fit-only age保持。公开来源LVD142M，非医学专用权重；原始SHA b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9，来源code7764ea0，不加载N03训练产物。
- 其余沿R107：原9:1名单、scale[.85,1]/8draw、G64P8/8rank8/accum1、pairlog温度1/448对、AdamW backbone1e-4/head-age1e-3/WD与clip1、SD.1、24k成功上限/800完整dev/patience4/绝对1600warmup和EMA；6h训练/8h整轮/9hgrant、Small默认12GiB/floor16/free20等现资源政策保持。mask_token沿既有实现冻结，不入Adam；其余骨干可训。公开源+架构+必要504+原DINO读出作为整体比较，不称纯架构因果。
- 直接锚点R107与R120 singlebest，完整selected双门严格通过才另定固定seed27；否则关闭此精确合同，不扫DINO来源/尺寸/LR/头/seed。完整freeze≥.996才官方5849一次；不读官方选方向。旧N03当前dev `.979925`属不同8:2/BCE合同的负先验，R118 DINOv1 R50阴性亦保留，不承诺收益或raw.999可达。
- R139无资格则跳过R140；有资格先固定R140，再R141。R141工程准备若未闭合，不冒称可启动或跳过CPU校验。

## 文件与验证边界

| 范围 | 必要工作 |
| --- | --- |
| `configs/R140-SCIENCE.json`及`experiment91.py` | 固定条件seed27身份、官方门/输入policy一致；一个新身份目标，不重像素测试 |
| `configs/R141-SCIENCE.json`、`config.py`、`experiment91.py` | 固定DINO当前八rank/age/504/source/参数与空Adam身份；旧N03和所有旧profile保持 |
| `dinov2_model.py`、`models.py` | 从明确public provenance旁的已实核官方代码asset导入；严格公开backbone与原head/age，补既有public_initialization_receipt/descriptor；None/旧N03保留原路径 |
| `tools/activation/prepare.py`、`tools/validation/check_candidate.py` | 现入口补self_supervised kind、DINO实际结构/参数组/head与一次正常formal，读取已验来源小件，不建新owner/launcher |
| `single91.py` | 已保存job的实际504/512进入同一freeze/验证几何，旧512值不变；不改selector或官方一次算法 |
| `tests/test_candidate_binding.py`及新`tests/test_dinov2_current.py` | 两项DINO受影响正向目标：实际官方工厂strict来源/504 finite/组/空Adam；真实配置、保存EMA/online strict重建及几何/旧合同保持。仓库无既有DINOv2测试文件，故只新增这两个目标的一件测试；只用合成图像/fixture，不提前真实数据forward/GPU预演 |

外部官方源码与公开权重按原来源清单核字节后传入`exp-v3/assets`；只传源码来源与小回执进Git，权重不进Git。动态身份沿现a365 CPU allocation本人0–11/nice19/隐藏GPU，与R139训练24–119互斥；不续旧grant、改活跃source或重跑旧safe。

## 执行步骤

- [x] 条件身份及两个DINO边界的必要正向目标，Pod内实际RED；原始输出保留。
- [x] 最小接线后只复跑受影响目标，9源compile；原pairlog/Gloo/AdamW/clip/EMA算法按精确源码身份复用。
- [x] 同步GitHub并回读commit/树，干净技术包及一次normalformal完成；参数/whole与正式运行身份实证。
- [x] R139完整终态及actualsafe后独立fresh一次activation/direct launch，真实八rank初态与首自然成功已证；本次接续超300秒未隐去。

## 核查重点

官方源码asset需逐文件SHA及导入位置一致；mask_token冻结与Adam唯一覆盖一致；504几何贯通freeze/official/predict；旧N03四rank配置不被全局放宽；原8draw/age和训练游标不因模型构造或准备而改写。测试优先覆盖这些实际变化，不添加平台、全套回归或审批门。

官方[固定revision说明](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md)列出无register ViT-S/14为21M参数、LVD142M无标签预训练及相应公开骨干；这些是来源/结构依据，不是本任务效果证据。新资产与CPU目标现在已有上方实际证据，formal或新GPU尚未执行。
