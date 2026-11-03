# 结构审阅与 fit-only 适配实施计划

**Goal:** 先核实封存难例在原图和真实512输入中保留的结构，再检验一个新增学习信号的fit-only适配；完整内部AUROC主判，附AP、原固定20/其余147的修复及新增错序，晋级后固定第二seed。用户本次目标AUROC≥.995且AP>.9；长期raw .999目标仍未证明。

**Architecture / Spec:** 本页落实本次用户给定顺序及“你自己决定”。复用原 `persistent_errors.py` 已封存组、生产 `prepare_image`/`ImageDataset._image_tensor` 和现有SSL→原监督链，不另建训练、保存、评估或监控。当前R221R2/c732667冻结运行不改。R222仅原双门通过后的seed27条件分支；R223既有代码/CPU证据保留为未启动备选，先完成本任务，不自动启动它。

**Tech Stack:** 现有Python3.12/Pillow/NumPy/torch；一切解析、图像处理、哈希和验证只在实时核验的现有Pod、nice19及隐藏GPU的准备池。

用户随后明确任务只有图片和CSV、其余由本会话完成。不再索要额外临床资料或人工审阅者；不假设存在患者/双眼/盘沿标注。仅按现有图像自建明确记录的技术结构审阅口径，标签无法从图像可靠判读时记uncertain。原要求中“熟悉原标注标准者”的独立审阅资格仍没有证据，不能以本会话辅助观察冒充。

## 身份与执行

11-03当前Paseo self `srv_KqW7XmJdXxGJ`，仅活动会话`4e0ce707-5c70-493a-8e54-796363eb740f`。不恢复旧32e3/a4、不派生Agent；本会话承接Git/准备及后续串行GPU监督，R221R2原owner进程和冻结site不改。实际Pod UID仍51fe3daf；原owner780575/start375200048和八rank出生身份、训练mask已活核，旧CPU executor lease空闲。新准备allocation `technical-prep/goal-20261103-4e0c/CPU-ALLOCATION.json` SHA e1fcfa50d322222ed401b17b6e13d080c3dfc030cbf828a3156231c6df82596d：实时有效120核，准备0–11/12核、12–23保留不执行，训练24–119/96核不变；不借用旧executor身份。新probe9a477c7d exit0，Python3.12.7/torch2.10/numpy1.26/Pillow12.3、GPU未初始化。首SSH直传因登录节点/Pod目录视图不同被拒，bootstrap缺文件未运行；经kubectl stdin传输后新allocation成功。首次observer误引用不存在的long-ema路径失败42751a7f，随后使用原真实eva02路径成功086fc408，3400/0skip/c4、八rank活跃；失败不重启训练。

## Task 1 — 隐藏标签的结构材料与先封存后揭盲

用户随后明确“当前会话独占24核份额”。必要最小边界增加`tools/environment/cpu_budget.py`及原`tests/test_cpu_budget.py`：只使既有planning/validation支持显式1或2位执行者，资源比例和互斥核验不变；先真实单owner RED，再原窄检查。新建exclusive allocation与新guard，不改e1fc、7c或GPU冻结。

实际单owner旧实现ValueError已观（e4ed0c94/exit1），最小两处校验调整后原窄文件12/12通过（6a4b3588/exit0，0.095秒）。实核120有效核，exclusive allocation SHA7588546c047c651a837d6dde2a6d400988d6842cb452a9091906a285c62226bd、新guard05f3b6b7；当前会话独占0–23/24核，无另一准备执行者。旧两CPU lease空闲后才生成，原GPU owner+八rank仍相同出生身份、训练24–119/96核及site不变。

Files: `tools/diagnostics/structure_review.py`、`tests/test_structure_review.py`、本页和`records/analysis/20261103-structure-review/`必要汇总；图像/名单/逐例评语只在exp-v3。

- [x] 先写窄测试，真实Pod RED：公开包不含名称/标签/预测/分组；原图SHA错拒绝；实际512视图一致；缺例/重复/改动图像拒绝，只有完整盲态评语先封存后才揭盲。
- [x] 最小实现 `prepare_packet` / `seal_review`，复用原固定20正例及原8+8随机对照，共36例，固定shuffle20261103。原组按原四锚点定义恢复并核原质量，不用新结果重选。保留原图原生像素及512输入，可手工定位后只作审阅局部放大，绝不将坐标输入训练。
- [x] 本会话在未看mapping时逐例查看；记录视盘完整性、盘沿/杯盘边界和邻近结构的可辨性、原图→512损失与不确定原因，评语SHA先封存再聚合。
- [x] 身份必须写`AI辅助结构审阅`，并非临床专家或熟悉原标注标准者；原逐例判读标准缺失，标签一致性保持uncertain，不杜撰专家、独立复核或错标结论。视盘可见≠疾病征象完整；原始信息不清与缩图损失分开。

实际4/4测试d124ee3c，36原图/缓存一致c3aed325；18全图展示和12原生局部展示均逐例查看，手工坐标仅展示。评语dfaa1d8b先封存后揭盲，最终addea60d exit0；[真实结果69c75f90](../../records/analysis/20261103-structure-review/README.md)：固定20视盘20/20均完整，3/20盘沿技术可辨性下降、3/20原生也不可靠；对照15/16完整、1例原始取景已出框，无新增出框；36标签均uncertain。细节损失真实，但临床决定性、标签一致性、唯一瓶颈未证。流式导回截断和manifest误自哈希首次失败及定向诊断保留，图片未改。原生路线须先在fit侧验证真实定位，不能用本dev坐标训练。

## Task 2 — 一个补充空间学习信号的候选

优先考察固定mask的局部RGB重建，再原R107监督合同：相对R115全局两视图不变性/方差/协方差，新增被遮挡位置的空间重建目标，而非延长旧VICReg。RETFound提供领域masked重建的研究依据（[论文](https://www.nature.com/articles/s41586-023-06555-x)），SimMIM提供轻量空间回归头依据（[论文](https://arxiv.org/abs/2111.09886)）；本项目小样本CNN适配不等于复现其模型、规模或收益。

Ruling（审阅后）：预登记R224/order139，条件R225/order140 seed27；只检验“原公开Small在fit-only遮挡空间RGB重建后，原R107监督合同是否改善完整dev排序”。保持1600成功适配更新，避免把延长R115当新信号；G64、每rank8、单视图、原增强/512/ImageNet，从16×16个32像素块中每图固定遮154块，归一化零遮挡，重建同一增强的RGB归一化输入，仅masked像素L1均值。原features+finalLN共享，轻量768→3072 1×1+PixelShuffle32头只用于适配、之后丢弃；无teacher/医学预训练/额外任务图像。复用R115均匀fit采样及首视图seed流、原适配LR/WD/clip/warmup160/cosine1600，监督空Adam从头重建、原head/age不动。它不增加512外的信息，不声称解决已见细节损失；CNN密集遮挡与RETFound/SimMIM原结构和规模不同，阴性同样关闭固定配方。

精确源码边界：小改现有`runtime/bundle/retrain/ssl_stage.py`复用同owner整段与派发，仅加不可缺的`runtime/bundle/retrain/masked_reconstruction.py`空间头/无泄漏mask及loss；原`runtime/r115_ssl_gate.py`读取配方声明头几何，保留旧VICReg门；`runtime/bundle/retrain/experiment91.py`补两绑定和候选/固定合同，`configs/R224-SCIENCE.json`与`R225-SCIENCE.json`独立身份；新`tests/test_masked_reconstruction.py`/`tests/test_masked_reconstruction_candidate.py`及旧`tests/test_ssl_stage.py`必要目标。现有prepare/check_candidate、训练launcher/distributed/save/inference/observer均复用，实核无新增必要点前不扩范围。

- [ ] 结合Task1证据固定单一可证伪问题及配方，不以AI审阅宣称标签一致已获证；若512确有决定性信息缺失，先评估可靠定位，不复活亮区裁剪/四角窗。
- [ ] 明确需改的现有SSL/初态/CPU消费点及精确文件边界后，先必要RED再最小实现；fit-only、无dev/官方/标签/age损失、公开fresh头/age保持、监督空优化器重建、原G64P8/LR/EMA/24k/800/patience4不变。
- [ ] 必要CPU/现有正式验证、Git同步及精确部署在当前训练期间推进；当前运行安全终态后仅唯一owner沿现有入口接续，不加GPU预演/旧clock重置。若准备未完成而超5分钟，记录真实原因与间隙，不虚报可运行。

## Task 3 — 完整结果与收益/代价

- [ ] 完整selected、实际成本、AP及固定20/其余147的修复/新增错序一并归档；仅严格超过R107 .9884405731067525和R120 .9894773092062407后做固定seed20261027，不把单次高分升级基线。
- [ ] 官方5849仍按现有完整selected≥.996门后一次；不得用官方调参，不能将subset/临时best当完成。保留目标active，直到要求逐项真实达成。

Ruling: 用户将审阅安排交本会话决定，先做明确标识的AI辅助盲态结构复核并记录资格缺口；不会用该结果冒充专家一致性。既有授权要求必要最小变更直接执行，无额外设计审批或新Agent。旧未提交fit-census/实验复盘草稿不混入本任务。
