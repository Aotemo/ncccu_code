# R201 后续公开 ViT-B 骨干低学习率对照

11-03 03:08已完成[4必要目标/3编译与两actual public fresh及两technical包](../../records/preparation/vitb-low-lr-20261103/README.md)。R202按3906410原包保持；R203首次技术包暴露run_name135>128，原失败保留，无formal/clock/GPU。只缩短未启动round名为118字符并同步其精确存储键；仅重跑/重编译一个受影响存储测试，三个候选目标与两fresh按相同SHA复用。57348fb实际Git回读及新未启动namefix技术包guard0，仍未normal/新clock/GPU。32e3唯一Git/GPU、53ab准备0–11与训练24–119互斥；不恢复/派生Agent，不新申请资源或例行审批。现役R199/82add93及既定R200/R201/8635b53冻结不改，不因本准备阻塞接续。

备选继续改变Small目标、增加新公开架构、或为已跑过的原生ViT-B只降低骨干LR。前两项已有固定队列/更多改动，本次选第三项：R191原Base-CLS完整AUROC `.9666919930498946`，其原backbone LR1e-4不证明对该表示合适，也不证明特征漂移是失败机制。发布者[Model Card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)强调冻结特征的迁移用途，建议谨慎微调；它没有给本任务LR1e-5的证据。本实验仍全训，仅固定十倍较低骨干LR，是自己的低扰动假设，不是冻结特征/作者微调配方复现，不扫描LR，不声称临床原因或收益。

R202/order115仅在完整有效R201唯一内部selected严格胜R107 `.9884405731067525`与R120 `.9894773092062407`且actualsafe后，同Small/fe6/512/原优化与固定wide2-peakgrad loss，seed20261027独立公开fresh复核；否则只技术准备。R203/order116独立直接R191原native ViT-B/DINOv3 public c6a5fb7d/seed20261029/512/FP32/ImageNet/原生CLS/原logistic/G64P8/default AdamW，仅job.backbone_lr从1e-4降为1e-5，head/age1e-3、WD1e-4、24k余弦/1600 warmup/EMA保持。不叠wide loss、WD=.05、gamma、8k、其他policy或旧checkpoint。公源/源图像语义、LayerScale/RoPE、注册token、末端Dropout/Linear/age及参数/分组/forward均保持R191。

拟永久范围：两SCI、原experiment91注册/严格来源/仅新profile LR身份/互斥；原ROUND-STORAGE-POLICY只给R203精确新round追加16GiB，与R191相同工程余量，默认12/旧例外/free16保持，原runtime预算函数不改。原candidate测试与storage policy函数测试窄验，todo/records。原model/config/engine/DDP/data/loss/EMA/保存/推理及来源加载不改，无新依赖/平台/下载/独立GPU预演。准备助手从已修7fc6b3dd版本扩必要身份/证据参数，继续读取冻结policy，不恢复按SCI编号重复写容量预期。

- [x] 在实核Pod由原R201/R191生成两SCI，目标身份及R203存储消费先RED；必要最小注册/精确单round policy后GREEN，原job仅seed或骨干LR变化、来源/结构/9:1/日程/早停/官方门保持，保存job与错误叠加拒绝。
- [x] 只核新受影响目标/编译；原LR/AdamW算法和模型/P8/EMA/cold按未变SHA复用。两实际公开fresh：R202原345全训/5空组/whole91507；R203原165全训/5空组/wholec349，实际骨干组initial_lr1e-5、head/age1e-3、默认betas/eps/WD及原RoPE非持久buffer语义。无生产图像forward，不把CPU当DCU峰值或吞吐保证。
- [x] 原storage预算函数按真实冻结policy做有界size/free fixture，R20316、R20212、旧16/默认保持，超cap及free15拒；旧32及全部原policy字段按相等核对，不实际写大文件。
- [x] 小回执、Git同步实际回读/两technical包已完成，R202为3906410、R203短名新stage为57348fb，原失败包及R202保持。

后续正常CPU/精确部署仍于前轮可行收尾前一次完成，actualsafe后唯一owner沿原入口串行接续；未执行前不称已通过，旧clock/token/grant及当前冻结包不改。

标签1=青光眼且仅据眼底照片，0仅任务阴性；不改标签或dev名单权重。24k上限/每800评估/p4早停/EMA1600/固定9:1及唯一内部AUROC选点保持，完整内部selected≥.996才官方一次，raw .999未达。
