# R164：原生 DeiT III Small 公开初始化

存储状态：本轮checkpoint/initial已按用户三模型保留策略完成[精确清单清理](../storage/20261102-closed-rounds/README.md)；完整结果、指标、配置和原身份记录保留，以下checkpoint路径为历史来源。

**完整阴性收口。** [最终结果](R164-RESULT.json)：7200 success=attempt/0skip/eval9/bad4、正常早停，唯一EMA c5@4000内部AUROC `.980575346391472`、AP `.7480948827714725`、error_twice32264，低于R107与R120。固定原生CLS配方关闭/no seed27，R165无资格且未formal/GPU；不把这一整体配方阴性外推所有ViT或临床机制。官方不足`.996`未跑/指标null，总体`.999`未达。

[原终态](R164-COMPLETION.json)与[CPU观测](R164-CPU-COMPLETION.json)证明completed/exit0/signals空、owner原start匹配现Z、8rank退出、锁无holder/设备users空/scan denied0/8PCI busy0、actualsafe true；初次done时owner尚活跃的观测也保留。所选tensoraf18a9f9/blob2c29449e、freeze元数据2784979f，均仅转存既有JSON/guard，无新checkpoint加载、forward/官方或安全重扫。以下为历史启动记录。

[真实八rank初态与自然首成功](R164-START.json)：观测running450 success=attempt、0skip/eval0；177全训/5空AdamW、初态64556b75与formal一致，首自然25 loss `1.20843505859375`有限，WORLD8/G64P8/448pairs。8PCI与训练24–119/96核匹配，owner520173/start362992028和8rank身份匹配；GPU链已实测能跑，完整AUROC收益待结果。

运行Git `f06130548f33c63f10466988386afc656b7547d7`、tree `b4db655ecb749baa8a5df504428878222d91e895`，SCI cb7e6a49、SITE c53cb60e；[18/18唯一正常formal](R164-PREPARATION.json)51.154秒、精确部署18.745秒、一次activation/direct launch0。[CPU准备](../preparation/deit3-20261102/README.md)5目标/8编译与原6loss/WORLD按SHA复用，无独立GPU预演。原公开176/1000头严格加载后原生位置重采样到512，CLS/LayerScale/norm/age一次；公开384与比赛512分列，不宣称完整transitive库pin、临床机制或实测DCU峰值。

[真实交接间隔](R164-HANDOFF-GAP.json)：R162训练完成事件至dispatch1779.721秒、最晚设备1837.213秒、首自然25记录1875.983秒，均超过300秒；整轮ended至三项为1707.668/1765.160/1803.931秒。原因是后续公开来源、原生适配与CPU/Git准备完成偏晚，原proxy/来源回执/配置遗漏失败均保留。该轮未满足用户5分钟要求；未跳必要检查、重启旧轮或改clock/token/grant，不以单次设备证明冒充连续利用率。

唯一Git/GPU仍32e3；aa0f准备根0–22/23核、peer23/1核与训练96核互斥。当前r164-pod-context.env绑定start-3，原f061冻结不热改。最多24000成功更新、每800内部dev、连续4轮无提升早停；完整selected≥.996才官方5849一次。raw `.999`未达。后续[R165/R166固定准备](../../todo/auroc-999/R165-R166-PREPARATION.md)在当前训练期间进行，不预判R165资格或新GPU可用。
