# R164：原生 DeiT III Small 公开初始化

[真实八rank初态与自然首成功](R164-START.json)：观测running450 success=attempt、0skip/eval0；177全训/5空AdamW、初态64556b75与formal一致，首自然25 loss `1.20843505859375`有限，WORLD8/G64P8/448pairs。8PCI与训练24–119/96核匹配，owner520173/start362992028和8rank身份匹配；GPU链已实测能跑，完整AUROC收益待结果。

运行Git `f06130548f33c63f10466988386afc656b7547d7`、tree `b4db655ecb749baa8a5df504428878222d91e895`，SCI cb7e6a49、SITE c53cb60e；[18/18唯一正常formal](R164-PREPARATION.json)51.154秒、精确部署18.745秒、一次activation/direct launch0。[CPU准备](../preparation/deit3-20261102/README.md)5目标/8编译与原6loss/WORLD按SHA复用，无独立GPU预演。原公开176/1000头严格加载后原生位置重采样到512，CLS/LayerScale/norm/age一次；公开384与比赛512分列，不宣称完整transitive库pin、临床机制或实测DCU峰值。

[真实交接间隔](R164-HANDOFF-GAP.json)：R162训练完成事件至dispatch1779.721秒、最晚设备1837.213秒、首自然25记录1875.983秒，均超过300秒；整轮ended至三项为1707.668/1765.160/1803.931秒。原因是后续公开来源、原生适配与CPU/Git准备完成偏晚，原proxy/来源回执/配置遗漏失败均保留。该轮未满足用户5分钟要求；未跳必要检查、重启旧轮或改clock/token/grant，不以单次设备证明冒充连续利用率。

唯一Git/GPU仍32e3；aa0f准备根0–22/23核、peer23/1核与训练96核互斥。当前r164-pod-context.env绑定start-3，原f061冻结不热改。最多24000成功更新、每800内部dev、连续4轮无提升早停；完整selected≥.996才官方5849一次。raw `.999`未达。后续[R165/R166固定准备](../../todo/auroc-999/R165-R166-PREPARATION.md)在当前训练期间进行，不预判R165资格或新GPU可用。
