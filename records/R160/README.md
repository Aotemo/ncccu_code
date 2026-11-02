# R160：固定批内primal AUC margin1

04:34:12.248165 UTC[真实首成功](R160-START.json)：100 success=attempt、0skip；八rank、8PCI和96核mask匹配。345全训/5组初态空AdamW，whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`与formal/R107一致。首自然25更新loss `.9626380801200867` finite，WORLD8/G64P8/448pairs，每rank8图。此证据证明GPU链可运行，尚无完整AUROC收益结论。

运行Git `3ef3b015c9b2781be9a8a1db6e681d159f3f1986` / tree `8fd1a32e09a1c6d2d3b5a30a4357304d8e17bf3e`；SCI `d675741a5196beb27752804efa8eaa974878e06456c25240f546e101b1876e2e`，SITE `621849b29ff25dbebe40e0ce3833cb9eb14837df4d6d96e26fc7f75f46cd8119`。[7目标/5compile与两public fresh](../preparation/aucmargin-20261102/README.md)按精确SHA复用；[唯一正式验收](R160-PREPARATION.json)19/19、85.698秒，精确部署11.040秒、SHA `9e3f73a2fe97e2877fe0ebd6a4ab84b5b0f5d94d2bea3eb48a6e613ac3661a02`；一次activation/direct launch0，无GPU预演。

直接R107/原公开fe6b/seed29/RGB512/原增强与采样，仅固定损失为448 raw差值的`mean((d-mean(d))²)+relu(1+mean(d))²`；两类批内方差+均值间隔，不是逐pair squared-hinge、完整PESG或总体无偏目标。无可训练loss参数、teacher、历史训练权重、dev名单或额外forward。[保存的实际512与loss](R160-ACTUAL-GEOMETRY.json)匹配科学配置。

[真实交接间隔](R160-HANDOFF-GAP.json)未达到5分钟要求：R158训练完成事件至dispatch438.776秒、最晚设备证明494.108秒、首自然25记录550.650秒；整轮ended至dispatch373.811秒。下一轮Git回读/技术包未在旧训练结束前收齐，加上EOF后源码补传及必要formal/部署，均发生在交接阶段。未跳检查或重置旧clock/grant，连续DCU利用率未测。后续在训练前段完成Git/技术包，压缩传输，结束前及时跟进既有进度；正式验收仍沿现有入口，不加GPU预演。

唯一Git/GPU owner32e3，owner503162/start362128728；训练24–119/96核，aa0f准备根0–22/23核、peer23/1核互斥。当前`r160-pod-context.env`绑定START3，原冻结包保持。最多24000成功更新、每800内部dev、连续4次无提升早停、原EMA/唯一选点；完整selected≥.996才官方5849一次，raw .999未达，继续。
