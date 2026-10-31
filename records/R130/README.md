# R130：用户主动暂停

R130 已按用户即时指令停止。冻结运行在 2026-11-01T01:43:49.895523+08:00 实核为安全退出；停止时成功/尝试更新均为 1503，跳过 0，已评估 1 个周期。训练状态为 `interrupted`，原因为 `owner_stop_requested`。

原训练 supervisor 收到一次定向 SIGTERM 后，8 个 rank 在安全边界退出。原控制器把非零中断码 130 归为 `done.failed`；这不是早停、完整实验失败或科学阴性。未生成 TRAIN_RESULT、ROUND_RESULT、唯一冻结选点，也未进行官方评估；R107/R120 两项科学门均未判定。

提交索引位于 exp-v3 下 `rounds/r130-small22k-seed20261029-scale085-pairlog-layerdecay08-24k-p4-9010-20261101/outputs/r130-small22k-seed20261029-scale085-pairlog-layerdecay08-24k-p4-9010-20261101-8c-convnext_small-s20261029-b64-p8-fp32/convnext-small/checkpoints/index.json`。`last` 为 `checkpoint-806c3832c05e487d88c938245ac1d3e2.pt`，索引记录 SHA-256 为 `08ac68114696b7bd698fb8e70658f7cfbeecf994088a023a115b7aa0591eb377`；文件及 sidecar 均存在且相互一致。previous、best、initial 和保护引用均保留。本次未读取或重新哈希大 checkpoint；代码保存合同包含模型、Adam、EMA、分布式 RNG 和游标，但这次未重新加载逐字段验证。

原 owner 为 Z 状态（不等于已回收），supervisor 与 8 个 rank 已不在；锁和设备用户为空、扫描拒绝 0、8 张卡空闲。实例未关闭，冻结源码、clock、token、checkpoint、source 和 cache 未改动。底层有显式恢复代码，但它要求原绝对截止时间一致；原时钟与 grant 未暂停，**不能据此保证跨设备直接续训**。待用户在另一设备明确恢复后，再核合法接续方案；当前不得启动本轮或新轮。

运行 Git：`c8c610dd5ee42e9de99473a83c15ae14b8189178`；源/正式包、首成功及停止证据的原小件 SHA 分别见 `R130-RESULT.json`。本轮没有完整 AUROC；既有单次最佳 R120 AUROC 0.9894773092062407，总目标 0.999 尚未达到。
