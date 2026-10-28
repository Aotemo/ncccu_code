# R93 运行实证

本轮已完成，原生终态及25项对账通过：6400次成功更新、6400次尝试、0跳步，8周期×online/EMA共16条内部评估。完整官方5849仅原生一次，raw AUROC为0.9887607594491606，未达0.999，总体目标尚未完成。

| 数据范围 | 唯一模型/指标 | 结果 |
| --- | --- | --- |
| 内部dev5140 | 按预定规则选择c3 EMA@2400 | AUROC0.9866344126546825，AP0.8167243430095755，error_twice22200 |
| 官方5849（191P/5658N） | 上述冻结模型，FP32/identity/raw logit，一次比较 | AUROC0.9887607594491606，AP0.8424437871775934，error_twice24292 |

内部完整轨迹见 DEV-METRICS-INTERNAL.csv / INTERNAL-DEV-SUMMARY.json；官方不参与选点或下一轮科学决策。FINAL-RECEIPT.json来自Pod原件，SHA为`e1d0e20895f3a8047f2bd4b8477875f85b36f32ab013e6994d4b95d9864cf205`：核了原生done、全部rank exit0、EMA copy1600/blend4800、唯一checkpoint、canonical一次预约、现有预测SHA/5849顺序及exact AUROC。owner已inactive且OS encoded exit0（保留Z事实），8rank均inactive、无GPU设备使用者、8卡idle、原实例锁inode不变且已释放。

纯训练1867.606秒，内部评估147.717秒，保存46.335秒；不把部署调试计入训练。FOUR-VS-EIGHT-SPEED.json保存中途5525更新的同口径比较：R92四卡0.516815秒/update，本轮约0.291897秒/update，约1.77倍吞吐；来源、LN及CPU资源条件不同，不能称严格同配置GPU基准。官方阶段耗时超过历史参考，未捕获足以定位慢阶段的动态证据，原因保留unknown。

- 冻结实现：`80e06f31c33bc6256f6b622e6ed162273d33f1ba`，Git tree `6593d39e4bcc55355fe673dba250ecda8f949fba`，Pod 实核 274 文件的集合、字节与 mode；见 DEPLOYMENT.json。
- 正式根：`/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r93-tiny22k-9010-20261028`；owner PID 2207，初始八 rank PID 2219–2226。实际初态八 rank 为公开 fresh，模型 tensor SHA `eba53feeafdf20bc86855a40648bc453358928d148078c6ae78b65f059846cc7`，Adam 状态条目 0；见 START-OBSERVATION.json。
- 科学合同：configs/R93-SCIENCE.json；固定 fit46256/dev5140，G64/P8，8×local8/accum1，FP32，800×8=6400 成功更新，warmup/EMA warm-copy1600。Tiny22K→1K 与 final LN1e-6 相比 R92 同时变化，不标作纯预训练来源消融。
- 准入：GPU-ADMISSION.json 真实 completed/exit0，50.692667722702026 秒，0 forward/0 update；实际八 PCI、原生初态、FP32 collective、native SO 映射及 CPU mask、owner/rank wait/reap 与退出均通过。此技术准入不证明训练吞吐或梯度，UPDATE-250.json 才证明正式训练已成功更新。
- 资源：集中 RESOURCE-POLICY 配置实时有效 120 CPU，验证池 24/训练池 96，互斥；训练 rank 各 11 核、control 8 核。A 观察与 B 下一轮 CPU 验证合计受验证池额度约束。SO 仅在 exp-v3，源 C 与产物及 CPU 构建回执均被 site pin；旧失败冻结包和回执保留。
- 一次性约束：ACTIVATION-EXIT.json、FORMAL-BINDING-AUDIT.json 记录唯一正式绑定；clock/token 不重置，不重新 activation/launch。内部 16 条评估选择唯一 checkpoint，完整官方 5849 只在训练/内部选点完成后原生一次比较，内部与官方分报。

以上 JSON/内部CSV均由已核验 Pod 原样普通字节传回，本机未执行解析、重哈希、模型或测试。选中模型tensor SHA为`7ed9a7d4597c5401b0b84ea30b389333f253b18343ab02071e5b47a1098e0093`；官方现有预测SHA为`c9402f3301b012fdfed3271bc679cafddb117492e79b4fc9b3ef84cab9857fff`。大权重、图像、缓存、样本级预测和大manifest留在exp-v3；冻结运行/clock/token/一次预约不改、不再启动。

