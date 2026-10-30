# R117：Swin V1 Base 公开 22K→1K 配对

固定 9:1 内部验证的唯一选点是 **EMA、cycle 7、成功更新 5600**：整体 AUROC `0.9850076641408516`，AP `0.805972999257521`。训练在 8800/8800 次尝试与成功更新、0 次跳过、11 次成对内部评估后依 patience 4 正常早停。该选点未严格超过预先固定的 R107 Small/同 seed 整体 AUROC `0.9884405731067525`，因此不进行 R117 seed 27 复核。此结论只来自完整内部结果。

内部选点低于用户规定的 `0.996` 门槛，完整冻结后按版本化政策未运行官方 5849 评估；官方状态为 `not_run_by_policy / threshold_not_met`，指标为空。2026-10-31 04:00:55 北京时间只读交接核到 round `completed`、原 owner 为同一 PID/start 的退出态、8 rank 全部退出、训练锁无持有、设备用户空且 8 卡 busy 0，资源 `safe=true`。

源码冻结为 Git `2429b9643632061668bc5b8c1d3c12e94198b5b0`。完整原始训练、冻结和安全回执保存在 exp-v3 对应 R117 round 与独立 technical-prep；这些原件含运行 token，不纳入 Git。内部 selected checkpoint tensor SHA 为 `b7822ab843eeb81827bd5fad1f8377a397d46447598db06dbea31e3a891d9319`；安全交接小回执 SHA 为 `96aad795d1c15e67a85e7935d375f12dca92c417cb045252ff7e4af54a1c6d07`。
