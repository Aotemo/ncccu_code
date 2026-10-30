# R116：ConvNeXt Base 公开 22K→1K 配对

固定 9:1 内部验证的唯一选点是 **EMA、cycle 4、成功更新 3200**：整体 AUROC `0.987677169289011`，AP `0.8222910002960596`。训练在 6400/6400 次尝试与成功更新、0 次跳过、8 次成对内部评估后依 patience 4 正常早停；训练组件耗时 4079.423 秒。该选点未严格超过预先固定的 R107 Small/同 seed 整体 AUROC `0.9884405731067525`，因此不进行 R116 seed 27 复核。此结论只来自完整内部结果。

R116 已绑定的原一次官方 5849 评估自然完成：AUROC `0.9891096145197737`，AP `0.8452081777306721`；该值单列，不参与选点或继续/停止决定。官方组件耗时 770.905 秒。2026-10-31 02:15:56 北京时间的只读交接核到 round `completed`、原 owner 为同一 PID/start 的退出态、8 rank 全部退出、训练锁无持有、设备用户空且 8 卡 busy 0，资源 `safe=true`。

源码冻结为 Git `7ba90d5686eb0d30efc56089e1f134119fef96cf`。完整原始训练、冻结、官方和安全回执保存在 exp-v3 对应 R116 round 与独立 technical-prep；这些原件含运行 token，不纳入 Git。内部 selected 的 checkpoint tensor SHA 为 `ca285aa69ddcf887c040fa79927b4835e5fef4a7d38c59a00173f0c688ae622d`；安全交接小回执 SHA 为 `a6313751d68e1ffe5db3e7f0ac3eb4865fac9fd78ce109f320563517b32dd7df`。
