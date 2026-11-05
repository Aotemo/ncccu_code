# F05：本地 512 缓存与加载并发

2026-11-05 用户经 c607 指示：R264 训练期间只读定位 loader、等待与 I/O，不修改现役；为五折及其后新轮把现有 512 缓存按 Pod 分别复制到本地盘，并在隔离测试后提高 loader 并发。此决定覆盖先前“仅做小基准、不复制全套”的限制。最终不得改变样本、变换、采样、模型、优化器、更新或官方隔离等科学值；worker 数与预取属于新轮版本化资源执行策略，仍须检查可复现性，且不得热改已运行 R264。

## 原缓存复制与验证

源为现有共享 `prepared512-v2-pillow12.3.0/train/sealed`，按固定原 9:1 manifest 的全部 51,396 train 行确定键和完整 namespace（含任务内 fit 和 dev，不含官方 valid）；源 manifest SHA `7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377`。每个 Pod 写入该 Pod UID 绑定的 `/tmp/technical-prep/<UID>/...` 新目录，不能覆盖共享缓存，也不能跨 Pod 复用本地盘。每行先核 sidecar 中源路径、源 SHA、准备策略/数据根与数组 SHA，限速读 NFS；将数组和原 sidecar 原样写临时文件、fsync 后原子提交，独立重读并核对本地 SHA。重复键、缺失、非普通文件、hash 不符、源变动、Pod UID/挂载变化或容量不足均记失败/unknown；无完整通过回执不得供训练。记录实际文件数、字节、逐文件 SHA 清单及目录身份；样本级清单只留 exp-v3 或本地盘，Git 只留汇总与小回执。历史 512 数组约 31,286,003,712 B/Pod，现有 1.3 TB 级本地余量仅为现场线索，复制前及过程中仍须实核，保留 ≥20 GiB。

两 Pod 可并行，但初始每 Pod 最多 4 个准备线程、总读速上限 24 MiB/s，CPU 分别限定已核验的 24 核准备池并 nice 19、隐藏设备；按每 64 对的读取 p95 和 Pod D 状态比例自适应：D 比例 ≥0.2 或 p95 ≥1 秒时减半速率/并发，重压 D ≥0.35 或 p95 ≥3 秒时暂停 30 秒后复查；低压连续 3 批逐步恢复且不超过初始上限。频繁异常或缺失不能自动跳过，必要时暂停，避免影响现役训练和 activation。复制、完整核验与 GPU 接续在同一 Pod 排队；无中断热改。

## 资源策略与实测

只读诊断核实 R264 真实 rank/loader 子进程数、状态及 D 状态，读取原 status 的累计 `data_wait_seconds` / `training_seconds`、更新数及 NFS 小批次读取耗时，区别一次快照、训练/评估阶段和整个 cgroup。当前旧策略 `loader_workers` 上限为 1，实际 worker 数需按运行证明；不凭单次 GPU 忙碌度确定瓶颈。

为新轮在独立资源配置中登记更高每 rank loader 数和 prefetch，保证八 rank+owner+loader 位于实时有效 CPU 的 80% 训练 mask 且不与 20% 准备 mask 交叠。先以固定输入、固定 sampler 顺序/随机种子在 CPU/隔离通道比对缓存字节、样本序列及数据加载吞吐，确认内存和进程数上限；性能报告含观察时间、批次数、实际均值/尾延迟，未测前不许诺提速。仅在源码已推送、Pod 本地副本通过逐件 SHA、正式 CPU/精确部署和现场容量检查后供未启动的新轮使用。

最小拟改范围：`configs/F05-PREPARED512-LOCAL.json`、`tools/diagnostics/f05_copy_cache.py` 及测试，现有 `runtime/bundle/retrain/site_binding.py`、`parallel_fresh_train.py`、`runtime/run_training.py` 和必要的资源配置/准备工具；具体新增在实现前登记。复用既有 cache 验证、CPU guard、GPU owner 和训练入口，不另建调度平台。
