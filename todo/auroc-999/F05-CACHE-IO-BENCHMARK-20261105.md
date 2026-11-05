# F05 512 缓存读取小基准

按 c607 2026-11-05 指令，仅在不延误五折准备、GPU activation、评估和清理的准备池空档执行；先基准，不复制整套缓存或改训练绑定。

事前固定原 9:1 manifest 的 fit 索引按原 manifest 顺序前 256 项，不按标签、图像内容或困难度挑选；沿现有 `PreparedImageCache` 精确定位同一 256 个 512 `.npy` 数组及其 sidecar，缺失或身份不符记失败/unknown，不替换样本。只复制这 256 对到本 Pod 独立 `/tmp/technical-prep/<UID>/...` 临时基准目录，并逐件校验 sidecar 声明的 array SHA 与 NFS/本地字节一致；不改原缓存或读取官方数据。

同一组分别计时：NFS 读取并完成相同 SHA、逐文件 `posix_fadvise(DONTNEED)` 后的 NFS 客户端尽力冷读并 SHA、复制后的 `/tmp` 读取并 SHA。系统/服务器缓存不可由本任务可靠清空，因此标为“客户端尽力冷读”，记录操作是否受支持；不得称端到端真实冷盘，也不清全机缓存。报告完整耗时、总字节、吞吐及 CPU 时间，明确本地复制成本未摊销、运行时原图 fingerprint 仍发 NFS stat，操作系统页缓存和批次大小会影响外推。计时顺序固定为 NFS 基线、NFS 尽力冷、复制、`/tmp`；无 GPU/模型前向或训练。

最小文件范围为 `tools/diagnostics/f05_cache_benchmark.py` 及对应窄测试；真实样本路径与逐文件散列只留 exp-v3 私有回执，Git 仅聚合值、输入身份、代码和小型回执。测后按实际收益、复制成本、容量和五折启动优先级决定是否复制全套；证据不足时保留 unknown，不把估算当成已提速。
