# R236R2 闭轮 checkpoint 清理

完整 [R236R2 结果](../../R236R2/R236R2-RESULT.json)039c2ec2 已随 `368b95dcbececa64287b759fb513db824741134a` 推送并 API 回读、确认无后续模型前向依赖；当前 R238 公开八 rank 和自然首成功后，沿已有精确清单流程执行。未重查旧 actualsafe。

[清单](PLAN-SUMMARY.json) f6a163da / CPU118faa7d 只含 R236R2 `outputs/*/*/{checkpoints,initial-checkpoint}/checkpoint-*.pt` 的 9 个真实文件，无跳过或拒读。结果以已推送 Git blob e43985d7 核对，KEEP-MODELS blob 1db4b06c / SHA4cb230ed 保持；活跃 R238、新准备包、R120 提交及其依赖、三永久模型、原数据/缓存/公开初始化受保护。

[删除](DELETE-RECEIPT.json) e1e5538c / CPU14db2bc1 完成 9 个，逻辑 6,733,347,211 bytes、删除前 allocated 6,733,350,400 bytes，无跳过、错误、训练信号或实例锁。只用真实 stat 和 sidecar SHA/声明 blob SHA，不加载或重哈希大权重；sidecar/index/lock/一次性标记、selected CSV/precision、freeze、完整适配结果等清单外文件保留。

[终检](FINAL-VERIFICATION.json) / [CPU guard](FINAL-CPU-RECEIPT.json) fe1df9f4 通过：清单文件不存在、sidecar SHA 原值一致，受保护文件 stat 和小文件 SHA 前后均为23cc1b0f。free 瞬时 31,521,243,136 bytes，不等于后端物理回收保证或下一次准入豁免；后续激活仍调用原严格 >20 GiB 检查。

完整清单、inventory、journal 和保护快照保留在 `exp-v3/technical-prep/fit-label-contrastive-20261104-4e0c/closed-r236r2-storage`。CPU3 仍 nice19、0–23 核、隐藏 GPU，与 R238 的24–119核互斥。
