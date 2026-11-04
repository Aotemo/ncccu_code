# R250/R251 草稿准备

统筹按前批条件批准 fresh R201 原 seed 与 fresh R172 新 seed `20261106`；[合同](../../../todo/auroc-999/R250-R251-FRESH-MEMBERS-20261104.md)。两 Pod 的公开 Small 权重绑定分别为 [A](BINDING-A.json) `0e57fc9dc7eb5023efcc2ca66a1f067d0f7b3b417c9708f2b341b4fc70ac56bc` 和 [B](BINDING-B.json) `9259b8ffe926603890c90d22713efd69b1f01f89adf6a4856e09cadaf4badfe4`；R250 原科学值不变，R251 仅改变 seed 和必要身份。

[R251 CPU 初始化推导](R251-INITIAL-CPU.json) `12a770efc023498dcb7586d01d7f6d7bbb784c859338d9dce4aa3bb90879de42` 从已核公开源经原 factory 得 tensor `16f453c2ab3c6864703c826831050a694623f20ff77a52924314d4bcde811809`，无 GPU 初始化或前向。旧配置与新 seed 的初次窄检查 `1ae5f99c8ec9e63ae48cdcafb95e7dbd468d11cbeebe6b8363e4f0eb46e7f5f8`；四份实际配置和旧、新身份/官方门的最终草稿窄检查 `7bfc9dab85d991f3749fbc4ee8a721fcd3ce2a270fdb78d12e098364d3c06de4`。正式 CPU、Git 精确部署、容量和实际 GPU 均待。

已推送/API 回读 `2742968868e9c3e5cffd4ae35dbf5c5208eb320c`；[增量](SOURCE-TRANSFER.json) `1f2e765897c30df7f9abcec20b58c4a45de9808e0c83f2162b1263b11d6edf46` 从独立 `631ce201` 快照更新，两个 Pod 的原全树校验器对3134文件预验证通过：[A](TREE-VERIFY-A.json) `6578659ff58f5bb887102cd0f8cff5251c7a964c9e193800b79c2fd4fa4eda08`，[B](TREE-VERIFY-B.json) `f92f9b277875e9a4a6543c2e7131752e2c848c993ecf29b334c70d91d8906580`。预验证不等于正式 18/18 或四变体部署，后者在 R248/R249 训练期间完成。

五项[容量与对侧身份隔离模拟](CAPACITY-TEST.json) `86684e0f9068cd13ae65f09a808c0634d47f9c95144d7944523e0c394bcfe4bd` 通过；[正式准备编排语法](FORMAL-ENTRY-CHECK.json) `d95ab98e21577326597129b63e1a6ce7e22b3feda072ea91033eb4413a378cc9` 和[自然首成功观察语法](OBSERVE-CHECK.json) `81c5b783a324a8486b6d44786ce143945d58f91f69be646a32c69509a89c0408` 检查通过，均不代表真实 GPU 运行或现场容量准入。

统筹为 R247 终态后的容量后备提前要求 R250/A 正式准备：[R250/A](R250-A-PREPARATION.json) 已通过18/18 CPU `eb5b2b222e29794088fd1b43d907ed6fd0ebf56f9f9c821ff8968708c965b51e`、精确部署 `0a719b342a3083a9e961432a04bd6cc825776256678b76fd1e7ffe7393a4ee7f`、SITE `c2be7b76a9e8226afb78d1928faae4b7c88e831886a44c848344e52def933bb9`，未 activation。跨批后备容量/身份[隔离测试](../fresh-members-248-249-20261104/CROSSBATCH-WINDOW-TEST.json) `45a5fe3ac2783b00b417e795602ddbb7d1abfe0b41dd5b81687c8c40615724f2` 为模拟，不代替现场安全和容量核验。
