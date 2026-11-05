# F05 新一批已闭轮非 selected 清理

2026-11-05 用户授权按既有规则清理已归档且 actualsafe 闭轮的非 selected checkpoint，全部 selected、KEEP 三权重、原始数据/共享缓存、公开资产及其最小依赖永久保护。

对 R253–R260 的只读逐文件计划已封存并在删除前报 c607：60 个候选，逻辑 `41,014,234,533 B`、文件系统已分配 `41,014,253,568 B`；计划时共享盘可用 `39,142,293,504 B`，本批未发现跳过项。计划 SHA `971063cd58118263022a4c185fa895ca59da27e6856362b312fdd4e497aff79d`，库存 SHA `5712e0ebce63ffe809f2976cb53b8e4c0cf1db8faa2f952d7302e79d34672450`，原件位于 `exp-v3/technical-prep/takeover-20261104-08ac/F05-CLEAN-253-260-v1/`。

两 Pod 的 60 项 open-file/owner/权限核验均 safe、计数均为 0；A/B 回执 `a4d2a3b483325648b54eb68d5c0c029b370485d555e3abe69c7c975b56dcefe6` / `4a38770c215bb1fb90b1c945c1f55dee56b833940615a9db0843eef5e9e71d5f`。dry-run 核 60/60、未删任何文件，回执 `c9d240ba1b11d467c75f3487d1e74cf9ab9ffd26abe357fb5b8d36ed5622416f`。随后逐项删除 60 个非 selected 文件，保护对象前后签名一致、跳过 0；删除回执 `b521054fcd21b964c077a09572afe2082556548aeaca7870578fcfc58ecd1a94`。实际共享盘可用量 `38,844,497,920 → 74,709,991,424 B`，增加 `35,865,493,504 B`。

R251/R252/R261/R262 与其他闭轮需另核各自归档 actualsafe 证据及实际 selected，不能因本批通过推定自动可删。预计分配字节不是已验证物理释放量；共享盘 free 差值可能含并发活动。
