# R121：固定 pairwise sigmoid 的 seed 20261027 复核

R121 工程完整，6400/6400 次成功/尝试更新、跳过 0、内部评估 8 轮，连续 4 轮未提升而早停。预定选择器唯一选中 EMA cycle 4、update 3200：内部 5140 dev AUROC `0.9869089490433972`、AP `0.8355471078657948`、error_twice `21744`，权重张量 SHA-256 `273cb362219e0c2569648a410aa0c9d9359af9d07f4f3f16676c9ae8db65c71f`。

预冻结的同 seed 参考 R108 AUROC 为 `0.9872328538178018`；R121 未严格胜出。此前 [R120 seed20261029](../R120/README.md) 的 `0.9894773092062407` 严格胜过 R107 `0.9884405731067525`，但两次固定 seed 的方向不一致。按预定规则停止这个固定 sigmoid surrogate 的主线升级，不选取单个较好 seed 或用 AP 改判。

R121 所选内部 AUROC 低于 `0.996`，官方 5849 按版本化政策记录 `not_run_by_policy / threshold_not_met`，没有官方指标。2026-10-31 08:31:36 北京时间的唯一终态只读回执显示 ROUND/done completed、原 owner 已退出、八个 rank 不在、锁和设备无占用且 safe=true。原冻结包、checkpoint、clock 和含 token 原件保留在 exp-v3，不纳入 Git。

[R121-RESULT.json](R121-RESULT.json) 是在既有 Pod 有效 CPU16 准备池内从原冻结结果、TRAIN、ROUND、done、R120 无令牌摘要及安全回执生成的小结。摘要 SHA-256 为 `d9f42be903608573a28c21e0000e4cd4859521069977e07d37c8ada64b46e97f`；原始四件 SHA-256 在摘要 `original_sha256` 中分列，08:30 进度原件 SHA `5644d80536516a028b892658569b4be5a1302e0b83d0d89944a40b3770934820`、CPU guard 回执 SHA `9a5c6f852bb3acec80219ff3be1ee5eb21ffd8762fd157e9474000e36df1bfe7` 单独记录。原件未改写。
