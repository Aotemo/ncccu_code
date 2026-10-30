# R120：固定 pairwise sigmoid，seed 20261029

R120 工程完整，8000/8000 次成功/尝试更新、跳过 0、内部评估 10 轮，因连续 4 轮未提升而早停。预定选择器唯一选中 EMA cycle 6、update 4800；内部 5140 dev AUROC `0.9894773092062407`、AP `0.8313831149805229`、error_twice `17478`，权重张量 SHA-256 `f21bffae88acbf7c216c4141c0ef0b9d990a624e3e92a1baf750a501c415fc85`。该完整选点严格超过预冻结 R107 的 `0.9884405731067525`，触发固定 seed 20261027 的 R121 复核；单个 seed 的收益尚不足以升级基线。

选点低于内部 AUROC `0.996`，官方 5849 按版本化政策记录 `not_run_by_policy / threshold_not_met`，没有官方指标。2026-10-31 07:03 北京时间的只读终态回执实证 round completed、训练 ranks 退出、原锁无持有者、设备无使用者、八卡空闲及 safe；原冻结包、checkpoint、clock 与原始回执保留在 exp-v3，不纳入 Git。

[R120-RESULT.json](R120-RESULT.json) 是在既有 Pod 有效 CPU16 准备池内从原冻结结果、TRAIN、ROUND、done 和安全回执生成的无令牌摘要。摘要 SHA-256 为 `545fa6c427afa625d0335c96391c17d162ed7bebef70f9834d70e441b509b53d`；原件各自 SHA-256 在摘要 `original_sha256` 中分列。原始完成回执并未被改写。
