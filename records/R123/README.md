# R123：从 fresh 初始化移除年龄残差

R123 工程完整：成功/尝试更新 7200/7200、跳过 0，内部评估 9 轮后连续 4 轮未提升而早停。预定选择器唯一选中 EMA cycle 5、update 4000，内部 5140 dev AUROC `0.9867584356723914`、AP `0.8255658133774761`、error_twice `21994`；权重张量 SHA-256 为 `bc86edde5547c8069a2117b176fe4f172cf93ccc1b66a58986b5943462b7e7b3`。

该 AUROC 低于预冻 R107 `0.9884405731067525`，也低于已记录的 R120 单次最佳 `0.9894773092062407`。按预定判据关闭整个无年龄残差合同，不做 seed20261027 复核或年龄分箱、非线性、正则权重及其他 seed 扫描。此结果只评价本轮固定消融，不能证明年龄对所有模型或数据划分都有益。

所选内部 AUROC 低于 `0.996`，官方 5849 按版本化政策记录 `not_run_by_policy / threshold_not_met`，没有官方指标。2026-10-31 12:03:28 北京时间的终态只读回执显示 ROUND/done completed、原 owner 为僵尸态、8 个 rank 已不在，锁和设备无占用且 safe=true。原冻结包、checkpoint、clock 和含 token 原件保留在 exp-v3，不纳入 Git。

[R123-RESULT.json](R123-RESULT.json) 是在既有 Pod 有效 CPU17 准备池内从原 freeze、TRAIN、ROUND、done 和终态安全回执生成的去 token 摘要；其 SHA-256 为 `b9bc6213a69c9b22660c59c913bd3c61d7d0f51ca564fe93f48425678d53ff8a`。原始四件 SHA-256 在摘要 `original_sha256` 中分列，12:00 进度原件 SHA 为 `362817e9b8d47223435f68332a9018dedb57caf4a03a1ba21374ba89c020a7b3`，当次归档 CPU guard 回执 SHA 为 `3c7cf542f7356d48e6c24c6e88a0723e4c199439b0a3a0045fb210dc05c580f9`。原件未改写。
