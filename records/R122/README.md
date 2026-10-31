# R122：训练光度增强整组 identity 消融

R122 工程完整，成功/尝试更新 5600/5600、跳过 0、内部评估 7 轮，连续 4 轮未提升后早停。预定选择器唯一选中 EMA cycle 3、update 2400：内部 5140 dev AUROC `0.9874796957462513`、AP `0.8211659433777129`、error_twice `20796`，权重张量 SHA-256 `4aa597b03788019483a0b348f108876435f264c19b3b0b1f9be302176ecfbb2f`。

内部 AUROC 低于预冻 R107 `0.9884405731067525`，也低于已记录的 R120 单次最佳 `0.9894773092062407`。按预定判据关闭整个 photometric identity 消融，不做 seed20261027 复核，也不拆亮度、对比度、色彩单因子或扫描幅度。`training_photometric` 缺省 None 的原增强路径仍可用；本轮显式 `identity-v1` 不成为后续默认。此结果不能证明某一增强因子单独有效或无效。

所选内部 AUROC 低于 `0.996`，官方 5849 按版本化政策记录 `not_run_by_policy / threshold_not_met`，没有官方指标。2026-10-31 10:02:02 北京时间的终态只读回执显示 ROUND/done completed、原 owner 为僵尸态、8 个 rank 已不在，锁和设备无占用且 safe=true。原冻结包、checkpoint、clock 和含 token 原件保留在 exp-v3，不纳入 Git。

[R122-RESULT.json](R122-RESULT.json) 是在既有 Pod 有效 CPU16 准备池内从原冻结结果、TRAIN、ROUND、done 和终态安全回执生成的去 token 小结。摘要 SHA-256 为 `4a3dbf4943e992441cd62e08fd60da3d3eb58781ff138186346eec3a291e7b69`；原始四件 SHA-256 在摘要 `original_sha256` 中分列，10:00 周期进度原件 SHA `6fd08ef4aceced9a24821c49c3abf1a310c894640425ce11521cc6691971f853`、CPU guard 回执 SHA `659ed4a11a5ce56db6e00860e3c9335a7abe865db186f834b689f1609c20bc90` 单独记录。原件未改写。
