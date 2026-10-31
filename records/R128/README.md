# R128：固定 1600 次 head+age 先行更新

R128 从 R107 的公开 ConvNeXt Small、seed20261029、pairwise logistic 合同重新初始化。前 1600 次成功更新只由 head 和 age 执行 AdamW 更新，骨干仍参与完整前后向与有限值检查；从第 1601 次起三者共同更新。正式初态与 R107 的 345 个张量逐键一致、Adam 状态为空，真实 8 rank 在更新 1、1600、1601 的阶段与优化器状态有原运行回执证明。首次精确源码打包遗漏 R127 归档文件的失败回执保留；修正后正式验收与唯一启动另有原件，不把失败覆写为成功。

训练成功/尝试更新 **8000/8000**，跳过 0 次，内部评估 10 轮，连续 4 轮无提升而早停。唯一选中 EMA cycle 6、update 4800：内部 AUROC `0.9882093845688875`、AP `0.8282870957132695`、`error_twice=19584`，张量 SHA-256 `0edb6ee48f74935e662fde8151255053e86ab5c1319c1554a8222865719b8be1`，checkpoint blob SHA-256 `7068d3f955640a326fc6dc64cb97fe36724d92b675e128c36e699f40dca7404b`。它低于预冻 R107 `0.9884405731067525`，也低于 R120 单次最佳 `0.9894773092062407`；固定的 1600 次策略本阶段结束，不启动 seed20261027。这不代表所有先训线性头或冻结策略都无效。R128 与 R107 的停止和选点更新数不同，不是等更新量比较。

选点未达内部 AUROC 0.996 门槛，官方 5849 按政策 `not_run_by_policy / threshold_not_met`，无官方指标。北京时间 2026-10-31 22:38:23.630 的单次只读观察记录 TRAIN exit0、ROUND/done completed、原 owner 为 Z 态、8 rank 已退出、锁及设备使用者为空、8 卡空闲、safe=true；Z 不表示进程已回收。冻结阶段结束至首次观察到 completed 相隔约 87.33 秒；原 done 标记的精确创建时刻未记录，不能把它解释为通知延迟或宣称固定节省时长。

[R128-RESULT.json](R128-RESULT.json) 是有效 Pod 准备池仅从既有原件生成的去令牌摘要，Pod SHA-256 `8a51098f90988130279d201039665ac06a5f6b0d619068c96fdc2b0a23fcd282`。原 freeze、TRAIN、ROUND、done、progress、CPU 与首成功回执的 SHA-256 分列于 `original_sha256`。训练冻结 Git `9caaf3e060da6b5d4d4291456094463e8cddf1be` 与后来的观察工具 Git `f0d6902ef2b8e8cd424cfbbbad33db4990bda376` 分开记录；后者未热改训练包。原 checkpoint、clock、运行令牌及完整回执留在 exp-v3，未纳入 Git 或改写。
