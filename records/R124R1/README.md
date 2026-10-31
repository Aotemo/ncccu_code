# R124R1：固定内容注意力汇聚，工程续接后的完整结果

原 R124 在 activation 前因 120 秒 live 观测过期而失败，没有 clock、GPU 或科学结果；其失败原件保持独立。R124R1 首次新正式 CPU 包虽有 20/20 内部检查，外层 guard 实测 13 个 runnable 线程超过本人 12 核份额，故没有用于启动。后续仅修正正式 CPU 检查的线程占用，独立新包的外层 guard exit 0、20/20 检查通过，再由唯一新 activation 和 fresh 完成此轮；科学合同仍是同一 order34。

训练成功/尝试更新 7200/7200、跳过 0；内部评估 9 轮，连续 4 轮无提升早停。预定选择器唯一选中 EMA cycle 5、update 4000，内部 AUROC `0.9861082179096462`、AP `0.8290710437486213`、error_twice `23074`，模型张量 SHA-256 `fc83bc5df278a4fc3f8449696c24aaf2d02622f6e47234269cff40561c47ff74`。AUROC 低于预冻 R107 `0.9884405731067525` 和已记录的 R120 单次最佳 `0.9894773092062407`，该固定 readout × 当前合同关闭，不做 seed20261027 或池化公式、温度、维数、LR 扫描。这只评价本轮固定组合。

终态已有状态记录 q 在线参数 768 个非零、L2 `3.7132256031036377`、Adam step 7200 且包含 exp_avg/exp_avg_sq；EMA q 768 个非零、L2 `3.5872137546539307`。所选内部 AUROC 低于 0.996，官方 5849 按政策 `not_run_by_policy / threshold_not_met`，没有官方指标。2026-10-31T14:33:14.994031+08:00 的只读回执显示 completed、原 owner 为 Z 态、8 rank 已退出、锁和设备使用者为空且 safe=true；Z 不代表已回收。

[R124R1-RESULT.json](R124R1-RESULT.json) 是现有 Pod 有效准备池内从原 freeze、TRAIN、ROUND、done 和终态安全回执生成的去 token 摘要，Pod SHA-256 `c6ed2eb82521febc0f3e666d4550f164536087dc0ff774b35c4e37212e40c65b`。四件原 SHA-256 在 `original_sha256` 分列；14:33 进度 SHA `756ce87b07f140676f7e4960f91613270afd9f1ffbfc5e9c085515389d655a42`、其 CPU guard 回执 SHA `7e706f678e266962339b9d9f5df7f8c3a47694b747f20c5a3248e42d59b0b99e`。原 checkpoint、clock、含 token 回执和旧失败原件留在 exp-v3，不纳入 Git，也未改写。
