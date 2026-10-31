# R125：第三阶段 GAP 线性残差读出

R125 是 R107 的固定同 seed 消融，只新增零初始化的 384 维第三阶段读出参数，恢复原年龄支路与 pairwise logistic。正式 CPU 模型验收 19/19 通过，与既有 seed29 初态的 345 个共同张量逐一相同，新参数初始全零；唯一新 activation/fresh 后 8 rank 实际首步成功。先前准备脚本仅在完成验收和部署后的汇总断言误用骨干参数数值而退出 1；该原失败保留，后续只读原验收回执生成准备摘要，没有重跑正式模型检查。

训练成功/尝试更新 6400/6400、跳过 0；内部评估 8 轮，连续 4 轮无提升早停。预定选择器唯一选中 EMA cycle 4、update 3200，内部 AUROC `0.9867572315654234`、AP `0.8252083971535648`、error_twice `21996`，张量 SHA-256 `fe81a1d83f6d6d92902e32de4e252780bf44adab39e7f72836374ba6562ef13a`、checkpoint blob SHA-256 `d24b42aab2ef47964d3f5a27132806d892b30544d7aa1606a7fae529e56c9d95`。AUROC 未严格超过预冻 R107 `0.9884405731067525` 或已记录的 R120 单次最佳 `0.9894773092062407`，该固定组合关闭，不做 seed20261027 或层号、组合、归一化、维数、系数、LR 扫描。

现有状态和指标没有第三阶段参数更新或 Adam 状态的直接统计，因此该项记为 `not_directly_observed`，不从初始全零或训练成功推断终态数值。所选内部 AUROC 低于 0.996，官方 5849 按政策 `not_run_by_policy / threshold_not_met`，没有官方指标。2026-10-31T16:02:58.840224+08:00 的只读快照显示 completed、原 owner 为 Z 态、8 rank 已退出、锁和设备使用者为空且 8 卡空闲，safe=true；Z 不代表已回收。

[R125-RESULT.json](R125-RESULT.json) 是现有 Pod 有效准备池内从原 freeze、TRAIN、ROUND、done 和终态安全回执生成的去 token 摘要，Pod SHA-256 `3f4ac4f4a62321cd1e80dd36562ec93f120a46766062b7549a893da4e315410a`。四件原 SHA-256 在 `original_sha256` 分列；进度回执 SHA-256 `c74b2d273b9337eaa78387c5c48683efa3dfa73cd81c7182c6efac1070527a06`、其 CPU guard 回执 SHA-256 `d0f1c557961f8501e53a3126adeb9cf1f593491da7cfd3493e0287d9013f85f3`。原 checkpoint、clock、含 token 原件及旧汇总失败原件留在 exp-v3，不纳入 Git，也未改写。
