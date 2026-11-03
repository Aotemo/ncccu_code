# R221工程失败

R221完整公开fresh八rank初态与CPU正式证据一致，237全训参数/五空AdamW/模型tensor51592a3d均通过。训练组件111.388秒后退出1，八rank均报告`Tiny noBN EMA live buffers differ`：生产无BN EMA检查未接纳EVA02静态非持久RoPE缓存。外层timeout为通用失败状态，原组件reason null、无监督信号，未触及长clock时限。

最后记录进度为0，无首成功更新回执、唯一selected或官方结果；optimizer.step位于失败的EMA.update之前，不能据最后日志宣称优化器未运行。该轮科学结果unknown，原clock/token/grant及全部失败原件保留，不重启。15:58:47唯一terminal-1 actualsafe已核，无rank/锁持有者/设备用户，八卡busy0；旧safe不再观测。

后续[R221R1](../../todo/auroc-999/R221R1-ENGINEERING-RETRY.md)为科学delta0的独立fresh工程身份，仅修复生产EMA静态缓存检查。当前CPU回归成功不等于新轮GPU已跑通；实际八rank及自然首成功另行记录。DCU失败后空闲超300秒，如实记录恢复时刻，不改原指标或以ineligible轮填补队列。
