# R250：R201 原配方 fresh 重训

Pod A 接续 R247，使用已推送源码 `2742968868e9c3e5cffd4ae35dbf5c5208eb320c`。正式 CPU 18/18 `eb5b2b222e29794088fd1b43d907ed6fd0ebf56f9f9c821ff8968708c965b51e`、部署 `0a719b342a3083a9e961432a04bd6cc825776256678b76fd1e7ffe7393a4ee7f`、SITE `c2be7b76a9e8226afb78d1928faae4b7c88e831886a44c848344e52def933bb9`；[现场容量](R250-WINDOW.json)已计 R249 剩余预算，唯一 [activation](R250-ACTIVATION.json) audit `a64da686d63ef43f2eaa5dab328abfe0d4172561592d5cc268221bdadfd42ba9`。

activation 客户端超时后，[首次只读对账](R250-RECON-1.json)确认 exit0、clock 一致且未派发；首次恢复入口因 `awk` 转义在 launch 前 exit2，[再次对账](R250-RECON-2.json)仍确认所有派发标记和进程不存在。随后重新核 UID/Ready 和保守容量，复用原 clock 首次 direct launch，未重复 activation 或重置 clock。

[真实首成功](R250-START.json) SHA `17f44ccd68794b7cfa30ffc1781990255f49c0f36d135921575f01be5ccb153a`：八 rank fresh 初态、345 参数/5 空 Adam 组、训练资源、G64/P8/448 对均通过，首自然记录 25 成功 loss `1.92756986618042`，观察时 100 成功/0 skip。官方留给最终冻结方案。

[交接计时](R250-HANDOFF.json) SHA `fe5af0d37fa08d74ae10a954b9f6e7dc77c8013cea4ec849b591406b8d730944`：R247 completed 事件至 R250 owner spawn `604.504` 秒，actualsafe 观察至 spawn `434.609` 秒，超过 5 分钟；原因包括 R247 安全收尾、R248 容量不足、activation 超时对账和上述命令错误。未连续采样 GPU idle，这些时间点差不能冒充实测空闲。

R250 已完整结束：`7200` 成功更新、`0` skip，c9 连续四次无改进早停；唯一选中 EMA 为 c5@4000，内部 dev AUROC `0.9889535226751404`、AP `0.8363466610846791`。原[安全终态](R250-COMPLETION-08ac.json) `81432866b991a7ca549e72d8ee5aa05c16d167d273fee87793af3defce6b7751` 记录 done completed、actualsafe、未读取官方；该数值与 R201 历史内部选中结果一致。新 selected 的完整模型构造复核和固定 F03 重算尚待，尚不能将本轮列为可部署融合成员。
