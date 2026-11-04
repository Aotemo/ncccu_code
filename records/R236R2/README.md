# R236R2：固定强光度NT-Xent完整阴性

11-04 后续状态：用户 [R120 test 提交](../submissions/R120-20261104/README.md)及下载往返SHA已完成交付；唯一GPU已接续独立 [R238](../R238/README.md)，实际公开八rank和首50通过。完整R2结果已368b95dc推送/API回读后，[9个无后续依赖checkpoint](../storage/closed-r236r2-20261104/README.md)按精确清单清理6.733GB、终检CPUfe1df9f4通过，三KEEP/当前依赖和所有selected预测/metadata保留；旧safe未重查。下文末尾“当前先执行R120”是执行时序记录。

[完整结果](R236R2-RESULT.json)039c2ec2：原监督12800成功=尝试、0skip/c16/bad4，唯一原AUROC selector选EMA c12@9600，内部5140 dev AUROC **0.9849896025363309**、AP **0.8104065152313845**、error_twice24932，tensor e155a8b888672b3f7ca32babc6e38099a7268770d2033d67cdf23dd8e93cf259。严格R107/R120双门均false，固定第二seed R237不eligible、未激活；官方5849按原≥.996门未跑。AP/局部修复不救AUROC，目标.995/AP>.9及总体raw.999未达到。

[原唯一安全终态](R236R2-COMPLETION.json)0449db24来自现成PROGRESS-r120-gpu-wait.json/guardb78dec36：原owner845023/start379621036已Z退出、八rank不在、锁无holder、device users空/拒读0、八PCI busy0。只读复用，未再扫描旧safe。[CPU归档](CLOSEOUT-CPU-EXPORT.json)由原helper完成（aec005c6），导出b51dbfc3；只读既有selected CSV/precision/freeze与完成回执，无模型、checkpoint加载或新forward。

相对R107，固定20错序修复1714/新增3729、净增2015，其余147修复1118/新增1969、净增851；整体修复2832/新增5698、净增2866。[配对汇总](PAIRED-REPORT.json)保留原2000次paired bootstrap和对R120/R115/R224/R226/R228/R230/R232/R234的完整比较。区间条件于既定训练及选点，反复使用同一dev的选择偏差仍在；这些结果不确认临床原因、标签错误或指标上限，不据固定难组做fit样本加权。

实际适配1600成功/0skip/dev0，无label/age，102400主图、204800视图/encoder forward、0 teacher；NT-Xent固定.1、128×128矩阵、1同图正对/126负视图和强光度消费通过。SSL首末日志1123.735秒，joint train component7365.463秒，component开始→SSL完成1215.290秒、SSL完成→原监督component结束6150.173秒；预算相同不代表算法计算或停止点相同。完整[适配结果](R236R2-SSL-RESULT.json)、[原监督初态与自然首25](R236R2-SUPERVISED-START.json)/[来源回执](R236R2-SUPERVISED-RECEIPT.json)证明derived9ce9f323、公开head-age不变/原factory RNG恢复、345全训/5空Adam及G64P8/448。

原冻结c8004a695989aa76ab17f7cb493c6c486f100438/tree1f6a6d9c、SCI266beb7f、SITE65bf8525/部署50f33058不改，2822全树及18/18实CPU通过。05:08:50.774 UTC自然首适配及[首75](R236R2-START.json)3d62e66b/guard8244ae79完成。[此前R234→R2接续](R236R2-HANDOFF-GAP.json)dispatch/首1/25分别2018.493/2121.741/2138.648秒，明确超过300；两次旧身份pre-clock存储失败、必要清理/源码观测补足和同步/部署、API EOF及传输均保留，第二次失败瞬间容量unknown。原错误草稿与新corrected导出分列，不改冻结或重做前向，不把时间间隔当连续利用率测量。

用户新增R120生成test提交，当前先由同一owner串行调用R120冻结预测入口，既有gpu-owner硬件额度仅作资源约束，不重开旧训练/clock/token；提交完成并退出后，按已备原入口接独立R238（3bd/SITE4bb2）。R237/R238和后备R240/R241优先合同保持；R240源f22/2873全树与两技术包已通过，技术包不是正式部署或GPU消费证据。原数据、512缓存、当前R120依赖及三KEEP保护；本轮checkpoint在完整归档并核后续依赖前不删，预测/precision/结果保留。执行仍仅4e，CPU3 24核0–23与GPU96核24–119互斥。
