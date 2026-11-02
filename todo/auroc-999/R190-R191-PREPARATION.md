# R189 之后的条件复核与固定 ViT-B CLS

R190已按37eeae6实际技术打包通过，SITE267e0af5、guard0m_e92pe/d1205862…103c exit0，未formal/clock/GPU。R191首次技术包nxk8yk6x/9686e73f…6c35 exit1：prepare.py未给Base格式写入expected_backbone_state_keys162，验证器正确拒绝。只补原162注入分支后，真实原prepare/load_config 1evksm6d/8d1b34fe…ec4e通过；[精确复用+修复证据](../../records/preparation/vit-base-20261102/PREPARE-FIX-CHECKS.json)共8目标/14编译，原模型P8等不重跑，sxb7gt7n/25488300…baba归档0。下一R191使用新未启动阶段并实际Git回读/技术包，原失败目录与R190/现役R189不改，尚无新formal/clock/GPU。

21:47全部必要CPU证据已完成：[7目标/13唯一源码编译](../../records/preparation/vit-base-20261102/README.md)、两actual public fresh/原noBN首warm-copy、精确source/SCI/guard SHA核对均通过；最终u5lfpyk_/6ef579b9…56f exit0。R190 whole8c62435a/SCI7fcf3418，R191 wholec3495f5a/SCIf41f43a2；正式配置已从Pod导回，无占位。R191仅新round16GiB例外，默认/旧例外/freefloor不改；完整24k按既有实际ViTS速率及4倍运算量场景24147秒<27750 soft，但非GPU耗时上界。当前Git同步/远端回读/技术包待完成，无新formal/clock/GPU；现役R1892450成功/0skip/eval3、EMA暂AUROC .97001894，原0f3冻结保持。下文为准备历史。

当前唯一GPU R184R1/cd7246a及既定R188/R189/0f3顺序和冻结不变。公开ViT-B来源已[实际完整验收](../../records/preparation/vit-base-source-20261102/README.md)，仅用当前20%准备池预备一个后续单模型；不派生或恢复Agent、不新建资源/GPU预演。

固定分支：R190/order103只在完整有效R189唯一内部dev选点严格超过R107 `.9884405731067525`及R120 `.9894773092062407`且actualsafe后，以原Small-CLS/3bf4720a/seed20261027独立fresh复核；不满足则仅技术准备。R191/order104为R189之后的独立固定seed20261029假设，保留原512/ImageNet/logistic/G64P8/AdamW/LR/EMA/age/9:1/24k-800-p4，仅将同家族Small/384维换为Base/768维和已验证公开c6a5fb7d初始化，使用相同固定CLS读出。Base仍12层/12头/5prefix/1024patch，原native final LN/Dropout.2/Linear768→1/age各一次，不拼特征、不新增teacher/参数扫描。若R190有资格，先完成并safe，再R191；总体raw .999已达则无需后备。

最小必要范围：models.py现原生ViT函数按精确Small/Base身份消费宽度及公开来源，旧Small描述/RNG/None行为保持；config.py只扩同族模型/原CLS/公开格式白名单；experiment91.py登记两固定id/order/source/shape与保存job身份；fresh.py、ema.py、single91.py、inference.py、r92_initial_gate.py、check_candidate.py只把现Small非持久RoPE路径显式扩到Base，不改算法、时钟、GPU调度或旧合同。configs/R190-SCIENCE.json、R191-SCIENCE.json；tests/test_deit3_model.py复用真实P8/保存冷图fixture并增加Base原生CLS oracle，tests/test_nonpersistent_buffers.py将同一真实EMA/冻结冷图fixture用于Base，tests/test_candidate_binding.py加两身份。现source profile已经验收，不复制训练、保存、评估、监控链路。

验证只覆盖受影响路径：新Base原生factory/逐tensor-RNG/独立CLS+age数值，真实P8反传/全部165有限梯度/5空Adam组/真实noBN EMA/CheckpointStore→inference冷图；Small原有None/CLS oracle兼容及共享非持久RoPE旧/新模型的真实EMA copy/blend/冻结冷图；两候选/保存job/source绑定与实际public fresh。旧不变loss/WORLD/采样/优化器数学证据复用，不重跑全套。CPU保存storage只作情景，非DCU峰值/吞吐上界。实际初态及最终SCI SHA才进Git，无占位；push/远端回读/技术包在当前或R189训练期间完成，正常formal在可行收尾前完成，actualsafe后沿一次入口接续。

21:39已实际完成6个必要目标：Base原生CLS/RNG/age oracle、P8/165有限梯度/5空Adam/真实noBN EMA/保存冷图及Small None/CLS兼容3目标（6sbt5vtu/781736f1…e92，78.845s）；Small/Base共享非持久RoPE真实copy/blend/冻结冷图及新候选身份3目标（zomn7j_h/44041cce…9dce，21.088s）。新2public fresh及实际首noBN warm-copy r2hy4d78/f44caa65…4f8a通过，R190 whole8c62435a/SCI7fcf3418，R191 wholec3495f5a/SCIf41f43a2。原传输覆盖顺序错误77axtw6o未执行到新模型断言，已保留；正确新oracle的真实RED w82b_1oh为384≠768，后续GREEN通过。0实图/0GPU，新SCI仍仅Pod私有，未Git冻结或打包。

最后一个必要工程范围扩为configs/ROUND-STORAGE-POLICY.json和tests/test_round_storage_policy.py：仅精确新round `r191-vitb-dinov3-seed20261029-scale085-pairlog-cls-storage16-24k-p4-9010-20261102` 用16GiB，原默认12GiB、旧所有例外及free16GiB不变。Base参数量与完整Adam/EMA保存增大，按既有R170保存前原子新blob再prune的已知情景预留，CPU最终报告将用实际参数数推导；这不是全部并发/故障峰值上界或16GiB必成功证明。R190及R189仍默认12GiB；旧冻结包不热改。该必要函数级检查及全部实际受影响源码compile/最终归档待完成，再同步Git/技术包。

原公开256与比赛512区分，CPU存储不是DCU峰值或吞吐保证；完整24k的时间预算须按现有时窗与实际资源准入，不以早停假设缩小预算。当前唯一GPU已R189/0f3，后续准备不阻塞既定接续。raw .999未达。
