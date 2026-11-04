# 已失效CPU验证副本清理

R238正式准备时真实free20,510,146,560低于原严格20 GiB准入；在R236R2继续训练期间，仅清理两个实际清单中的历史CPU-only初态fixture。每项由已通过的`CPU_initial_checkpoint_strict_roundtrip_emptyAdam`、fixture-only标志、实际小侧车SHA/advertised blob SHA及唯一regular inode证明。对应完整轮次结果已归档且无后续forward依赖，或固定第二seed已不eligible/从未激活，或activation在formal clock/root/launcher前失败；原失败与所有CPU结论保留。

[第一清单](PLAN-SUMMARY.json)a684c6b6/[删除](DELETE-RECEIPT.json)c620e016共7个、2,771,504,125逻辑字节；[第二清单](SECOND-PLAN-SUMMARY.json)0987a6a6/[删除](SECOND-DELETE-RECEIPT.json)8439349d共8个、3,167,433,880字节。[汇总](SUMMARY.json)实计15个、5,938,938,005逻辑字节及原5,938,942,976 allocated字节。两次0skip/error，无open inode或扫描拒绝；保护文件stat与小SHA均前后一致c18655db。原侧车/index/锁/one-shot标记保持，未加载或重哈希大blob、未发训练信号或取训练锁。

[第一终检](FINAL-VERIFICATION.json)/guard4b7bd942和[第二终检](SECOND-FINAL-VERIFICATION.json)/guard58b72f14通过，CPU证据见对应FINAL-CPU-RECEIPT；小metadata导出guard6fdf84fd。当前R236R2、R237/R238正式包及fixture和三KEEP N06/R17/R8-C25受保护，原数据、当前缓存和待用公开权重不在候选。free点观测多次随并行checkpoint/共享存储变化，末次25,652,363,264仅为瞬时值，不声称后端物理回收或保证未来准入；下一次activation前仍直接调用已发布同一`verify_storage`先核真实值，不放宽20 GiB门。

私有范围为`fit-label-contrastive-20261104-4e0c/{old-CPU-fixture-storage,other-CPU-fixture-storage}`和相应PLAN/DELETE/verify helper，复用原闭轮精确清单流程，仅替换当前范围/结论和保护身份；六个第二批及未来R2闭轮helper已在第二plan guard内编译，未来完整结果未读。本轮活跃checkpoint不作为CPU fixture删除；这次去除的旧fixture不能再称可加载验证资产，其原成功身份/配置/结论仍保留。
