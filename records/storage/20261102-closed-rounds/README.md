# 三个已关闭轮次的受控 checkpoint 清理

按用户“三模型保留，其余先存结果再清理”授权，实际删除R164、R166、R168共25个checkpoint/initial文件；各轮完整结果和指标此前已同步，源Git`6ec73abd252535ae99f01bcfcda600a31d148c6c`的原件Git blob及SHA在Pod内核对。[精确清单](PLAN.json)SHA`9e7941ff78b380bb9432e404fba6c7e2182b0e727dc65aaffe74a54d2cbfc040`与[清单汇总](PLAN-SUMMARY.json)保留每个路径、实际stat身份、已提交侧车SHA及其advertised blob SHA。没有重hash或加载大权重。

[执行回执](DELETE-RECEIPT.json)记录25/25实删、0跳过、error null，1.521秒完成；同prep的`DELETE-JOURNAL.jsonl`原件50项before/unlinked严格对应，其SHA见回执。[最终复核guard](CPU-FINALIZE-GUARD.json)exit0确认25路径均不存在、侧车与结果SHA保持。pre-unlink logical11,580,704,146字节、allocated11,580,709,888字节；共享文件系统free38,677,774,336→50,258,247,680，点时差11,580,473,344字节。allocated与共享FS点时变化分列，不把它当后端物理回收量保证或重复累计exp-v2/v3同一FS。

保护对象前后stat/小SHA相同，SHA均`8c4f716635074bbd2560e0caca8e6e60e0db0f91287f1a9d575d0a1cdd33c8cc`：[前](PROTECTED-BEFORE.json)、[后](PROTECTED-AFTER.json)。N06/R17/R8-C25、当前R172及其冻结配置、失败R170、比较锚点R107/R108/R120、R130全部、公开初始化、原始数据、共用缓存及全部technical-prep保护。仅清单三轮的独占regular inode可删；实际open fd/maps无拒读或目标模型owner，逐文件重新核stat/侧车/已归档结果和父目录无链接，再unlink确认可见消失。侧车、index、锁、one-shot和结果原件保持；当前R172在另一受保护路径和训练96核池继续，无训练信号或实例锁。

沿既有manifest删除流程，为新清单明确当前绑定、三轮路径范围和活跃R172保护；没有调用旧环境固定入口或恢复旧进程。原件在exp-v3`technical-prep/storage-cleanup-20261102-32e3`，dca3根0–22/23核、nice19/隐藏GPU，准备池与训练24–119互斥。[清单guard](CPU-PLAN-GUARD.json)、[删除guard](CPU-DELETE-GUARD.json)及最终复核均exit0。历史实验结果未改，原checkpoint路径现只作历史身份记录。
