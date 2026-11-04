# 2026-11-04 冗余 CPU 验证权重清理

按用户本次明确授权，实际删除 **50份 CPU-only、零训练更新的验证权重**，逻辑字节21,816,588,622（21.82 GB / 20.32 GiB），删除前allocated21,816,601,088字节。共享NFS可用空间从首批删除前28,014,804,992升至最终核对时49,701,453,824字节（49.70 GB），点时增加21,686,648,832字节；期间有其他写入，不能把该点差或allocated字节声称为后端物理回收量。

| 批次 | 实际删除 | 逻辑字节 | 精确清单及删除回执 |
| --- | ---: | ---: | --- |
| 首批 | 39 | 17,903,707,769 | [PLAN](targeted-PLAN-SUMMARY.json)、[DELETE](targeted-DELETE-RECEIPT.json) |
| 补批 | 11 | 3,912,880,853 | [PLAN](supplement-PLAN-SUMMARY.json)、[DELETE](supplement-DELETE-RECEIPT.json) |

所有目标均位于实际清单内的 `technical-prep/<旧ready>/formal-cpu/CPU-FIXTURE-NOT-FOR-TRAINING/initial-checkpoint/checkpoint-*.pt`；没有递归删除目录。每项核对实际regular唯一inode/stat、CPU strict-roundtrip/emptyAdam通过及fixture_only/零训练更新、小侧车SHA和原广告blob身份，未重哈希或加载模型payload。保留原侧车、CPU回执、配置、指标、索引和一次性标记；历史通过不代表删除后仍可加载该fixture。

首批使用已推送31da652的completed/actual_safe结果；补批另核实8个历史 `completed_not_target` 的真实progress.completed与actual_safe、1个R184已安全结束的工程失败fixture，以及2个由R217/R219完整结果确认不eligible且实际未创建round根的R218/R220准备fixture。工程失败及未启动项不虚构科学指标。全部来源JSON逐项匹配该commit的Git blob，原结果/状态/指标未改。

[最终独立核对](FINAL-VERIFICATION.json) SHA `5f9e1981e935bbf5cfb36c5479cb33de095e1cd3bf8f9cf70c4cee50b270d503`：两清单50项均实际不存在，逐项journal的before/unlinked成对，删除过程0skip/0error；220个其余保护路径的stat及适用的小文件SHA保持。三KEEP N06/R17/R8-C25实际权重stat未变，无新加载或前向，不将此登记为新的加载验收。四个plan/delete CPU回执和[最终CPU](FINAL-CPU-RECEIPT.json)均exit0、树退出、无affinity违规。

全部runtime round受保护，包括未同步完整结果的R238、旧Pod移除后unknown的R240；R236之后prep、goal/fit-label-contrastive/fit-label-quota/fit-supmin/dual-pod新准备、原始图像、assets/公开权重、共享缓存、R120提交及模型依赖均不在删除目标。未发训练信号、不取训练锁、不重查旧actualsafe、不重置clock/token或一次性标记。

定向盘点65份fixture中，另15份暂未删除，精确路径见最终核对：5份参考/诊断范围（R107/R108/R120/R130/R134R1）、5份本批尚未匹配充分闭轮/CPU final证据的旧准备（R150/R170/R184旧ready/R221/R221R1），以及5份R236R2至R240受保护准备。这里的暂保护不建立新的永久模型或教学例外，也不把未匹配等同于整个环境缺少证据。

本次使用现有新PodA UID04203d31、Python3.12.7、有效CPU120的明确12/12准备分配：4e为0–11，c617仅12–23，nice19/隐藏GPU/interop1；与每Pod计划96核训练池互斥，不使用旧51fe绑定或B份额。[新allocation原件](CPU-ALLOCATION.json) SHA9fca3bbf，当前协作范围见[双Pod任务](../../../todo/exp-v3-dual-pod/README.md)。唯一Git与双GPU owner仍为4e；本会话未做GPU、资产迁移或Git写操作。

失败与调整原样保留：首次budget继承旧policy SHA725b、传入当前policy而被校验拒绝，仅对新context按Pod实核0b1a5f95改pin后生成新allocation；第一次广泛只读盘点陷于NFS元数据等待，观测时268秒仅约3秒CPU，确认自有PID155/start381217639/argv后只发一次TERM停止该进程，[原CPU回执](INVENTORY-INTERRUPTED-CPU.json)exit−15/树退出保留，未执行删除。随后改为已知fixture路径定向盘点，18.1秒完成/ed83ab1f通过。一次运输调用的远端shell引用失败发生在目标写入与Python执行前，改用既有tar/stdin传输；不将失败写成通过。

仅导回小JSON回执，不导出模型或样本。[10件导出身份](EXPORT-MANIFEST.json)及[实际下载副本返回Pod的SHA复核](EXPORT-ROUNDTRIP-VERIFICATION.json)ce7ef27c通过，[导出CPU](EXPORT-CPU-RECEIPT.json)785a908d exit0。原清单、journal、保护快照、临时helper源码与所有原件均在 `exp-v3/technical-prep/storage-cleanup-20261104-c617-04203d31`；复用原精确清单清理链，没有新增长期存储平台。对应[任务收口](../../../todo/storage-retention/CLEANUP-20261104-c617.md)由c617维护，Git同步由4e统一处理。
