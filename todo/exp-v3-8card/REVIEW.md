# 本轮固定差异审阅

2026-10-28，规划按当前文件边界做文本审阅。对象为工作区`work/exp-v3-8card-review-20261028/FULL-DIFF.patch`及`FULL-DIFF-v2.patch`，包含tracked与untracked正文；v1保留，v2包含末端部署关联修复和真实CPU小回执。此时未stage、未发布、未GPU。B两split/最终迁移回执及当前文档为待纳入的末次小delta，最终提交仍由A统一冻结。

以下范围未发现新增的具体阻碍：

| 范围 | 审阅结论 |
| --- | --- |
| R93科学与生产初始化 | 精确公开Tiny22K→1K来源、fresh binary+age、23LN1e-6、18GELU none、无BN；明确R92 finalLN1e-12差异，不称纯预训练来源效应 |
| 拓扑与训练语义 | reference32、G64/P8、8rank×8×1；RankBatchSampler仍分片同一全局样本/增强种子序列；LR/EMA随成功更新计数，800×8与warmcopy1600不因加卡改变 |
| 选点与一次性控制 | 完整16条内部dev按冻结排序选唯一checkpoint；canonical训练目录持有官方一次预约，发生在官方读取/模型构造前；旧运行与技术fixture不能变正式初始化 |
| 当前绑定与CPU | 显式site/UID/Ready/源码/设备身份；rank与控制进程使用获准mask，子进程不绕过合计20%；activation/launch不覆盖已有身份 |
| cache迁移接口 | PreparedImageCache与迁移工具的policy、namespace、key、新source_file字段相符；arrays不重建，当前目标raw/NPY以内容SHA证明，train/valid索引分离；使用exp-v3目标，无旧路径fallback |
| 技术GPU入口 | 本轮0update/0forward/no-data native初始化与FP32 collective，独立一次性目录、300秒界限、精确进程退出与资源释放；不宣称梯度/吞吐或整轮时限已验 |

协调提出的commit→tree关联缺口已关闭：`freeze_deployment.py`消费pin过SHA的实际GET commit JSON，核commit.sha与commit.tree.sha，再核完整remote tree及实际文件bytes/mode；commitA/treeB误配拒绝。现有`tests/test_deployment_tree.py`新增窄case，A报告Pod3/3、source unchanged；旧2项及真实RED原件保留。相关原件见`records/environment/20261028-8card/pod-51fe3daf/{COMMIT-TREE-LINK-CHECKS.json,COMMIT-TREE-LINK-CPU-TREE.json}`。

末次迁移delta已核对：`records/migration/20261028/RELOCATION.json`为passed/complete，train51396、valid5849、120380文件/75373650642B，原FREEZE与canonical未改；`CPU-FINAL-RECEIPT.json`为exit0、task_tree_exited=true、无affinity违规。B报告全部源码/测试/回执冻结，无待改项或在跑任务；Pod实算SHA及份额释放记录见本任务README。该delta纳入A最终具体提交集合，未重跑已通过来源/迁移项。

本审阅没有执行本机Python、解析、哈希或测试，也没有派生新Agent；文本结论不替代A生产18/18、启动29/29、准入合同11/11、部署3/3及B来源/迁移各自的Pod实证。未重跑旧通过项，未新增检查范围。GPU和正式R93仍须等待具体Git同步/远端commit与tree复读、Pod实际SHA冻结；尚未取得的结果不能标通过。
