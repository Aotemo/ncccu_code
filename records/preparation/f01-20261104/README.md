# F01固定logit网格的必要CPU检查

源码范围仅`tools/diagnostics/ensemble_dev.py`、`configs/F01-LOGIT-GRID.json`、`tests/test_ensemble_dev.py`，方法和候选已在06ace3c发布的F01登记，当前R240R1/R242保持C4冻结运行。

在新Pod B原12核0–11准备池、nice19、GPU隐藏执行原target runner。RED guard `8963fdabce7ffdfeda4c36c34efe3cc055f80d8582dac4a6b96ee02e8246da5e` 因“F01 fixed logit grid behavior is missing”按预期失败；GREEN guard `0f05b081953c7dcfcb282d38d973ea56f388e8373d09988fef57e1e9530b91e6` exit0，两独立排序例通过：互补错误能改善到AUROC/AP1且并列时选更大R120权重，相同排序保留R120顶点。完整模块已实际导入并编译，原度量/selected/precision/错序实现从C4按源身份复用，不重跑其全套检查。

这不是实际成员加载或融合收益证明。工具将先核真实blob/EMA tensor、通用源及原selected指标，核新源码/配置/测试的Git blob bytes/mode后才计算21点；这些实际结果仍待源码发布后运行。没有图像forward、官方读取、GPU计算或当前训练变更。

e2ac11b已发布并API核commit/tree和三文件实际bytes/mode。第一次真实CPU预检d629170f在R120 EMA消费者读取规则时因缺少`NCCCU_SITE_BINDING`拒绝，在计算任何融合指标前结束；不是权重缺失或模型失败。按原R120提交已通过的加载方式，最小补入每个成员自己的历史model-identity site并在选择后恢复环境；当前CPU Pod身份仍由独立CPU通道核验，不把旧模型site当当前执行绑定。原失败目录/回执保留，新运行另用analysis-v2，不改网格或成员。
