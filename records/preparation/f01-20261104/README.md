# F01固定logit网格的必要CPU检查

源码范围仅`tools/diagnostics/ensemble_dev.py`、`configs/F01-LOGIT-GRID.json`、`tests/test_ensemble_dev.py`，方法和候选已在06ace3c发布的F01登记，当前R240R1/R242保持C4冻结运行。

在新Pod B原12核0–11准备池、nice19、GPU隐藏执行原target runner。RED guard `8963fdabce7ffdfeda4c36c34efe3cc055f80d8582dac4a6b96ee02e8246da5e` 因“F01 fixed logit grid behavior is missing”按预期失败；GREEN guard `0f05b081953c7dcfcb282d38d973ea56f388e8373d09988fef57e1e9530b91e6` exit0，两独立排序例通过：互补错误能改善到AUROC/AP1且并列时选更大R120权重，相同排序保留R120顶点。完整模块已实际导入并编译，原度量/selected/precision/错序实现从C4按源身份复用，不重跑其全套检查。

这不是实际成员加载或融合收益证明。工具将先核真实blob/EMA tensor、通用源及原selected指标，核新源码/配置/测试的Git blob bytes/mode后才计算21点；这些实际结果仍待源码发布后运行。没有图像forward、官方读取、GPU计算或当前训练变更。
