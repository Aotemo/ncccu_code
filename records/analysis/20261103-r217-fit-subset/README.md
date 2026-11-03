# R217所选EMA：原封存256图fit子集

已按13:43[预声明](../../../todo/auroc-999/R217-R219-FIT-SUBSET-20261103.md)，在R219真实首成功后，用原R213封存的同256张fit图计算一次。原名单SHAf5f774b4及manifest7cc0f8b7逐字节核对、indices完全相同；原PCG64/seed20261103、128正例+128阴性，无重选。R217/6fb491a实际唯一EMA c13@10400/tensor46e343fa/blob14fe3b40，原512/ImageNet/age，复用inference._load_model→engine.evaluate，CPU batch4/workers0/torch5/interop1，828.221秒完成。7c7af22c/root0–11/nice19/隐藏GPU，与R219训练24–119互斥。

[精简结果](SUMMARY.json) SHA7373b739e010b56b22bef3b428a30c504d1c6239cc47575841e4972117ce3279：子集AUROC **.99822998046875**、AP .9982425340702886、错序质量29（error_twice58）；同模型原完整dev AUROC **.9875435134155578**，点差+.0106864671。2000次分层图像percentile bootstrap/seed20261103给子集条件AUROC区间[.9952987671,.9998168945]，复用paired_evidence常数零参考delta区间加.5；不是fit-dev差值区间。

256平衡子集不是完整46256 fit，AP/阈值指标不具有dev患病率。区间条件于已选模型与该样本，不涵盖seed、患者相关或反复dev选点偏差。CPU FP32未新增与原DCU逐字节前向对照。较高子集分离与dev点差不能单独证明过拟合、难图病因、错标或可达上限；完整fit和逐例结构标注仍缺证，不据此扫描EMA或改变冻结队列。

[封存](CPU-SEAL.json)j_ebher_/635dbc6e、[实际CPU推理](CPU-EVALUATE.json)_pnhq7zo/fb2b3117及[归档复算](CPU-ARCHIVE.json)18u4jrpu/dd1792b6均exit0，树退出且无违规。实际[一次性producer](PRODUCER.py)只替换R213调用的真实身份并增加原名单exact核对，不新建维护评估链。模型tensor前后完全一致，无GPU/官方读或训练选点变化。名单、预测CSV、封存及完整原件仅在`exp-v3/technical-prep/r217-fit-subset-20261103-32e3`；该fit诊断的临时checkpoint依赖已完成，R217还待固定组汇总及清理终核。
