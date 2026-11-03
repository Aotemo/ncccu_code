# R213 所选 EMA：预封存 fit 子集

已按[前向前声明](../../../todo/auroc-999/R213-FIT-SUBSET-20261103.md)完成：原封存fit46256中PCG64/seed20261103不放回抽128正例+128阴性，256名单先封存，SHAf5f774b4。只用原R213/936源码、原SITE52d85df1、唯一EMA c6@4800/tensor2d3e2466/blobdc3ca40d，复用inference加载及engine.evaluate；CPU batch4/workers0/5线程、原512/age/ImageNet，182.032秒完成。d837/root0–11/nice19/隐藏GPU，与R215训练24–119互斥，未读官方或更改训练/选点。

[精简结果](SUMMARY.json) SHA `b4b59827283cd5043c8a1a00dad3360599e4fd47b0a8ecb0c6b69672913f8fa4`：子集AUROC **.9971923828125**，错序对质量46（error_twice92）；同模型原完整dev AUROC **.9830353369271913**，点差+.0141570459。2000次分层图像percentile bootstrap/seed20261103给该子集条件AUROC区间[.9929809570,.9996963501]。调用既有paired_evidence、常数零参考的delta区间加.5得到单模型区间，不是另一模型比较，也不是fit-dev差值区间。

该256平衡子集不代表完整fit；AP/阈值不能与原dev类别比例直接比较。模型、样本与已选点条件下的区间不包含seed、患者或反复dev选点偏差。CPU FP32没有新作与原DCU逐字节前向对照。较高的子集表现与dev差距可以作为后续泛化假设的描述线索，不能单独认定过拟合、临床根因、错标或可达上限；完整fit与逐例结构审阅仍缺证。R217既定EMA值不因本结果重调。

[封存回执](CPU-SEAL.json) wd68lha6/a573e834、[实际推理](CPU-EVALUATE.json) 3y2nv7tt/920ce374、[归档与指标/区间复算](CPU-ARCHIVE.json) 8ik4_g4l/46f7a234均exit0、任务树退出且无违规。实际执行[一次性调用源码](PRODUCER.py) SHA96e55bc9保留供复核，未另建维护性评估链；样本名单、CSV及完整原件只在`exp-v3/technical-prep/r213-fit-subset-20261103-32e3`。模型张量前后exact，本次已完成其临时checkpoint依赖，后续可按真实清单清理。
