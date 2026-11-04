# F01：固定21点logit融合结果

[实际结果](F01-RESULT.json)与[真实成员预检](F01-PREFLIGHT.json)由新Pod B的12核准备池完成，[整树CPU回执](F01-CPU-RECEIPT.json)c70f49f3通过。R120/R238实际blob及EMA tensor与原selected相符，通用Small ImageNet源、固定dev5140、identity/FP32及precision核验通过；未构建模型或forward，未读官方标签。

21个预登记组合中选择R120 0.6、R238 0.4，raw内部AUROC **0.9896254143633104**、AP **0.8330962923453213**。相对R120错序8739→8616，修复550/新增427、净减少123；固定20修复359/新增288、净少71，其余147修复191/新增139、净少52。改善较小，未达到raw0.995或AP > 0.9，不视作官方改善或总体完成；完整网格保存在结果中，样本级fused CSV仅存exp-v3。

源码f018e2e/tree948afc99已push/API回读，三个新文件实际Git blob bytes/mode在运行前核验，原度量和selected/precision实现从C4按身份复用。首次e2ac11b的EMA预检缺成员model-identity site而由原消费者拒绝（d629170f），未计算融合指标；最小修正采用原R120入口的模型身份绑定，当前CPU执行身份不替换。原失败目录保留，成功输出另在`technical-prep/f01-20261104-4e0c/analysis-v2`，结果SHA2be1d9bf。

实际整棵CPU任务12.682秒，结果内6.412秒计时从模型加载前开始，不含前面的导入与Git核验；不把该区间当端到端准备时间。读取已存在的selected checkpoint与dev预测，训练/成员参数和checkpoint选择均未改变；只有内部dev上的融合权重选择。重复使用dev的选择偏差仍存在，不能据单次小幅提升推广。R120/R238所选EMA及最小可加载依赖继续保护。
