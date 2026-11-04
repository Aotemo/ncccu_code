# fit标签固定类配额准备

[事前合同](../../../todo/auroc-999/R240-R241-FIT-LABEL-QUOTA-20261104.md)151896ae已同步/API回读，在R238尚未激活/无其dev之前登记。唯一处理为R238均匀fit流→原已有类排列流固定global64中8正/56负，保留同一SupCon、强光度/双视图、公开Small/MLP、1600预算、学习RNG和原监督完整合同。复用RankBatchSampler/两reference microbatch和八rank连续分片，只重新附上R238原augmentation seed公式；实际indices/像素随处理变化，不称与R238逐draw像素相同。

[CPU汇总](CPU-CHECKS.json)39dc67e8：两缺行为REDc48adc66后，仅`ssl_stage.py`/`experiment91.py`两必要runtime文件实现，两个目标/三编译GREEN0a4d78cf通过。独立手算class permutation、两microbatch顺序、八rank合并与每draw seed oracle，70步跨排列边界、重复/交错迭代互不影响；dev/age访问trap、两身份/原完整监督job/学习域、旧recipe及新quota完成回执边界通过。

实际fit流/消费guardd587adcb通过：CPU模拟八rank完整1600×64，共102400主图draw、每步8正/56负，12800正draw/89600负draw，覆盖46256图（1490正/44766负）。正例610张各8次、880张各9次；负例44698张各2次、68张各3次，重复暴露是代价。真实rank0的8×2×3×512×512/首批4正标签索引对齐，global首批8正；新配额一次CPU损失计数forward验证128anchor、正对15/111及负视图16/112，未重跑旧loss数学/梯度目标、Gloo梯度或模型P8。标签表/gather、原UniformFitSampler等33范围、含原loss与rank sampler的14整文件相同；复用f3fcced0的原Gloo梯度和93ac00ab的原模型P8真实证据。没有模型/checkpoint加载或GPU预演；CPU八流不冒充真实DDP消费。

两SCI为68d06946/8f652c43，源码/config发布在本提交，精确全树/技术包及原正式部署尚待；当前R2与R237/R238原冻结包保持。原完整优先分支按合同处理后才有R240资格，R241仍严格全门后；无AUROC收益或晋级结论。当前R236R2[8475/c10/bad2/0skip](../../R236R2/R236R2-PROGRESS-quota-checks-ready.json)，原owner及八rank活跃，完整selected待。

[导出回执](PREPARATION-RECEIPT.json)/guardde5cb7d8仅小metadata及两SCI，私有范围为`fit-label-quota-20261104-4e0c/{prepare-fit-quota-cpu.py,finish-fit-quota-cpu.py,export-fit-quota-preparation.py}`与临时source，沿原单元runner/数据消费/observer，不建新训练/保存/评估/启动/监控链。主Agent自审采样、quota、种子、计数/旧分支和完整原监督合同；按用户要求无子Agent/独立审阅者。CPU3仍独占24核0–23/nice19/隐藏GPU，与当前96训练核互斥；仅任务图像/CSV，无医学权重/teacher/新资源，目标未达。
