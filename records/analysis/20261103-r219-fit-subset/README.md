# R219同封存256图fit诊断

原13:43预声明的R213同256个fit index，128阳/128阴、PCG64 seed20261103，不按dev错图重选。R219完整唯一EMA c6@4800/tensor677606c3/blobade2ad01，seal1e9310bf、原封存f5f774b4；原CPU inference._load_model→evaluate/physical4/workers0/torch5/interop1，831.037秒完成。

实际子集AUROC .997314453125、AP .9971116153401626；原完整内部dev .9877759060603908/AP .8358105199595313。模型tensor前后相同，256图/无GPU/无官方/无训练选点变化。原EVALUATION SHA b19f0d91，guardwuzx793u/a7f2a7f9 exit0；producer260151bb仅替换前一已验证producer的source/selected/result身份，原先pin_memory无GPU警告保留。

这是平衡子集，非完整46256 fit；AP/阈值不能按dev患病率直接比较，AUROC点差不是机制或上限证明。本记录尚未新算bootstrap区间，不包含seed/患者相关/自适应dev选点不确定性；不据此猜临床病因或改标签。模型forward依赖已完成，后续固定20仅读保留CSV；权重按真实清单解除临时保护，历史blob身份不代表当前仍可加载。原始样本级预测仅在exp-v3。
