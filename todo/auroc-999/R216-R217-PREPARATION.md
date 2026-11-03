# R216/R217：条件 SigLIP2 复核与较长 EMA 窗口

沿已启用brainstorming/writing-plans/TDD，由32e3在20%准备池最小实现；不恢复/派生Agent，不追加审批。当前R215/c9bbde5/SCI68703241/SITEd155冻结，真实8rank与自然首25已通过。11:52单次sysfs快照八卡均100%，非连续利用率证据；raw .999未达。

R216/order129仅完整R215唯一所选严格胜R107 `.9884405731067525`及R120 `.9894773092062407`且actualsafe后，用同SigLIP2/public MAP/unit/原配方独立seed20261027复核；否则technical。来源、固定512与所有科学合同保持，精确轮名24GiB。不是无条件seed扫描。

R217/order130为直接原R107 Small/public fe6/seed29的一项独立假设：原前1600次成功更新EMA逐步复制不变，之后仅将EMA decay .999改为.9999；原在线模型/单次forward/G64P8/448 logistic/AdamW-LR-WD/采样增强/age/globalclip/24k-800-p4及双variant内部dev唯一选点不变。使用当前已有EMA更新、保存、加载、评估链，不增加教师、checkpoint融合或推理模型；只在新候选metadata中登记准确规则，旧规则与当前冻结不改。

依据为[EMA权重研究](https://arxiv.org/abs/2411.18704)对平均轨迹、隐式正则与泛化的研究；它不证明本任务.9999优于.999，也不证明难图或临床根因。固定此一个衰减，不扫描系数/窗口或叠加新优化器/损失/ROI。更长窗口可能滞后和欠拟合；必须完整内部AUROC评判，难组不能替代总体。fit缺证仍unknown，后续同所选模型fit侧诊断不作为事后改变此冻结值的理由。

范围：仅`experiment91.py`的精确候选/来源/metadata EMA绑定、`tests/test_candidate_binding.py`、新增`tests/test_long_ema.py`、精确R216存储例外和既有storage函数目标，以及R216/R217 SCI。无需改模型、data、loss/gather、optimizer、EMA数学实现、启动、保存、评估、监控或fresh来源映射。先候选绑定与生产EMA边界/解析数值目标RED，再最小GREEN；两actual public fresh、必要compile、Git同步回读、technical及后备normal在R215训练期间完成。源树按现有部署器要求完整传输，可压缩I/O，不再用不完整树绕过核验；真实8rank首更新接管GPU证明。

11:55计划固定，以上新增源码/SCI尚未实现/训练，未声称收益。R215最近有效800/0skip/eval1/bad0暂内部`.9534745108616469`；当前d837/root0–11与训练24–119、保留peer12–23互斥。R211清理正在现有manifest/deleter边界内准备，不涉及R213所选临时诊断依赖或R215任何产物。
