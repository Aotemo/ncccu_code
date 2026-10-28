# R95：24000上限与内部AUROC早停

2026-10-29用户明确要求最多24000步、连续4轮指标无提升早停，并直接选择“内部 dev AUROC”。科学合同据此确定；R95尚未运行，A正在既有20%准备池完成受影响实现。R94冻结`212b41f9d3aa922519e62182701bfbea527c852d`继续原6400步合同，不热改其源码、配置、clock或token。

唯一配置为[`configs/R95-SCIENCE.json`](../../configs/R95-SCIENCE.json)，对应[`runtime/PLAN.md`](../../runtime/PLAN.md)：profile=`single-tiny22k-bc48a87f-fixed-9010-24k-earlystop-v1`，candidate=`r95-convnext-tiny-imagenet22k-to-1k-24k-earlystop-9010`，order5。消费者读取同一SCIENCE文件及实际SHA，不另拟身份或散写科学常量。

| 合同 | R95语义 |
| --- | --- |
| 更新与评估 | 最多24000次成功更新；每800次完成一轮内部dev，最多30轮，每轮online/EMA各评一次 |
| 早停 | 每轮分数为两variant较高的内部AUROC；首轮建立基线，严格高于历史最好值才刷新并清零坏轮计数；相同或更低计一次坏轮，连续4轮早停；`min_delta=0`、`min_cycles=1` |
| LR与EMA | backbone初始LR1e-4、head/age1e-3；warmup和EMA warmcopy仍是绝对1600次成功更新，首blend1601；cosine horizon改为24000，尾倍率0.01，EMA decay0.999 |
| 完成与选点 | 达到最大预算或在完整周期边界合法早停均为completed；保存实际成功更新数、已完成周期及停止原因。只在实际已完成C轮的2C条内部结果中按error_twice升序、AP降序、update升序、EMA优先选唯一checkpoint；不要求固定16条或必须跑满24000 |
| 墙钟 | 现有CLOCK_POLICY配置total16200、binding1800、train hard10800、freeze300、official1200、cleanup及soft margin150秒；新运行只生成一次时钟，不能据此声称租期或吞吐有保证 |

其余公开fresh来源、固定9:1 fit46256/dev5140、seed、512/FP32、G64/P8、优化器和模型结构沿用R94。初态仍从公开权重新建，不加载R93/R94 checkpoint。最大预算、早停和cosine曲线同时变化，因此本轮不是LR-only或等预算对照；不预定科学收益。官方完整5849仍只在内部唯一选点冻结后评一次，不能参与早停、调参或其他训练决策。

A负责runtime/configs/tools、必要受影响CPU检查、唯一Git同步与新冻结，R94实际终态和GPU退场核实后串行启动R95。沿既有直接训练入口，由真实8rank初态/组/空Adam/资源检查接管，不新增独立零forward GPU预演、继承桥、套件或文档批准门。B按[准备清单](NEXT-PREPARATION.md)完成既有配置/身份/干净打包/调用材料；不碰A文件，不重复来源、迁移、模型或native验证。
