# R230/R231 — fit-only EMA 跨视图分布目标

R228仍按冻结be5b5fa8运行；本页在其完整结果前固定后备问题：仅以fit图片，令同一公开fresh学生匹配自身历史EMA教师的跨视图类别分布，能否比R115的同时刻方差/协方差目标改善后续完整dev排序。不是延长R115，也不从旧任务模型、医学teacher或额外图片取得目标。R230/order145 seed20261029；R231/order146仅原严格双门后的固定seed20261027。

依据：[DINO原论文](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)、[作者训练实现](https://github.com/facebookresearch/dino/blob/main/main_dino.py)、[作者原型头](https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)。固定使用其三层2048/256瓶颈、65536原型、瓶颈及原型行L2归一化、固定单位原型scale、无BN；学生温度.1、教师.04、center EMA .9，教师动量.9995→1余弦。作者为较小batch举.9995/256的建议，本项目G64采用该值是事前推断，不是已证最优或论文G64结果。首个完整fit epoch（floor(len(fit)/64)次成功更新）取消原型梯度，其余学生参数照常训练；全程不恢复已有状态。

保持原R115均匀fit/G64/P8/两独立原增强视图/512、1600成功更新、原学习率/WD/clip/warm160及随机域。新增信号是stop-gradient、居中/锐化且随历史EMA变化的跨视图分布，替代VICReg同时刻全局统计；临时原型头、教师和center是该目标的必要组成，不声称纯单组件因果比较。本方案是公开Small上的固定DINO核心适配：不复现原ViT、多crop、裁剪、优化器日程、训练规模或结果，不把论文语义性质当作眼底收益保证。选择保留完整512全局输入，遵守旧亮区/四角窗阴性及结构审阅的不确定性。

临时教师只是同一fresh学生的无梯度EMA副本，无独立优化器/外部初始化，eval禁用随机depth；最后丢弃教师、原型头和center，仍只有一个最终监督模型。只传学生encoder/final-LN derived，原公开head/age保持、监督空AdamW/EMA及完整R107 24k/800/patience4保持。SSL不用label/age/dev/官方。每步仍64图片/128唯一增强视图，但学生128+教师128=256次encoder-view forward，不能称与R115等算力。

为防只看loss误认收益，原日志同时记录教师逐图目标熵、平均目标分布熵、动量与首epoch原型冻结状态；这些只诊断fit训练，不挑SSL checkpoint或调参。初态核教师/学生state完全一致、无存储共享、教师梯度禁用；结束核各rank教师/center一致并丢弃，监督原门及唯一选点保持。

精确代码范围：新增`runtime/bundle/retrain/ema_assignment.py`实现原型头/EMA目标及loss；`ssl_stage.py`仅接入同一既有fit循环和回执；`experiment91.py`加两固定身份；`runtime/r115_ssl_gate.py`仅为此recipe核辅助教师初态；`configs/R230-SCIENCE.json`/`R231-SCIENCE.json`；`tests/test_ema_assignment.py`及既有SSL门测试的必要部分。data/sampler/gather、监督训练/保存/评估/监控/启动平台不重建，活跃R228冻结不改。CPU准备独立`technical-prep/ema-assignment-20261103-4e0c`，原24核0–23/nice19/隐藏GPU，与96核训练互斥，不派生Agent。

- [x] 先Pod两项缺失行为RED；GREEN含解析交叉熵独立oracle/教师及center stop-grad、动量及center算术、原型归一化与首epoch冻结/解冻、公开head-age及RNG保持、两SCI完整R107合同、实际教师初态门。
- [x] 最小实现、19必要目标/5编译，真实公开Small/P8双视图/65536头/349有限梯度/学生更新/教师EMA/冷derived通过；722步原型冻结，监督5空组，CPU fixture退出删除，不进训练。证据e58e363e/guard65e9ed62，143.844秒，见[准备回执](../../records/preparation/ema-assignment-20261103/README.md)。
- [ ] Git同步/API回读/精确源和技术包，正式CPU及已有入口准备在R228运行期间完成；不新增GPU预演。
- [ ] R228完整selected双门过则优先R229，否则唯一actualsafe后R230；R231只在R230完整双门后激活。完整内部AUROC主判，AP与原固定20/其余147修复/新增错序及真实成本并列，比较R107/R120/R115/R224/R226/完整R228；seed27比较R108/R121，官方仍原≥.996一次门。

后备固定设计已实现并通过必要CPU验证，尚未正式部署、GPU启动或证明收益；当前唯一GPU仍R228。必要最小改动按用户既有授权直接实施，无额外任务数据、资源或审批门。
