# R230/R231 fit-only EMA assignment CPU准备

[事前问题](../../../todo/auroc-999/R230-R231-FIT-EMA-ASSIGNMENT-20261103.md)先于R228完整结果以dfa0c0d同步。新信号为同一fresh学生的无梯度EMA教师、居中/锐化跨视图分布；只用fit图片，不使用label/age/dev/官方、已有任务模型或额外任务数据。公开Small、均匀G64/P8、原两独立512增强、1600更新和原R115优化器/LR/随机域保持；完整R107监督合同保持。

[19项受影响检查](CPU-CHECKS.json)e58e363e已过：5项新概率/梯度/EMA/原型冻结和候选门、4项既有SSL、8项空间重建兼容、2项既有候选，另5文件编译。先[两项缺失行为RED](RED.json)，再GREEN 81e0c02d guard；兼容导出guard0e4d285b。源码及配置SHA、未改data/gather/监督/保存/评估/资源/入口/监控链的精确身份见回执。不重复运行整个项目测试或独立GPU预演。

[实际公开Small/P8 CPU](ACTUAL-P8-CPU.json)143.844秒，guard65e9ed62/exit0：344公开张量strict exact、完整初态50c6f55c；8个真实fit图片/16独立视图，学生和教师各16 encoder-view forward。65536原型头、349有限学生梯度后按规则取消原型梯度；首完整fit epoch为722步，原型不变且没有Adam状态；教师初态与学生完全相同、无共享参数/可训练张量，更新后教师和center变化且无梯度。临时fixture学生encoder e5ba70d2只用于验证，原head-age不变，冷derived精确、监督5组空AdamW；fixture退出删除，没有进入训练。

CPU在同一实核Pod的独占准备0–23/24核、nice19/隐藏GPU通道执行，数值线程12/interop1，与训练24–119/96核互斥；guard最终通过，没有放宽额度。单rank CPU未实测八rank全局64×65536 collective，真实八rank初态和自然首成功由已有生产门证明。

头为无BN的768→2048→2048→256→65536，瓶颈及原型行归一化，scale固定1；学生温度.1/教师.04，center动量.9，教师.9995→1余弦，首完整fit epoch取消原型梯度。依据和小batch动量推断边界见事前计划。它不是原DINO完整配方或眼底收益复现；临时教师/头/center丢弃后只保留一个学生encoder进入原监督流程。每步128唯一增强视图但256 encoder-view forward，不能声称与R115等算力。

此时只有CPU可运行性证据，Git精确部署与正式包仍待，未activation/clock/dispatch。R228活跃冻结be5b5fa8不改；完整R228严格超过R107和R120才用已准备R229，否则唯一actualsafe后接R230。R231仅R230完整双门后固定seed27。内部AUROC主判，AP和固定20/其余147修复、新增错序、实际成本并列；官方仍原门。目标尚未达到。
