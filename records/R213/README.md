# R213：原生作者 EfficientNetV2-RW-S

已[完整结束](R213-RESULT.json)：8000 success=attempt、10次内部dev、连续4轮无提升、零跳过；唯一EMA c6@4800内部AUROC`.9830353369271913`、AP`.7949732693925452`、error_twice28178。严格R107/R120双门false，R214不eligible，官方按阈值未跑；raw .999未达。tensor2d3e2466/blobdc3ca40d/CSV4c7c3ea4/precision28227e91；原freeze.status=null保持。[唯一actualsafe](R213-COMPLETION.json)/guardg3i17om6/73fa8618已证owner inactive、8rank退出及设备无人占用，旧safe不重查；[归档](R213-CPU-COMPLETION-ARCHIVE.json)emqd9wmz/2431f7380。后续[R215](../R215/README.md)真实首成功已证，本轮所选暂保护至已选模型fit侧诊断与结果归档完成。以下为原启动证据。

11:04历史快照3875/0skip/eval4/bad0，暂best EMA c4@3200内部`.9830160712157026`。[真实初态与首25](R213-START.json)已证459参数张量全训、五组/各rank空state、whole cb874611/params f46b7d1d/buffers eb77f82b与CPU一致，112公开BN保持eval、336buffer固定。首25 loss`.6861116290092468`有限，WORLD8/G64P8/448pairs、各rank8primary/8view符合原合同。generic非Lion optimizer身份字段不等于独立实测每rank逐组AdamW kwargs。

直接原R107的512/FP32/ImageNet/G64P8/448原logistic/原AdamW-LR-WD/增强/age/globalclip/24k-800-p4/EMA1600，仅改原生作者efficientnetv2_rw_s.ra2_in1k及匹配ImageNet-1K公开fresh、固定公开BN。完整794原生1000类先strict/exact加载，再仅改Dropout.2+Linear1792→1，native.drop_rate=0，外层age一次。它是架构/公开初始化/固定BN组合假设，源288/test384不替代实际512，不声称纯架构因果、临床机制或收益。

source936dde0812c5c2a06e468cf6cc2f359fc4f2ea1a/tree06cc0c7cdab574f978f220e2163fbdc91cb8fbf1；SCI7f6de4e13109a6e0262fd9dfa91d2a5247fc6df34e13d55eaf3cf08348c69420；actual SITE52d85df16fdbdf387d8aa97b1ee6aaf026d567128c42aec24fe032f228f3faf9。[normal19/19](../preparation/efficientnetv2-20261103/NORMAL-R213-LEASE2.json)在R211训练期间完成，guardkfsbv092/1d166f17；7受影响目标/12唯一源码compile与两fresh按SHA[复用](R213-CPU-REUSE.json)，不重复验收。manifest1d41e822/cpu-prep68d88144/deploymentf4d7fa24；默认12GiB/free16GiB不变。

原lease1正常CPU已过，但完整clock窗口不足按原逻辑拒绝，未activation/GPU；另建未启动lease2，只复用精确technical/source材料，不改旧allocation或回执。lease2窗口guard2c0klwgg/91745676、一次activation guardbmbe92o8/7cf14c0c、显式八设备direct launch均0；原四调用在同一次SSH中依次执行、逐项保留退出检查，无新启动平台/GPU预演。start1 unknown原件保留，start2/guard99o2efuv/5dc76327为实际首成功。唯一owner730484/start373000787；八rank730496–730503，训练24–119/96核，rank24–111各11核、control112–119；准备d837/root0–11与保留peer12–23互斥。当前本地r213-pod-context.env绑定lease2/start2，旧r211 env不再作为当前入口。

R211[完整所选](../R211/R211-RESULT.json)双门false，R212不eligible；actualsafe唯一原件后才接续。[实际间隔](R213-HANDOFF-GAP.json)：整轮结束至dispatch116.095秒/设备183.371秒/首25记录232.264秒；末训练日志至三项186.056/253.332/302.225秒，最后一项略超300如实保留。连续DCU利用率未测。32e3唯一Git/GPU；不恢复/派生Agent，不发训练信号、不热改当前冻结。
