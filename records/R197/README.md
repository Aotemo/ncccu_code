# R197：原 Small sigmoid 的独立 fit gamma 增强

[9个非保留权重已按真实清单删除](../storage/closed-r197-20261103/README.md)，约6.73GB；三模型KEEP及当前R199保护快照保持，CSV/precision/侧车/选点和一次性控制记录保留。无新forward、训练信号或旧safe重查。

[原封存难组补充](../analysis/20261102-persistent-errors/R197-FIXED20-SUMMARY.json)复算与完整指标一致：固定20正例错序质量6334、其余147为3241、总9575；比R107难组少289却其余多264，净少25，两组仍均高于R120。无新forward/官方或分组/标签改变，不据局部结果改采用门。

11-03 02:45完整[终态](R197-RESULT.json)：10400 success=attempt、0skip、13次评估、bad4正常早停，唯一EMA c9@7200内部AUROC `.9884706757809537` / AP `.8262997760790266` / error_twice19150。严格胜R107但未胜R120，R198条件复核不eligible；官方不足.996未跑，freeze原status=null保留，raw .999未达。固定gamma配方本轮未达采用双门，不外推全部光度方法。选中tensor6251937d、blobbe7c5591、CSV76554b38、precisionafc4acea；没有新增forward/改标签。[实际安全观测](R197-COMPLETION.json)与[guardm1jbs9bo](R197-CPU-COMPLETION.json)均完整：owner Z/nonactive、八rank缺席、锁holders/device users空、denied0及八卡busy0，封存后不重查旧safe。已沿原入口串行接续[R199](../R199/README.md)，本段覆盖下文早期running状态。

11-03 01:35:40.963 UTC[既有监控单次快照](R197-DEVICE-POINT.json)八绑定PCI全部busy100%，非连续利用率；[guardi9s5j806](R197-CPU-DEVICE-POINT.json) / `2ab3c4f81f69c14927d117066fc4dab3b4338020eb2eeaa6bbafefec763dce56` exit0，0 GPU API/模型/forward、未改runtime。01:42最近2525 success=attempt/0skip/eval3/bad0，暂best EMA c3@2400内部AUROC `.9863695091217123` / AP `.8258739741425536`，尚无完整selected。

[真实八rank初态与首自然25](R197-START.json)已通过：345全训、5组空原生AdamW、whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`与formal一致，WORLD8/G64P8/448对、各rank8图，有限loss `.9216874241828918`。11-03 01:23最近175 success=attempt/0skip/eval0，八rank活跃，尚无完整selected或官方结果；raw `.999`未达。

仅直接R120原Small/public fe6/seed20261029/512/FP32/ImageNet/sigmoid新增fit `gamma-reciprocal075-v1`。原八次sample-key几何/光度抽样后，以独立流等概率选`.75,1,4/3`，同RGB查表`round(255*(i/255)**gamma)`，转张量/原normalization前应用；eval不变。原backbone1e-4/head-age1e-3/default AdamW/WD1e-4与原24000余弦保持，不叠半LR、8000跨度或后续WD=.05。最多24000成功更新/每800内部评估/p4早停/EMA1600与唯一内部AUROC选择保持；固定曝光鲁棒性是假设，未证明临床机制或收益。

冻结commit `436ae5b72dd51bbd567e7e5df60762e44ccb70bf` / tree `3613123ab20606aa68e427026de1fc5bcb2f1c64`、SCI `6000c2a8366ce1464032e1bc0e5801075dead535b6f868cd3c969ee4344f892c`、actual SITE `8bbbdd4394aeba44933d6151a804cfe632322cfcda6eb19c328072a0c8fb9298`。完整344张量公开fe6来源与原R120相同，未用本任务checkpoint/teacher/医学初始化。[正常CPU18/18与精确部署](R197-PREPARATION.json)在R195/bad3仍训练时一次完成，[五受影响目标/五编译](../preparation/gamma-20261102/README.md)按[精确SHA复用](R197-CPU-REUSE.json)，原像素/独立RNG/配置/schema及两actual fresh有CPU证据；未重跑未变模型/P8/EMA/cold/WORLD或独立GPU预演。[准备](R197-CPU-PREPARE.json)、[一次activation](R197-CPU-ACTIVATION.json)、[自然首成功观测](R197-CPU-FIRST-SUCCESS.json)guard均0。默认12GiB/free16GiB保持。

R195完整6400/c8/bad4、EMA c4 AUROC `.9861756478998568`，严格两门false/官方未跑/actualsafe后，R196不eligible，独立本轮。[实际交接](R197-HANDOFF-GAP.json)：整轮结束→dispatch107.931秒/最晚八卡设备初始化169.297秒/首25记录245.468秒，均≤300；从末次训练日志起分别182.964/244.331/320.502秒，最后一项仍>300。整轮结束→安全观测36.680秒，安全观测→dispatch71.251秒；未独立分解owner收尾与观测处理，连续DCU空闲未测，不声称全部5分钟要求已证。后续进一步缩短终态发现和临界段处理，不改旧clock/token/grant或跳过必要检查。

唯一Git/GPU owner32e3；01:07实核peer idle/无activeTurn，未恢复/派生Agent。准备53ab/root0–11、peer12–23保留，与训练24–119/96核互斥。新只读`work/r137-20261101-a/r197-pod-context.env`由实际start-2/当次allocation和UID生成，不source旧绑定。归档[guardjda0hmwn](R197-CPU-HANDOFF-ARCHIVE.json) / `357343d478870107d9f5666bc42d6c44da5baf5040dedb5676f9821f20e7aa5e` exit0，无新forward/模型/官方。后续R198条件gamma seed27/R199独立原R120 WD=.05仅82add93两technical包，R200/R201仅预登记；当前436冻结不改。
