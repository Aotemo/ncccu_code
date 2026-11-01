# R150R1 独立工程重试

唯一Git/Pod/GPU owner32e3，2026-11-01 21:35 UTC实时Paseo只见此活跃Agent。R150已原始safe，重试只修复`prime_reducer`的3/6输入；没有可比较科学结果，仍沿`configs/R150-SCIENCE.json`/order62，不改原R150身份或clock/token/grant。

精确生产范围`runtime/bundle/retrain/distributed_engine.py`；回归范围现`tests/test_candidate_binding.py`新增单目标，RED真实复现、GREEN通过。原3目标及未改源码按SHA/AST证据复用，见[准备](../../records/preparation/native6-ddp-repair-20261101/README.md)。CPU-only32画布不是GPU验证；新正常formal512一次后，真实八rank初态/自然首成功必须尽快核实。

新round为`r150r1-small22k-seed20261029-scale085-pairlog-native6-24k-p4-9010-20261101`，全部新产物位于既有exp-v3。Git提交实际push/回读后精确部署，使用原prepare/formal/activate/direct launch，无新训练/监控平台或GPU预演。预热、采样、16次本地增强调用对应同8值、单Small单forward、FP32/G64P8、AdamW/age/EMA/max24k800p4/官方>=.996一次均保持。

prep仍00202581本人CPU0–22/23核，槽23保留，与训练24–119/96核互斥；23:24 UTC到期前须为新CPU阶段另立实核分配，不能改旧分配。现R148完整8000/eval10/bad4、唯一EMA c6@4800 AUROC .9872244250690254，两门false，关闭联合p0/no27；R149未formal/activation/GPU。R150实际空闲已>300秒，如实保留原因，不跳过必要检查。raw .999未达，持续推进。
