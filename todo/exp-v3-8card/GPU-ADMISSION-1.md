# 首次 GPU 准入：失败保留

最新用户资源纠正：执行Agent准备计算20%，模型训练及其GPU准入整树80%。此前将20%扩大到GPU全树是协调/规划的误读；以下记录仅描述20af/attempt1当时执行和失败事实，不是新阶段资源标准。旧源码、allocation和失败原件不改，新阶段使用独立80%绑定；helper全128仍超新合同。

本轮首次技术准入使用已同步实现`20af511b0177c0f1576c8c630baaf1865d8001f9`、tree `85fc6860bec2b5827a990e864ca444e0472550b4`，部署根为`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/r93-frozen-20af511b0177`，attempt目录为其`gpu-admission-1`。远端ref/commit/tree与Pod实际bytes/mode/SHA校验先完成，再执行GPU。当前Pod UID仍为`51fe3daf-ba0d-46e8-919c-99caec599c99`。

正式身份CPU包生成exit0，初态与已验生产模型一致：PACKAGE-MANIFEST SHA `0359f02e978f5d2803addad834980c213d8e4a2bfdd156aa415352fb68666212`，CPU-PREP SHA `7d7a57390e3476d62fea472261ee103442a564c03be43bc419cb9f3ddc83242b`。这不是GPU通过回执。

实际失败证据保存在`records/environment/20261028-8card/gpu-admission-20af511b0177/`，服务器原件仍在exp-v3：

- 首次wall49.280953秒；8个rank原始FAILURE均在最终thread_masks断言失败，各有一个native helper线程mask为0..127，超出当时各自获准CPU mask，触发当时20%门。保留该历史失败，不将旧20%限制继续作为训练目标。
- 外层CPU树审计向owner发SIGTERM，owner以-15退出；原owner没有在该信号路径有界wait/reap其子进程，OWNER-RESULT缺失、rank退出码未收齐。此为继发退出管理缺陷，与CPU越界分别处理。
- 后续只读诊断中owner已消失，8rank为PID1下Z；device_users为空，8卡busy0、每卡used2207744B。GPU资源空闲不能改写成all_exited，也不能补造成功rank proof。
- `HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0`实际已设置；容器默认`NCCL_IGNORE_CPU_AFFINITY`为空，rank0原日志出现affinity03。现有证据尚不能确定哪个库创建或扩宽了逃逸线程，不能把失败误称为NCCL未初始化或直接断言具体库根因。
- native模型与FP32 collective执行已走至后置mask门，但没有完整成功proof，因此本次始终为`failed_or_resource_unknown_do_not_retry`。无数据forward、无成功更新、无正式R93 clock/token；不将技术状态作为正式初始化。

只读诊断原件为`failed-admission-diagnostic-1/DIAGNOSTIC.json`，Pod SHA `d59e43400ac31665789d02b555ab1add7d13391490dba0ccb744e3eeca53f6d3`。失败、退出审计、逐rank FAILURE与执行env均保留，不覆盖20af冻结包或attempt1。

后续范围：A继续必要的CPU亲和性根因取证；A1仅在既有准入入口和对应测试两个文件中修复TERM后的有界退出收集，父A统一做必要Pod CPU验收。成功与失败都记录真实exit code，Z不能当完整回收。不杀PID1、不重启资源、不热改旧包、不重派旧attempt、不增加benchmark，不重来源/迁移等未受影响检查。修复具体差异和真实CPU证据齐后先同步新版本、实核部署SHA，再根据明确的旧进程/设备状态决定独立新准入身份；R93保持未启动。
