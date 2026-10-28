# 新阶段续验与文件内容证明

新 Pod 的受影响10项检查、显式阶段绑定和完整 strict 已真实通过。来源语义与旧八项不重复执行。生产 binary+age 工厂、参数组、空 Adam 和迁移后路径由A独立验收。

## 真实失败与最小修正

首次新增检查9/10通过，同长度原位改写未被拒绝。原件 [REVIEW-TESTS.json](pod-51fe3daf/REVIEW-TESTS.json) 保持 false，Pod SHA `fedb168ecc1d862b1085cc5817a70f70ebf76c9fc0f18cf01a14edcc08cbb178`。

仅做一次 scratch 诊断：[FILE-IDENTITY-DIAGNOSTIC.json](pod-51fe3daf/FILE-IDENTITY-DIAGNOSTIC.json)，Pod SHA `c0e1cbf845747a08e5964e31834c2c6a618eb44a8c2161077a69ee50213221fd`。三字节 `{}\n` 改为 `[]\n` 后，SHA 从 `ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356` 变为 `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`，但 write、utime、fsync 后完整七项身份仍相同，mtime/ctime 仍为 `1793184067000000000`。实际 checker 输入、源码和 SHA 都在回执中。未 sleep、改长度、skip 或扩展 FS 实验。

该证据说明 stat 只能拒绝已知变化，不能独立证明当前内容。经协调、规划与A确认，最小修正采用 `fresh_pod_sha256`：完整元数据仍记录并参与快速拒绝，匹配后必须核当前 SHA；未知/缺少 evidence.kind 拒绝。原同长度回归保持三字节、同 inode，不延时，修正后真实通过。

## 单一阶段接口

`stage_binding.py` 读取成对 path+SHA 固定的旧 SOURCE-CHAIN，新阶段不改写历史。`strict_load.py` 接收成对 `--stage-binding/--stage-binding-sha`。同 UID 路径保留全部五项身份门；跨 UID 无显式绑定直接拒绝。

- `schema_version: 1`，`stage: pretraining-strict`。
- `source_receipt` 恰为 `path, sha256`，与 strict 参数一致，旧 SHA 为 `0d46797900fdef0a479f68845ace7a1ce4ad83165cccfd22a7b4e91bacb75414`。
- `execution` 恰为 `namespace, pod, uid, container, python`，逐字段匹配本次 cpu_context。
- `cpu_allocation` 恰为 `path, sha256, executor_id, threads, mask`，mask 取 context.channel_mask。affinity/torch/后代审计留在执行与通道回执。
- `assets` 完整11件：9件 files、provenance、historical_converter。身份恰为 `path, resolved, bytes, device, inode, mtime_ns, ctime_ns`；当前 SHA 必须与历史 pin 相同。
- evidence 只接受 `fresh_pod_sha256` 与旧链精确 SHA。新设备/inode 等变化如实记录，不写回旧链；`unchanged_shared_object`、未知或缺 kind 拒绝。
- `runtime_snapshot` 固定为旧链中的 Git `cb939c8e5078a9c74a26158a78c4aa033009042a` 与逐文件 SHA，不依赖A的临时 runtime。

stage 创建与 strict 使用点各独立核11件当前内容，每次 `228873307` 字节，均如实记录 weight_blob_hashes_recomputed=true。读取前后核 path/fstat、长度和实际 SHA；strict 后 stat 检查明确不称内容证明。未重做来源语义、旧八项、官方或全历史。

## 环境与结果

新 Pod `acfwvv9ejn/vcjob-2115385083737251842-acfwvv9ejn-44186-worker-0`，UID `51fe3daf-ba0d-46e8-919c-99caec599c99`，container `vcjob-2115385083737251842-acfwvv9ejn-44186`，Python `/usr/local/python3.12/bin/python3.12` 3.12.7。这些是实际回执身份，不是工具默认值。

A与规划实核 effective120核，总20%=24，B12核 mask12..23、A12核 mask0..11。B自身 probe 成功后执行；CPU-ALLOCATION-2.json pin SHA `25fe957398e81748c170d929b7b662b24270d1012e7f2940ec28ed315d9385ad`。B只消费通道分配，不建立第二全局预算。nice19、隐藏 GPU、无额外 workers，strict 实际 torch intra12/interop1。来源任务已退出并报告释放；后续迁移须核当时的新份额，不能据此复占。

新 prep `/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/pretraining-cpu-20261028-51fe3daf-b`。10项检查0 failure/0 error，实际测试用时5.772秒；stage、strict 和182键映射 SHA 见 [README](README.md)。

实际11件路径及七项 stat 见新 stage/strict 原件。公开原始/转换权重仍读历史 r16/assets，恢复 README 与 converter 仍读旧独立 prep；新设备号132、旧链设备号161均保留。不能扩称随后复制的新路径已通过。

实际工厂为安装的 timm/models/convnext.py（timm1.0.27，SHA `a2ef5eb22cc1039dadcddcbf7b243034e98a553630349483302ed21d09fdb50f`）和 torchvision/models/convnext.py（0.25.0，SHA `5c00c412bc79e5c650f887dfe0d21aa8028903a3bbeae1a95c3b0fdfbdaf55c2`）。source/tv182严格加载双方 missing/unexpected 为空；180backbone、18gamma变形与FP32数值完全匹配；23LN均1e-6、18GELU none、22卷积非tensor和18SD匹配、BN/buffer0、native avgpool768。新binary原生Linear仅为无age演示，不是生产工厂准入。

| 验证源码 | 新 Pod 实际 SHA256 |
| --- | --- |
| tools/pretraining/source_chain.py | `5d183e1a0949e913956d57647f6b819846696c34fcb1b4f5d1c82c1510d76473` |
| tools/pretraining/strict_load.py | `c6a16e38ce1593b8675a5f3068d1e833e5326170b131840d3702f7b1316c019c` |
| tools/pretraining/stage_binding.py | `82c49b1bec1b3084afe2a50501a0e61c62789ec9841ad0ff627da5da3d46c47e` |
| tests/test_pretraining_sources.py | `a864eda00f18a240fe99e6136edeaedc310c54f618b53a427070eb34d3b8de94` |

SHA 来自实际 Pod 回执，本机只编辑/展示/传输。验证源码在终态后未改；原件未改写。A唯一负责集成提交和GPU；B无Git写入、GPU或正式训练权限。
