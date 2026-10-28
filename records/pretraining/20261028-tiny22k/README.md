# Tiny22K→1K 来源与 CPU 验收接续

2026-10-28，执行B的独立任务。**来源链与独立 timm/TV 工厂的完整 strict-load 已通过。生产 binary+age 工厂、参数组、空 Adam 和正式部署仍由A独立验收，不能据此启动 R93。** 未做 GPU、模型前向、数据集读取、训练 activation 或 Git 写入。

协调后续审查指出跨 Pod 续验缺口及同长度改写的 SHA 复用风险。新 Pod 的真实诊断证明七项 stat 相同仍可发生内容变化，因此新阶段与 strict 使用点分别对必要11件资产核当前 SHA，采用 `fresh_pod_sha256` 证据。修正后的10项窄检查、新阶段绑定和完整 strict 均真实通过，见 [REVIEW-FOLLOWUP.md](REVIEW-FOLLOWUP.md)。历史四件回执、9/10失败回执与 FS 诊断均保留原样。

## 实际结果

- 精确来源为 `timm/convnext_tiny.fb_in22k_ft_in1k`，revision `bc48a87f119bae1b06418f5340bfa97ef75d4609`。模型卡、API、LFS 原 blob、配置、下载回执、许可证和上游通知的链路已经核对。旧来源语义检查引用执行A的独立盘点；新 Pod 为补足实际内容证据，在 stage 和 strict 两阶段分别核必要11件 SHA，未重做来源语义或全历史检查。
- 缺失 README 从本机历史原件原样传入独立 exp-v3 准备目录，在旧实核 Pod 内所得 SHA 为 `5829109fcf28ec11af3f28fe4477a8319a4aaf39aa67a3ab7b4499e8d1720d4d`，与原 provenance 完全相同。未修改历史资产目录，未用其他 revision 替代。
- 历史 converter 的 Pod SHA 为 `82e74a0056e0b9035fbd06a9c354889384186bf480492e113bb937b35ce83bb5`，匹配 provenance。源码对照为 Git `cb939c8e5078a9c74a26158a78c4aa033009042a` 的独立固定快照；完整文件 SHA 见来源回执，不受A的后续文本修改影响。
- 第一次完整验收在 GELU 检查处失败。旧失败回执仍为 `passed=false`；新的成功和正式182键映射保存在独立的新 Pod 子目录，未覆盖旧结果。
- 窄诊断确认 source 为 `timm.layers.activations.GELU`，实际 `forward` 为 `return F.gelu(input)`；TV 为 `torch.nn.GELU(approximate='none')`。原验证器只查 `isinstance(nn.GELU)`，漏掉了自定义类。修复只涉及观察器：精确类型、已核 forward SHA、实际 F 绑定与默认 none 语义；tanh 和未知实现仍拒绝，生产 GELU 未改。
- 修复后的八项窄测试在旧 Pod 真实通过。随后旧 Pod `NotFound`，**旧 `STRICT-LOAD-2` 未执行**。A与协调后来实核新既有 Pod 的 Ready/UID、解释器、资源和预算，B通过自身通道 probe 后才开始新阶段。未创建计算资源、未回退本机或登录节点执行。
- 新 strict 对原始 timm182与转换 TV182张量严格加载，双方 missing/unexpected 均为空；182项逐键、shape、FP32有限性及数值完全一致，含18个 gamma 的 `[C]→[C,1,1]` 映射。双方23处 LN 均为 `1e-6`，18处 GELU 均为 none，22处卷积非张量配置和18处 stochastic depth 匹配，BN/buffer 均为0。原生 avgpool/768特征与1000类头通过；仅丢弃公开分类头后，新建原生 binary Linear 的演示保留180个 backbone 张量。演示不附加 age，也不替代生产工厂验收。

## 新 Pod 成功回执

服务器根目录：`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/pretraining-cpu-20261028-51fe3daf-b`。本机只原样传回文本，未在本机或登录节点重算 SHA。新 UID 为 `51fe3daf-ba0d-46e8-919c-99caec599c99`；所有新阶段明确 pin 下述旧来源回执，不把其历史 UID 当当前身份。

| 回执 | 新 Pod 返回的 SHA256 | 结果 |
| --- | --- | --- |
| [REVIEW-TESTS-2.json](pod-51fe3daf/REVIEW-TESTS-2.json) | `bb801bd7752eb4a910d9a4b047efcade58835bb11eee7dfc29ebca7a92980587` | 受影响10项通过，0 failure/0 error |
| [STAGE-BINDING.json](pod-51fe3daf/STAGE-BINDING.json) | `7277013030069484cca9eac7d5ebc674225a54751edfa890338ed4780bd04a5d` | 新执行身份、预算、11件当前内容与固定快照绑定通过 |
| [STRICT-LOAD.json](pod-51fe3daf/STRICT-LOAD.json) | `9d74f10363d622b21e760811a6cfc323c42c63436c2d19ebb36e4a8d848c94f6` | 修复后的完整来源/转换/工厂验收通过 |
| [TENSOR-MAPPING.json](pod-51fe3daf/TENSOR-MAPPING.json) | `7741247a9be209399de643a591c94e05303254128abf3578a9dccb1be26db8c2` | 182项映射明细，无权重值 |
| [CPU-STRICT-RECEIPT.json](pod-51fe3daf/CPU-STRICT-RECEIPT.json) | `b1890ba1ca1e68653004ea2084220857462bfddec2bc3fbe33727ba9d97c8b8f` | 所有 strict 子进程退出，exit0，通道配额审计 |

## 旧 Pod 原始回执

服务器根目录：`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/pretraining-cpu-20261028-execution-b-5d11d2c`。以下四件文本已经原样传回本目录；SHA 是旧实核 Pod 产生并返回的身份，传回后未在本机或登录节点重算。

| 回执 | 旧 Pod 返回的 SHA256 | 结果 |
| --- | --- | --- |
| [SOURCE-CHAIN.json](SOURCE-CHAIN.json) | `0d46797900fdef0a479f68845ace7a1ce4ad83165cccfd22a7b4e91bacb75414` | 来源链、恢复 README、converter 与固定快照通过 |
| [STRICT-LOAD.json](STRICT-LOAD.json) | `ab29731f11f38f4a37c32c45c1ded10910a9b8cb283419db872851152d005dc9` | 首次失败，保留 GELU 原因 |
| [ACTIVATION-DIAGNOSTIC.json](ACTIVATION-DIAGNOSTIC.json) | `11a913b723384184fd9ab187cd1d1e0c6585514f1d00fc3e5e1b9c89f24ce405` | 激活模块及实际源码观察通过；不是模型准入 |
| [TESTS-2.json](TESTS-2.json) | `032841cd319bab2f097ba4f6fad091365a6e300558a2445a35ad7e3cb8651efc` | 八项窄测试通过，0 failure/0 error |

旧 Pod UID 为 `f9fe6762-7127-479b-8e80-09ce2487d754`，仅是上述历史回执身份，不是执行默认值。失败 checker 原件另存服务器根目录的 `strict_load.failed1.py`，SHA 见诊断回执。

原始 timm 激活源码文件 SHA 为 `effc50cc017066d60f3e8bc9f4a65b2765ad374a1b4da7bf023c1db99e8f1cd9`，其 forward SHA 为 `375052b2c2b44c570274dee239db60f55afda25d413c98c827561f63a0ca5a77`。TV 激活源码文件 SHA 为 `f335544f19b07e279dd729ba4fbc1de0226797fdf9fa2e790670546499df0efa`。完整实际类型、repr 和代码片段见诊断回执。

## 继续边界

生产工厂、fresh binary+age、参数组和空 Adam 由A在固定的新候选源码上集成验收；独立 timm/TV 的来源验收不替代该步骤。候选末端与内部 LN 均为 `1e-6`，R92 末端为 `1e-12`，不能混记或称只改变预训练。正式资产到 exp-v3、GitHub 同步与部署身份仍由A处理。

用户后来调整为两个执行 Agent 合计使用实时有效 CPU 的20%，覆盖旧两线程限制。新 Pod 实核 effective120，B分配12核、mask12..23；这是本次显式分配，不是工具默认。新回执记录 nice19、隐藏 CUDA/HIP/ROCR、torch intra12/interop1，未启动额外 workers。工具只消费A通道注入的身份、预算和 affinity。旧四件回执仍对应旧限制与其源码 SHA；没有为提高额度重跑旧八项或来源语义。

新 strict 终态后B重CPU任务已退出，份额已报告释放。A负责后续生产工厂集成、实际依赖迁移、GitHub 同步与冻结部署；任何新资产路径变化都需其新身份，不能把本回执对旧资产路径的验证扩称迁移后加载通过。B无 GPU 与 Git 写权限。
