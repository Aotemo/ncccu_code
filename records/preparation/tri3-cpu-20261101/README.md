# 三处 Tri3 的 CPU 技术准备

本次仅准备结果无关的条件候选，未注册新 SCIENCE、profile、order、正式目录或 GPU grant，未运行 normalformal、activation 或训练。R132 继续使用冻结 runGit `c11a0829f9f6e75b4f03872248008ee7cd9aae00`；其完整双门通过时固定 seed27 仍优先。[授权、科学边界及停止条件](../../../todo/auroc-999/NEXT-PREPARATION-20261101.md)是后续入口。

实际 RED 为缺少实现导致的两项 FAIL、0 ERROR；最小实现后 GREEN 两项 2/2、0 fail/error、4.795 秒，6 个受改 Python 文件 compile 通过。未重跑旧 loss/Gloo、LoRA、fixture 或全套检查。规划完成唯一只读审查，本次 CPU 范围无 Critical/Important 或待修 Minor。

- [CPU-CHECKS.json](CPU-CHECKS.json)：GREEN 原件，SHA `d4403e5d4f233cd3b8644dd14339183e423e4e9112b56d40d878e177147fc096`，包含实际 6 源 SHA 和静态资源计数。
- [RED-CPU-CHECKS.json](RED-CPU-CHECKS.json)：保留预期失败，不覆盖旧证据。
- [CPU-GUARD.json](CPU-GUARD.json)：SHA `8ae50201c73fd9cbaeb0f8ec3839e508fe52df4226a03291b2d893c9d8827f72`，672 独立软件绑定下本人 12 核（0–11）、nice19、隐藏 GPU、exit0；根 12–23 保留未用，与训练 24–119 互斥。
- [STRUCTURE-AND-REBUILD.json](STRUCTURE-AND-REBUILD.json)：提取实际测试打印的 native 结构和保存后重建证据，保留输出来源 SHA。

第一项以独立 binomial 窗口和手工反射索引核值、解析输入梯度及边缘；native `initialize=False` 实查 345 个原参数对象、名称、值、RNG 和全可训状态保持，新增模块零参数、零 buffer，三处尾部尺寸正确。这不是公开 fresh 参数加载或整模型图像前向。

第二项实查 None 旧结构、RNG 和 job_contract 身份，并经真实配置、CheckpointStore 和 `inference._load_model`（显式小 toy_factory）完成 strict state 与实际结构描述；重建的三处 LN/Conv/Tri3 stage 逐一执行并零容差匹配。默认生产 factory 的可选 hook 仅经源码审阅，本次不证明完整公开 Small、EMA variant、GPU 或独立 standalone export 可用。

按实际三处通道与固定空间尺寸静态计数：新增卷积 MAC 为每图 `2587336704`、每 global64 forward `165589549056`；最大单个新增显式 FP32 张量 per-rank P8 为 `102242304 B`。不含 pooling、backward、workspace 或同时存活张量，不是峰值或新增峰值实测。R107 历史 6400 步小件时序的环境一致性未重新核验，不外推当前后端完整 24k、30 轮双 variant 和保存预算，也不靠早停兜底。CUDA 反射 padding 反向可能不确定，CPU 检查不证明 BW 后端严格确定性；实际吞吐、峰值、满程壁钟及 100ms 仍 unknown。

本次仅沿现有源码进行可选 policy、模型附加、描述和重建接线，所有已注册正式 profile 仍禁止非空 Tri3；None 不添加字段、模块或 RNG 消费。未修改优化器、EMA、通信、loss、保存算法、活跃冻结包及任何旧 clock/token/grant，所有旧模型和依赖保持保护。源包同步核验另存 technical-prep 原件，不为自引用证据追加提交。
