# R153 / R154 固定后续 CPU 证据

本准备由唯一owner32e3在当前Pod的b781分配、nice19、23核0–22和隐藏GPU下完成；R152仍使用b58冻结、96核24–119和全部8卡。未新normalformal、grant、activation或GPU，不称下一轮已启动。规则与精确文件范围见[固定计划](../../../todo/auroc-999/R153-R154-PREPARATION.md)。

[CPU-CHECKS.json](CPU-CHECKS.json) SHA `24fc6daca152d5a614999d2cdc483e5579b8a0940f81a2817045e1d37a7ff0c7`，实际3/3受影响目标、12源编译通过；[原始最终guard](CPU-FINAL.json) SHA `6b197d4f861707c266af69f55751a9b9b53e57c98dd43b7ddcfb0114d1c146c7`。覆盖两SCIENCE在配置/保存合同中的模型、seed与几何消费、旧R151/R152绑定及768派生cache兼容、实际V2 Small 512前向、EMA1600/1601/24000静态缓冲区逐位保持、既有CheckpointStore保存与实际inference入口严格冷重建，输出相等。CPU八rank fixture/normalformal及真实GPU初态仍待各自一次现有正式链。

| 分支 | fresh whole | 全训tensor / 元素 | 空AdamW组 |
|---|---|---|---|
| 条件R153 / seed27 / RGB768 | `91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67` | 345 / 49455459 | 5 |
| 独立R154 / seed29 / Swin V2 Small RGB512 | `4ff58b9c91fb82b9577bcd7de7711d3eb07e24bd05b8dcacd94c54c117b58726` | 402 / 48969213 | 5 |

两初态由生产factory实测，未加载历史训练权重、未前向数据图像；[R153原件](R153-PUBLIC-FRESH-INITIAL.json) SHA `130c5778e9ca9c251378ddb7c37249560436b4eb91b817bd243ccc424c05ee0a`，[R154原件](R154-PUBLIC-FRESH-INITIAL.json) SHA `c52fb6e510ad5d4c5c607a53755816f6bb8a60bc523c83b98a24c0101cd7f242`。R153只在完整R152 selected同时严格过R107与R120且actualsafe后才有GPU资格。

R154公开源是[固定revision模型卡](https://huggingface.co/timm/swinv2_small_window8_256.ms_in1k/blob/db3a0c7e5d97487e7e874c943838569b84fe8e54/README.md)声明的MIT/ImageNet1K。源权重 `fe5783254b12041fb00864de00e51f00826f318a8077023ac472a175f5c882a6`、200976925字节，实际LFS身份匹配；源436项（425参数、11源窗口mask）完整核验，q/v bias精确拼成q/zero-k/v，401目标参数加48确定位置buffer共449项。沿现swin_source/swin_strict扩固定选项，实际timm1.0.27与TorchVision0.25的2×3×256×256完整1000类FP32前向max/mean误差均0、top1一致；转换保存再严格重载及逐张量精确核验。[严格原件](STRICT-RECEIPT-2.json) SHA `d731a0d121d5f58513bd4b06de5396a2da731d294415cb8b6034c782149910bd`，转换权重 `e2052f3313bce47b14c404b18b0ca31e3e1f44cf9563cb2c9fe93faa3ac38d42`，provenance `c9471d7261940e7730259988ae976aa26f0cdc5d3dae46c1cc097059d44c2518`；48个静态buffer摘要 `e03d8ca9249cb44539c4cd13937ecbee3329a3d00d704dbe13590d6f3c7477d0`。48包括24个整数索引和24个FP32坐标表，EMA原样复制这些确定常量，可训练参数仍用原公式。

原失败均保留：[RED](RED.json)为新增模型尚未注册的实际失败；Pod旧proxy下载超时与直接TV工厂重复stochastic_depth关键字错误由CPU-CHECKS中原guard定位。首次转换在写mapping回执时缺`passed`字段，[失败原件](STRICT-RECEIPT.json)及[部分产物保留清单](PRESERVED-PARTIAL.json)未覆盖；补字段后第二次严格链才成功。最终测试在补实际保存/推理检查后重新执行一次，不将早期报告冒作最终源码通过。

预算仅作历史工程情景：R117原8800步组件wall3843.433秒，满24000线性10482.09秒，×2.5为26205.225秒，小于原新轮train软27750秒；R107原官方组件823.401秒×2.5为2058.503秒，小于official软2250秒。不是V2当前GPU benchmark或耗时/内存保证，不假设早停；集中8.75h整轮/7.75h训练与原9h allocation策略保持，R152原clock不改。完整内部唯一选点>=.996才官方一次，raw .999未达。

2026-11-01 23:48:16 UTC实际Git同步与技术包核对完成：[源码04172db46eccc884f5d4e242c1235d066c8657af](https://github.com/Aotemo/ncccu_code/commit/04172db46eccc884f5d4e242c1235d066c8657af)，tree `c3e626dc38fc8c528b271ae718b6ebecc7986462`；实际远端commit/tree回读、12源/两SCIENCE/集中配置SHA匹配。沿既有prepare构建R153/R154技术包各5.917/5.858秒，技术SITE为`65a1c451bebb158a84a851108687b95980aa0e569b25c7335185cc1fcbc99958` / `881a98e9aecbba35e35845417b34ac6d5eb5cca358c2f94e251c381aad3ff7ec`。现有formal调用参数wrapper与首成功metadata读取器仅适配两新身份和动态参数计数，Pod编译/入口bash语法通过；[NEXT-READY](NEXT-READY.json)、[CPU-READY](CPU-READY.json) SHA `1d8d0c091be5ce4f8b79efea67ed73fb39731e52a8585e138aa729b75bac1ba8`。仍未normalformal/grant/activation/GPU；正常formal和精确运行部署只在前轮进入freeze时执行。

2026-11-02 00:36:34 UTC增加必要的新架构启动路径证据，原04172db生产源码未改：[PRIMING-CPU](PRIMING-CPU.json)、[CPU-PRIMING](CPU-PRIMING.json) SHA `8f41efeada6acf4667822b9dee03c411d635d69f4078826e377f54193f728fbf`。消费上述真实R154技术SITE/TRAIN_CONFIG，经现resolve_topology获得world8/per-rank8；实际公开fresh whole4ff与正式合同匹配。现prime_reducer在CPU上对8×3×512×512执行原两次训练前后向，44.632秒、804次有限梯度回调覆盖全部402参数；模型/48静态buffer/模式及Python、NumPy、torch RNG逐位恢复，grad清空，AdamW空。不曾初始化实际DDP/collective/GPU、推进成功更新或读取数据图像；真实8rank与自然首成功仍待新一轮实际启动。原3目标/12编译证据不重跑，不改变其含义。

01:38 UTC用户新分析会话按[任务页](../../../todo/README.md)四文件边界仅用CPU23。新未启动轮的调用wrapper另存`RUN-SWINV2-NEXT-PEER.sh`，只替换准备分配路径与SHA为aa0f（A0–22/23核、peer23/1核），原b56 wrapper原件保留；不改04172db运行源码/两SCIENCE/旧GPU绑定或clock。实核bash语法、精确仅两处绑定替换和120核三池互斥通过，[NEXT-BINDING](../persistent-peer-20261102/NEXT-BINDING.json)，新wrapper SHA `41eb34d4ee433e5b9f0fe20ff9bf1f91bf81a7462b1e5358f69c90dd382398d7`，[CPU guard](../persistent-peer-20261102/CPU-NEXT-BINDING.json) SHA `47819b9de42f523db66344130dff713d403cff89b2a4f7fac082f7284418ef5f`。仍未新normalformal/grant/activation/GPU；不要求互斥peer退出。
