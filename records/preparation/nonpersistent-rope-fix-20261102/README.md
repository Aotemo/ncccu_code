# 非持久 RoPE 的持久化及 EMA 修复

[实际CPU检查](CPU-CHECKS.json)为1个必要回归目标通过、5个受影响源码编译；nice19/隐藏GPU/5de root0–11。新目标在旧实现真实报 `Tiny noBN EMA live buffers differ`（[RED](CPU-RED.json)）；中间一次fixture漏保存EMA，冷加载origin缺variant/tensor身份而失败（[原件](CPU-FIXTURE-FAILURE.json)），只补fixture实际EMA状态后[GREEN](CPU-GREEN.json)6.117秒通过。没有为这个fixture错误改生产inference或原指标。

实际原生ViT512图、固定16值RoPE、真实noBN EMA规则 warm-copy1600/blend1601、下一步state往返exact及篡改RoPE拒绝均执行；真实CheckpointStore→freeze_training→validate_freeze→load_frozen_model→validate_frozen_checkpoint链通过，冷图RoPE与模型state严格一致。checkpoint持久buffer0件/空摘要、完整模型normalization仍1件/固定摘要；不把RoPE注册为持久、不新增参数/随机采样。EMA参数混合公式及warmup/decay不改。

仅外部source/data admission、科学checkpoint validator和指标选择用fixture替身，未冒充完整公开fresh或实际八rank准入；没有图像forward/GPU/P8复测。生产r92初态门和既有check_candidate现在共用相同持久buffer合同；实际正式CPU及八rank仍待独立新身份执行。原R184/当前R186冻结包及全部clock/token/grant不改，不据静态/CPU证明AUROC改善。

归档guard `0nzmdnne` exit0，SHA493d61de8bea9deb5495ab3439b4ac75d222c952dc324011372b2f88b50219db，原件在exp-v3/technical-prep/cpu-renewal-20261102-1651-32e3/tasks/32e3b079-81b0-4f66-9604-b32491599f44-0nzmdnne/CPU-RECEIPT.json。CPU-CHECKS逐件保存实际源SHA与三次guard/OUTPUT身份，不重跑既有P8或旧全套。
