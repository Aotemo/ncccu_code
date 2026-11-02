# R155/R156最小准备与真实CPU证据

[CPU-CHECKS](CPU-CHECKS.json)，Pod SHA `cfb553354f9fe66841c25f5d1ed646ca4b440daa7f537b3c2ddcd6fb8bc628ef`：6个受影响目标通过、6源compile、0failure/0error。最初4个loss目标8.578秒全部通过；独立复核指出原WORLD fixture的选中索引只是参考，随后补上每rank逐64个global logit梯度（gather SUM前）的独立解析对照，仅重测该WORLD目标，并运行新配置/保存loss身份、旧sigmoid目标，共3目标7.761秒。另3个原loss目标及scalar reference的AST与生产loss SHA未变，直接复用。最终[实际CPU guard](CPU-RECEIPT.json) SHA `c182007dd86bb04616a4304bcc6bded11b2258cbd1b64cd76a373f88d80d21ce`，exit0。

检查覆盖8×56全448pairs、非同分解析值/逐logit梯度、共享平移和bias梯度0、全同分/第2名边界同分/极值有限、真实两rank WORLD gather与默认DDP参数标度及各rank实际logit梯度一致。旧None/BCE、logistic、sigmoid值与梯度保持；新kind直接进入原PAIRWISE_LOSS_KINDS，config/engine/distributed_engine源码未改，无额外模型forward或新平台。

[R155实际public fresh](R155-PUBLIC-FRESH-INITIAL.json) whole `51ea8834a559b09050081432354f59360c9fa751437efc387f6788bf9ad38218`，402参数/48静态buffer，buffer摘要与R154相同；[R156实际public fresh](R156-PUBLIC-FRESH-INITIAL.json) whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，345参数/0buffer，与R107相同。均原生产initialize=True严格公开加载、全参数可训练、5组空AdamW，未加载实验checkpoint或做图像forward。首次测量辅助脚本误读只在experiment91绑定时附加的可选回执属性而exit1；原失败及draft保留，仅修辅助脚本，第二独立输出目录测量成功，不改生产模型。最初新loss不支持的RED exit1同样保留。

用户另开的CPU23会话只读AST与数学核对未发现阻断问题，没有重复测试、写Git或GPU；其原fixture限制在最终WORLD目标补齐。全部计算仍根0–22/23核或peer23/1核、aa0f、nice19、隐藏GPU，与R154训练24–119/96核互斥，各执行者自身串行；6阶段原始guard/output逐件SHA已核。

SCIENCE SHA：R155 `63a94a917e2d866423d209883a9c7884ed49d40661b6a924065d025eba94488a`；R156 `ec65ee71c6a55c5009b9d79302462ae0fba48acc5030eee0b15b8b60a39fbfb9`。R155只在R154完整双门及safe后复现seed27；否则独立directR107的R156固定`.5全8均值+.5稳定top2均值`。原完整内部选点、24k/800/p4、官方≥.996一次与raw .999目标保持，未见新训练收益。

工程预算不变：train27900秒/total31500秒/9h allocation。R155同结构继承原Swin历史24k×2.5情景26205.225秒，R156原R107训练耗时外推满24k×2情景24154.892秒，均低于train soft27750秒；原official×2.5耗时2058.503秒低于soft2250秒。它们是历史情景，不是当前GPU实测或保证，不依赖早停兜底，不热改R154 clock。

源码/配置已实际push并远端回读Git `874fefff678d99b2174df929331b9bb38f4b1ca6` / tree `c575ce4fb760000da02576815d1f016be57d9cf1`。02:26:35 UTC[技术包回执](NEXT-READY.json)证明6源、两SCIENCE、集中策略与CPU通过字节一致；现有prepare/check/freeze/activation/launch入口仅注入新绑定，临时调用脚本语法已过。[实际guard](CPU-NEXT-READY.json) SHA `de7579f12c91aa38c8672553548bc74b2fd0ebe46c29bb9d37fbb4113fb82590`，exit0。R155技术SITE `235955aea51783814c9cfe3483b3ce89be6981fc628acc4bbb58ab2277146a9e`，R156 `b95d2f705ed90288f54820f8722e22cb70e297d5ec0fb1693988326c01b61478`；均未normalformal/grant/clock/activation/GPU，仍待R154完整结果沿既定分支接续，不重跑已通过窄验。
