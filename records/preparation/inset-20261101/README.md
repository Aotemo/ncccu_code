# R146 固定全图加原生局部嵌图：CPU与双缓存准备

两项必要CPU实际RED2FAIL/0ERROR→GREEN2/2、9受影响源compile通过，GPU未初始化。[CPU-CHECKS](CPU-CHECKS.json) SHA `64dafcbf7c6130d002bc0ed8aa8508f602bc824914cf7510448a66fd5b69f1ed`、原[guard](CPU-GUARD.json) SHA `5647e33d0e96b6a3e3f106574305131ed5c7f567e9bd16ad0f2fd70722a6e706`及OUTPUT SHA一致。像素oracle、原8draw/全局RNG、真实serial/parallel/predict消费相同保存policy、合成valid在原最终门内应走的sealed raw路径均已检查；新SCI整件对R107差异和保存policy往返通过。模型/loss/optimizer/EMA与父级89a逐一SHA保持，没有公开模型重复前向或GPU预演。

[SCIENCE](../../../configs/R146-SCIENCE.json) SHA `f119629c317b3cafc4a914b4ea6cb4b9f7b8c35d2530e500e1456b7543600915`固定：global prepared RGB居中等比bilinear放入512黑画布，既定Native亮区ROI缩为256方图覆盖左上角(0,0)，然后原8draw/512增强。直接R107/public fresh Small/seed29/原3RGB/单编码，初态50c与其余科学合同保持；不是多模型分数融合。256及左上角一次预定，不按dev逐样本或partial扫描。完整双门未胜关闭这份精确配方，不扫位置/大小/ROI/seed。

[PAIR-CACHE-HANDOFF](PAIR-CACHE-HANDOFF.json) SHA `98b5e6978e014acac87e3df5a448a276ee073d6a36013af4fbf8358431d2d693`实证fit46256+dev5140共51396行，两份cache各完整只读复读校验，用时118.405秒、cache_images_written=0；private整份索引SHA `bbe4fddabafd68acd37752eb3b49741b6d99796ee18c52e830df9ac1951972ac`留Pod。原cache类、prepare/亮区定位/保守边界AST与11d相同；当前只核本候选必需输入，不恢复旧FIT/census。未读实际官方图，官方仅完整freeze≥.996后在既有一次门中按相同政策准备原图。

准备使用00202581本人0–22/nice19/隐藏GPU，R144冻结11d/训练24–119不改。仅在现有prepare/prepare_gpu_site验证新联合回执，在原run_training给consumer注入第二只读cache env；不新建训练、保存、评估或缓存平台。当前无新formal/grant/clock/activation/GPU，raw .999未达。
