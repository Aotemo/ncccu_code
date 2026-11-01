# R151条件六通道seed27与独立R152全局768准备

**Git与技术包已实际完成。** 22:20:59.773920 UTC[技术回执](TECHNICAL-READY.json)绑定已push并实际回读的commit`b58f4a99f2b0f43e581f150e2a134e5ee80be396`/tree`2d6f614d01adeffdd263be043e8b3cca54324159`。两技术包各5.932/6.061秒，R151 SITE`c6307173a4d138c0868f913af8a82551342ce1d43af528235b90a05c1b053b88`，R152 SITE`6291e2c9588a421b50724a19850ed80b7e31705f2055e82faa7dcecfaf223176`；实际asset尺寸分别512/768、CLOCK31500/600/27900/2400消费匹配，现PREPARE/OBSERVE Python语法及原分模式RUN shell语法通过。正常formal、grant、clock、activation、GPU均未创建；[原CPU guard](CPU-TECHNICAL.json)exit0，SHA`d7e3db40d1f8ca523059c6210849e7b1c2ed06c8eb2987cf36a50d994fafa13e`。下文“尚待同步”是CPU阶段历史。

[CPU-CHECKS](CPU-CHECKS.json) SHA`ef6ee3f4e91ee6f5ba79777f465a6e8db9dc0f2e2a4d331abadf2d3b56a52893`覆盖三个必要目标，7受改Python源编译通过。原RED为1 PASS/2预期FAIL：真实Small768前向、24×24末层、严格重建及推理几何已PASS；随后仅修复新绑定/资产尺寸及768仅fit-dev缓存的官方sealed raw绕行，像素目标与最终科学身份目标各自PASS。未变模型目标、测试模块与直接factory依赖按精确SHA复用，原exit1保留，不写成一次完整3/3 GREEN。

像素目标验证原生768准备、原8draw增强和只读cache一致、官方valid绕过fit/dev cache且实际校验原始SHA；官方图像均为合成fixture，未读真实官方。`prepare.py`沿现asset key读job的prepared_size；其余训练/启动/保存/评估/监控链复用，未新建平台。

两个实际公开fresh CPU测量均无图像forward/GPU、无旧训练权重。R151 whole`cd6207be4a20230164fe62243d293ae00aa7175c2e8d40d7d54fa3425f286b8c`、shared RGB`91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67`；R152 whole仍`50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`。均345全训/5组空Adam。R151只在R150R1完整双门与safe后合格，否则不正式运行；R152直接R107，单RGB768、无ROI或六通道叠加，见[固定计划](../../../todo/auroc-999/R151-R152-PREPARATION.md)。

global768完整缓存51396行（46256fit/5140dev），18有界批次逐条写入+只读实证，合计739.717秒；[handoff](GLOBAL768-CACHE-HANDOFF.json) SHA`605569da9823b2eb445f924a82ee4615f31ed2785d695ea6f866a78a27c2e251`绑定每批原CPU guard/索引SHA、same manifest与当前Pod。新finalizer只汇集真实索引/覆盖，重新打开图像0；原cache/prepare核心AST与实际生成源码相同，明确区分生成字节与新验证接线。收尾可用112.95GB，高于64GiB checkpoint预留。当前512全局/Native缓存均不改；[R151派生回执](NATIVE6-CACHE-HANDOFF.json)只更新当前source pins，完整图像覆盖复用原已证cache，未再写图像。

固定768满24000历史情景来自原R107真实6400 timing，并非当前GPU benchmark：[预算](BUDGET-SCENARIO.json)约24928秒，加10%27421秒；[最终补充](BUDGET-SUPPLEMENT.json)含一次官方历史耗时情景2038秒、像素缩放峰值约19/21.7GB，均是情景且不保证当前后端/原始官方解码成本。只为未启动源码包集中设置train27900/total31500/binding600/official2400秒，各窗口总和31350<=31500<原9h allocation32400；原R150R1/旧clock和grant不改，不启动后续租或重置，不靠早停藏成本。旧预算报告的配置身份保留在Pod ORIGINAL-BUDGET副本，最终配置身份以补充与CPU-CHECKS为准。

源码与配置在CPU证据形成时尚待同步和技术打包，实际后续完成见页首。正常formal/GPU当前均未执行；未来只有真实八rank初态及自然首成功才可称新GPU链可跑，未加独立GPU预演。当前唯一训练R150R1继续冻结c8a24cd，raw .999未达。
