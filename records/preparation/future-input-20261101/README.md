# 后续条件分支的两项 CPU 准备

R137冻结`7b48f49`继续八卡训练时，由当前唯一owner32e3在实际a365 allocation本人0–11、nice19、隐藏GPU的独立`technical-prep/future-input-20261101-32e3`完成。原R137及其clock/grant/冻结包未改；另外12个prep核保留不用，训练96核与准备池互斥。

两项新增目标均实际通过：像素/裁剪/原None与8draw保持，以及真实serial evaluation、parallel evaluation和prediction consumer消费同一保存input policy；R138固定Lion seed27和R139固定局部对比度的SCIENCE/candidate/age/job序列化身份与对应原基线逐字段比较。仅合成32像素图和小模型，不读取真实图像、public模型或样本预测，不运行GPU。原Lion五个算法函数源码及age/loss/EMA/clip依赖与已过R137版本完全一致，因此复用原数值/Gloo/保存证据，不重跑旧套件。

真实RED为两FAIL/0ERROR。GREEN1保留一测试ERROR（遗漏evaluator必需variant）、一FAIL（seed文本替换误改公开来源回执日期）；GREEN2保留两测试ERROR（误用prediction函数名、缺省fixture未含initialization）。只修相应配置/测试，生产七源从GREEN1后未改。最终GREEN3两项分别1/1、3.410秒和2.568秒，guard均exit0，9份受改Python源compile通过。未将先前失败写成通过，也没有全suite通过声明。

[CPU-CHECKS.json](CPU-CHECKS.json) SHA `a82f8a06aa7b27c58c02e561b67c94729bf77e4873f2cb8309ca5452d5991006`；[最终编译/回执核对CPU树](CPU-GUARD.json) SHA `4246497ddea5db964c2ae79958c77f061845422989ddbb7df0afa41032b56842`。配置SHA分别为R138 `cd5099601e603ea3f89866037dcd4906b7123b81a560dc16c2bc5ef302dfd03d`、R139 `17661d8bb45ecc9177654630b479d8d9fedb2dcef70a1c3f2352b21241a9307c`。这些是CPU准备证据，不等于真实公开fresh formal、GPU准入、activation或新训练已执行。

接续及精确范围见[当前后续准备](../../../todo/auroc-999/NEXT-PREPARATION-20261101.md)。R138只有完整R137严格胜两预冻锚点且actualsafe后才执行；否则跳过未运行的预留编号，采用独立R107/AdamW的R139输入政策。后续唯一normalformal、正式资源/clock、activation与真实八rank首成功仍须在采用时完成；当前不以partial dev或准备完成判科学收益。旧census六件/未知analysis保持隔离。
