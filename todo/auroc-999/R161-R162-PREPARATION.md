# R160期间提前准备

实际接续：R160完整6400/0skip/eval8/bad4，唯一EMA c4@3200 AUROC `.9842021165792284`、两门false/官方未跑且actualsafe；R161未formal/GPU。R162以冻结Git608e71b/SCI5ddf唯一formal20/20、精确部署和一次activation/direct launch0，05:29:52 UTC真实八rank/345初态/5空Adam及100成功0skip、首自然25 finite已证，见[实际启动](../../records/R162/README.md)。当前GPU仅R162，下面“未formal/GPU”是此前准备阶段。

05:02:32 UTC实际Git608e71b/tree069bacc已同步回读，[两技术包及入口语法](../../records/preparation/cosine-head-20261102/NEXT-READY.json)完成；3新目标/7compile及原6loss/WORLD按精确SHA消费，未新normalformal/grant/clock/GPU。完整R160结果决定R161或独立R162，旧训练收尾期间沿原正常formal/精确部署，actualsafe后立即一次activation/direct launch；不等待本文后续归档。

R161/order73仅在R160完整工程有效selected同时严格胜R107 `.9884405731067525`及R120 `.9894773092062407`后复核原固定批内margin1/seed20261027；否则不formal/activation/GPU。独立R162/order74直接回R107/public fe6b fresh/seed29/原512及logistic，只把原末端Linear改为固定余弦Linear：`16 * dot(x/max(norm(x),1e-12), w/max(norm(w),1e-12)) + b`，x为原GAP→LN→Flatten→Dropout的输出，之后只加原age一次。scale16、eps1e-12固定，不扫scale/温度/位置或叠R160。

假设是检验去除末端特征/分类权重范数对打分的影响，尚无证据表明范数是难图根因。归一化思路参考[NormFace原论文](https://arxiv.org/abs/1704.06369)，数值定义用[Torch2.10 normalize](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.normalize.html)。本二元pairlog及age配方不是论文softmax训练复现，不转用其人脸权重或收益。16等于`√(768/3)`，只作为预先固定的维数尺度，不声称实际公开特征满足各向同性或初始函数保持；原参数张量可相同而前向函数改变。

最小范围：现config.py精确可选policy/None省略；models.py在原函数内增加懒定义的Linear子类适配器、复用原weight/bias对象且不采样RNG，完整network forward不拆；experiment91.py两身份/互斥/描述与保存绑定；inference.py核无状态读出重建；check_candidate.py核真实头及冷加载；tests/test_cosine_head.py两必要目标、原candidate目标与R161/R162两配置。无新参数/buffer/优化器组、额外模型forward、teacher、数据源、临床ROI或平台；旧训练/通信/优化器/EMA/保存/评估/监控链保持。

先实际RED，CPU独立标量值/解析输入和weight/bias梯度、范数缩放、零输入有限性、对象/名称/RNG保留；再原生factory/age一次/保存→真实inference冷加载/无policy兼容及配置身份窄验。复用SHA未变的WORLD/loss证据，不重全套或GPU预演。两fresh初态实核；当前训练前段完成push/回读、压缩传输和技术包，旧轮进入安全终态后沿现有正式验收/一次activation/launch接续，以真实八rank和首自然成功证明GPU链。

根32e3唯一Git/GPU，aa0f准备0–22/23核、peer23/1核与训练24–119/96核互斥，不恢复/新建Agent。当前R160的3ef3b015冻结不改。原24k成功/800dev/p4/EMA/唯一selector及官方完整selected≥.996一次保持；raw .999未达。R162完整未胜R107则关闭固定配方，同时胜R120才固定seed27资格；不以AP/难20组或官方救门。[3新目标/7compile及两public fresh](../../records/preparation/cosine-head-20261102/README.md)已实际通过，旧6loss/WORLD按SHA复用；当前待实际Git/技术包，尚未normalformal/grant/GPU，不称收益。
