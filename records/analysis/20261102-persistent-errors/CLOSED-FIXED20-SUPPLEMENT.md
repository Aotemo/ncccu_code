# 已结束模型的固定难组补充

[实际汇总](CLOSED-FIXED20-SUMMARY.json)SHA `4a0166433bd49ac577a4a7fe21f1a511d788eb7746a41982bfa32c56ca62fdc3`，复用原4锚点封存的20正例，未用新轮重新分组。原PRIVATE-SELECTION SHA702680…0a3b及诊断源码a93ead…3bc5相符；4锚点组质量逐一吻合。不重读原14个CSV，各新轮原selected CSV/precision只读一次并核已归档SHA、行序/标签、EMA步数/tensor身份，原函数复算AUROC/AP/error_twice及两类贡献守恒均一致。

| 完整唯一选点 | 内部 AUROC | 固定20正例错序质量 | 其余147正例 | 总错序质量 |
| --- | --- | ---: | ---: | ---: |
| R107（原封存） | .9884405731067525 | 6623 | 2977 | 9600 |
| R120（原封存） | .9894773092062407 | 5829 | 2910 | 8739 |
| R186 EMA c5@4000 | .983750576466211 | 8554 | 4941 | 13495 |
| R184R1 EMA c5@4000 | .9791930315921545 | 10709 | 6571 | 17280 |
| R189 EMA c5@4000 | .977164111350996 | 11492 | 7473 | 18965 |
| R191 EMA c9@7200 | .9666919930498946 | 15920 | 11742 | 27662 |
| R193 EMA c4@3200 | .9838854364466322 | 9297 | 4086 | 13383 |
| R195 EMA c4@3200 | .9861756478998568 | 8336 | 3145 | 11481 |

单位是错序对质量，同分半计，不是错图数。新三轮固定20组占各轮总错误的63.39%/61.97%/60.60%；占比下降不表示已改善，其绝对错序量和其余147组均高于R107。R189相对同Small DINOv3 meanpatch的R184R1，固定20组多783、其余多902，合计多1685，描述与完整AUROC方向一致。不能从这些数值推临床结构、错标或优化机制，也不以局部组替代完整AUROC。

用户来源明确1=青光眼，原标签仅依据眼底照片；0只称本任务阴性。没有新图像/模型/checkpoint/forward/官方读取，无逐例权重、标签或当前R191/后续冻结合同改动。R193的骨干LR减半仍是独立固定假设，这份描述不证明其机制或收益。

[CPU回执](CPU-CLOSED-FIXED20.json)tekkrv7n/`32496a777b9039d5d96fc8b7fca1d8ec848372a9155e2122a421d81e4be27159` exit0，53ab准备root0–11/12核、nice19/隐藏GPU；原件 `exp-v3/technical-prep/closed-fixed20-20261102-32e3`，私有CSV临时快照已随任务退出清除，原8件分析不改。

R191结束后独立补充的[原件汇总](R191-FIXED20-SUMMARY.json)SHA `4b05d48a82776213cd928702f6110bd543559c184503d1cd88ad5596d45bc963`沿同一原4anchor/同一20组，未重读前三新轮或原14CSV。只读R191新selected CSV/precision各一次，manifest SHA、行序/标签、EMA7200/tensor与已同步95b522b结果相符，原函数复算AUROC/AP/error_twice及两类总贡献守恒。固定20组较R107多9297、其余147组多8765、总计多18062，固定组占57.55%；同样不能把占比下降说成改善，也不能把整轮恶化只归因于这20组。无新模型/图像/forward/官方，无新权重或SCI变更。

该次[CPU回执](CPU-R191-FIXED20.json)3whvfkwn/`051464501a48e8b28762bba2d5ff32ac2508975cc49d09e5a8bae763008e8001` exit0，仍53ab/root0–11/nice19/隐藏GPU，与当前R193训练24–119互斥。仅复用原单次supplement helper并将新轮列表限定R191，私有临时CSV已退出清除；原件在`exp-v3/technical-prep/r191-fixed20-20261102-32e3`。这份描述不建立临床机制、不证明当前半LR或后续固定衰减会收益。

R193完整收口后单独[复算汇总](R193-FIXED20-SUMMARY.json)SHA `615ce9a3eb645211e0fe24edc26eb6e51e387782df141c47389e13953e13f712`，仍由原四anchor定义同一20组，不读此前各轮CSV。仅新R193 selected CSV/precision各一次，原已提交a5a8505 SHA、manifest/行序/标签、EMA3200/tensor均匹配；AUROC/AP/error_twice及两类贡献守恒一致。固定20组较R107多2674，其余147组多1109，合计多3783；较R120分别多3468/1176、合计多4644。固定组占69.47%，两组绝对错序均未改善；这不是错误图片数或学习率/临床机制证明。

该次[CPU回执](CPU-R193-FIXED20.json)xji4j8py/`ef646ba5169a0fb40931571e2c6b2b158876dbe4dacaed385a94349c55c2e946` exit0，53ab/root0–11/nice19/隐藏GPU，与当前R195训练池互斥；原件`exp-v3/technical-prep/r193-fixed20-20261103-32e3`。0图片/模型/checkpoint/forward/官方，私有CSV快照已退出清除，原标签与当前693/后续436固定合同不改。

R195完整收口后[同组补充](R195-FIXED20-SUMMARY.json)SHA `8812540d91d152d27a8ca12e9cad34ae1bed5fc19a3d6711cd0df421e8a39113`仍用原四anchor，未按R195重分组。新selected CSV/precision各读一次，已同步b2aa1bd的来源SHA、manifest/5140行序标签、EMA3200/tensor均一致，原函数复算AUROC/AP/error_twice及两类贡献守恒。固定20组8336、其余147组3145、总11481；比R107分别多1713/168、总多1881，固定组占72.61%，两组也均高于R120。较早衰减没有解决封存难组或完整AUROC；这些是错序对质量，不是错图数或临床/优化机制证据。

[CPU回执](R195-FIXED20-CPU.json)usmtd727 / `a6a5252583c67d68f04e8ae1d146b5cfa285f600e432b5f8a166a0b4a41ce1d4` exit0，当前53ab/root0–11/nice19/隐藏GPU，与R197训练池互斥。私有原件`exp-v3/technical-prep/r195-fixed20-20261103-32e3`；无旧14CSV重读、图像/模型/checkpoint/新forward/官方，临时CSV快照已退出清除，原标签及现436/后续82add93冻结保持。
