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
| R197 EMA c9@7200 | .9884706757809537 | 6334 | 3241 | 9575 |
| R199 EMA c9@7200 | .9858601718742286 | 8101 | 3642 | 11743 |
| R201 EMA c5@4000 | .9889535226751404 | 6092 | 3082 | 9174 |
| R203 EMA c6@4800 | .9816433892721294 | 8293 | 6952 | 15245 |
| R205 EMA c3@2400 | .9860335632776274 | 8291 | 3308 | 11599 |

单位是错序对质量，同分半计，不是错图数。新三轮固定20组占各轮总错误的63.39%/61.97%/60.60%；占比下降不表示已改善，其绝对错序量和其余147组均高于R107。R189相对同Small DINOv3 meanpatch的R184R1，固定20组多783、其余多902，合计多1685，描述与完整AUROC方向一致。不能从这些数值推临床结构、错标或优化机制，也不以局部组替代完整AUROC。

用户来源明确1=青光眼，原标签仅依据眼底照片；0只称本任务阴性。没有新图像/模型/checkpoint/forward/官方读取，无逐例权重、标签或当前R191/后续冻结合同改动。R193的骨干LR减半仍是独立固定假设，这份描述不证明其机制或收益。

[CPU回执](CPU-CLOSED-FIXED20.json)tekkrv7n/`32496a777b9039d5d96fc8b7fca1d8ec848372a9155e2122a421d81e4be27159` exit0，53ab准备root0–11/12核、nice19/隐藏GPU；原件 `exp-v3/technical-prep/closed-fixed20-20261102-32e3`，私有CSV临时快照已随任务退出清除，原8件分析不改。

R191结束后独立补充的[原件汇总](R191-FIXED20-SUMMARY.json)SHA `4b05d48a82776213cd928702f6110bd543559c184503d1cd88ad5596d45bc963`沿同一原4anchor/同一20组，未重读前三新轮或原14CSV。只读R191新selected CSV/precision各一次，manifest SHA、行序/标签、EMA7200/tensor与已同步95b522b结果相符，原函数复算AUROC/AP/error_twice及两类总贡献守恒。固定20组较R107多9297、其余147组多8765、总计多18062，固定组占57.55%；同样不能把占比下降说成改善，也不能把整轮恶化只归因于这20组。无新模型/图像/forward/官方，无新权重或SCI变更。

该次[CPU回执](CPU-R191-FIXED20.json)3whvfkwn/`051464501a48e8b28762bba2d5ff32ac2508975cc49d09e5a8bae763008e8001` exit0，仍53ab/root0–11/nice19/隐藏GPU，与当前R193训练24–119互斥。仅复用原单次supplement helper并将新轮列表限定R191，私有临时CSV已退出清除；原件在`exp-v3/technical-prep/r191-fixed20-20261102-32e3`。这份描述不建立临床机制、不证明当前半LR或后续固定衰减会收益。

R193完整收口后单独[复算汇总](R193-FIXED20-SUMMARY.json)SHA `615ce9a3eb645211e0fe24edc26eb6e51e387782df141c47389e13953e13f712`，仍由原四anchor定义同一20组，不读此前各轮CSV。仅新R193 selected CSV/precision各一次，原已提交a5a8505 SHA、manifest/行序/标签、EMA3200/tensor均匹配；AUROC/AP/error_twice及两类贡献守恒一致。固定20组较R107多2674，其余147组多1109，合计多3783；较R120分别多3468/1176、合计多4644。固定组占69.47%，两组绝对错序均未改善；这不是错误图片数或学习率/临床机制证明。

该次[CPU回执](CPU-R193-FIXED20.json)xji4j8py/`ef646ba5169a0fb40931571e2c6b2b158876dbe4dacaed385a94349c55c2e946` exit0，53ab/root0–11/nice19/隐藏GPU，与当前R195训练池互斥；原件`exp-v3/technical-prep/r193-fixed20-20261103-32e3`。0图片/模型/checkpoint/forward/官方，私有CSV快照已退出清除，原标签与当前693/后续436固定合同不改。

R195完整收口后[同组补充](R195-FIXED20-SUMMARY.json)SHA `8812540d91d152d27a8ca12e9cad34ae1bed5fc19a3d6711cd0df421e8a39113`仍用原四anchor，未按R195重分组。新selected CSV/precision各读一次，已同步b2aa1bd的来源SHA、manifest/5140行序标签、EMA3200/tensor均一致，原函数复算AUROC/AP/error_twice及两类贡献守恒。固定20组8336、其余147组3145、总11481；比R107分别多1713/168、总多1881，固定组占72.61%，两组也均高于R120。较早衰减没有解决封存难组或完整AUROC；这些是错序对质量，不是错图数或临床/优化机制证据。

[CPU回执](R195-FIXED20-CPU.json)usmtd727 / `a6a5252583c67d68f04e8ae1d146b5cfa285f600e432b5f8a166a0b4a41ce1d4` exit0，当前53ab/root0–11/nice19/隐藏GPU，与R197训练池互斥。私有原件`exp-v3/technical-prep/r195-fixed20-20261103-32e3`；无旧14CSV重读、图像/模型/checkpoint/新forward/官方，临时CSV快照已退出清除，原标签及现436/后续82add93冻结保持。

R197完整收口后[原组复算](R197-FIXED20-SUMMARY.json)SHA `4fd3fc9195dd812db73bf88342850a7b39044c09a597df55967708dbefc52828`仍不重新分组；只读本轮selected CSV/precision各一次，76554b38/afc4acea及manifest/5140行序标签、EMA7200/tensor6251937d匹配已归档结果，原函数复算AUROC/AP/error_twice与两类贡献守恒。固定20正例6334、其余147组3241、总9575；比R107分别少289/多264、合计少25，固定组占66.15%。比R120两组仍多505/331、总多836，因此局部难组减少不替代完整采用双门，也不证明亮度或临床/优化机制。单位为错序对质量、同分半计，不是错误图片数。

[CPU回执](R197-FIXED20-CPU.json)huryzkcn / `0cdb4cbbea35a68dd96bb62c77292e091a1317b1cbe8fd4eb1c342153e082e06` exit0，53ab/root0–11/nice19/隐藏GPU，与R199训练24–119互斥；原件`exp-v3/technical-prep/r197-fixed20-20261103-32e3`。无旧CSV重读、图像/模型/checkpoint/forward/官方，私有临时CSV已退出清除，原标签及现82/后续8635等冻结合同不改。

R199完整收口后[同组复算](R199-FIXED20-SUMMARY.json)SHA `ba1e4ddd85c112671ffac4c5bbd3c1fa6c0ef68b4732169f3ae779a8c6a6807c`沿原四anchor、原20组；新selected CSV/precision各读一次，d6512bbb/30828eaa与manifest/5140行序标签、EMA7200/tensor20f1cbec均匹配已归档完整结果，原函数复算AUROC/AP/error_twice及两类贡献守恒一致。固定20组8101、其余147组3642、总11743；比R107多1478/665、总多2143，固定组占68.99%；比R120多2272/732、总多3004。固定WD.05未改善完整AUROC或这两组，不能据它外推所有weight decay或建立临床/优化机制。

[CPU回执](R199-FIXED20-CPU.json)65e7p1n_ / `851a59378c09eba74f3abe61dfc2d7db8810477227c01e58cda61fa64794fa06` exit0，ec5/root0–11/nice19/隐藏GPU，与R201训练24–119互斥；原件`exp-v3/technical-prep/r199-fixed20-20261103-32e3`。无旧CSV重读、图像/模型/checkpoint/forward/官方，私有CSV快照随任务退出清除；原标签与R201/后续冻结合同均保持。

R201完整收口后[原组复算](R201-FIXED20-SUMMARY.json)SHA `4128aba9eb80573abbca1180cc59c22a1adaaa820c7086dc13c434c439664343`仍沿原四anchor定义的20组。新selected CSV/precision各只读一次，79a5f8dd/4165fa9c与已同步81694b0结果及manifest/5140行序标签、EMA4000/tensora0b54a08一致；原函数复算AUROC/AP/error_twice及两类贡献守恒。固定20正例6092、其余147组3082、总9174；比R107少531/多105、合计少426，固定组约占66.4%。局部减少与完整R107门方向一致，但完整R120门仍未通过，不以难组或AP救门，也不证明梯度尺度/临床机制。

[CPU回执](R201-FIXED20-CPU.json)3gqaqcfn / `3457cd393db8720afad5d1d67e824587b10bd2e973e72d2b26b3a6ef314c46d2` exit0，ec5/root0–11/nice19/隐藏GPU，与R203训练24–119互斥；原件`exp-v3/technical-prep/r201-fixed20-20261103-32e3`。无旧CSV重读、图像/模型/checkpoint/forward/官方，私有CSV快照已退出清除，原标签及现573/后续7ead冻结合同保持。

R203完整收口后[同组复算](R203-FIXED20-SUMMARY.json)SHA `c7701348bbf965e4eead1ae913a36f4bc06209a6f65be0560cf19b12284a29a8`仍沿原四anchor定义的20组；本轮selected CSV/precision各读一次，3379a3bf/db58a7e6与已同步0e331a8结果及manifest/5140行序标签、EMA4800/tensorb418341f一致，原函数复算AUROC/AP/error_twice及两类贡献守恒。固定20正例8293、其余147组6952、总15245；比R107多1670/3975、合计多5645，固定组占54.40%。占比下降不表示改善，两组绝对错序均恶化，不能把完整退步仅归于原20组，亦非临床或优化机制证明。

[CPU回执](R203-FIXED20-CPU-GUARD.json)3s66ji9k / `5a617ab56832404f640365cd751b68d21e0a3a2c686bfa449da8a5e780731e29` exit0，ec5/root0–11/nice19/隐藏GPU，与R205训练24–119互斥；原件`exp-v3/technical-prep/r203-fixed20-20261103-32e3`。无旧CSV重读、图像/模型/checkpoint/forward/官方，私有CSV快照已退出清除，标签及R205冻结合同保持。

R205完整收口后[同组复算](R205-FIXED20-SUMMARY.json)SHA `f639d04abd0f3e58b3fa86b4630470b9f6bf386f0f7c3c5c63137e9908f4d79d`仍沿原四anchor、原20组。本轮selected CSV/precision各读一次，ecf728c6/3385836b、manifest/5140行序标签、EMA2400/tensor7aa325b2与已同步4f82d59完整结果一致，原函数复算AUROC/AP/error_twice和两类贡献守恒。固定20正例8291、其余147组3308、总11599；比R107多1668/331、合计多1999，固定组占71.48%。本固定困难负例加权未改善完整AUROC或这两组，不外推所有困难样本方法，也不建立临床或优化机制。

[CPU回执](R205-FIXED20-CPU-GUARD.json)zut07tft / `8e9ee061c667fa05f5483b25a041686d03cfc386ec2adc383f30fecf3f58befe` exit0，ec5/root0–11/nice19/隐藏GPU，与R207训练24–119互斥；原件`exp-v3/technical-prep/r205-fixed20-20261103-32e3`。无旧CSV重读、图像/模型/checkpoint/forward/官方，私有CSV快照已退出清除，原标签及R207冻结合同不改。
