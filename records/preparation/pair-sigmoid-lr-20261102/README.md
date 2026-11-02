# R192/R193：固定分支与骨干学习率准备

[CPU-CHECKS](CPU-CHECKS.json)实证4目标/3个受影响源码编译通过；两个新候选/config/保存job身份、R193原生AdamW的5空组与参数归属/0、799、1599、1600、23999步原LR公式、原R190/R191候选兼容、精确新存储例外均通过。新源码experiment91 SHA `9734a2ac7cc0ae11bc390683cc4f97ce48fdb5c63e5b610f6aeba9bb755224d3`，模型、loss、optimizer、scheduler、EMA、WORLD、保存/推理实现未改；原P8/EMA/cold证据按文件SHA复用，未重跑，非新的DCU性能证明。

R192/order105是完整有效R191严格胜R107 `.9884405731067525`与R120 `.9894773092062407`且actualsafe后的条件同Base-CLS seed27复核。[实际public fresh](R192-PUBLIC-FRESH-INITIAL.json)为whole `6a261e48579f7651b5a887ff6889a140262adfc93a08c9e0c5832bda94b8f6cc`/165全训/5空Adam，SCI `689c3b57010fafc21a8d3165eca6ca237f91bfebb11d5e0bbbbf260820d7065d`。仅新精确round16GiB例外；原所有例外/default12GiB/free16GiB保持。

独立R193/order106直接原R120 Small/fe6b/seed29/512/pair-sigmoid，唯一job delta是backbone_lr `1e-4→5e-5`；head/age1e-3、原AdamW/WD/采样/增强/G64P8/EMA/1600绝对warmup/24k-800-p4/内部selector保持。[实际public fresh](R193-PUBLIC-FRESH-INITIAL.json)whole仍 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`/345全训/5空Adam，SCI `0985f5763473658f599e72bb2dd93fc5d81e7a94bcdefed2ffe36092dec3463c`。默认12GiB。R96是Tiny+BCE、R130逐层衰减，本固定Small+sigmoid未据其结果冒称已验证或机制已证；阴性不扫LR/seed。

必要guard均在5de准备root0–11/12核、nice19、隐藏GPU执行：新两目标fybgkf62/`5ba78dff1973383bc21081c36854547ce386a251b426c9f174648510db1ddece`、两actual fresh s4jle8b7/`d92fbfd37e6fa97c25d3e48b87cf9364b9311555e881b226addf2b942be0ff41`、旧候选/存储vdkawv7_/`6836e7ab6f1490d0e9979d81abfd2d8c03e5507d786b4582a9c3e3d79d34269e`均exit0；最终归档x9eqkxst/`942fe6d6ef305b6646b33f2f7754b300f2175583baff0ca7a19869a8c58331c1` exit0。候选未登记的RED12kba4tu与一次私有fresh helper的语法失败_uj7vcgo均原样留回执；语法失败发生在任何执行前，改用修正V2后两fresh成功，不改失败原件。

原件在 `exp-v3/technical-prep/pair-sigmoid-lr-20261102-32e3`，0图像forward/0官方/0GPU/0新clock。此页记录CPU准备，不等同Git已同步、technical包、formal或启动；这些后续按实际回执更新。R191原a77冻结不改，32e3唯一Git/GPU owner，无新Agent或资源。raw .999未达。
