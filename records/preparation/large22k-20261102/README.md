# 原生 ConvNeXt Large 准备证据

沿用既有 Base 来源下载/strict、原生模型工厂、fresh、保存/冷加载及部署链，显式增加 large224 profile；旧 Base 默认入口保持。固定公开非 gated `timm/convnext_large.fb_in22k_ft_in1k` / `dc2e53fad264aa1fb4e5940e0d1c9b79ac6e0854`、Apache2 完整许可、791,192,465 B 原件 LFS `fd951098…422277` 已实际核对，转换 `b27d6640…044f21`。完整1000类344张量与两 factory strict、41 LN/36 GELU-SD/40 Conv/无BN-buffer及池化/1536宽度等非张量语义通过；没有来源前向数值对照。

四个受影响目标及11编译通过。原RED两失败；第一次GREEN中source映射目标通过，真实CheckpointStore冷加载发现inference白名单遗漏Large。仅加一项后只复测模型目标通过：独立原生factory/RNG、真实512/P8全345有限梯度、5空Adam全参数分组、EMA、age一次、保存→冷图exact。随机初始化fixture与真实公开strict/fresh分列；CPU保存storage 15,313,271,040 B不是DCU峰值或上界。既有loss/WORLD/engine/EMA/保存按相同源码复用，不重跑。

两份真实public fresh：R179为Small/fe6/seed27，whole `91507a65…6d0d67`；R180为Large/dc2/seed29，whole `3360682e…1450e0`，均345全训/5空Adam/无数据前向或GPU初始化。R179只在完整R178严格过R107/R120且安全时有资格；R180独立沿原R107的512/ImageNet输入/原logistic/G64P8/优化与EMA合同，仅骨干及匹配公开初始化共同改变，不是容量纯因果消融或来源224配方复现。

仅新R180精确round存储上限32GiB，原默认12GiB、旧例外、free16GiB及原clock/resource不变；函数fixture实核29GiB通过/33GiB拒绝/free15GiB拒绝，并核R178默认12GiB。没有实际写入33GiB或峰值必成功保证。[旧轮35件清理](../../storage/closed-r170-r172-r174-r176-20261102/README.md)已核保护文件不变。HTTP2/HTTP1整件传输与Pod直连代理失败、一次SSH分段中断如实保留；95段最终重组整件LFS精确一致才进入原来源链。

CPU与来源原件位于exp-v3/technical-prep/large22k-20261102-32e3；本目录只保存精简真实回执。GPU可运行须由之后冻结部署的真实八rank/首自然成功更新证明。本准备不证明吞吐、临床机制或AUROC收益；raw .999未达，完整selected≥.996才官方5849一次。

正式包前发现R180 SCI冗余stage_widths不在原公开handoff.structure内，会被原逐字段交接检查拒绝。仅删SCI该字段、修正对应断言并补实际来源交接目标，0.059秒PASS/CPU整树0；原生/source的stage-width检查不改。最终SCI为`9ea7429c…0ba442`，其余解析字段与原34e6c689完全一致，whole3360682e不变。旧strict/fresh及4目标回执保留原样；原fresh报告不是新SCI字节的直接证据，确切不变范围与实际handoff复核见CPU-CHECKS-HANDOFF-FIX，之后正常CPU须以最终SCI再构造fresh。增加1窄目标/1编译，不重复原大模型前向；累计5目标、11个唯一受影响源码编译覆盖。17deee5只同步过，未formal/clock/GPU，未热改任何运行。
