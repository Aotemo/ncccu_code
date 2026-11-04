# 两个新Pod的实际环境绑定

[实际双Pod首成功汇总](DUAL-START-SUMMARY.json)已由Pod CPU导出：两边各8rank、原生SSL公开初态50c6及自然首更新均通过，导出CPU740cb1de；监督模型CPU初态345全训/空Adam，监督原生初态仍待自然交接核验。当时R240R1适配750/1600、R242为1250/1600，完整内部dev结果仍待。当前训练使用c4b77a2冻结包；用户后续允许合规融合/禁止TTA及raw目标0.995的规则更新只作用于后续计划，未热改现役。

用户选择两独立8卡实验并行；A UID04203d31/Node10.5.15.26，B UID37e3fb34/Node10.5.15.35。各120有效CPU、8张DCU，每卡实测68,702,699,520字节显存。实际Python3.12.7、torch2.10.0/torchvision0.25.0、numpy1.26.4、Pillow12.3.0；两个96核训练池分别与本Pod24核准备池互斥。A准备allocation9fca3bbf，B f48cb4b7，初始root0–11/peer12–23，未改旧allocation。

两Pod的同NFS新UID资产续验分别CPU45a0d759/12b7cd0e通过：必要23个metadata/公开Small文件新SHA，51396训练缓存及5849官方缓存receipt SHA、raw fingerprint和array大小。缓存大字节证明继承被pin的原迁移/绑定，生产读取每个array仍检查SHA；没有冒充重新拷贝或重哈希全部大字节，没有读取官方标签语义。A complete020df177/nativefc5a1d96；B complete7924a47c/native8beb1cf1，完整原件留在`exp-v3/technical-prep/dual-pod-20261104-4e0c/{pod-a,pod-b}/assets-current`。

首轮技术打包因公开Small handoff仍绑定旧51fe UID被原检查正确拒绝：A guardfe93572e、B b27e6eee，activation/GPU均未开始，失败原件保留。随后逐Pod重新核original权重SHA，绑定已通过的当前converted/provenance字节证明，明确复用原精确strict结构回执a4cc6fec而没有重建模型或forward。[A窄检查](POD-A-PUBLIC-BINDING-CPU.json)/CPUcf6a5f59及[B窄检查](POD-B-PUBLIC-BINDING-CPU.json)/CPU94e837bd证明旧UID拒绝、新UID通过。新`R240R1-SCIENCE.json`和`R242-DUALPOD-SCIENCE.json`只修改`public_asset_receipt`；其余科学值与R240/R242逐键相同。原配置/旧回执未改。

双Pod部署将使用同一已测SupMin源码（125ceea8的运行文件保持原字节）。新CPU检查已覆盖旧默认SupCon数学/梯度不变；固定配额完整1600流与像素/P8检查按原精确配置复用。两边仍分别执行原正式18/18和freeze/一次activation/直接launch；新正式状态见[todo](../../../todo/exp-v3-dual-pod/README.md)，不得将本环境续验当作已训练或完整科学结果。
