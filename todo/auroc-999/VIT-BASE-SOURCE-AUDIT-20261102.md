# R188/R189 之后的单一自然 ViT-B 来源可行性核查

21:16完整公开来源验收已通过，[原件与CPU证据](../../records/preparation/vit-base-source-20261102/README.md)：bin LFS ed8a1265/342624338B、完整162张量strict/exact/FP32有限，source pi0yumhw/de039c79…6c14 exit0；provenance36847019/handoff f0dbbd7e。原3profile字典、默认API和常量实核保持，1source compile；未重复旧全量加载。当前只扩来源工具，不实现/注册Base训练factory或新round；原生512/许可CPU证明不等于GPU可跑或AUROC收益。当前R184R1仍运行，21:16实观2925成功/0skip/eval3/bad0，暂EMA `.9750773939753712`。

当前唯一GPU为R184R1/cd7246a，既定R188条件mean复核及R189独立CLS/0f3队列不改。仅在20%准备池核发布者[timm/vit_base_patch16_dinov3.lvd1689m](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m)一个公开自然LVD来源和现有timm原生512工厂；不扫描其他模型，不注册新round/clock/GPU，不改当前生产源码。旧ViT-S来源任务的单源范围及原始证据保持，此页为独立后续可行性任务。

依据限于可检验的同家族容量假设：发布者模型卡给出ViT-B的768维特征，原ViT-S为384；不将公开自然图像指标或容量当成本任务AUROC收益、临床机制或DCU性能证据。模型卡说明QKV零bias在timm省略、原生FP32 RoPE与原版BF16存在数值区别，沿已固定timm语义核实际结构，不宣称原Meta逐字节forward复现。

必要范围先限repo外审计文本与本页：只读registry固定revision/public/non-gated/bin LFS身份，再读固定配置/模型卡/完整许可；在当前5de准备0–11池构造随机原生512模型，核参数/前缀/层与head维度、非持久RoPE、原生库文件，0图像/0forward/0GPU。无代码或权重准入结论前不下载大bin、不改现factory；若来源/结构不合适，记录后停止该假设。后续若采用，先登记精确必要现有函数范围、完整public strict及受影响CPU证据，再同步Git并独立fresh；不增加训练/保存/监控平台。

21:11实际registry `_uftvirx`/exit0：revision `c6a5fb7d12bbd3cf3b0079253141c3332aaed7da`，public/non-gated，bin342,624,338 B/LFS `ed8a1265847086c913535fe7d84c0156a68dc98c0edd52298b8bb4bb7bdcbbb1`。固定配置、模型卡与完整许可后，原生512/162张量/85,641,216元素/768维/12层12头/5prefix/无QKV bias/noBN构图guard32vldjel SHA `df074abc06f96a23f785f75afa863159f5d4db4425eed7b46775daeda416166b` exit0；原生RoPE仍FP32[16]且SHA192c9d93与Small一致，未读大bin或做forward/GPU。来源审计原件在exp-v3/technical-prep/vit-base-source-audit-20261102-32e3/。

来源与原生结构未见可行性阻断，下一必要范围只扩现有 `tools/pretraining/deit3_source.py` 一个显式vitdinobase profile及DINO分支宽度消费；旧API默认deit3、旧Small的384默认/原件身份/输出语义保留。复用原fetch、完整张量strict/exact/FP32、同4库文件与许可链；完整bin按LFS验证后才接受source。不实现或注册Base训练factory，不改当前任何GPU队列。raw .999未达。
