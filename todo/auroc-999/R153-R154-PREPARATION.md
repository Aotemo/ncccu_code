# R152训练期间的固定后续准备

唯一owner32e3继续使用互斥20%准备池，不派生Agent；R152的b58冻结、clock、grant和GPU绑定不改。CPU任务全局串行，新b781分配有效至2026-11-02 05:11:20 UTC，23核0–22，训练96核24–119。

R153/order65仅在R152完整工程有效selected同时严格胜R107 .9884405731067525及R120 singlebest .9894773092062407且actualsafe后，采用相同全局RGB768的固定seed20261027。必须真实核公开fresh初态，预期原三通道whole91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67；不加载任何旧实验checkpoint。未过则不formal/activation/GPU。

独立R154/order66候选为Swin V2 Small、公开ImageNet1K、原RGB512，直接对照R107，不叠加768、ROI、六通道或其他近期失败改法。先核公开模型卡适用许可、不可变revision、原权重和完整1000类转换；来源或CPU可运行证据不满足则不采用，不伪填redistribution或准入。Swin V1 Base的R117 .9850076641408516是明确负先验；V2 Small不是同一来源/宽度，不能称纯V1→V2因果或保证提升。

来源已实核HF模型卡MIT、ImageNet1K、revision db3a0c7e5d97487e7e874c943838569b84fe8e54；Pod旧代理下载超时原回执保留，授权本机仅作原始字节relay，全部解析/哈希/模型计算仍在Pod。最小必要范围为现有swin_model.py（固定V2 Small结构/精确公开加载/静态位置缓冲区）、models.py（仅factory分派和来源核验）、config.py（模型名白名单）、experiment91.py（两固定身份及现有结构检查）、ema.py和single91.py（已知整数位置索引及浮点坐标表原样复制/唯一freeze校验）、现check_candidate.py及candidate测试、两SCIENCE。现tools/pretraining/swin_source.py与swin_strict.py只增固定V2 Small选项，复用原下载relay/身份/SHA/逐张量映射/严格加载及源目标前向数值检查，不另建转换链或修改旧R117资产。临时relay和测量脚本仅放本次technical-prep。复用现有训练、启动、保存、评估和观测链，不新增平台。

保持单模型、固定9:1、原scale.85–1/旋转/光度、FP32/G64P8/world8、原AdamW/LR/WD/age及24k成功/800评估/patience4/1600warmcopy。仅适配V2本身必需的静态浮点缓冲区复制，不改变可训练参数EMA公式。完整内部唯一选点>=.996才官方5849一次；raw .999仍是总体目标。满24k预算和实际CPU/内存须可满足当前未启动轮的新7.75h训练/8.75h整轮及9h分配，不能靠早停兜底。

调用点实查补充：现inference.py仅扩模型白名单，现r92_initial_gate.py仅优先消费新static_buffer_count=48，现check_candidate.py同样区分48总缓冲区与24整数索引并复用原Swin检查。否则正常正式链会把V2误当24缓冲区或拒绝架构；这三个已有消费者属于本候选不可缺少的适配，不新增推理/初态链。

先在准备池完成必要CPU实际factory、512前向、EMA边界及严格冷重建，复用字节一致的已通过证据。源码/配置实际push、远端回读和精确部署后，原唯一GPU入口于前轮actualsafe后一次activation/launch；真实八rank初态与自然首成功证明GPU链可跑，不另做GPU预演。

实际准备已通过[3目标/12编译与公开源严格链](../../records/preparation/swinv2-20261101/README.md)。R154仅一次固定source/window8/RGB512/seed29配方，完整工程有效selected未严格胜R107则关闭本合同；只过R107为有限正向，不以AP救门；同时严格过R120才另作固定seed27，无尺寸/window/LR/seed补考。历史2.5倍预算情景可容满24k且不假设早停，具体数值与不确定性在原CPU回执中。新GPU allocation只在前轮进入freeze时创建；其9h与完整8.75h T0预算仅15分钟余量，不提前长时间占用分配。
