# R94：ConvNeXt Tiny，backbone LR 单变量减半，固定 9:1

本规格随源码维护；实际 Pod、Git commit、路径、CPU 分配、clock/token 只由当次显式绑定产生。`configs/R94-SCIENCE.json` 保存本轮科学配置，prepare/check_candidate 必须显式传入 `--science-config configs/R94-SCIENCE.json`，site 固定其绝对路径和 SHA，package 保留相同 bytes 的 SCIENCE.json。模型及参数组消费者从该配置取得 LR，不回退 R93 默认值。CPU 准备不创建正式 clock、训练锁或 GPU 进程。按最新用户指令，LR-only本轮不做独立零forward GPU预演，明确记为not_run_for_lr_only；新Git/实际SHA冻结后一次activation/launch，由真实训练八rank初态、参数组、空Adam和实际mask核验接管，保留实时空闲/锁检查。

本轮依据 R93 完整16条内部 dev 和既定主选点规则：c3 EMA@2400，AUROC0.9866344126546825、AP0.8167243430095755、error_twice22200。唯一科学变量为 backbone LR2e-4→1e-4；不根据官方结果选择本变量。精确 profile 为 `single-tiny22k-bc48a87f-fixed-9010-lr1e4-v1`，candidate 为 `r94-convnext-tiny-imagenet22k-to-1k-lr1e4-9010`，order4。R93 配置与其80e冻结副本保留。

候选为公开 `timm/convnext_tiny.fb_in22k_ft_in1k`，revision `bc48a87f119bae1b06418f5340bfa97ef75d4609`。原 blob SHA 为 `8cb65dc650a7fb25eaadbff3eb69559d02f83cdc5afdfb3785d1889cdc3e3518`，转换权重 SHA 为 `1ec1dd0b7045237eafe1c8f1becd58456cc811326c97ec416ae8b2e5ede860ed`，provenance SHA 为 `40c14c8bd780d4feb88ae96d7e9def1ef78499daf93ae4412eff78502780233f`。来源链、182 键映射及非张量结构须消费真实验收回执；生产工厂仍须完整 1000 类 strict 后再新建 binary head 和零 age。无 N06、历史 checkpoint、teacher、BN 或优化器参数来源。

原生 avgpool、768 特征、final/internal LN epsilon 均为 1e-6、dropout 0.2、stochastic depth 0.1、单线性 head+两维 age；无 BN、q 或 attention。上述来源与结构均同 R93。seed20261027，head 使用 `nn.Linear` 原生初始化，age 为两个 FP32 零值；Adam 状态为空，online=EMA。初态 tensor 必须仍为配置中固定的 R93 控制值 `eba53feeafdf20bc86855a40648bc453358928d148078c6ae78b65f059846cc7`，不加载 R93 checkpoint。

固定 fit46256/dev5140，来自原始 train 的同一 9:1。stage1 的 18 冻结文件及 FREEZE 保持原 bytes；FREEZE SHA `90e063dca3ab22225308043ba4b0de379688675ef00606d00b8b36e2135bc0c6`，manifest blob SHA `7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377`。物理迁移路径另建清单，不改原科学身份或重新拟合 age。

512 输入与 prepared512，ImageNet normalization、既有增强、FP32 无 autocast/缩放。reference physical32；八 rank 各8、accumulation1，全局64/P8。保留全局两批 reference32 的顺序和增强种子后再 shard；不强制每 rank 一个正例，不承诺跨 world 的完整浮点/RNG 轨迹相同。

S800×8=6400 成功更新，warmup1600，cosine 尾倍率0.01，无 skip/早停；backbone LR1e-4、最后 linear 与 age LR1e-3，矩阵 WD1e-4、bias/LN 向量 WD0、age WD1e-4，clip1/BCE/no mixup。末端 LN 属 backbone；AdamW 参数收缩也随该 LR 改变。EMA FP32 warmcopy1600、decay0.999，首 blend1601，最终 copy1600/blend4800。LR 与 EMA 都用成功更新计数。

完整16条内部 dev 读数依 error_twice 升序、AP 降序、成功 update 升序、最后 EMA 优先选唯一 checkpoint。冻结 source/job/split/age/variant/blob/tensor 后，官方完整5849只做一次最终比较，reservation 在模型构造及官方数据读取前绑定 canonical 训练根；换输出或复制 freeze 不能重入。官方不参与训练、age、早停、难例或选点，内部/官方分报。raw AUROC 未达0.999不称总体完成。

所有新非源码对象位于显式 exp-v3 工作根。数据与 prepared arrays 只按实核清单迁移；缓存新 namespace/索引绑定新数据根和当前文件身份，科学图像 bytes 不改。三份保留模型不作训练依赖。CPU 准备 A/B 合计不超过实时有效 CPU 的20%；GPU阶段也只用新显式 owner/rank/loader/control masks，不能回退全量 affinity。旧 Pod、旧 token/clock 和旧冻结运行不恢复。

墙钟 policy 沿当前保守预算：total12600、binding1800、train hard7200、freeze300、official1200、cleanup/soft margin150 秒。它不是吞吐或租期保证。正式只创建一次T0/D/H/token，train_H=min(train_start+train_hard,D-freeze-official-cleanup)，freeze_H=min(freeze_start+freeze_hard,D-official-cleanup)，官方仅在尾窗足够时开始；不自动延长、改周期或重置。失败/unknown保留真实终态，禁止再次activation/launch或官方前向。
