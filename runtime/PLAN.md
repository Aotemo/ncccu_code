# R102：ConvNeXt Small22K，最多24000步与连续4轮无提升早停，固定9:1

R104以R103 Small/seed20261029为直接对照，仅配置`training_scale=[0.85,1.0]`替代旧缺省`[0.94,1.0]`；读取`configs/R104-SCIENCE.json`。其余增强顺序/随机抽样数、验证居中视图和以下模型/数据/时钟合同不变；本次只试一个下限，不据难组或AP补救整体内部AUROC未胜的结论。工厂和seed不改，原生正式包自然核对R103已实测初态，不另做control-only构造。精确身份见`todo/auroc-999/R104-DECISION.md`。

R103继续下述Small合同，直接对照R102仅将训练seed改为预先固定的20261029；实际入口显式选择`configs/R103-SCIENCE.json`及其新profile/candidate，以下R102的seed20261027是原对照，不适用于R103。其余科学值、资产、工程预算和真实初态/一次性流程不变；新seed整体tensor控制必须重新实测，不能复用R102或Tiny的控制。精确身份见R103配置及`todo/auroc-999/R103-DECISION.md`。

本规格随源码维护；实际Pod、Git commit、路径、CPU分配、clock/token只由当次显式绑定产生。`configs/R102-SCIENCE.json`保存本轮明确科学合同，prepare/check_candidate显式传入此配置，site固定其绝对路径和SHA，package保留相同bytes的SCIENCE.json。消费者读取配置中的科学值。历史运行与冻结包均不改。新fresh运行不增加独立GPU预演前置，由真实训练八rank初态、参数组、空Adam与资源校验接管；保留实际Git冻结、实时身份/独占GPU、一次activation和官方一次。

本轮直接对照预登记R95/BCE/seed20261027，仅成对更换架构和对应公开checkpoint为Small22K，不称纯容量消融。精确profile为`single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-v1`，candidate为`r102-convnext-small-imagenet22k-to-1k-24k-p4-9010`，order12。P4采样不执行，其余R95科学值保留，官方结果不参与本计划。

候选为公开 `timm/convnext_small.fb_in22k_ft_in1k`，revision `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b`。原blob、转换权重、provenance的真实SHA及实际结构计数只从本轮Small来源/strict/handoff回执进入pinned SCIENCE，不沿用Tiny常量或远端广告SHA。生产工厂须完整1000类strict后新建binary head和零age。无N06、历史训练checkpoint、teacher、BN或优化器参数来源。

原生avgpool、768特征、final/internal LN epsilon均为1e-6、dropout0.2、显式stochastic depth0.1（不用TorchVision Small默认0.4）、单线性head+两维age；无BN、q或attention。seed20261027，head使用`nn.Linear`原生初始化，age为两个FP32零值；Adam状态为空，online=EMA。Small整体初态由相同生产CPU工厂实测并写入SCIENCE，不能复制Tiny的eba或18313控制值；正式CPU包和真实8rank初态均须匹配本轮新控制。

固定 fit46256/dev5140，来自原始 train 的同一 9:1。stage1 的 18 冻结文件及 FREEZE 保持原 bytes；FREEZE SHA `90e063dca3ab22225308043ba4b0de379688675ef00606d00b8b36e2135bc0c6`，manifest blob SHA `7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377`。物理迁移路径另建清单，不改原科学身份或重新拟合 age。

512 输入与 prepared512，ImageNet normalization、既有增强、FP32 无 autocast/缩放。reference physical32；八 rank 各8、accumulation1，全局64/P8。保留全局两批 reference32 的顺序和增强种子后再 shard；不强制每 rank 一个正例，不承诺跨 world 的完整浮点/RNG 轨迹相同。

最多S800×30=24000成功更新，warmup1600，cosine尾倍率0.01按24000预算计算，不拉伸已跑R94的曲线。每800次成功更新完成一次内部dev轮次，online和EMA各评一次；监控该轮两者较高AUROC。首轮只建立基线，严格高于历史最好值才算提升（min_delta=0），否则坏轮计数加1，连续4轮未提升就早停，提升则清零。patience按完整轮次计，不把两个variant计成两轮；不继承旧min_cycles8。早停计数和最好分数随checkpoint保存。无skip；backbone LR1e-4、最后linear与age LR1e-3，矩阵WD1e-4、bias/LN向量WD0、age WD1e-4，clip1/BCE/no mixup。EMA FP32 warmcopy绝对1600、decay0.999、首blend1601，不因30轮变为6000。LR与EMA仍由成功更新计数驱动。

最大预算完成或合法早停均为completed。在实际完成的全部轮次中，online/EMA成对内部读数依error_twice升序、AP降序、成功update升序、最后EMA优先选唯一checkpoint；不再要求固定16条或必须跑满24000才允许选点。冻结source/job/split/age/variant/blob/tensor后，官方完整5849只做一次最终比较，reservation在模型构造及官方数据读取前绑定canonical训练根；换输出或复制freeze不能重入。官方不参与训练、age、早停、难例或选点，内部/官方分报。raw AUROC未达0.999不称总体完成。

所有新非源码对象位于显式 exp-v3 工作根。数据与 prepared arrays 只按实核清单迁移；缓存新 namespace/索引绑定新数据根和当前文件身份，科学图像 bytes 不改。三份保留模型不作训练依赖。CPU 准备 A/B 合计不超过实时有效 CPU 的20%；GPU阶段也只用新显式 owner/rank/loader/control masks，不能回退全量 affinity。旧 Pod、旧 token/clock 和旧冻结运行不恢复。

R102 Small 的新墙钟policy为total28800、binding1800、train hard21600、freeze300、official1200、cleanup/soft margin150秒；GPU分配上限从集中RESOURCE-POLICY读取32400秒。24000成功更新和最多30轮验证的科学时钟不变，这些只是更大模型的工程预算，不是实测吞吐或云资源租期保证。正式只创建一次T0/D/H/token，train_H=min(train_start+train_hard,D-freeze-official-cleanup)，freeze_H=min(freeze_start+freeze_hard,D-official-cleanup)，官方仅在尾窗足够时开始；不自动延长、改周期或重置。失败/unknown保留真实终态，禁止再次activation/launch或官方前向。
