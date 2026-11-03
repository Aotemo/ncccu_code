# R222条件EVA02复核 / R223原始官方DINOv2 Base准备

16:23预声明；17:06工程恢复后接续对象明确为当前唯一GPU R221R2/c732667，最近1425成功/0跳步/八rank活跃，超过R1失败点。R221/R1均工程failed，不作科学阴性。R219同封存256图fit已完成；仍仅既有7c/root0–11准备池与训练24–119互斥。R222/order137，R223/order138，不派发第二CPU任务或GPU。

R221R2完整唯一内部dev所选必须严格超过R107 .9884405731067525及R120 .9894773092062407，才允许R222同EVA02配方seed20261027独立fresh复核；否则不启动R222，接准备完备的R223。该条件不恢复已因R219阴性而不eligible的R220。当前R222/R223都无最终SCI/公开fresh/正式CPU/clock/activation/启动资格。

R223候选仅原始官方自然图像`dinov2_vitb14`无register骨干/LVD142M，代码固定7764ea0f912e53c92e82eb78a2a1631e92725fc8；[官方说明](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md)列出普通DINOv2代码及权重Apache2、ViT-B/14下载，和另行许可的医学分支分开。本轮仅考虑普通自然源。原[timm镜像筛查](DINOV2-BASE-SOURCE-SCREEN-20261103.md)的许可不一致身份继续未采用，不改其原metadata或用新解释覆盖。原[R141 Small合同](../../records/R141/README.md)完整.9418572868阴性保持，不重开Small/504/固定旧头初始化。

原始官方URL `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth`，346378731字节/S3 versionId `UgXoSvH_JJMe1OVbpL.ucwhEuZ6APOrb`。Pod实测完整SHA0b8b82f8/provenance de08471a/strict91d62c19，175原张量FP32 finite/strict exact、0图像forward，guardm5ii10ey exit0；这是实际字节校验，不称作者发布SHA。复用177固定原始公开代码文件与Apache2许可，不读取Small或任何历史任务训练权重，不使用医学预训练/本任务teacher。第一次传输误将session返回当完成导致超过拟定8流并有4次断连，原失败保留，四份分别重传0后以完整原件SHA验收。

固定拟议输入仍原512/缓存/增强/ImageNet归一化；仅在模型内向右/下补6个归一化零像素得到原生518/37×37，不裁剪/resize。按原始官方特征接口固定final-LN CLS与mean-patch拼接1536维，二分类Dropout .2/原生Linear→1及age各一次；原未用mask_token冻结，原12 blocks/LayerScale/位置/MLP/FP32 math attention保持，不扫描池化或补边。该容量、公开源和必要输入/头联合假设不作单一机制或临床病因声称。若官方结构实际不符，停止此候选并记录，不静默换合同。

拟议源合同其余直接R107：seed29/G64P8/448 logistic/默认AdamW-LR-WD/globalclip/EMA1600-.999/最多24000成功更新、每800评估、连续4轮无提升/唯一dev选点及官方.996门后一次5849，官方不参与准备或调参。R222只改R221R2科学seed/独立身份；两轮精确16GiB存储条目，默认12/free16不变。当前冻结R221R2不热改。

必要文件边界：优先小改现有`dinov2_model.py`的显式Base工厂/读出，保留旧Small默认；`models.py/config.py/inference.py/experiment91.py/fresh.py`和既有`validation/check_candidate.py`只补不可缺注册/身份/冷加载消费点。`activation/prepare.py`原通用kind消费可复用，无需改动。新`tools/pretraining/dinov2_base_source.py`仅来源核验/登记，复用source_chain，不另建训练/启动/保存/评估/监控链；新`tests/test_dinov2_base.py`和`tests/test_dinov2_base_candidate.py`、两SCI及原精确存储测试/配置，真实小回执归`records/preparation/dinov2-base-official-20261103/`。若实核发现额外必需文件，先在此更新精确范围。

- [x] 完整官方原件字节/固定177代码资产与许可SHA、原175张量FP32 finite/strict exact，零图像forward；不将网页或下载头作该证据。
- [ ] 新模型/候选先真实RED，再最小实现；CPU native oracle核pad6/37网格/CLS+mean/旧Small默认与RNG，P8实际反传/177可训参数及五空组、生产EMA warmcopy/blend、真实CheckpointStore→inference冷图exact。masked参数不纳入optimizer，不把CPU storage当DCU峰值。
- [ ] 两实际public-fresh初态/必要窄目标与源码编译、按SHA复用旧实现证据；本机/登录节点不执行Python/哈希/解析/测试，不与当前fit准备任务争用owner lease。
- [ ] Git同步/完整commit-tree回读/精确部署及既有正式CPU在当前训练期间完成。完整所选双门与actualsafe决定分支，只沿既有入口串行接续，真实八rank初态/首成功证明可跑，无GPU预演、新资源、审批、Agent或旧clock/token/grant重置。
