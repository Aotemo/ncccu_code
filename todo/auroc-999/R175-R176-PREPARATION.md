# R175/R176：CLIP 固定复核与 Small 自监督公开初始化

R174已完整6400/c8/bad4、所选`.9855170013883353`、双门false/官方未跑/actualsafe，R175无资格且未normalformal/activation/GPU。[R176已实际接续](../../records/R176/README.md)，13:05实观50 success=attempt/0skip/eval0，八rank/345全训/5空Adam/自然首25有限。唯一Git/GPU仍32e3；719a准备root0–11/12核与保留peer12–23/12核、GPU24–119/96核互斥。运行e9a7f1d冻结，后续准备不热改当前包。

R175/order87只在完整工程有效R174唯一selected严格超过R107 `.9884405731067525` 和R120 `.9894773092062407`、actualsafe后，执行完全相同Base/public CLIP/seed20261027固定复核；未满足不normalformal/activation/GPU。不因R170旧部分结果判本轮，不改R174科学合同。

独立R176/order88直接R107 Small/seed20261029/512/ImageNet输入/原pair-logistic/G64P8/default AdamW/LR/EMA/age/24k-800-p4，只换完整骨干的公开fresh初始化为`timm/convnext_small.dinov3_lvd1689m`固定revision `c497e64d5d483b3ad42eac93e2ef9d1799f9547c`。来源为自然LVD-1689M的自监督DINOv3、由ViT-7B蒸馏，不是医学专用预训练，也不等同R141的ViT/DINOv2。仅检验此固定初始化组合，不先声称临床机制或收益，不扫描LR/normalization/分辨率/seed。

真实来源准备已在PodCPU核registry public/non-gated和固定commit/model card/config；完整原bin197948066字节、LFS SHA `587d3fd7f07b9a2dc5a27276b45444826898c94dd2bdaeebf7d901c89dc96ce5`匹配。原生342无分类头张量完整strict并逐张量映射至原TorchVision Small的342骨干，FP32有限exact、36 gamma reshape、41 LN/36 GELU/36 SD/40 Conv、0BN/buffer及pool/norm_pre/pre_logits非张量语义已核，无forward。来源224/center/crop1仅是来源属性，比赛仍512；不得补造两张public分类头张量或将342写成344。生产接入、转换资产、fresh/cold与SCI均已完成，下列CPU及实际训练证据分列。

许可为DINOv3自有协议，不能称Apache2。固定发布者声明指向该许可，Meta官方完整文本已经Git blob `f531b1e6b5ab2318957bbf8ad1bda9f800a23e17`核字节，SHA `25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e`；有使用/修改/分发授权，分发附许可、研究发表注明来源等条款保留，不将其概括为无条件许可或臆断法律结论。权重/数据不进Git，许可和原件位于exp-v3；预训练数据与本任务重叠未知。

最小范围：扩现`tools/pretraining/small_source.py`可选固定profile，旧224/384入口默认和行为保持；`small_strict.py`复用原映射/非张量核验而允许已登记的342无头来源；`runtime/bundle/retrain/models.py`与`config.py`只为固定Small自监督格式加入完整严格加载，不改后续网络/forward；`experiment91.py`及两SCI登记身份/互斥/选点绑定，现`tools/validation/check_candidate.py`仅在既有backbone_only断言名单加入该身份，不增验证链；扩`tests/test_pretraining_sources.py`与`tests/test_candidate_binding.py`，必要生产加载/RNG/冷加载目标在`tests/test_dino3_small.py`。其他训练/启动/保存/评估/监控/loss/DDP/EMA/optimizer实现不变，不建新平台。

生产加载保留原生1000头构造所消耗的RNG，仅临时取下原对象，完整strict载342骨干后恢复原对象，再走原二分类头构造；不加载R107/R170或任何训练checkpoint、teacher/BN。CPU只验证实际受影响来源/配置/加载/保存冷图，原loss/WORLD按相同源码证据复用；真实八rank与自然首更新接管GPU可运行证明，无额外GPU预演。必须实际Git同步与精确部署后才能接续，R174完整结果决定唯一分支，raw .999未达。

调用点核对补充两处当前实验不可缺少的范围：`fresh.py`现有按架构限定初始化kind的条件，仅允许该固定Small自监督格式继续保持self_supervised身份；不改预算/优化/梯度或旧来源。R175为新独立Base seed27身份，复用R174的16GiB工程额度需在原`ROUND-STORAGE-POLICY.json`仅增该精确新round名，并扩现`test_round_storage_policy.py`验证，旧/当前round不热改。

原`tools/activation/prepare.py`生产配置只为此新无头格式补传`expected_backbone_state_keys=342`，保持同一个打包入口；其他格式与训练流程不改。生产模型窄测2目标已PASS：合成公开骨干fixture验证完整strict/二分类头与年龄RNG/真实512 P8的345有限梯度、0optimizer steps和CheckpointStore→inference冷图exact；正式metadata回调在该fixture中stub，真实公开字节、固定来源及正式SCI绑定另由实际来源验收/两fresh/候选目标验证，不把合成P8写成实图或DCU峰值/吞吐实测。

- [x] 完整来源strict/非张量、两public fresh、5实际目标及12编译通过；[精简真实证据](../../records/preparation/small-dino3-20261102/README.md)含原失败索引，未虚报firsttry。
- [x] Git e9a7f1d/tree1399aa09已push/实际回读，两technical包在R174训练期间完成；R175仅技术包。
- [x] R176正常CPU18/18及精确部署在R174训练期间完成，窗口余量32057.710≥31560秒；R174完整阴性/actualsafe后一次activation/direct launch0，实际八rank初态/自然首25已证。不为R175创建formal clock/grant或GPU。
