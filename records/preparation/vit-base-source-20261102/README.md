# 自然 DINOv3 ViT-B 公开来源与 CPU 可行性

[发布者单一来源](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m)固定revision `c6a5fb7d12bbd3cf3b0079253141c3332aaed7da`，public/non-gated、LVD-1689M自然图像DINOv3公开蒸馏来源；不是自身实验teacher、医学权重或N06。完整DINOv3许可SHA `25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e`。模型卡只提供一般来源/容量依据，不证明本任务收益。

[实际来源验收](SOURCE-CHECKS.json)及[完整handoff](PUBLIC-HANDOFF.json)确认342,624,338 B原始bin的LFS SHA `ed8a1265847086c913535fe7d84c0156a68dc98c0edd52298b8bb4bb7bdcbbb1`，全部162张量经原生factory strict/exact/FP32有限；0forward、0数据、GPU未初始化。原件在exp-v3/assets/public/vit-base-dino3-c6a5fb7d/，无转换，[provenance](PROVENANCE.json) SHA36847019…52f0e、handoff f0dbbd7e…1c088。公开源256与未来比赛512区分，不声称逐字节复现原Meta BF16 RoPE。

[原生512随机工厂](SOURCE-FACTORY-AUDIT.json)实际162参数/state张量、85,641,216元素、768维/12层12头/1CLS+4register、无QKV bias、native final LN eps1e-5、head/fc_norm Identity、noBN。唯一非persistent FP32 RoPE[16]摘要192c9d93与原Small相同，仍不能只凭state_dict证明完整函数。所列4个库文件及timm1.0.27实核，未pin整个transitive包；未做GPU显存/吞吐证明。

最小源码仅给原 `tools/pretraining/deit3_source.py` 增显式vitdinobase profile及宽度消费；完整来源API实跑通过，原DeiT3/Small DINO/CAFormer三profile字典和原prepare默认签名逐项相等、旧常量保持，1文件compile。原全量旧权重验证不重复，当前训练factory/保存/EMA/WORLD/启动未改；新来源不等于已实现或可启动训练候选。新profile文件SHA `7fa48427b413198744d9d50b871367f721a5d30a27f2b3095708d0fcdc4e0e27`。

[来源CPU回执](CPU-SOURCE-GUARD.json) pi0yumhw/`de039c79f490aa5988bbbed19d0833c1b2f809efc75222f27df44b5fc0476c14`、[原生512回执](CPU-NATIVE-GUARD.json)32vldjel/df074abc…166b均exit0/整树退出/无affinity违规；registry _uftvirx/f16a4f64…aecef亦0。均在当前5de root0–11准备池、nice19、隐藏GPU执行，与训练24–119互斥。当前R184R1及后续R188/R189冻结/顺序不改，不新增round、clock、GPU或资源；raw .999未达。
