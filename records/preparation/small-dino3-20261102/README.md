# R175/R176：Small DINOv3 公开初始化准备

[真实CPU证据](CPU-CHECKS.json)含本次3个来源/候选/存储目标与此前本次2个模型/配置目标，共5项实际目标通过、12源编译通过。模型目标使用合成公开骨干并stub正式metadata回调，验证完整342加载、原生二分类头/年龄RNG、真实512/P8的345有限梯度、0optimizer steps及真实CheckpointStore→inference冷图exact；不称为实图、DCU峰值或独立八rank证明。正式来源和SCI绑定由下面的完整来源验收及两次无mock公开fresh分别核实。loss/engine/DDP/EMA/persistence/inference实现SHA保持，沿原证据复用，不重复GPU预演。

[来源汇总](SOURCE-SUMMARY.json)：[发布者](https://huggingface.co/timm/convnext_small.dinov3_lvd1689m)固定revision `c497e64d5d483b3ad42eac93e2ef9d1799f9547c`、public/non-gated；完整原bin197948066字节，SHA `587d3fd7f07b9a2dc5a27276b45444826898c94dd2bdaeebf7d901c89dc96ce5`。完整342无头张量两factory strict/FP32有限exact映射，36 gamma reshape、41 LN、36 GELU/SD、40 Conv、0BN/buffer及pool/norm_pre/pre_logits非张量语义通过，**无timm→TV前向数值对比**。转换SHA `33fdc1b52cbfefa8f1f8fa35de081e60526e6baa2ab99a167e8a1105fc8c071e`、provenance `bdf428cd42138e329cbbbafda538f11746f2a7060dc845222fd0377ff97dcd9e`。源为自然LVD-1689M自监督DINOv3/ViT-7B蒸馏，来源224/center/crop1仅为属性；候选仍原512/ImageNet输入。许可为DINOv3自有协议，不能称Apache2，完整许可SHA `25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e`保留，适用条款不简化为无条件许可；预训练重叠未知。

[真实两fresh](PUBLIC-FRESH-SUMMARY.json)：R175 Base seed27 whole`2d6f2eefb533ff0976071648b10c39ca91a6983c8155241201840ceb74d2512f`；R176 Small DINOv3 seed29 whole`126ddef0d745516515716a252ad2e68847486b5163897470c51c8eed303ec333`。两者均345全训/5空原生AdamW组、原betas `.9,.999`，公开完整344/342分别验证，均无图像forward/旧训练checkpoint/官方读取或GPU初始化。R176只替换直接R107的完整公开初始化，输入、loss、采样、LR/EMA/age/24k-800-p4及后续网络不变；R175仅完整有效R174严格超过R107/R120且safe后同配方seed27，沿现有集中policy为新精确Base round登记16GiB，R176仍默认12GiB。当前R174的6ec冻结包/policy不改。

原件在`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/small-dino3-20261102-32e3/`。11个原CPU任务的SHA/exit/OUTPUT SHA及无越界/任务树退出已逐件核入CPU-CHECKS：包括来源/模型预期RED、首次严格转换的Identity.in_features旧报告假设、错误适配器漏passed，以及三次合成fixture的正式站点/回执假设错误。均保留原失败回执；转换原件另存`converted-failed-1`，修正后完整strict通过，不把这些工程失败当科学阴性或覆盖为firsttry成功。

准备719a/root0–11共12核、保留但未恢复的closed peer12–23共12核，与训练24–119共96核互斥，nice19/隐藏GPU。两SCI已用实际fresh SHA定稿；现进行Git同步与技术包，尚无新normalformal/clock/activation/GPU。后备正常CPU在R174收尾窗口完成，完整结果/actualsafe后沿现有入口串行接续；raw .999未达，未证明临床机制或AUROC收益。
