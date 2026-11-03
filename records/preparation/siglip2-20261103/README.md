# R214/R215 受影响 CPU 准备

[7必要目标/12唯一源码compile](CPU-CHECKS.json)实际通过，最终guard48enz91b/452c44c0 exit0；正式GPU尚未启动。目标为原生图/RNG/单age oracle、P8全部有限梯度/五空AdamW组/EMA复制/真实CheckpointStore→inference冷图、两候选绑定、封存unit与imagenet绑定、精确存储边界及两actual public fresh。P8用合成512输入，generic EMA fixture只验证R8_RULE第1步复制，正式1600/.999由实际fresh/后续formal消费。CPU不证明DCU内存或吞吐。

[完整原件strict](PUBLIC-SOURCE-STRICT.json)为固定timm/vit_base_patch16_siglip_512.v2_webli revision3c4b2d3989d7cfae00a09fff172c0a6cf8cbb32a，bin374141074 B/SHAe6cebc19、162完整公开张量FP32有限/strict/exact、93520128参数、无buffer/BN，public tensor45f38421。Apache2声明与canonical许可SHAcfc7749b固定；[provenance](PROVENANCE.json)c93b8f30及[handoff](PUBLIC-HANDOFF.json)8ec651b8。仅零forward来源加载，不称来源前向数值对齐；两timm1.0.27文件pin而非整个transitive包。公开MAP latent、q/kv/proj、pool LN/MLP、全部位置与Transformer保留；head原Identity，追加Dropout.2+Linear768→1，age一次。

[R214 actual fresh](R214-PUBLIC-FRESH-INITIAL.json)为原EfficientNetV2公开BN配方seed27，wholef1144444/params952189f2/bufferseb77f82b；459全训/五空AdamW组/336固定buffer。仅完整R213严格双门成立才GPU，否则technical。

[R215 actual fresh](R215-PUBLIC-FRESH-INITIAL.json)whole与params均13e6317f、空buffer摘要e3b0c442；165全训/93520899参数/五空AdamW组。原R107512几何及采样/增强/logistic/LR-WD/age/globalclip/EMA1600/24k-800-p4保持，现成unit实现(.5,.5,.5)归一化；这是架构、公开预训练与匹配归一化组合，不是纯架构或临床根因证明。公开crop/interpolation属性不等于实际比赛预处理。精确R215轮名24GiB，默认12/free16保持，原子保存float场景只作容量估算、不是峰值保证。

原生模型未注册、模型卡WebLI大小写假设、测试fixture漏显式EMA rule、配置未绑定、封存写死ImageNet和未登记24GiB的真实失败回执均保留。fixture修正后P8独立GREEN；原生oracle最终取得独立干净回执，不把初次组合命令exit1写成全通过。只修当前受影响分支；原data/loss/gather/optimizer/EMA/训练与保存机制不变。当前R213冻结936不动，无新GPU预演、实图/官方前向、训练信号或新资源。raw .999未达。
