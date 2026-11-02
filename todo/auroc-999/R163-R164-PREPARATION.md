# R162期间的下一固定准备

R162已完整安全结束，所选内部AUROC `.9875302682389093`低于R107/R120，原固定余弦16关闭；R163/order75固定seed27身份保留但无资格，不formal/GPU。独立R164/order76为非医学ImageNet22K→1K监督公开DeiT III Small、原512/9:1/G64P8/FP32/age/原logistic/AdamW/LR/EMA/24k-800-p4。这是模型与公开初始化整体配方，不从R141的DINOv2阴性推断所有ViT，也不把收益归为已证临床机制。两SCIENCE已写入，R164实际fresh张量64556b75f7fb9b0f1946089b5a0115de38348f049aba6252b49f3cd8bf5c9a93；仍须最终受影响CPU证据、Git同步及精确部署，不将登记当启动。

一手来源：[timm模型卡](https://huggingface.co/timm/deit3_small_patch16_384.fb_in22k_ft_in1k)、[原作者DeiT III](https://github.com/facebookresearch/deit/blob/main/README_revenge.md)、[现有timm1.0.27模型定义](https://raw.githubusercontent.com/huggingface/pytorch-image-models/v1.0.27/timm/models/deit.py)。三者声明Apache2.0；registry为原ImageNet mean/std、公开384/crop1.0。公开源384与比赛512分列；固定使用原生位置重采样到512，需先核实际依赖、完整public strict/非张量语义及保存冷加载，不能把网页广告当实测。

准备均在本人独立`technical-prep/deit3-20261102-32e3`，aa0f根0–22/23核、nice19、隐藏GPU，与peer23及训练24–119互斥。必要CPU合成512/P8前后向、原工厂参考、EMA及真实保存冷图已通过，不读真实图像/官方、不加载历史训练checkpoint、不做GPU预演。训练、启动、保存、评估和监控仍复用既有链。

原生构造与固定registry已实核：timm1.0.27、176个public张量、384维/12层6头/25LN/0buffer，set_input_size原生把24×24位置网格变32×32；固定revision8d59b2810f277c19d0cdd1b1e691d93c24aab6db，bin实际SHA88140b2a114ec4db0e3588af052162b87fcdd1bb92d5e85dd93f7d2adc1fb526/88885197 B，176全1000头strict/exact/FP32有限，零转换/零来源forward。原Pod proxy503、来源回执写入失败及拒绝覆盖旧原件的失败均保留，修正后guard6a928e39通过；公开provenance23453a05、handoffb7b1aa1b。仅pin版本及两个原生定义文件，不称pin整个transitive timm包。

后续不可缺少的最小范围限定：tools/pretraining/deit3_source.py复用small_source.fetch/source_chain核固定原件与native full1000，零转换；现models.py增原生VisionTransformer小子类/严格来源与二元age包装，config.py加模型枚举；experiment91.py增两身份/原生描述和保存绑定，inference.py只加模型允许项/实际图核验，check_candidate.py复用原验收仅增native差异；tests/test_deit3_model.py两必要原生目标、原candidate目标及两SCIENCE。现engine/distributed_engine/EMA/r92_initial_gate/训练启动保存评估监控均不改，无新平台。先RED再CPU合成512的原生参考/梯度/P8/保存冷加载，按实测存储与历史时间情景核预算；这是CPU可运行证据，不是GPU峰值/吞吐保证。允许这部分必要CPU合成forward，仍无真实图像或GPU预演。未验证不部署，当前608冻结不动。

最终5受影响目标/8编译及更新后的公开fresh初始化回执已真实通过，原6 loss/WORLD目标按SHA复用，见[精简证据](../../records/preparation/deit3-20261102/README.md)。配置format白名单遗漏先失败，最小修复后窄重试通过；另补kind及来源身份后核最终5/8，均不重写旧回执。现待实际Git同步/回读与唯一R164正常formal/部署，不重跑旧R162或对其调参。
