# 自然图像 DINOv3 ViT-S 来源先验核查

本页只是R181/R182之后的来源与原生库可行性准备，不改变唯一活跃R180或既定接续，不注册新round、clock或GPU授权。单一候选为发布者[timm/vit_small_patch16_dinov3.lvd1689m](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m)，不扫描plus/base/其他权重。R141 DINOv2及R164/R166 DeiT3阴性是明确反证，不凭换Transformer预设收益。

16:58实核公共registry：revision `3bf4720a82ec2066db88137180ff1f83a675cef0`，public/non-gated，bin 86,407,250 B / LFS `b7e7b7d1518a4dad2d92304ff47967c3dfee806f863552a06870de1e7498e7a2`，声明DINOv3许可与自然LVD来源。完整许可、原件和strict加载尚未实核，不能称已接受公开初始化。

实际Pod/timm1.0.27随机原生factory、512、avg池、SD.1/head-drop0构造成功：21,586,944元素/162参数及state张量，patch16/384/12层/6头、1CLS+4register、无QKV bias、原final LN eps1e-5、fc_norm/head为Identity；无forward、图像或预训练权重读取、GPU未初始化。原生有一个非persistent FP32 `rope.periods[16]`，不能沿用ConvNeXt的buffer0或仅凭state_dict宣称完整函数身份；如后续采用必须显式绑定其原生生成/保存冷图及精确库来源，不截为BF16或重采样旧已训练位置。

库实核eva.py SHA23314ef5…0a12d，pos_embed_sincos.py 8d1751ad…cb914，attention.py608f054b…a947，patch_embed.py7b7e670f…55aa；只pin所列文件，未声称全部transitive包。原件在`exp-v3/technical-prep/vit-dino3-source-audit-20261102-32e3/`；首次probe因读取不存在的报告属性`num_reg_tokens`失败（guardwrxbx5m3/b08f5b3a…4cd79 exit1），按原生reg_token实际shape修正报告后guardjjg_0_rz/ab044715…8a211 exit0；没有修改任何生产模型或将失败写成通过。

上述16:58阶段仅有repo外审计脚本及本页，无生产改动（历史时点）。后续若采取，只沿现native来源验收、models/config/fresh/experiment91、candidate检查和现保存/推理路径作必要分支；固定原R107/seed29/512/原logistic/G64P8/优化EMA/age，网络采用完整原生features及发布者avg读出，不与Large、ROI、MIL、margin等叠加。先做完整source strict与受影响CPU验收/资源情景；不可运行就不准入，不能依赖早停规避满额预算。

不使用医学权重/教师/N06，官方隔离；该核查不能证明临床根因、AUROC收益、DCU峰值或吞吐。raw .999未达。

17:07已通过完整bin LFS/配置/许可及完整162张量strict/exact/FP32有限、非persistent RoPE语义核查；provenance6193971f…aed8、handoffc899c713…7fb2，来源forward0/GPU未初始化。真实RED为旧API不接受source_profile；仅给原native来源入口加显式vitdino3 profile及models库pin后，新来源guard_rbxe5r7/9a52d2f7…61d85 exit0。原DeiT3三参数默认也实际完整176/full1000 strict、原provenance语义相同，guard6__73aww/d7dc5d10…64770 exit0，旧资产未改。当前只改tools/pretraining/deit3_source.py与models.py必要库常量/检查函数；新训练factory、fresh、P8及冷图尚未实现/验证，仍非可训练候选。
