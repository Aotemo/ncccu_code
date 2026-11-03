# DINOv2 Base 固定镜像筛查：未采用

[真实Pod筛查](NATIVE-SCREEN.json) SHA `2a768b1ce55d17f26ec07d759982c8a2df7949da84527ee1b8e66f9a90ea4591`；[registry guard](CPU-REGISTRY.json)6yi0iv47/c2f5b859和[native guard](CPU-NATIVE.json)uicgea0y/40cf89bc均exit0。原件 `exp-v3/technical-prep/dinov2-base-source-screen-20261103-32e3`，d837/root0–11/nice19/隐藏GPU，与当前R217训练24–119互斥。

固定镜像`timm/vit_base_patch14_dinov2.lvd142m` revision `4685c99dabffe5affac90bd99dbffd25801ae58d`，public/non-gated；config/card完整字节匹配该revision的Git blob与size。registry/card标Apache2，但[该固定config](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m/blob/4685c99dabffe5affac90bd99dbffd25801ae58d/config.json)的pretrained_cfg.license为CC-BY-NC-4.0，不能把首个registry筛查当完整来源接纳。[原始DINOv2项目](https://github.com/facebookresearch/dinov2#license)明确其普通代码和模型权重为Apache2；因此这里只将本固定镜像来源身份保留未采用，未推断所有DINOv2版本均不可用，也未给出法律结论。若以后采用其他原始来源须独立闭合真实身份，当前无新候选。

实际timm1.0.27原生随机factory为174张量/86,579,712参数/0buffer，12层/768特征/CLS池化/Identity头，patch14、严格518输入、37×37网格、位置shape1×1370×768、dynamic_img_size和dynamic_img_pad均False。故默认图不能直接接现512合同；未执行或选择padding/重采样改法，未偷偷改比赛输入。仅记录vision_transformer.py和patch_embed.py两文件SHA，不称整个transitive库逐文件pin。

未下载/加载公开大权重、未图像forward或读取数据/训练checkpoint/官方，未注册候选或改变R218/R219队列，不证明生产可运行、GPU资源、临床机制或AUROC收益。本次仅metadata与原生结构筛查，不写成public strict/fresh/生产CPU验收。
