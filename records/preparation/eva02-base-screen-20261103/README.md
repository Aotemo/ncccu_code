# EVA02 Base：固定metadata与原生结构筛查

[实际结果](NATIVE-SCREEN.json) SHA `4741a4c90ae31de3043e3966115049cf28fbd9cae743bde69bc6aa2e0fdce6e7`；[registry guard](CPU-REGISTRY.json)9fx51adp/b1f1a9b0及[native guard](CPU-NATIVE.json)7i_7z5av/0cf38d4d均exit0。原件`exp-v3/technical-prep/eva02-base-source-screen-20261103-32e3`，d837/root0–11/nice19/隐藏GPU，与当前R217训练24–119互斥。

固定`timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` revision `81063ecfe9c381a16a19d06f396d6c7011aa426a`；public/non-gated，registry/card/config均MIT，config/card字节及size与该revision Git blob一致。[发布者模型卡](https://huggingface.co/timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k)描述自然ImageNet22k MIM与22k→1k微调；其预训练EVA-CLIP teacher属于公开来源配方，不是本任务teacher或旧checkpoint。原bin348,553,277 B、LFS期望SHAdbbfef82仅来自metadata，尚未下载或实核大权重，不当成public strict证据。

实际timm1.0.27随机原生factory：236参数/state张量、87,117,544元素；12层/768特征/原生avg池化/Linear768→1000，SwiGLU含MLP LayerNorm，final norm为Identity而fc_norm为LayerNorm。原生严格448/patch14/32×32，位置1×1025×768；一个非持久FP32 `rope.pos_embed` buffer为1024×128，ref_feat_shape16×16。源配置为CLIP mean/std与448/squash，不能默认为原ImageNet normalization，也不能将来源448变成比赛输入。512的显式padding/位置与RoPE处理尚未设计或测试。

四个被读取库文件SHA已记录，不称整个transitive库固定。0图像forward、未读数据/训练checkpoint/官方或初始化GPU，未加载公开权重、未复现原生来源前向数值、未注册候选/修改runtime或R218/R219队列。不能据此声称生产可跑、DCU内存/吞吐、临床机制或AUROC收益。当前仅保留来源研究结果，是否采用另依后续证据。
