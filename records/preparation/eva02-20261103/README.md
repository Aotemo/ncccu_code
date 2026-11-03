# R220条件复核 / R221原生EVA02 Base CPU准备

[实际汇总](CPU-CHECKS.json) SHA28bfe302de427f04751053e55ba2ac83d166da1057e6ad98404b76f5dffeba78：10必要目标、13个项目源码编译，原始回执/output SHA及互斥CPU资源均核验，最终[归档guard](FINAL-ARCHIVE-CPU.json)ypf1q_2x/41d1191c exit0。准备7c7af22c/root0–11/nice19/隐藏GPU，与R219训练24–119互斥；当前R219冻结87cc919完全保持。无新GPU预演、clock或收益声明，尚待正式CPU包及精确部署。

固定[发布者自然图像模型](https://huggingface.co/timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k) revision81063ecf，registry/card/config MIT一致，完整[作者许可](https://github.com/baaivision/EVA/blob/5623d08bb08a8268c7bc18e1d9b8b4bd651fe4d9/LICENSE) SHA80c5e100。原件348,553,277B/SHAdbbfef82已在Pod核完整LFS；原生448/full1000的[236张量strict/exact/FP32](PUBLIC-SOURCE-STRICT.json) SHA2a4cc9f3通过，37LN及SwiGLU/MLP LN/原生avg→fc_norm/非持久FP32 RoPE均核。[provenance](PUBLIC-PROVENANCE.json)9e4e701e、[handoff](PUBLIC-HANDOFF.json)e1060093；来源预训练的EVA-CLIP是发布者配方，不是本任务teacher。

三项模型测试113.012秒PASS：原生448→518的位置网格32→37、保留CLS前缀、比赛512右/下补6个归一化零像素、独立patch/pool oracle及RNG/age一次；P8全237参数有限梯度、五空AdamW组、EMA、真实CheckpointStore→inference冷图exact。源CLIP mean/std由现有数据接口显式消费，原输入缓存/增强/采样/age/448 logistic/默认AdamW-LR-WD/globalclip/EMA1600-.999/24k-800-p4保持。原作者448/squash预处理未被冒称复现；架构、初始化和所需归一化整体比较，不作机制归因。CLS仍在注意力序列中，仅末端patch均值排除前缀。

两份零图像前向公开工厂初态：[R220](R220-PUBLIC-FRESH-INITIAL.json) seed27/345参数/whole55fcad7c，原Small12k；[R221](R221-PUBLIC-FRESH-INITIAL.json) seed29/237参数/86,614,275元素/whole51592a3d、live buffer1/持久buffer0/shaafe09042。此处初态测量先于最终冻结包fresh元数据，正式CPU准入另核完整生产合同。新候选及受影响R218/R219绑定窄验GREEN；精确R22116GiB函数fixture验证13GiB通过/17GiB拒绝/free15GiB拒绝，旧默认12GiB/free16GiB保持。CPU测试不证明DCU峰值、吞吐或八rank可运行，后者由真实初态及首成功证明。

初次配置未注册模型的RED、R220汇总属性缺失及候选fixture缺science失败均保留，修正限相应登记/一次性汇总或fixture，无科学训练尝试。单流传输慢且直连失败，改同一既有主机的有界8路字节传输；21块完成后只在Pod[合并核全SHA](TRANSPORT-ASSEMBLY.json)，未量化或更换原件。参数、源/归一化/SCI身份均由实际SHA绑定；四个timm文件pin不代表整个transitive包。

完整R219严格超过R107/R120才R220，否则独立R221；[合同](../../../todo/auroc-999/R220-R221-PREPARATION.md)不改变当前选点、官方隔离或旧token/grant。raw AUROC .999未达。
