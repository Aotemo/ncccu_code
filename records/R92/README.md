# R92 当前源码与准备记录

ConvNeXt Tiny ImageNet1K，固定 9:1，公开 fresh 初始化；与 R91 沿同一份 fit/dev、年龄、6400 更新及单模型评估流程。R91 使用固定 BN；Tiny 无 BN，末端 LayerNorm eps 为 1e-12、内部为 1e-6，backbone LR 2e-4、linear/age LR 1e-3。

本目录保存服务器导回的实际部署清单和 29 项 CPU 检查回执；检查已由执行 A 在指定 Pod 完成，仓库迁入未重复模型套件。source digest：

`57ef83209692eb79e682e04e2b3fe7a060225856f673505ab67396080216d51e`

源包来自：

`/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/technical-prep/resume9010-A-r92-prep-20261028-attempt3/package`

实际正式根：

`/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/rounds/R92-TINY1K-9010`

当前已完成唯一绑定/launch 和真实四 rank 初始门，仍沿原 token 串行运行。此记录不表示完整训练完成，不填写尚未产出的 dev/官方指标。最终选定模型和官方结果由真实最终交付接续。

`runtime/` 由上面的通过包原样迁入。R91 源码在提交 `c04e8dddd3d7a9195c317e584094910fcde66178`；不为 R91/R92 分别维护两份完整源码目录。
