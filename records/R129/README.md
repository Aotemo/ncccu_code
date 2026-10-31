# R129：ConvNeXt Small 公开 ImageNet-22K-only 初始化

R129 直接沿 R107 的 seed20261029、固定 9:1、单图 512、pairwise logistic 合同重新初始化，只把公开 ConvNeXt Small 权重改为 `timm/convnext_small.fb_in22k` 的固定 revision `4cde7ae59c8f7fcff3f9547de274819f54863dfd`。实际 K=21841 分类头先完整严格加载，随后只丢弃公开分类层并新建 binary head；原始与转换权重 SHA、严格来源回执身份见 [R129-RESULT.json](R129-RESULT.json)。同源祖先谱系未获证明，本结果不能解释为精确撤销同一个 checkpoint 的 1K 微调阶段。

新来源、RNG 和绑定三项窄验通过 3/3；原生正式 CPU 验收 18/18。新鲜初态 345 个张量，whole tensor SHA-256 `515f4e6340cc2c8c07934dbd971977c810f1aa89fee181da6af2d0787464540a`，Adam 状态为空，8 rank 首批真实读取了原 rotation/photometric 缺省、每 rank 8 主样本、四元组和 age 特征。首次外层包装器错传 formal-cpu 路径，exit 1，未进入真实 launcher；当时 pod-admission、dispatch、controller-start、controller.log、ACTUAL_FRESH_INITIAL 五项启动/派发证据均不存在，错误原件保留。根明确授权后，在同一已 activation 的 round 和原 clock 上唯一一次正确调用真实 launcher 并 exit 0，没有二次 activation 或补还等待时间。首个成功观察为 150/150，最终结果另记为 6400/6400，不能把前者当终态。

完整训练成功/尝试更新 **6400/6400**，跳过 0，内部评估 8 轮，连续 4 轮无提升而早停。唯一选中 EMA cycle 4、update 3200：内部 AUROC `0.9884188991813276`、AP `0.8233186216195478`、`error_twice=19236`。selected tensor SHA-256 `173a9c1cf719310d502f44c4a9762c55df9796b3ca69bd0cd2c89842e3299671`，checkpoint blob SHA-256 `c816cf8776a2ee1bd75ebcd7c49809c5e28720fd321c67781ff34db16c8a7f52`，两种摘要不混。它严格低于预冻 R107 `0.9884405731067525`，也低于 R120 单次最佳 `0.9894773092062407`；本固定 22K-only 合同结束，不启动 seed20261027。这不证明 22K-only 来源普遍无效，也不支持公开权重之间同祖先的因果归因。

选点未达内部 AUROC 0.996，官方 5849 按政策 `not_run_by_policy / threshold_not_met`，无官方指标。北京时间 2026-11-01T00:36:08.608798+08:00 的同一次终态观察记录 TRAIN exit0、ROUND/done completed、原 owner 为 Z 态、8 rank 已退出、锁及设备使用者为空、8 卡空闲，safe=true；Z 不表示进程已回收。总体 raw AUROC 0.999 目标尚未达到。

[R129-RESULT.json](R129-RESULT.json) 是有效 Pod 准备池从既有小回执生成的去令牌摘要。原 freeze、TRAIN、ROUND、done、终态观察和 CPU、首成功与包装器缺席证据的 SHA-256 分列于 `original_sha256`。训练冻结 Git `04cdb75c170476a7fc80febeb4b4940b74cfa156` 与归档文件不同；原 checkpoint、clock、运行令牌和完整回执留在 exp-v3，未纳入 Git 或改写。
