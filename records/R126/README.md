# R126：按优化器角色分别裁剪梯度

R126 在 R107 的 Small、seed20261029 和 pairwise logistic 合同上，仅把原全局梯度范数上限 1 改为 backbone、head、age 三角色各自的 L2 上限 1。新策略在梯度修改前做 WORLD finite 共识。新项窄验 3/3、正式 CPU 19/19 通过；实际公开 fresh 初态张量 SHA-256 为 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，空 Adam。准备脚本在正式验收和部署已成功后因旧的 backbone 数量断言退出 1，原失败保留；后续只用已有正式回执生成 exit0 摘要，没有重跑验收。新 activation 与 launch 各一次。

8 rank 的首个成功更新均记录 WORLD finite。backbone 的 342 个参数张量、49,454,688 个元素，裁前范数 `13.748153686523438`、系数 `0.07273703813552856`；head 的 2 个张量、769 个元素，裁前范数 `4.393540382385254`、系数 `0.22760683298110962`；age 的 1 个张量、2 个元素，裁前范数 `0.21580223739147186`、系数 `1.0`。因此首步 backbone 与 head 实际触发裁剪，age 未触发。这仅是首步的直接证据，不外推到训练全程的裁剪频率。

训练成功/尝试更新 6400/6400、跳过 0，内部评估 8 轮，连续 4 轮无提升而早停。唯一选中 EMA cycle 4、update 3200，内部 AUROC `0.9867283329981902`、AP `0.8252218982626471`、error_twice `22044`；张量 SHA-256 `d4294587db120b143898e07aaa6916b0bda097d5e60e28945a9a7b8e053b7351`，checkpoint blob SHA-256 `0e25aeff2e17470ea0340de9568ba4c99c8161deeb030cded1341e3c21ee0dbc`。AUROC 未严格超过预冻 R107 `0.9884405731067525` 或 R120 单次最佳 `0.9894773092062407`，固定策略关闭，不做 seed20261027 或裁剪阈值补考。唯一内部结果低于 0.996，官方 5849 按政策 `not_run_by_policy / threshold_not_met`，没有官方指标。

2026-10-31T17:34:48.499230+08:00 的只读快照显示训练与整轮 completed、TRAIN exit0、原 owner 为 Z 态、8 rank 已退出、锁和设备使用者为空、8 卡空闲，safe=true；Z 不表示已回收。[R126-RESULT.json](R126-RESULT.json) 是有效 Pod 准备池内从现有原件生成的去令牌摘要，Pod SHA-256 `60a733c241914d616fcd389aa2892304c1a20524db2b06e47750c11f99153127`。原 freeze、TRAIN、ROUND、done、progress、CPU 回执的 SHA-256 分列于 `original_sha256`；原 checkpoint、clock、运行令牌及旧失败原件仍在 exp-v3，没有纳入 Git 或改写。
