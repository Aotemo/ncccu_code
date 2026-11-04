# R249：R197 固定 gamma 配方 fresh 重训

Pod B 在 R246 完整 actualsafe 后接续，使用已推送源码 `631ce2012e18c4ef6bee5f57e3fc6a6746195641`；[正式准备](R249-PREPARATION.json)通过 18/18，CPU SHA `79351253c3d4400b1c2447558070767ddba307b6fc45207880fdab676bfd22b9`、部署 `45fb27c036fa0892ddd13d7e0a54e69f36a9d8b23b4c7689cf06e826563ee549`、SITE `194544d5e96418efd880481254df2f9f43377d2c08bdcec8fcc5788969aeef82`。[现场容量](R249-WINDOW.json)按 R247 剩余预算通过 single-only，唯一 activation audit `251200df145f3bb980a586f5f918848a92d590ad841002d5cb94c73761543f14`，launch 命令 exit 0。

[真实首成功](R249-START.json)回执 SHA `525e52f212f6b865639ade6860b87b772b6b6c06124f1729378f6ff9575dd7c6`：八 rank 初态、345 参数/5 空 Adam 组、训练资源、G64/P8/448 对均通过，首自然记录 25 成功 loss `0.9216874241828918`，观察时 125 成功/0 skip。官方留给最终冻结方案。

[完整安全终态](R249-COMPLETION-08ac.json)回执 `23079339323370ca412aba4d4f9d61be45698f3c99f5ce03d7b4100f000b4942`：10400 次成功、0 skip，c13 连续 4 次无提升结束；唯一 selected EMA 为 c9@7200，内部 dev AUROC `0.9884706757809537`、AP `0.8262997760790266`，张量 SHA `6251937d3d778ed4be5d0cc11fc505fc4c2cc897d0ae18ff31ddfb6c8dce8a73`。这是内部结果，selected 模型严格加载及 F04 实际导出大小仍待复核。

R246 完成事件至 R249 owner spawn 282.696 秒，actualsafe 观察至 spawn 176.213 秒；未连续采样 GPU idle，故两者只是时间点间隔。回执 `e5a9989fb4683f342e277dd5476ecc8c04e05d7449c804d522598f16c769b170`。
