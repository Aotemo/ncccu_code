# R199 训练期间准备池续期

03:21实际Ready/UID、解释器、依赖、affinity/cpuset/cgroup quota及owner/八rank出生身份复核后，沿既有renew helper创建[新准备allocation](CPU-ALLOCATION.json)，SHA `ec5cb66f3548814547e0cfb0be891f75fd2206cdbc27a20d16cda3a780c91396`，有效至11-03 09:21:03.296 UTC。实际effective CPU仍120，集中RESOURCE-POLICY725b2b90的准备1/5仍24核，双方各12核；32e3为0–11，peer12–23只保留、未唤醒，与活跃R199训练24–119/96核互斥。03:19实时Paseo核peer idle/无activeTurn，没有派生Agent或新资源。

原53ab allocation、R199 SITEa1aa7710/GPU allocation/clock/token/进程不热改；只给后续未启动CPU任务使用新ec5。真实PID/start_ticks及affinity检查通过，Python3.12.7/Torch2.10.0+DTK/TV0.25/timm1.0.27/numpy1.26.4/Pillow12.3均实读。该环境测量使用原nice19/隐藏GPU bootstrap，不运行模型/图像/训练或新平台。

[新租约首用](CPU-FIRST-USE.json)xwdmq5l5 / `0b9ee26b5413fb5a5d41aedaeaa79837c04a350b56e3893ea4eb9656adc65079` exit0，观察R1994100 success=attempt/0skip/eval5/bad2、八rank活跃。后续CPU前缀已改用ec5，私有`r199-renewed-pod-context.env`按当前实核字段绑定；旧env和回执保留。此记录不把CPU首用当GPU性能或科学收益证明。
