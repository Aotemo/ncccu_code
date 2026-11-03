# 同一 Pod 的 CPU 准备池续验

14:34:52 UTC实际新建[CPU-ALLOCATION.json](CPU-ALLOCATION.json)，SHA `7c7af22ca15c3b20cd4e88016dd11e73f842530d4330dba485599b2dcb791e7e`，有效至20:34:52 UTC。原d837 allocation未修改；实核仍有效120核、准备24核，两既有执行者仍各12核/互斥0–11与12–23，训练池24–119/96核保持。当前owner及8rank出生身份和mask已读核，依赖版本与原分配一致，无模型/GPU计算或训练信号。

[原始续验摘要](RENEWAL-SUMMARY.json) SHAf2d3be33，CPU策略沿冻结724参考；未来GPU包0b1仅准入TTL不同，80/20份额相同。14:34只读Paseo实核root32e3 running、peer a4 idle/activeTurn null；未恢复、派生、提示或发消息。原生分配函数先实测未收窄容器拓扑，再将该小型规划进程收窄至既有root mask，沿已验证入口生成新准备分配。

[新池首次受控观测](CPU-FIRST-GUARD.json)ld3tzplt/ef816d6a exit0；[核验](VERIFICATION.json) SHAb03e7ef5及[归档guard](CPU-ARCHIVE-GUARD.json)v06zb4i5/6411ed4d exit0。旧d837与当前R217 SITE4c4 SHA保持，task tree退出/无信号/affinity违规0/线程预算合规。只更新后续CPU通道和未执行调用的外层分配引用，原GPU包/grant/clock/token/现场不改；当前只读环境为私有`r217-cpu1435-pod-context.env`，旧env保留。原件`exp-v3/technical-prep/cpu-renewal-20261103-1435-32e3`。
