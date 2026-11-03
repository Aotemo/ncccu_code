# 后续准备池续验

08:57实时UID51fe/Ready门、Python3.12.7、affinity/cpuset与quota实核后生成[新allocation](CPU-ALLOCATION.json)，SHA `d8370ed3a789939559098f3ad2f2b2dbec80f3455aa69e75a69b1901db432b66`，有效至1793717868.5687554（14:57:48.569 UTC）。版本化policy725b2b90保持：有效120核，准备24核初始平分root0–11/12核与peer保留12–23/12核；现役训练24–119/96核互斥。旧ec5与R209 GPU allocation、clock/token/grant均不改。

[原生guard probe](CPU-PROBE.json) `_w275p_z` / `c95e84bda5aaff2d31747dbbbea1b0ada1044d7264dc528f02ccbea49d917916` exit0，实际affinity0–11/intra12/interop1、依赖版本相符、GPU_initialized=false、树退出/无affinity违规。新准备执行才消费d837；旧回执保留其ec5身份。

[续验摘要](RENEWAL-SUMMARY.json)记录真实R209 owner和8rank PID/start/mask仍有效，当前SITE SHA37c08d4e与原冻结pin相同。字段`active_GPU_site_sha256_before_after`保存的是本次当前SHA对既有冻结pin的核对值，不当作两次独立新哈希采样。续验工具沿既有分配函数：先读实时完整拓扑，再把这个小型规划进程收窄到原root mask，生成新文件并核两池；不是训练或GPU准入执行。Paseo08:56实观仅32e3 running、peer idle/activeTurn null（更新08:42、last user08:36），未唤醒/派生或发送消息。root旧prep lease稳定跨新旧绑定，后续任务继续原串行guard。
