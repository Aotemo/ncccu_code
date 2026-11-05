# R263R1：保留失败现场后的同科学 fresh 接续

2026-11-05 经 c607 决定，R263/A 的首次后台 activation 和 launch 返回 0，但控制器在 owner 创建前以 live Pod 证明超过 120 秒窗口失败；原 round/clock/token/claim 及失败回执保留，禁止重派或重置。R263R1 只使用全新的 round、grant、clock、token 和一次性 claim，科学配置、R134R1 配方、seed `20261119`、公开 fresh 初态及内部 selector 与 R263 完全相同。round ID 为 `r263-r1-small-mixstyle-seed20261119-pairlog-24k-p4-9010-20261104`，沿用原候选 profile、权重来源和科学文件；新 round 仅属于最终冻结融合成员，不触发单轮官方评估。

优先 A；须重新核证旧失败无进程、锁、设备用户及 8 卡空闲，实测失败日志/dispatch/admission SHA，不能以 launch exit 0 代替首成功。后续按原链新建正式 grant、完成完整正式 CPU 检查和 Git 精确部署；容量及 B 当前 peer 的合并预算现场重核。复用只限字节及身份实际相同的科学证据，不复用旧激活状态。

## 一次性后台链修复

现有预检快照在 activation 前取得，activation 可能耗时超过 90 秒，而控制器再次使用同一快照并要求年龄不超过 120 秒。仅对新部署的接续入口修改：activation 成功后，取得并验证当前 Pod UID、Ready、namespace/name/container 的新 Kubernetes 快照，再以该快照实际接收时刻进入原 launch 和 controller `verify_live`；新快照与回执独立保存，旧快照不覆盖。若 API 不通、证明不匹配或等待超过明确限时，不 launch，保留一次性 claim 和失败/unknown 回执，不重复派发、不松动 120 秒门。通过隔离测试模拟 activation 延迟 90 秒以上、失效快照和 API 失败；随后只用于 R263R1 及将来新包，不热改现役 R264。

受影响最小文件：`tools/activation/async_handoff.sh`、`tools/activation/validate_launch_pod.py`、`tools/environment/refresh_launch_pod.sh`、`tests/test_async_handoff.py`、`tests/test_launch_snapshot.py`、`tests/test_refresh_launch_pod.py`、`runtime/bundle/retrain/experiment91.py` 的新 round 官方隔离登记和 `tests/test_r263r1_identity.py`。为遵守本地源码树要求，还需对 `tools/activation/prepare.py`、`activate.py`、`freeze_deployment.py`、`local_snapshot.py` 做最小来源绑定：将本地源码中 SCIENCE 的同字节副本绑定到 work root，核全树的本地 UID 路径与回执，再经原链部署；相关窄测试先行。外层只做有限次短时只读/幂等观察或投递新快照；非幂等 claim/activation/launch 超时先对账，不重试。
