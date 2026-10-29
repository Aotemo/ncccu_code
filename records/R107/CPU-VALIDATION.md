# R107 必要CPU验收

本次仅验证新增pairwise目标、真实两进程Gloo/DDP梯度与受影响配置。原生产模型不在这些数学fixture中构造；新Git正式原生CPU包与真实8rank训练另按原入口执行。

首轮5项中4项通过（独立数值/解析梯度、极值有限性、配置绑定及loss身份roundtrip），Gloo项超过父75秒界限，原 [CPU-CHECKS-FAILED.json](CPU-CHECKS-FAILED.json) 保持false。仅补fixture阶段日志后仍超时，[GLOO-CHECKS-2.json](GLOO-CHECKS-2.json) 保持false，完整阶段原文见 [GLOO-FAILED-2.txt](GLOO-FAILED-2.txt)：两rank的Torch导入约2.31秒，FileStore初始化至约64.20秒；之后可微gather、DDP backward、完整64样本参考梯度断言、报告和destroy均完成，但进程未在总界内退出。两份报告为8P/56N、448对，loss0.4406687915325165、权重梯度约[-0.63797355,0.01856655,-0.01000074]、bias梯度0。

唯一修正只在测试fixture：FileStore rendezvous换为同Pod loopback TCP、仅选一次端口，不重试；生产loss/gather、数学断言、梯度容差和父75秒/group40秒界限不变。仅复验该Gloo方法，实际1/1通过、6.416秒，两个子进程正常退出；见 [GLOO-CHECKS-3.json](GLOO-CHECKS-3.json) 和 [CPU-GLOO-GUARD-3.json](CPU-GLOO-GUARD-3.json)。这支持初始化路径延迟解释，不将底层共享FS缓存机制当作已证。前4项不重跑；生产loss源SHA始终为 `ac6c9d6bb253ed018ea09cb7cc36939ba61377825d53ff2a946f92d7fae130a3`。

CPU8 A份额0..11/nice19/隐藏GPU，Gloo子进程各Torch intra/inter1及OMP/MKL/OPENBLAS1，总树仍受A12预算约束。没有生产模型/图像/官方前向，Gloo通过不冒称NCCL已通过；实际8rank pairwise训练首成功更新将沿原流程验证。原件根 `/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/r107-update-a`。

原首轮JSON Pod SHA `7e61e438236710179b846eebd0a95861d2b66858d0811d6894a10c9b24a1978f`；诊断第二轮 `47a531d0f7444053f8ffaeb40f7da36bc2f6371a855408e70bdd47a4734de848`；TCP复验 `c9edbf6d4e232367856c90690b55429512d8bfe4b090c68eb5643423ea8b8890`。原失败不覆盖、不改为成功。本机仅文本与原始字节传输，不解析或重哈希。
