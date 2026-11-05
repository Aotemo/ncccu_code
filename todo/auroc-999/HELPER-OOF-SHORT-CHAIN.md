# OOF 单连接短链接续：纯假调用测试

该项由 08ac 实现并提供待测 wrapper 快照；助手只准备独立合成测试与本说明，不修改主源码、不派 GPU、不调用真实 SSH 或 kubectl。c607 已取消 active-peer 新门，该测试不要求新增 active-peer 分支。

短链在同一次受限 SSH 内实时核 Pod 身份、以 CPU guard 生成该 Pod 的新 safe/window，再在身份仍匹配且全部 CPU 步骤成功后，经明确的设备环境执行一次既有 OOF claim。真实资产及设备准入仍由 08ac 负责，测试只验证传输控制流，不能替代真实 GPU 回执。

拟覆盖：初次和二次 UID/Ready 不符、CPU guard 非零或无有效完成回执、safe/window 生成或绑定失败时均不派发；成功路径的显式 Pod UID 和三组设备可见性为 `0,1,2,3,4,5,6,7`；窗口输入和实际派发共用一次身份；单次派发成功、非零失败或连接结果未知时不得自动重试或另造 claim。测试使用临时目录与 PATH 内的假 `ssh` / `kubectl` / CPU / GPU 入口，并核实际调用次序和次数。

2026-11-05 在 B Pod 的 12–23 mask、nice 19、隐藏设备 guard 下，使用 08ac 固定的两份脚本快照运行 `tests/test_f05_oof_short_chain.py`：**6/6** 通过，CPU guard exit 0，回执 SHA `8e9f424ee7120e117b52a9265e032a3099461e77b202d02a25cf08d801a230e3`。实际外包装 SHA `c7182ffb71402ec70f13760a498ab38d0a8c3ccf0d27faac03fe5f0ed4ce1989`、Pod stage SHA `9083094d1c1207e183c4109bb283b7ee9c2a66f60f7fd30b7b726d1b3d429c85`、测试 SHA `accd0cd289387b0b8e3ba76f4a3c63d25696d2cf403f51871f0577bee78a1482`。

随后补强重入测试：更换 live、safe 和 window 路径，但保留已经使用的 claim；确认 Pod stage 在再次运行 CPU guard 之前拒绝，且失败或模拟 unknown 均没有第二次派发。相同两份 wrapper 身份下复测 **6/6**，最终测试 SHA `bceb9220e09c34d8c36adb1d52da518b6ef2647a72ec3022a475c223328a93b6`，CPU guard 回执 SHA `81da53a05ea8fadecb23e297693449b648b5df58797b1cc583a21cfdfb8df1f6`；前次回执仍保留。

测试在独立临时目录运行这两段 shell，仅把外包装的私有和工作目录前缀替换为 fixture 路径；通过假 `ssh`、`kubectl` 和 Python 分别验证上述失败分支、正确顺序及 safe/window 实际 SHA 绑定。假程序还核显式 GPU 环境和 claim 路径，验证派发失败或模拟 unknown 后同一请求不再发生第二次派发。测试没有真实 SSH、kubectl、模型或 GPU 调用，不能证明实际 Pod 观察、远端设备、真正 claim 或后续 owner 生命周期。主线 OOF 曾因 peer 观察超过 120 秒在 GPU 前失败，该原失败回执保留。
