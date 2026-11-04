# R246：R172全图与四局部配方的独立fresh

唯一owner在Pod B接续R244 safe后，按固定R172科学值从通用Small公开源fresh训练。实际[正式CPU与精确部署](R246-PREPARATION.json)18/18，CPU SHA `3c74fc535f85f12d390605acfe4157aa40f0da0c70ebf1759de6d52ccc2b859c`、部署 `25bb358dd1c7b1e3e9205d5d01d6c51ea590a44ba63597a6b8f19249a6849f32`，Git `66f4b808`；[唯一activation](R246-ACTIVATION.json) audit `cfc7e2380d5402dbe1dce435bb62aa137bcd5432cf5fe04a5b7436c22c928c82`。

[真实八rank及首成功](R246-START.json)证明fresh tensor `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`、345参数/5空Adam组、G64/P8/448对及训练资源绑定；首25 loss `0.7223862409591675`，观测时50成功/0 skip，回执SHA `e985fcb1d77bfae70df017ac6200b511e4849519f32d171c6ba4219d9f6ba584`。该首25 loss与原R172记录相同，不据此断言后续轨迹或完整结果。官方留给最终冻结方案，当前仅运行快照。

交接未满足5分钟：R244末条训练事件至R246 owner spawn约1533秒；未连续采样设备利用率。原因包括SSH旧复用通道失联、分片/共享文件系统RPC等待及全树校验发现中断传输截断的历史记录，修复后重新验全树；原失败记录保留。初次和第二次启动观察为unknown，直至自然首成功才确认可跑。
