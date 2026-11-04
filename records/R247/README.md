# R247：R182自然图像DINOv3 Large配方的独立fresh

[完整及安全终态](R247-COMPLETION-08ac.json)：6400 成功/6400 尝试、0 skip，8 次评估、连续 4 次未提升正常早停；唯一 selected EMA c4@3200 内部 AUROC `0.9868174369138257`、AP `0.8119164253097094`、error_twice `21896`，与历史 R182 数值一致。done completed、原 owner 和 rank 退出且资源安全；官方未读。回执 SHA `9f44acb50a2305ba13ea434a30c6a5c0b04150956676888217135462761bbd91`。[冻结模型严格构造加载](R247-STRICT-LOAD.json)通过，回执 `7c753fca5b0095e7881ae912b1dc08e79657326f9d5c71cf3d16f158af10aa31`；其单模 AUROC 低于固定 F03 的 0.988 准入门槛，重算不纳入。

唯一owner在Pod A接续R245 safe后，按固定R182科学值从通用LVD-1689M DINOv3公开源fresh训练。实际[正式CPU与精确部署](R247-PREPARATION.json)18/18，CPU SHA `d31824d2e530a73108309483d23b2ebc1fd3b4093550b2a205a01e2f23ab8fea`、部署 `6a519fef8197025123b7ec94d2059e72a6e6f2bb68c8ab0ec3da354cdf1d52de`，Git `66f4b808`；[唯一activation](R247-ACTIVATION.json) audit `45aa652cd63a982cef2ab4c106ca39ed003199fc63249f04771c923a6d7ecee9`。

[首次自然观察](R247-START.json)证明八rank fresh tensor `864775ca1be0d6be0c6b42ae6b06b2ded0b45a825762a85a4617afc8eee54b33`、345参数/5空Adam组、G64/P8/448对及训练资源绑定；首25 loss `0.6626197695732117`，25成功/0 skip，回执SHA `dc4bdef98f46d338317077172d8f5706a77096656000d53058b73f28e9a693f1`。该首25 loss与原R182记录相同，不据此断言后续轨迹或完整结果。官方留给最终冻结方案，当前仅运行快照。

本轮交接超过5分钟：共享盘RPC、源码传输及全树校验修复造成延迟。activation调用传输超时后只读对账，确认唯一activation实际完成且未dispatch；重新核Pod身份后对既有clock首次direct launch，没有重置token或重复activation。早期观察为unknown/等待初始化屏障，直至自然首成功才确认可跑。
