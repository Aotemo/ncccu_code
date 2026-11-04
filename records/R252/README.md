# R252：R201 原配方新 seed 的 fresh Small

Pod B 在 R249 完整 actualsafe 后接续，源码 `d5ac9940f5bcb50c5d20b5ec3796a6ed8f460f19`，GitHub 3215 文件[全树核验](R252-TREE.json)回执 `21e91ac319712c35f5ea390b3b0d96c8bdebd6ca062b98bd8546929059b93d0b`。[正式准备](R252-PREPARATION.json)通过 18/18，CPU `e004bfef6aeeb648ae541bf40f06b49b8060c5ad9d0e7e3558c6307ee9a98037`，部署 `b39931c793434366cfae62b345ff7e9252fd382d93c5ab6cd897db5090aec2e6`，SITE `0815729c15ca1a6ab5345dcb888537b91517d1ce55782ace126bd0c82c8326c1`。[现场容量](R252-WINDOW.json)包含 R251 剩余预算，可用 `121046564864 B` > 需求 `44439643521 B`；唯一 activation audit `895e155c85df571d09fc25d149df957c98e0a43735535a60bddd50a7ec0fbef3`，launch exit 0。

[首成功](R252-START.json)回执 `2b97158d2f93a8eb3a37d530c6dc57a170d60b4717ece569d40025691aa35397`：八 rank、345 参数张量、5 空 Adam 组、资源绑定、G64/P8/448 和新 seed 初态 `5b5ae7aa3de118f98aee9da89e6d289f262eb49db0e25e1000ae6133c529fc1d` 均通过；首条自然记录 25 成功 loss `2.024000883102417`，观察时 50 成功、0 skip。训练仍在运行，尚无完整结果；官方留给最终冻结方案。

R249 完成事件到 R252 owner spawn `2271.506 s`、actualsafe 观察到 spawn `1977.898 s`，见[接续回执](R252-HANDOFF.json) `4ce28a26d4db470b132d0ef50383a2935664a209593506fa342f5e245a70d30f`。未连续采样 GPU idle，上述为事件间隔；等待包括新增体积约束后重新准备 fresh 轮次、NFS/传输和正式验证。
