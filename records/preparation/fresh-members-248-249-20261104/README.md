# R248/R249 接续准备

已核两 Pod 公开来源及原科学参数不变；[Pod A](BINDING-A.json) SHA `3e095ded3d0d9f1133c3a3b9768fbce7f37c47905aee7f524619b0c2a9842c3a`，CPU `90b19d0b833ffc0d376ae372ec61df9ddfd5f8bb6714d33292bb831187703eeb`；[Pod B](BINDING-B.json) SHA `7bdc6cef3b0fb77604374dd254326740df81b6725589cd97137ef6c0fc343749`，CPU `4ee859b6d96de81fb1a9da1e854d760c7a5618d78ff60c0e08f0f1f68de63367`。R248 公开 Large 原始/转换/来源文件每 Pod 重算 SHA 并复用可追溯严格结构证明；R249 Small 沿当前 Pod 的原公开资产绑定。各科学配置仅变资产回执、最终官方评估门和比较说明。

修改的官方隔离及历史 seed 注册窄检查在 Pod B 的独立源码草稿通过，检查回执 `6b72b7556e377ef0d3bc5598f6cf13876b6991abcda782ba36f54b897248620b`，CPU `f69a7247e0677ab251f0b15e47110c17e6c05abc283352c4c49233761875dfbf`；不等于正式 18/18 或全树部署。当前 R247 Large 资源时点采样 `3c46c5125c1f1b689e8d2c6357510af8dfaec3e2cb214380527993d967bb7d87`，CPU `116406694cdb99f7a169709de3802e32543c63a9d1abfdb8657824784f9e3e28`：最高已用 20,783,206,400 B/卡，约 0.95296 秒/更新（含一次内部 dev），并非峰值保证。共享 free 66,047,705,088 B，低于两新轮 32+12 GiB 和 20 GiB 余量之和；须按授权核真实清单并清理或分时启动，启动时重核。

Git `631ce2012e18c4ef6bee5f57e3fc6a6746195641` 已 push/API 回读，tree `dd9135aa88bbb81bd297209bc668f5b02389ff3d`；从已核验 `66f4b808` 完整树创建独立副本，传入44个变更文件的增量 SHA `1184ce674cbf51f16fb992786df163164795f0472b45bea520aab283e0af7fa7`，每个 Pod 及变体独立通过完整 tree 精确部署。四个准备回执均为18/18，且未建立 clock、未 activation：

| 变体 | 正式 CPU SHA | 部署 SHA | SITE SHA |
|---|---|---|---|
| [R248/A](R248-A-PREPARATION.json) | `7ee5133b2cbbeb86d627c037a5d7820c8da32658b799bd816ec29b8c95600b2f` | `8557e85afab276862ce3233e0ff239ceae4ea4bde46577f19fa780b3f247dea2` | `d943778d4c0e29c7f63b54a47a2faf593f4fdfcd8dfb3b4b5fa3406ffe49d92e` |
| [R248/B](R248-B-PREPARATION.json) | `82f61435d4a78c8c85aa2d7aa1b031698e74fb8482e71c33c3da40f42a8dd748` | `87dbf7163a2a955d5db24c2e17102ee2a2e1dc580976d87bd8373bac9beef3fa` | `5cfc9a966f6704bb9d339cd691caffee593471b795fef2b6772539996fdf7435` |
| [R249/A](R249-A-PREPARATION.json) | `0650746762bf54ab8e097c99703ebbbf3848849560e4d5c99229aa05b0abd3c8` | `b12f422dd511022594f198814232af27a674069a4325ce1e42b86b4bfb241601` | `d7dc14ba3983de384b5bed75a1f1078dc68bfbac5361d04f3d10ab011bce1452` |
| [R249/B](R249-B-PREPARATION.json) | `79351253c3d4400b1c2447558070767ddba307b6fc45207880fdab676bfd22b9` | `45fb27c036fa0892ddd13d7e0a54e69f36a9d8b23b4c7689cf06e826563ee549` | `194544d5e96418efd880481254df2f9f43377d2c08bdcec8fcc5788969aeef82` |

旧闭轮至 R236R2 非 selected 复查 [0项](OLDER-INVENTORY.json)，SHA `25a39157115472b3232f74e14a05c3510a8d1b27ddc47cd6b22e91902526dbe2`。统筹授权后，[R244/R245 非 selected 清理](../../storage/closed-r244-r245-nonselected-20261104/README.md)完成，free 点测 71,388,102,656 B；两现役仍消耗空间，须等对应安全终态、重核并行和活跃轮剩余预算。

14:14 [容量投影](CAPACITY-PROJECTION-1413.json) `0c1106597a170395b253c681f1e6c25ca0bf0322938c1cf509bc16ab45261994`：free 68,454,187,008 B，现役 R246/R247 的剩余上限分别约 8.49/20.19 GB。统筹批准 R246 先终态接 R249/B，R247 先终态接 R248/A；第二个安全 Pod 接余下成员，启动时重新核容量。一次广域只读 `du` 超时并留下孤儿；第一次身份核验拒绝发信号，CPU 失败回执 `d55ed77996c48b4407e344860d071e6148e1a633f2f363e0c8ecd90273e15336`，复核 PID 已自行变为 cmdline 为空的 `du` 僵尸，不再执行、未发信号，[状态回执](DU-STATE.json) `1d1fe96350491a1f3cde779de6b18d243d26e43b4e672512912fb3adc7b56d42`。此后不再扫描整个 technical-prep。

Pod B 在隐藏设备的准备池禁用 credential helper 和提示后，匿名 GitHub `ls-remote` 30秒超时、exit 124；回执 `235282a3b22819fb7a573fd53b4b088ac41df84c574b54b3e96faffc9256a901`。代理信息仅保留脱敏结果，未复制凭据。后续使用已校验旧树与增量传输，并保持完整 tree 验证。

[容量入口隔离测试](CAPACITY-WINDOW-TEST.json) `d59257083522a48096ae10949ab775884162f7a0e0f42e696a5001935e5238d5`：Small 或 Large 单轮接续时计入另一活跃轮的剩余上限，等于门槛被拒，另一新轮存在时保留其剩余预算；均为模拟数据、不代表生产容量或实际 activation，临时 fixture 已清除。

现用入口在此基础上复核已有对侧新轮的 SITE/RUNTIME、Pod UID 和 commit；[五项隔离测试](CAPACITY-WINDOW-TEST-v3.json) `2b26c7371302169214698288b3d46c370ef2fa106b15c4eb65dbbadff10aa967` 包含错误对侧 Pod 身份拒绝，仍不替代实际启动前检查。
