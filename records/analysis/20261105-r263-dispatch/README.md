# R263/A：控制器在 owner 前拒绝过期现场证明

2026-11-05 R263/A 的一次性后台 activation 与 launch 调用均返回 0；launch 的 0 只表示已 spawn。控制器在创建 owner、rank 和 `done.json` 之前失败，日志报告 `live Pod UID/Ready/hostname/interpreter/freshness differs`。Pod 快照接收时间 `1793856541`，dispatch `1793856665.076307`，控制器错误日志末次写入 `1793856667`，对应年龄 `124.076307` 秒和 `126` 秒；原 live 校验上限为 120 秒。该快照超过门限是已观察到的充分拒绝原因，原有规则未放宽。

控制器原日志 SHA `d90defc0d605731dbf831b4f9becd232b27c92c3e0e06af073b18ff898d3834a`，dispatch SHA `f2e2ec73258c6241163ad071fe66fcd66462831d7c56b065aaa0f88f3c9aff08`，admission SHA `4da4250f760734d31de6c60fc43691ce3d5dd0ffb402360d5ce126d99f58f2c3`。只读失败观察 SHA `11b9858c24a7f866e7411a3a10f3fd5e009b5791ee8c57a32c84d836d0d2b4d7`、CPU 回执 `233bfddd85b56d5af430fe132811ce3b886035334302eed5e6adbf02cf5683c9`：PID 104430 为 Z，未见 owner/rank，锁无持有者、设备用户为 0 且 denied 0，三次独立快照八卡 busy 都是 0。三次快照不是连续空闲时长测量。

本轮无成功更新或 selected；未发送信号、未读取官方数据。原失败 round、clock、claim 与回执保留。后续 R263R1 使用新身份，并仅在完成事前登记的 live 快照修复、隔离测试、正式 CPU/部署和安全容量检查后启动。
