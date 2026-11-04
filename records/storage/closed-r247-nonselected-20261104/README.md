# R247 已闭轮非 selected 清理

R247 完整 actualsafe 并归档到已回读 Git commit `426c9306fd0c90209a401f3b02a21a5eca4b1788`；R250/A 启动后按预告[逐文件计划](PLAN.json) `6281b63b17e95d38dce34babb455ed11390c9aea286dd2fc62c8460972b4b6b2` 和[清单](INVENTORY.json) `bd7c80bcfeb176451bb8d43ba35f4c7f908f21fe1cf23891e6a03bad25e1f81a` 清理 7 个非 selected，预计分配量 `20,413,728,768 B`。R247 selected、其余轮次和 KEEP 均保护。

[删除回执](DELETE-RECEIPT.json) `327602eb4d860a210e517fed36d75d7733ac623aa1e046f98407d24de07ecc80`：7/7、0 跳过/0 错误，free 点测 `57,730,400,256→78,144,077,824 B`，保护对象 stat 和小文件 SHA 前后一致；未对训练发信号或取得训练锁。共享盘变化可能包含并行活动，大 checkpoint 本体未重算哈希。
