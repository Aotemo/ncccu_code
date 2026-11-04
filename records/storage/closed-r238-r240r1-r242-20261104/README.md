# 三个闭轮非选中 checkpoint 清理

2026-11-04按统筹授权，在R244/R245仍运行时，仅按[清单摘要](PLAN-SUMMARY.json)删除已完成且历史安全终态的R238/R240R1/R242的22个非selected checkpoint。PLAN SHA `98d3bef55f82e47858549652432bd5299ab4e8821884ffbb38a3ceb2f24fc7a7`，inventory SHA `46d34c4eed53b698cb93f66f8e04dc7a8acffeed71392d4469c103596b98aaca`，规划CPU `ce0accce968d7ea4c9aa0bd00d443b889b4f3ba1c562333e73df3a7fca413cd1`。

[删除回执](DELETE-RECEIPT.json) SHA `2ef88672a4d6eb92a5509abfa3b4f0cf1376e76716d4ca54986dcb1a37015706`，CPU `145ba3e9e30a7b9e41fa977f5a6eb051fcb4d78bb7b1d1b49d7e727395d15b29`：22项、逻辑16,239,260,911 bytes、删除前allocated16,239,266,816 bytes，0 skip/错误。三个selected和sidecar、三KEEP及其他round与technical-prep受保护，保护stat/小SHA前后一致；未发训练信号、不取训练锁、不重查历史安全终态。free点测24,526,192,640→40,765,489,152 bytes，受并发活动影响，不等于后端真实回收量或并行容量准入。
