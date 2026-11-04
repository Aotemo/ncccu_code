# 2026-11-04 存储盘点：进行中

NFS 位于 `exp-v3`：首批清理前后两次点测分别为可用 `68,992,106,496` 和 `104,341,700,608 B`，总量 `482,110,078,976 B`。训练持续写盘，点测不保证未来容量。Pod A/B 根目录及 `/tmp` 分别在本地文件系统，早先回执 [A](POD-A-LOCAL.json) / [B](POD-B-LOCAL.json) 可用 `1,394,906,005,504` / `1,400,216,129,536 B`。

在逐文件 [计划](ARCHIVED-SAFE-PLAN-SUMMARY.json) SHA `05e045705fb25b0d4621487a24d52b42f6cf557ce5c36b6c922035875cafa95c` 告知 c607、双 Pod 打开文件与进程检查 [A](ARCHIVED-SAFE-OPEN-A.json) / [B](ARCHIVED-SAFE-OPEN-B.json) 通过且 [演练](ARCHIVED-SAFE-DRY.json) 50/50 通过后，已删除历史 actualsafe 轮的 50 个非 selected checkpoint，分配 `35,349,656,064 B`；[实际回执](ARCHIVED-SAFE-DELETE.json) SHA `e39d7b450037dfa8b086ffbc4828d20c57d65f86d550b26420dccef9029aeaf9`，保护对象前后 stat/小文件 SHA 一致。完整逐文件清单和操作日志留在 exp-v3，不包括训练数据或权重进 Git。

R247 更早按单独计划已清理 7 个非 selected checkpoint，分配 `20,413,728,768 B`，见[原回执](../closed-r247-nonselected-20261104/README.md)。全量文件扫描、其余归属与不确定项仍在核验，不能把这些批次称为全部清理完成；保留所有选中权重及其依赖、活动轮和待用资产。
