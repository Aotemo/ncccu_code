# 已结束 R205 的冗余权重清理

[清单汇总](PLAN-SUMMARY.json)与[实际回执](DELETE-RECEIPT.json)记录7个闭合R205权重删除，0跳过/0错误，逻辑字节5,148,977,719、预先分配字节5,148,980,224。完整结果由实际GitHub回读4f82d59/tree8307cf6证明；唯一选点、同次actualsafe及封存难组均已归档，未重查旧safe或加载模型。精确plan SHA `f9b7fe20e311cd6515af60ebf9aa06f1fe480a3fb3f8f241bfad6dd0a44c938b`，inventory `adb463e6de915b36f82dcc86a0f4441ccb72d9f2ad60c0228c3e2d52d4f207a0`，receipt SHA `5b567249181e109e5f145e1220675c1a53fb71d1a766048db61edb0c20907de4`。

只删精确清单内6个训练blob及1个initial blob。N06/R17/R8-C25、当前R207的SITE/SCI/配置及所有非候选目录保护；保护stat及小文件SHA前后同为 `5fd80cf8de81980f312d57d9518512456e6cb536d3d5f6f411d9d6f2cc345480`。原CSV/precision、sidecar/index、freeze/metrics、锁/oneshot、原始数据、共享缓存及实际待用公开预训练保留。逐项核唯一regular inode/stat、无symlink/hardlink、实际sidecar SHA、未打开/映射；进程扫描无拒绝，未取训练锁或发训练信号，权重payload未重哈希或加载。

删除前静态diff发现私有生成器的lowercase轮次断言残留r203；当时仅只读plan已执行，删除未开始。仅将实际DELETE断言绑定r205，保留原plan/inventory及旧准备回执，不重写历史SHA；[受影响脚本编译guardq7fmompd](CPU-DELETE-COMPILE.json) / `97e035f9b614e5367ad3b521d2180171a8637acd353ce827c0b1e78b0c4be666` exit0后，才一次执行同一精确清单。实际DELETE源SHA `dbffc336c3486ff92c7b7520f4003c942cee5aff90a209d6dac860bc8e82c589`，原始准备输出中的47ffffb7属于修正前私有脚本，不冒充实际执行源码。

共享文件系统free38,474,350,592→43,623,907,328字节，差值5,149,556,736，仅两时点观测，不保证后端物理回收。原件`exp-v3/technical-prep/r205-closed-storage-20261103-32e3`；最终核对14条journal action/path、7文件实际消失、全部sidecar及归档结果SHA一致。

[计划guardzyitt5p6](CPU-PLAN-GUARD.json) / f9a9056e、[删除guardsak6rc97](CPU-DELETE-GUARD.json) / 59adead5、[最终guardj1t992sr](CPU-FINAL-GUARD.json) / `e0cb32ff1ac8f00b5b4be9896deb674007d6f1c6fc53b672cbdfe78504829ca7`均exit0；ec5/root0–11/nice19/隐藏GPU，与R207训练24–119互斥，无图像/forward/官方或现役修改。
