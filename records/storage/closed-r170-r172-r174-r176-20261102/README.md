# 已归档四轮 checkpoint 清理

在唯一活跃 R178 训练期间，按用户“保留三模型、提取精简结论后清理其余 checkpoint”的授权，复用既有逐文件清单与删除函数；源码结果核对远端 commit `30023dadf504245fc5cbf2cec76b91d97d4f448f` 的真实 Git blob。R172/R174/R176 均完整归档且 historical actual_safe；R170 是已归档存储工程失败、selected null，不是科学阴性，其独立复验 R174 已完整且安全。因此释放这四个旧轮 checkpoint 的临时保护，保留全部失败/指标/选点/侧车/index/锁及 one-shot 元数据。

实际清单 35 件，逻辑 36,507,477,730 B、删除前 allocated 36,507,485,184 B；35/35 unlink 经后查确认，无跳过/未知访问/活跃候选 owner，70 条逐件前后 journal 一致。共享文件系统可用空间 21,333,278,720 → 57,841,549,312 B；这个时点差值不是独占物理回收量证明。

KEEP-MODELS 三个永久模型、R107/R108/R120/R130 等其余轮、当前 R178、公开初始化、原图/数据/共享缓存及所有非候选根目录均保护。逐候选核 regular/nlink1/无 symlink、精确 inode/stat、侧车 SHA/广告 blob SHA、/proc fd 与 mmap 无引用；未重 hash 或加载大 checkpoint。保护文件 stat/小文件 SHA 前后相同，未获取训练锁、未发信号，也未改运行包、clock/token/grant。完整原件在 exp-v3 technical-prep/large22k-20261102-32e3/closed-storage；本目录只归档必要小清单及真实回执。
