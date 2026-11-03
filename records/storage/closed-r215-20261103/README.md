# R215 冗余 checkpoint 清理

完整结果已在Git `b8b8bbe6a21e60620a858c299172e1de9b5247fc`归档，原唯一actualsafe保留不重查；固定组描述完成且无后续模型forward依赖。按用户仅永久保留N06/R17/R8-C25规则，沿原manifest/deleter清理本轮8个训练checkpoint及1个initial，未加载或重哈希大权重。

[真实清单](PLAN-SUMMARY.json) SHAda115995，原PLAN SHA `f976d5c33316ae641a3e9d62d627e6b1c416fd296b977da0657a948184c81c36`，9件全部仅在已关闭R215树，0遗漏/拒绝。已逐路径review并编译原删除函数，仅改轮次/提交身份。初次绑定脚本因旧注释含相邻轮次名而逆替换校验失败，[原失败guard](CPU-PLAN-FAILED.json)aopx3yg5/8196958b exit1保留，未执行删除；改为单次最长键替换并核逆变换，在独立bound2生成manifest，旧失败目录不改。

[删除回执](DELETE-RECEIPT.json) SHA `04e5b493f55a0c9a7e8f33171fe027d76ea5540e49984d6fef90ee334c9923fc`：9件全部unlink，逻辑12,723,697,075 B、原allocated12,723,698,688 B，0skip/0error。[计划guard](CPU-PLAN-GUARD.json)50x26gmg/dbe6effa、[删除guard](CPU-DELETE-GUARD.json)puvsy_zb/79ffcfe0、[最终只读校验](CPU-FINAL-GUARD.json)_r9_ggwv/6329652a均exit0。18条before/unlinked日志与9路径完全吻合，文件缺席、sidecar/归档结果仍匹配，保护前后摘要同为e129c5ea。

当前R217/SITE4c4、所有其他轮次、technical-prep/原始数据/共用缓存/公开初始化及三KEEP全部保护；无训练信号/锁/GPU或模型命令。所选R215权重现已删除，结果及预测保留，不能再称本轮所选可加载。共享文件系统空闲点观测23,048,749,056→35,772,170,240 B，不等同后端物理回收。原件 `exp-v3/technical-prep/r215-closed-storage-20261103-32e3-bound2`；d837/root0–11/nice19/隐藏GPU，与训练24–119互斥。
