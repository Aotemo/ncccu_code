# R178 冗余权重受控清理

R178完整6400/c8/bad4、两门false及actualsafe的结果已在9199bf8归档并由2a12873实际远端tree核到；不重查旧safe、不加载或复算checkpoint。复用既有清单/精确stat、侧车SHA、打开文件/映射扫描和逐文件unlink流程，只将候选缩到R178一个已关闭轮次。

实际删除8件checkpoint，logical 5,941,126,657 B、unlink前allocated 5,941,129,216 B。共享盘点时free 26,440,892,416→32,382,124,032 B，差5,941,231,616 B含同期共享活动，不能当后端物理回收保证。保护文件stat及小SHA前后相同907aba1a…de107；无denied、跳过、错误或候选活跃owner，无训练信号或实例锁。R180、KEEP三件、其他轮次、数据/缓存/公开源/当前准备依赖均保护；侧车、index、锁、clock、token和one-shot未删改。

实际原件：`exp-v3/technical-prep/large-dino3-20261102-32e3/closed-r178-storage/`；PLAN SHA875b7708…25ea4，INVENTORY f74e1ed1…560a，journal15246a7a…1236。删除回执8bb6d31d…690d9，计划/删除guard各exit0；最终逐项核8个路径已不存在、侧车和归档结果SHA不变，guard `l0f97v0z` SHA3a34072c…f589 exit0。这里只保存小型汇总及两真实guard，完整清单与journal留原件。

当前准备使用新5de56a8f CPU绑定（实时120核，root0–11/peer12–23与训练24–119互斥），原719a及R180冻结GPU绑定保持原件。没有因清理启动或重启实验。
