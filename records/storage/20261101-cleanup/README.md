# 2026-11-01 受控清理实证

**最终三批收口：实删100,652个文件、385个目录。** 前两批372个模型/fixture/运输文件与独立第三批旧prepared768缓存100,280个文件分列，目标旧cache路径现已不存在。[最终汇总](ALL-BATCHES-RECEIPT.json)实际sharedFS free **22,961,717,248→283,262,320,640B**，点时差 **260,300,603,392B**；全部实删文件allocated **260,478,643,200B**、logical **260,451,279,845B**另列。原始数据、当前prepared512、模型和诊断依赖未改；以下前两批空间读数保留其实际时点。

用户授权清除训练冗余产物，A在 full-fit census 两项RED已结束、诊断GPU尚未启动的自然边界串行执行。先将R134R1完整终态实际同步为 `4968edc784d2eb11c8aadae4774b365b78a7855d`，再按新Pod清单清理；未运行旧硬编码prune入口、未中断训练/诊断、未新建资源、Agent或timer。

两批实际删除 **372个文件**，均无执行期跳过或错误；[完整路径及inode清单](DELETED-FILES.json)、[第一批原回执](DELETE-BATCH1-RECEIPT.json)、[第二批原回执](DELETE-BATCH2-RECEIPT.json)保留真实分类。

| 类别 | 实删数量 | 删除前 allocated bytes |
| --- | ---: | ---: |
| 已完成历史轮 checkpoint/initial | 320 | 209384450560 |
| 完全通过的formal CPU fixture | 49 | 17531599872 |
| 过期源码运输包 | 3 | 24514560 |
| 合计 | 372 | 226940564992 |

[汇总回执](CLEANUP-RECEIPT.json)的最终空间观测：同一共享文件系统可用空间从 **22,961,717,248B** 到 **249,898,729,472B**，实际点时差 **226,937,012,224B**。exp-v2与exp-v3为同一文件系统，不重复相加。logical合计226940464520B；allocated为独占普通文件删除前的`st_blocks*512`，不冒充后台物理回收量。第二批刚退出时free只137502916608B，随后汇总时为249898729472B；保留两个实际时点，不据差值推断具体后台机制或排除共享盘并发影响。

两批均重新核对确切路径、device/inode/size/mtime/nlink=1、无符号链接祖先、当前开放fd/maps和已归档终态；大checkpoint只核stat及实际小sidecar SHA/广告blob SHA，没有hash/load大blob。[保护摘要](PRESERVATION-SUMMARY.json)与原前后快照SHA一致；锁、index、sidecar、clock/token/grant、one-shot、失败和成本小件全部保持原inode/原记录，未改旧冻结包。历史index仍可记录已删除的非保留blob，不据其存在声称旧轮可恢复。

继续保留：KEEP三唯一权重及sidecar/12份minimal records；中断R130全部产物和依赖；R120及R107/R108比较依赖；R134R1选中EMA@3200确切checkpoint、必要sidecar/index/freeze、选中dev预测/precision、job/manifest/age/ASSETS/SITE与49c8完整源码，以及其initial和当前prep；未完成census合同/JSON/tests/RED/work/prep；原始数据、共享prepared512/cache、待用公开来源及所有身份unknown路径。此次不重新声明这些大模型已完成payload哈希或加载验证。

[39轮终态与49项CPU fixture摘要](CLOSED-ROUNDS-AND-CPU-FIXTURES.json)在删除前由原小件提取，未删唯一科学结论。扫描到的model/package中仍留94项，见[剩余分类](REMAINING-SUMMARY.json)：包括保护对象、公共/历史来源、归属未明运输包、R114旧失败链缺完整终态和4项无完整CPU final receipt的fixture；未扩扫、修复或盲删。

完整inventory、两份精确PLAN、逐文件删除journal、保护快照、工具源码及失败运输/首次PLAN定位断言原件保存在Pod：`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/storage-cleanup-20261101-7c17`。inventory SHA `7fe671ccb07d61b8cb8ac9925b6aa0c20ee446767bbcbc46109365957830678e`；两PLAN SHA `38b5b1f881ae2b4405a35dc14d6ef1a21111ed4e567a757e6ab3015a2ab3942c`、`8d6753d1a9f76e6ad94c64989afc9a937b2c39ecc818d1a4fd12999d3738c576`。两批[CPU guard1](DELETE-BATCH1-CPU-RECEIPT.json)/[guard2](DELETE-BATCH2-CPU-RECEIPT.json)均exit0，实际A12/0–11/nice19/隐藏GPU；动态UID/Ready及d29入口有效，未占训练锁或发送训练信号。

## 旧prepared768独立第三批

精确历史路径为`exp-v2/auroc998-20261025/coordination-huazhong8-20261025/r24/prepared768-v2-pillow12.3.0`，初次lstat确认目录/inode9868483150；2000项软上限不足以作为检查过重的证据，随后针对同一目录在25.508秒完成[100665项完整清单摘要](LEGACY-PREPARED768-MANIFEST-SUMMARY.json)，无链接/多链接/挂载或当前引用冲突。R134R1/R130/R120/R107/R108实际ASSETS均指向独立prepared512 inode12952720392，未读取任何图像payload。

首次串行逐文件打开目录链/flush出现真实I/O瓶颈；清理会话明确授权后，仅对确证PID381474/start353149553/argv+固定manifest匹配的自有cache子进程发一次SIGINT。[受控停止](LEGACY-PREPARED768-CONTROLLED-STOP.json)与[原CPU exit-2](LEGACY-PREPARED768-INTERRUPTED-CPU-RECEIPT.json)真实保留，guard/训练/诊断未被发信号。停止后[逐项核对](LEGACY-PREPARED768-PARTIAL.json)确认18,614条unlinked均实际missing、无未配对before或缺after消失项；旧journal `b62f0fc3…`不改，不伪填成功。

新独立future软件CPU allocation SHA `2171235fd9d06311af95898e8dbada0d8f5064e590f2b40f3a13df2c198d15e8`，实际effective120、prepcap24、A12/0–11/root12–23备用未用，valid_until1793515756.22722（14:49:16.22722北京时间）。[原件](FUTURE-CPU-ALLOCATION.json)独立于d29，旧entry/期限/所有训练clock及GPUgrant均不修改。仅对余81,666文件采用385个经inode核对的目录FD缓存、6worker和批次刷新日志，逐项no-follow/stat/nlink/mtime/ctime及周期FD/maps/保护/空间检查保留；[实际接续](LEGACY-PREPARED768-DELETE-RECEIPT.json)52.06秒完成、errors[]/users[]/denied0/保护一致、旧journal一致，合计100,280文件与385目录实删，target_exists_after=false。[新CPU guard](LEGACY-PREPARED768-CPU-RECEIPT.json) exit0。第三批manifest SHA `52574c43…`、剩余manifest `3304ee9f…`，完整清单/journal仅留Pod；没有重扫其他树、测试、模型加载或GPU。

旧清理exit137历史不改写；本次只独立核实并清完该精确prepared768遗留。原早期汇总中“未重审prepared768”按其生成时点保留，已由本节及最终汇总覆盖。未知analysis目录和未完成census源码/测试/合同均未纳入本次storage提交。
