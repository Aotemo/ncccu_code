# F05 三汤 dev 的 CPU 前向：门 (a)/(b) 结果（2026-11-05）

状态：**门 (a) 通过；门 (b) 不通过，按任务书停止，未跑 5140 全量。** 未读官方、未做 TTA、未初始化 GPU。

## 门 (a)：同一 CPU 路径 vs 既有 GPU 记录

用 R120 selected（`construct_member` 严格重建、EMA、CPU）在 `dev` 前 512 张上重算，与 `dev-cycle-0006.csv` 逐样本比对：

| 量 | 值 |
| --- | --- |
| 样本数 | 512（dev 顺序、name/label 全对齐） |
| Spearman（logit） | **1.0** |
| Pearson（logit） | 0.99999999999985 |
| max \|Δlogit\| | 1.106e-04 |
| mean \|Δlogit\| | 2.274e-05 |
| max \|Δprobability\| | **1.879e-05** |
| mean \|Δprobability\| | 5.23e-08 |
| 判定（Spearman ≥ .9999 且 max \|Δp\| ≤ 1e-3） | **通过** |

说明：差异量级与 fp32 累加顺序/线程数不同一致，不含任何精度降级；参考 CSV SHA `e6f9f640a513e723c29e5fe9ca00c5c93baf6399f68f682229652ad34c5a7ab1`。
512 张耗时 979.53 s（12 线程）⇒ **1.913 s/张**。

回执 `GATE-A-20261105.json` SHA `e2beefc133f24835608165cabe2afc324bade427d0feac5af33d9d232e739935`；线程 torch 12 / interop 1。本次运行的本机 SSH 因 10 分钟工具超时先断开，但 Pod 内进程继续跑完并写出该文件；其 CPU guard 收据位于 `tasks/…-qap0ovph/`，为容器用户 0700，登录用户不可读，故门 (a) 的 guard SHA 记 **unknown**（不补造）。

## 门 (b)：吞吐

| 量 | 值 |
| --- | --- |
| 预热 | 8 张 |
| 计时 | 72 张、batch 8 |
| 秒/张 | **1.7409**（125.35 s / 72） |
| 三汤估算（3×5140） | **26,845 s = 7.46 小时** |
| 阈值（3 小时） | **不通过** |

单汤估算 8,948 s ≈ 2.49 h。门 (a) 独立测得的 1.913 s/张 与之一致（含首轮冷启动与 NFS 抖动），说明该量级稳定、不是偶发。

回执 `GATE-B-20261105.json` SHA `f36942dafb857b1a30aaf1f3831fb2d11b272022bbde9f2af1c9fca330f74b5e`；CPU guard `02d51bca177d8607cab634ca3c532b0d545eee17a5a955b62cc0a54f8d524a3d`；12 线程、mask 12–23、nice 19、设备隐藏。

汤身份（只读核验，与 `R120-UNIFORM-STANDALONE.json` 的 `state_sha256` 一致）：`model.pt` `1943b980ee5a0ade77bce57595642a3514251fbc473bf7f088ec0d587ae6a472`、`model.json` `a6a29c3f37098babf7b56306c67bee671159630cd96c900f8429a3b0fb2b0d87`、载入后 tensor `59f6ea5f854afd8e606d33ee003ba439ea9cd8bc34fd652963f6f15a8b3b7901`、variant `ema`。

## 为什么慢（只描述，不改口径）

- 模型 `convnext_small`、输入 `512×512`、fp32。按 512 边长、12 线程的 CPU 前向，1.7–1.9 s/张属于该规模的正常量级；不是缓存或 I/O 问题（batch 8、prepared512 本地缓存、无 NFS 解码热点）。
- 保持与既有记录同一数值口径意味着不能改 bf16、channels_last、降低分辨率或加 TTA 近似；这些都会破坏门的可比性。

## 可选续法（需 c607/主执行者决定，本项不自作主张）

1. **GPU 单视图 dev（设计路径）**：既有 `f05_soup_dev` 的 3/3/2 八 rank `--soup-mode` 评估一次即可在分钟级完成，且天然与 F05 选员口径一致；需要某个 Pod 安全终态且 owner 派发。
2. **借用空闲训练池（2026-11-05 授权）**：若某 Pod GPU 空闲且安全终态，可按实时 affinity 申请更大 mask（最多 100% 有效 CPU）。但 `convnext_small@512` 受内存带宽限制，扩容收益需实测；建议先按新 mask 跑 72 张计时门，再决定是否开跑 7 h 级任务。
3. **保持 12 线程**：需约 7.5 h 连续空闲，且这期间不得与训练抢 I/O，实际排期风险高，不推荐。

## 回执与文件

- 脚本（在助手 prep 根，未进 Git）：`soup_dev_cpu.py` SHA `56e4bf1e4f9b1a5616c2027fe46eabfc61f3571bf84e8792ba15b04acf0f0ca3`；`soup_gate_a.py` SHA `5af69e811e2b0d82c06dcf093d976008038ff1c3a29163f8271de9c64458f2a3`。
- 目录：`…/helper-20261105-5ae33039/{GATE-A,GATE-B}-20261105.json`；未生成 `SOUP-DEV-CPU/`（按门结论停止）。
- 三次 CPU guard 回执：门 (a) `bcd1fc2f`（首次 interop 报错，无结果）→ 正式门 (a) 完成；门 (b) 首次缺 `NCCCU_SITE_BINDING`（`5756b9b6`）、二次 Path 序列化（`b438ded0`）均无结果；正式门 (b) `02d51bca`。失败件保留，不覆盖。
