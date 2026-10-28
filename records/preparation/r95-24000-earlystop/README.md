# R95：24000步与内部AUROC早停的准备证据

用户明确要求最多24000成功更新、连续4轮指标无提升早停，并选择内部dev AUROC。当前实现每800步完成一轮，轮内online/EMA取较高AUROC；首轮基线，严格提升清零，否则累加，连续4轮停止。warmup和EMA warmcopy保留1600，cosine horizon改为24000。新方案用于后续fresh R95；活跃212b R94冻结包不改，不重复activation/launch或官方前向。

本次在实时UID/Ready核验通过的既有Pod内，使用allocation3显式A12 mask0..11、nice19、隐藏全部GPU执行，和R94训练24..119及B12..23互斥。本机仅传输原件，无解析或重哈希。权威准备目录：`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/r95-24000-earlystop-20261029-a`。

| 原件 | 实际证据 |
| --- | --- |
| CHECKS.json / CHECKS-OUTPUT.txt | 13/13通过：新增4项覆盖首基线后4次无提升、EMA严格提升清零、保存状态接续、最大预算及实际轮数选点；9项为受影响候选/配置绑定检查。无模型forward、优化器更新、官方读取。RECEIPT SHA `2239f44b3790ccd1947a1157106e279071e3ba8afd64bed80fc35961fa23a2b5`，source unchanged，unittest 0.417秒 |
| CPU-CHECKS.json | 原guard回执SHA `20a0cdbf744154cc6fb9496bf79b9258662f48583f1125981554ef1eb095b616`，exit0 |
| CPU-PREPARE.json | 现成prepare配置生成实际exit0；SHA `d2610b30c4010946625ece3e53930948fc6f971e878ae6300947fa39cb29b0f5`。只生成technical package，不创建正式clock或GPU进程 |
| CONFIGURATION-PACKAGE.json | 新package的30轮、24000/800/1600、patience4、min_cycles1、min_delta0、配置bytes pin、EMA warmcopy1600及无正式clock均实核通过。SHA `2adb24d90904c3205f9e7c7cde28c6dc493b6a43ca910e105de74cc44b54e212`；SCIENCE SHA `ebb03883e571538f8352eca0d3613d452073956e91608ddfa39045a38c5e9dcb` |
| CPU-CONFIGURATION.json | package只读配置核验guard回执SHA `bd1ba5ff3c09b8065d1f94794d334a1ccb7624c7d2bb57ca5a9d9a17c4104a4f`，exit0 |

这些是新实现的CPU与配置生成证据，不是R95训练或收益证据；未做独立零forward GPU预演。正式source/Git身份、资源绑定和一次性clock由同步后的新冻结包生成；R94安全终态及GPU交接核实后才串行运行R95。旧来源、75GB迁移、native模型和历史套件没有重复执行。
