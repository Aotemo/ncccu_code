# R221R1：仅修复生产EMA静态RoPE检查

16:16已[真实跑通](../../records/R221R1/README.md)：5窄目标15编译、原10目标身份复用、7d744e2远端回读、正式20/20与精确部署，随后窗口/一次activation/direct均0；start5八rank初态及自然首25通过，最近100成功/0跳步，尚无完整所选。R221故障至新设备1137.143秒/首25为1193.645秒，未满足300秒，原因和原件保持；后文为修复前后合同记录。

R221在实际八rank初态全部匹配后、首成功更新回执前失败。八份rank失败原件一致报`Tiny noBN EMA live buffers differ`：生产`batch_norm_buffers=0`检查只允许DINOv3的非持久periods，遗漏EVA02非持久`network.rope.pos_embed`。外层`timeout`是非零退出的通用状态，不是clock不足；训练组件111.388秒退出1、reason null、无监督信号。最后已记录进度为0、无首成功回执/完整selected/官方；源码中optimizer.step先于报错的EMA.update，不能说没有执行过优化器指令，也不能判科学阴性。

R221原冻结f64f166/SCI3a205228/clock/token/grant/失败原件保持。15:58:47 terminal-1已实核唯一actualsafe，后续不重复观测旧safe。R219完整8000/c10所选.9877759060603908，双门false，R220仍不eligible。

R221R1采用独立fresh身份/order135，科学delta0：原公开源、512补6边/CLIP归一化、seed29/G64P8/448 logistic/AdamW/LR/WD/age/增强/EMA1600-.999/24000-800-p4及官方隔离均与R221完全一致。最小修复只允许EVA02原生描述验证后的静态RoPE，核在线/EMA缓存与SCI固定SHA一致且不进入参数EMA；不改权重、梯度或缓存值。原EMA缓存遗漏测试使用通用规则，新增生产规则测试覆盖warmcopy、blend、缓存不变及篡改拒绝，不把旧测试称为覆盖生产分支。

精确文件范围：`runtime/bundle/retrain/ema.py`、`runtime/bundle/retrain/experiment91.py`、`tests/test_eva02_model.py`、`tests/test_eva02_candidate.py`、`tests/test_round_storage_policy.py`、新`configs/R221R1-SCIENCE.json`和精确round的`configs/ROUND-STORAGE-POLICY.json`16GiB条目，及必要小型CPU/失败回执。既有准备/部署/启动/评估链仅更换明确身份和证据参数，不新增平台。先复现生产规则失败、最小修复、实际受影响CPU验证，再Git同步/精确部署/正式CPU及真实八rank首成功；不追加GPU预演或用户审批。

当前7c实核准备份额root0–11/nice19/隐藏GPU，保留12–23，与训练24–119互斥；不得用ineligible轮填补空闲。R221失败后空闲已超300秒，如实记录实际恢复时间，不跳过必要验证。尚未完成的新轮步骤不写为通过，raw .999未达。
