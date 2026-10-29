# R103：Small预定seed20261029匹配复核

直接对照R102，仅训练seed20261027改为预先固定的20261029；固定splitseed20261024及清单不变。与Tiny R95/R101共同补齐两个架构、两个预登记seed，不是seed搜索或挑最好seed。实际Git `44b883ece72f9428bad9992a60919b48d4ca01ad`，Small整体fresh初态 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`。

实际10400成功更新、attempted10400、skip0，共13完整周期；第10–13轮连续4轮无提升，正常早停。唯一内部选点c9 EMA@7200，AUROC **0.9840901346312001**、AP **0.8089351087388285**、error_twice26426。完整26条内部及唯一选择见[SINGLE-MODEL-FREEZE.json](SINGLE-MODEL-FREEZE.json)，训练终态见[TRAIN_RESULT.json](TRAIN_RESULT.json)。与R102实际训练长度不同，不能称等计算量；两seed不足以证明显著或稳定收益。

官方完整5849仅原一次最终比较，AUROC **0.9890744514092079**、AP **0.8131226498592974**，不参与R104科学选择；原结果及train/freeze/official三个组件exit0见[ROUND_RESULT.json](ROUND_RESULT.json)。raw AUROC仍未达到0.999，整体目标未完成。

12:28:45北京实核ROUND/done completed/complete，所有rank和组件inactive、设备idle无使用者、原锁释放。owner86325为Z，不称reaped；无信号或实例重启。安全终态原件为`technical-prep/r103-frozen-44b883ece72f/TERMINAL-OBSERVATION-2.json`，Pod SHA `99ce032ab07d978503e90e73f696b3d05a7df242e655b7c8e33b3502f9047f37`。

Pod原件根为`/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r103-small22k-seed20261029-24k-p4-9010-20261029`。freeze SHA `552f29f35d1f73e5733afb3d264b9bb543a57caf52d4857173d72b3c81d381fa`；TRAIN SHA `8e01269893a5e22247492056fc47ccb31c42bdb057a663ceb3ddc07a17939ee3`。本机仅普通字节传输和文本读取，未重新评估、解析或哈希。
