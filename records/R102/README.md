# R102：Small公开表征组合

直接对照预登记R95/seed20261027，成对更换ConvNeXt Small架构和fe6b358a公开checkpoint；不是纯参数量消融。BCE、512、显式SD0.1、dropout0.2、backbone1e-4、head/age各1e-3、固定9:1、24000/800/patience4与绝对warmup/EMA1600保持。实际Git为`620baa9d524f4d5978b79cbc3ad3b9c9ea4837b1`，整体fresh初态为`91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67`。

实际5600成功更新、attempted5600、skip0，7完整周期；c4–7连续4轮未刷新，合法早停。内部唯一选点为c3 EMA@2400，AUROC **0.9867391699609026**、AP **0.8333424024491386**、error_twice22026。完整14条内部及唯一选择见[SINGLE-MODEL-FREEZE.json](SINGLE-MODEL-FREEZE.json)，训练终态见[TRAIN_RESULT.json](TRAIN_RESULT.json)。主指标略高于直接基线R95，但单seed不足以确认稳定收益；R94不同horizon的AUROC0.9867415781748388仍略高，不能称全历史纪录。

官方完整5849仅原一次最终比较：AUROC **0.9879658880813711**、AP **0.8431683677756685**，不参与后续配置选择；原结果及三个组件exit0见[ROUND_RESULT.json](ROUND_RESULT.json)。raw AUROC未达0.999，整体目标未完成。

10:33:52北京实核整轮completed/complete、ranks及组件inactive、设备idle无使用者和原锁释放。owner77982为Z，不称reaped；未发送信号或重启实例。安全终态原件为`technical-prep/r102-frozen-620baa9d524f/TERMINAL-OBSERVATION-1.json`，Pod SHA `bbff9df627de08e3d45108530f4364f287b96c48a732b53be9fd3e2e6e9f4580`。

Pod原件根为`/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r102-small22k-24k-p4-9010-20261029`。freeze SHA `4a9ccb7646de8c63ec872e0ceb71fae0c200d1bd9c59ec6ca32f33690d9f68a4`；TRAIN SHA `59a72f04012fb728a119d0d7f7ce46b76544ca2f537f48a668624829311b0864`；ROUND SHA `cf2b3acef8dc305fa2e75b8228e8553429e4f49c49d39459b75c65354f251262`。仅普通字节传回和文本阅读，未重算预测、指标或模型哈希。
