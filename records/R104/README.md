# R104：Small seed29训练尺度下限0.85

直接对照R103，仅将训练尺度下限0.94改为0.85，上限1；其他增强调用顺序/随机抽样数、验证视图、模型、seed及固定9:1/24000-800-p4-1600保持。实际Git为`018fc627d1a93915bfec9a85b03441bcd28974af`，原生正式CPU及8rank真实初态均匹配原50c6f55c控制，未加载旧训练checkpoint。

实际5600成功更新、attempted5600、skip0，7完整周期；第4–7轮连续4轮未刷新，正常早停。内部唯一选择c3 EMA@2400，AUROC **0.9850823187728706**、AP **0.8116753198258198**、error_twice24778，完整14条见[SINGLE-MODEL-FREEZE.json](SINGLE-MODEL-FREEZE.json)，训练终态见[TRAIN_RESULT.json](TRAIN_RESULT.json)。整体内部AUROC高于匹配R103的0.9840901346312001；不是全历史纪录，也不凭单seed确认机制或显著性。两轮选点更新数和实际训练长度不同，不称等计算量比较。预定固定组辅助结果另见[聚合分析](../analysis/20261029-internal-errors/README.md)，不改变主结论或选点。

官方完整5849仅原一次最终比较：AUROC **0.9878233849490783**、AP **0.8258091877865354**，与内部独立分报、不参与R105选择；原结果与三个组件exit0见[ROUND_RESULT.json](ROUND_RESULT.json)。raw AUROC仍未到0.999，整体目标未完成。

13:39:47北京实核整轮completed/complete，ranks及组件inactive、设备idle无用户、原锁释放；owner95427为Z，不称reaped。未发信号或重启实例。安全终态原件`technical-prep/r104-frozen-018fc627d1a9/TERMINAL-OBSERVATION-2.json`，Pod SHA `225b5f71de13644956a5888858f3525acdb31c57114ed2df8b0ad3e9565f4dad`。

原件根`/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r104-small22k-scale085-24k-p4-9010-20261029`。freeze SHA `e810b994214e64f287a39a02cfa6cc767f878ab35f82dbe42992c5e0398236ae`；TRAIN SHA `bd250a25db76b88ad0f104dc3f95ca604077d9af23ed6f685a5b0d2abebd3f76`；ROUND SHA `07d1b2a4b1e888063e2a97421d4fd904bd55854267b92e64ea415734edfdd837`。本机仅普通字节传输及文本读取，未重新前向、解析、哈希或计算指标。
