# R101：预登记的一次训练 seed 复核

直接对照R95/BCE，仅训练seed由20261027改为预先固定的20261029；splitseed20261024、固定fit46256/dev5140、LR、dropout、24000/800/patience4及绝对warmup/EMA1600保留。不是seed搜索，不选择更高seed作为新的对照。

实际Git为`a215f6fb76fc25b24e1108274cb81f266c500e7d`。新鲜整体初态为`18313b8b6d240b080eef7af214b41306b83a347135a2b7f5958addafc79877dd`，原生18项CPU和真实8rank初态通过。训练完成6400次成功更新、attempted6400、skip0，共8完整周期；c5–8连续4轮未刷新，合法早停。

内部唯一选点为c4 EMA@3200，AUROC **0.9843911613732117**、AP **0.8278718323176861**、error_twice25926。完整16条内部汇总及选点见[SINGLE-MODEL-FREEZE.json](SINGLE-MODEL-FREEZE.json)，训练终态见[TRAIN_RESULT.json](TRAIN_RESULT.json)。该一次seed复核未胜R95内部AUROC，不据两个seed推稳定性或挑最好seed。

官方完整5849只做原一次最终比较：AUROC **0.9889671113874808**、AP **0.8548787074889032**；与内部指标分报，不参与后续科学选择。原结果及三个组件真实exit0见[ROUND_RESULT.json](ROUND_RESULT.json)。raw AUROC未达0.999，整体目标未完成。

08:10:47北京实核整轮completed/complete；rank及组件inactive、设备无使用者、原锁释放。owner69349为Z，不能称已被PID1回收；不影响已实核资源安全交接，不发送信号或重启实例。

Pod原件根为`/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r101-tiny22k-seed20261029-24k-p4-9010-20261029`。freeze SHA为`6783af788175b731b96c018c010f4a45aa60e7bc28c8bea9207bb424f54ca604`，TRAIN SHA为`f198c07f18cbf684a341c2ebba6a7cedceccca4092c05586d1cc9ed45e603225`。安全终态原件在`technical-prep/r101-frozen-a215f6fb76fc/TERMINAL-OBSERVATION-2.json`，SHA为`5b573d526907733e59baa70621ac75a74a471b8b7b7af5588a88348af9b3ec79`。本机仅普通字节传输和文本读取，未重算模型或官方预测。
