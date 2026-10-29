# R107：全局批内 pairwise logistic

直接对照预先固定的 R104 Small seed20261029、单视图、scale[.85,1]，唯一学习目标由 BCE 改为固定温度1的纯 `mean softplus(z_negative-z_positive)`。全局64主图、8正56负、448对；默认 WORLD 可微 logits gather 与 DDP 平均配合，不额外乘除 world size。实际首50成功更新已记录64图/64视图/448对；公开 fresh 初态50c、空Adam，Git `2c75e7d6664cdbf0328c0aeb8dbc83655edec001`，不续旧训练权重。

实际6400成功=attempted、skip0、8完整周期，c5–8四轮未刷新后正常早停。完整16内部记录唯一选择 **c4 EMA@3200**：AUROC **0.9884405731067525**、AP **0.8274326888154083**、error_twice19200。对照R104原selector所选c3EMA，AUROC增量约+.0033582543、AP+.015757369；不是只挑c4同行放大增益，也非全部周期均改善。仅本固定seed/split和选择政策下的结果，不证明机制、显著性、稳定泛化或校准。原件见 [SINGLE-MODEL-FREEZE.json](SINGLE-MODEL-FREEZE.json) 和 [TRAIN_RESULT.json](TRAIN_RESULT.json)。

既有固定20/其余147正例错序质量分别8782→6334、3607→3266，整体12389→9600（修复4816/新增2027/净减少2789）；两组同向，主要净收益集中固定20。分组源于早期内部结果、选点轮次不同，属于描述性辅助分解，不改选点，不是病例数、机制或独立显著性证据。聚合原件 `technical-prep/r107-error-decomposition-20261029-b/SUMMARY.json` Pod SHA `117f2e1e34c27f7b3d378475d29cd406824a3a7ef5fa04b8d9d452830cd8de31`，样本表/图片未入Git。

原生日志训练2718.788秒、验证146.153秒、保存136.526秒（27次），TRAIN组件wall3220.652秒；训练段按实际成功步数归一为0.4248106436秒/步。不同早停长度/选点和保存次数，不当等步数benchmark或普遍吞吐保证。每卡最高峰值allocated8443409920B、reserved9661579264B，总显存68702699520B，无OOM。

完整官方5849仅原一次独立比较：AUROC **0.9888033253198455**、AP **0.8379487816044738**，不用于R108选择。三组件均exit0/硬期限内/signals[]，见 [ROUND_RESULT.json](ROUND_RESULT.json)。raw AUROC仍未达0.999，整体目标未完成。

19:08:53北京实际核整轮completed/complete、rank及组件inactive、device idle无使用者、原锁释放；owner121114为Z，不称reaped。运行根 `exp-v3/rounds/r107-small22k-seed20261029-scale085-pairlog-24k-p4-9010-20261029`。freeze原Pod SHA `ccf40d8eaa4b134c02c67651988a7c584179edb8576d5d88893776b7c9b0f435`，TRAIN `b84ab2c676dad4ed76f54796e643d47ba9b6b108a935dabebdead9138a7c93be`，ROUND `55c9ea1ad3db96fa773e4d3d9cd4fa6830ae4ee792b332f5238a331af05adda5`；终态 `technical-prep/r107-frozen-2c75e7d6664c/TERMINAL-OBSERVATION-2.json` SHA `4186bcef783f8c334f1acb07476f9ee2d1e9712ff89a7c20cca5f8797b7e5f7a`。本机仅文本读取/普通字节传输，无新前向、重官方、解析或重哈希。
