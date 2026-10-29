# R106：单图双视图平均BCE

直接对照预定R104，唯一新增 `training_views=2`；Small seed20261029、scale[.85,1]、BCE、512/FP32、G64/P8、固定9:1、24000/800/patience4/绝对warmup-EMA1600保持。每步64主样本、128增强视图，单concat forward/mean BCE/backward及一次optimizer/LR/EMA；实际首25更新日志各rank8主样本/16视图。Git `39c153ee220eea527a20e0022e89edff00adbed0`，正式CPU和真实8rank初态均匹配50c控制、空Adam，不续旧checkpoint。

实际8800成功=attempted更新、skip0、11周期，在第8–11轮连续无提升后合法早停；完整22内部记录的唯一选择为c7 EMA@5600，AUROC **0.9842033206861964**、AP **0.8140529507628876**、error_twice26238。对照R104主指标0.9850823187728706未胜，AP略升不挽救预定主判据。完整原件见 [SINGLE-MODEL-FREEZE.json](SINGLE-MODEL-FREEZE.json) 与 [TRAIN_RESULT.json](TRAIN_RESULT.json)。单seed、不同选点和早停长度，不诊断机制或显著性。

B固定20/其余147正例的错序质量分别8782→9140、3607→3979，整体12389→13119（修复5072、新增5802，净增730）；两组均退化。仅既有selected点的描述分解，非病例人数或独立证据。聚合原件 `technical-prep/r106-error-decomposition-20261029-b/SUMMARY.json` Pod SHA `f0303c66542e757d32339e920a7eb7bd2914e3ef1b245dcb2c440bdcdae6507b`；不含样本表或图像。

已有native计时：R106训练6473.757秒、验证204.444秒、保存184.990秒（37次）、TRAIN组件墙钟7103.695秒；R104分别2363.294/126.941/119.114（24次）/2828.015秒。按各自实际成功步数归一的训练段平均为0.735654对0.422017秒/步，比1.743187；不是等步数benchmark、整轮wall比或普遍吞吐保证。双视图每卡峰值allocated最高15738070528B、reserved最高16806576128B，总显存68702699520B，无OOM。

完整官方5849仅原一次最终独立比较：AUROC **0.9860337676902833**、AP **0.8306009685595881**，不用于R107科学选择。三组件均exit0/硬期限内/signals[]，见 [ROUND_RESULT.json](ROUND_RESULT.json)。raw AUROC未达0.999，整体目标未完成。

17:26:35北京实际核整轮completed/complete、ranks与组件inactive、设备idle无使用者、原锁释放；owner112189为Z，不称已reaped。运行根 `/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r106-small22k-seed20261029-scale085-views2-24k-p4-9010-20261029`。freeze Pod SHA `09500603403acc77f76b0010f08019b06468846768d149899e870895e204cb1a`；TRAIN `b89de22738e316158338be4ceae0565f30463685d18334e58a0400c833c6bb87`；ROUND `62f8de8d105026fb5d7dda45b182073900d1adfb02a84c83b7402572cb1ef391`；终态 `technical-prep/r106-frozen-39c153ee220e/TERMINAL-OBSERVATION-2.json` SHA `9dad96758550fdc6e705ae9d4c55cc08b112e0f934889dd33890f25359e38990`。本机仅文本读取与原始字节传输，没有新前向、解析或指标重算。
