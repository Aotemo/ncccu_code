# R105：Small seed27训练尺度下限0.85

直接对照预固定 R102，仅新增 `training_scale=[0.85,1.0]`；也与 R104 仅训练 seed 不同。Small、公开 fresh、BCE、LR、固定9:1、24000/800/patience4/绝对 warmup-EMA1600 均保持。实际 Git `2f3cb4617de2939d003f9af86c22bf4fe52ef625`，正式 CPU 和8rank初态匹配 seed27 的91507控制，不加载旧训练 checkpoint。

实际5600成功=attempted更新、skip0、7完整周期，第4–7轮未刷新而正常早停。唯一内部选择 c3 **online**@2400，AUROC **0.987161811506687**、AP **0.8058457289282671**、error_twice21324。相对匹配R102 AUROC增加0.000422641545784419，AP下降约0.02749667；不是所有指标均改善。完整14条及选择见 [SINGLE-MODEL-FREEZE.json](SINGLE-MODEL-FREEZE.json)，正常训练退出见 [TRAIN_RESULT.json](TRAIN_RESULT.json)。固定20正例错序净少1612，其余147正例净多1261，整体净少351；这是错序对而非病例人数，只作描述性辅助，不证明机制或显著性。

原一次完整官方5849独立分报：AUROC **0.9857247024553104**、AP **0.817842592873573**；不用于R106选择。三个组件正常退出及唯一官方结果引用见 [ROUND_RESULT.json](ROUND_RESULT.json)。raw AUROC仍未达到0.999，整体目标未完成。

14:57:36北京实际终态completed/complete；所有rank与三个组件inactive，设备idle无使用者、原锁释放，资源可交接。owner103812为Z，不称已reaped。现有训练status记录每卡peak_allocated8443408896B、最高peak_reserved9359589376B、total68702699520B；这不是后续双视图显存实测。

权威运行根 `/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r105-small22k-scale085-24k-p4-9010-20261029`；freeze Pod SHA `82b582ea83446f79fd5f00e864b56a4d5a55f737e2441557ff9457006f8c7081`，TRAIN `f61af0168808c3a78d16fa8cd213079b807afda7a1e41d4afe5f3e5aaa044bf9`，ROUND `37f45700acf7a8758ab43537b6d515f0be6e640d71b8cd819dc6f3db7298c05d`。安全观测 `technical-prep/r105-frozen-2f3cb4617de2/TERMINAL-OBSERVATION-2.json` SHA `9bbacdec24f7f5dd94f1b7e26a69a22ed9c65f9b06991bec84c337e8c14a7162`。本机仅普通字节传输与文本读取，未重新前向、解析或重算指标。
