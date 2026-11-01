# R150R1 fresh 工程重试完整安全结束

23:04:41.887195 UTC[完整原始观测](R150R1-COMPLETION.json)确认正常early_stopping：8800success=attempt/0skip/eval11/bad4；唯一EMA c7@5600内部AUROC`.9859781743570972`、AP`.8099265137055353`、error_twice23290，tensor`ab44ff07e64e6a33924a51c5f5c19824e15f9049934f3dce19789c6406593b0b`。[结果](R150R1-RESULT.json)两门false，关闭精确六通道方法/no27，R151仅技术准备未formal/grant/activation/GPU。官方threshold_not_met未跑，指标null；freeze.status原始null保持。owner Z身份匹配、8rank退出、锁/设备users空、denied0、8PCI busy0，actualsafe true，不重查。原clock/token/grant及所有依赖保护；下方为启动历史，当前唯一训练已由独立R152承接。

21:44:51.589315 UTC[实际首成功](R150R1-START.json)：running50success=attempt/0skip/eval0；八rank345全训/5组空AdamW/whole`ea92356b21fb0cbc94f28344866859f827e4cd745c57b88bb35d1bb8a9b654e1`与正常formal一致，自然log25 finite loss`.6516661047935486`、WORLD8/G64P8/448pairs。[八rank真实六通道结构](R150R1-ACTUAL-NATIVE6.json)均为stem`[96,6,4,4]`；owner458671/start359674682与八rank458691–458698的birth/argv/mask/8PCI匹配。无新模型、额外forward或checkpoint加载来观察。

冻结runGit`c8a24cde14bb702459db83734aafa17d2c6a6d12`、tree`93d369bf5c9661283e5e3062efb9e5664c743e4c`；同R150 SCI`4ef1433632b1728cfc4b23a610def0f9c648418d0d415ce85d5289f5bbc1dfdb`/order62，SITE`c69373c1f9474e48a8544a638c98607d729f08e893e4a4d828f018d9b85b28ff`。唯一生产修复是按实际job给DDP预热构造3/6通道；[CPU回归](../preparation/native6-ddp-repair-20261101/README.md)真实RED复现/单目标GREEN、旧3目标精确SHA/AST复用。[新正常formal](R150R1-PREPARATION.json)唯一20/20、122.113秒，[实际部署](R150R1-DEPLOYMENT.json)11.134秒，一次独立activation/direct launch0；原R150 clock/token/grant及全部失败原件不改、不重启。

[实际接续间隔](R150R1-HANDOFF-GAP.json)：原R150失败结束至新dispatch1183.166秒、至最晚八卡设备初始化1238.390秒、至首自然25记录1292.682秒，均超过300秒。原因包括预热错误、故障发现与修复迟延及必要fresh验证/部署/启动，未跳检查，也不把事件间隔说成连续利用率。

当前入口`work/r137-20261101-a/r150r1-pod-context.env`绑定START3；唯一Git/Pod/GPU owner32e3。训练CPU24–119/96核，准备00202581本人0–22/23核、槽23保留不用，两个池互斥。仍按原单Small、global+Native六通道、FP32/G64P8/EMA/24k800p4/唯一完整selector/官方>=.996一次；暂无完整selector或官方，raw .999未达，继续训练并同步准备后续。
