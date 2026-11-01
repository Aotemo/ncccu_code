# 固定原生亮区输入：CPU与全fit/dev资产准备

**技术接续包已就绪。** 实际运行源码 `11d9b4f2df2f93acbc4c6c00a626362219029d1f`/tree `d52037074b13a354893d4fed1a08ee8ae4e3f542`已push并远端回读，8受验源及SCI字节匹配。首次技术生成的run_name为129字符，被既有128限制拒绝；只缩短运行round名称，旧包/失败guard保留，SCI SHA不变，见[修正实证](NAME-FIX.json)。新round为`r144-small22k-seed20261029-scale085-pairlog-nativebright-half-24k-p4-9010-20261101`，生成run_name122字符；[v2技术包](TECHNICAL-READY.json)实际6.042秒完成、SITE SHA `4ab9a8d09bd206a6b5f75191c30d347d398d529ba0a73860be1302862e5c92ad`，helper compile/bash语法校验通过，[guard](TECHNICAL-READY-CPU.json)0。当前没有新normalformal/grant/clock/activation/GPU；唯一待执行脚本为本机`work/r137-20261101-a/RUN-NATIVE-NEXT-V2.sh R144`，旧长名包不部署。

三个唯一目标均实际通过：像素/独立只读缓存及真实serial/parallel/predict两项RED2FAIL→GREEN2/2，科学身份/派生资产绑定一项RED1FAIL→GREEN1/1；8个受影响Python源compile通过，GPU未初始化。完整[CPU-CHECKS](CPU-CHECKS.json) SHA `9a72d68753bc1108393a17924cdc380678a254e401f363d50aa9213e943435cc`；两轮原GREEN/RED、原guard及OUTPUT SHA均实核。像素目标之后只向preprocessing加入单独验证的资产核验函数，移去该函数能恢复先前整个源文件的精确SHA，其他像素受验源保持，不重跑已过路径。实际R144配置对R107整件差异及保存policy往返也已检查；没有构造公开模型或执行GPU预演。

固定规则为同R143 green/blur10/内侧10%/首row-major最大或平坦几何中心/半短边方框，移至原始RGB经原保守裁边之后、prepare512之前；复用既有PreparedImageCache的新namespace，单个Small/public fresh/seed29，其余训练链不变。合成1024棋盘样本证明原生裁剪保留实际像素细节，None旧像素与旧namespace、原8draw/RNG保持，缓存命中不回原图，训练只读miss仍失败。官方路径仅在原完整freeze≥.996的一次门内读取sealed原图并同policy准备；测试只用合成valid目录，未读实际官方图。

[CACHE-HANDOFF](CACHE-HANDOFF.json) SHA `f057453c1ead8ef8f1f492732f02c12fc29cd90f68d70bcdcd5deab0b5738425`实证fit46256+dev5140全51396原train行，19批均核原sealed SHA、写入新cache后只读复读。私有整份索引SHA `ac5b0febe4b2687ccfccb142e077f435603da33a3896360e512cb404ea6a07a0`留Pod，不入Git；每条覆盖序号经全表核对。实际批处理CPU墙时合计919.928秒，首批17:07:01.094239 UTC、末批17:47:52.103508 UTC；合计处理时间不等于含准备/交接间隔的总跨度。free由255589351424B至212685815808B，包含并行训练产物变化，不能视作cache独占字节。

前9批使用原12核本人份额/8workers；实时仅一Agent后，在不修改旧allocation或GPU绑定的前提下，以既有guard规划接口生成[新准备allocation](CACHE-CPU-ALLOCATION.json) SHA `00202581c67988f8bc7b11843562299f954be444dfbd18d2c7c12b11dd9a7423`：effective120/总cap24，本人0–22/23核、历史规划槽23/1核保留不用；后续22workers/nice19/隐藏GPU，与训练24–119互斥。两份重叠的prep分配串行使用。首bootstrap误用identity键在新分配写入前KeyError，原源/log留Pod；V2只改为NCCCU_POD_UID后通过，guard与集中比例未改。

[R144 SCIENCE](../../../configs/R144-SCIENCE.json) SHA `f6173696a103dec9c3de15cab1832e90ce0b5c5da14b07400594f1d96b2712b5`固定引用完整资产与初态50c；旧全球cache及当前R143冻结7d不改。此记录只证明CPU/资产准备，当前未normalformal/activation/新GPU，不等同R144训练链或raw .999已达；完整selected双门、官方隔离及接续见[todo](../../../todo/auroc-999/NEXT-PREPARATION-20261101.md)。
