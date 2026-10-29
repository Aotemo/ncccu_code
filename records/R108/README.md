# R108：预先固定 seed27 的 pairwise 配对复核

直接匹配 BCE 基线 R105；相对 R107 只换训练seed20261027及自然fresh初态/身份，复用 Small、单视图scale[.85,1]、固定六字段全局pairwise logistic，64主图/8正56负/448对。Git `e92311a04e0ee7432eee38d8e119a9b0918c0745`；正式原生CPU18项及真实8rank初态均匹配91507控制、空Adam，无checkpoint恢复。首次初始化快照0更新/initial=null原件保留，后一次真实175成功更新确认64图/448对；不把spawn等同成功更新。首次传输中AppleDouble侧文件使文件集合检查失败，旧目录/失败guard/差集原件保留；同Git tree的干净独立`r108-frozen-e92311a04e0e-2`467文件校验通过后才一次activation/launch，无科学源码改动或第二提交。

6400成功=attempted、skip0、8完整周期，c5–8连续4轮未刷新而正常早停。唯一选择 **c4 EMA@3200**，内部AUROC **0.9872328538178018**、AP **0.8354670944168833**、error_twice21206。匹配 R105 原selector所选c3 **online**@2400，AUROC仅增+.00007104231111476622，AP增+.029621365488616047；不是改用R105 EMA、跨seed峰值或同c4行比较。两预固定seed配对方向同向但幅度异质；不同选点/停步、固定dev自适应和仅两seed，不构成显著性/普遍收益或机制证明。完整内部原件见 [SINGLE-MODEL-FREEZE.json](SINGLE-MODEL-FREEZE.json)、[TRAIN_RESULT.json](TRAIN_RESULT.json)。

B限定固定20/其余147正例错序质量：6653→7723（净增1070）、4009→2880（净减1129）；整体10662→10603，修复3497/新增3438/净少59，存在组间抵消。不能说难组或两组均改善，AP升幅也不等同该分解的原因。仅原selected CSV的描述性辅助解释，不重选组/点，不将pair当独立病例或用辅助组救主结论。聚合 `technical-prep/r108-error-decomposition-20261029-b/SUMMARY.json` Pod SHA `7c9b5f2b05b40eefde9af6d92fd06847e92f405154b9b7a2246042a648f04e5c`；图像/病例表/样本级预测不入Git。

原生日志训练2713.9104秒、验证145.6366秒、保存137.2609秒（27次）、TRAIN组件wall3216.6345秒；在Pod内按6400成功更新归一训练段0.4240485001335037秒/步，非benchmark或稳定吞吐外推。实际每卡最高allocated8443409920B/reserved9661579264B，总显存68702699520B。

完整官方5849只沿原一次流程独立比较：AUROC **0.9898970831274441**、AP **0.8525293647569135**，不用于R109科学选择。三组件exit0/硬期限内/signals[]，见 [ROUND_RESULT.json](ROUND_RESULT.json)。raw AUROC未达0.999，整体目标未完成。

20:34:24北京实际核整轮completed/complete、rank和组件inactive、设备idle无使用者、原锁释放；owner129568为Z，不称reaped。运行根 `exp-v3/rounds/r108-small22k-seed20261027-scale085-pairlog-24k-p4-9010-20261029`；权威prep为`technical-prep/r108-frozen-e92311a04e0e-2`，不是初次AppleDouble失败目录。freeze Pod SHA `5a57b0931f1f988a0e47590ba33205feefa69c5f9b9a83fc291c76c6e15e8a7b`；TRAIN `fe199888262071d3ba8474ce8b7d3f95897f5964fdd9b82f41b7f486e94f2dc3`；ROUND `bce88bc064cc955b793b2c2a705bae2c2b6b438dab83a660b7d0e21a10c46b0f`；TERMINAL-OBSERVATION-2 SHA `dbffbba60e861801eec6939539950fe66d499ed150277c1487aa494608a3a3a9`，guard `f813a78481027ee394eacf5a593f1d3e8b74ebae2b068253c26c0bf295f51fff`。本机仅文本与普通字节传输，无解析、重哈希或新前向/官方重评。
