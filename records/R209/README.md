# R209：原生 AMSGrad 的独立 fresh

09:19[完整终态](R209-RESULT.json)8800success=attempt/0skip/eval11/bad4、早停，TRAIN与整轮均completed/组件exit0/signals[]/errornull，唯一EMA c7@5600内部AUROC`.9867066590727653`、AP`.8166657875843961`、error_twice22080。严格胜R107及R120均false，关闭此固定AMSGrad配方/no seed27/R210不eligible，不扫beta/max公式/LR/WD/seed；不外推全部AMSGrad方法必然无效。官方正常not_run_by_policy/threshold_not_met/evaluatedfalse/metricsnull，freeze.status原null保留，不冒写completed；raw .999未达。

[唯一安全观测](R209-COMPLETION.json)cont28/SHA501a275e，[guard6par19li](R209-CPU-COMPLETION.json)/a7d7aced exit0：owner712175同start Z inactive、8rank缺失、锁holders/设备users空、denied0、8PCI齐且busy0。该历史safe不重查。tensor f6259ed5/blob6a81aae7/CSV960db511/precision5429b114及完整SHA在原件；归档不加载或重哈希大权重、不新增forward。实际已串行接续[R211](../R211/README.md)，整轮结束至新dispatch102.804秒/设备169.122秒/首25记录249.651秒；末训练日志至首25为315.833秒超300如实保留，连续利用率未测。[归档guard8riur6z7](R209-CPU-ARCHIVE.json)/93e78c2c exit0；后续原封存难组描述与冗余权重清理尚待，下文为运行期证据。

08:41[既有sysfs单次设备快照](R209-DEVICE-POINT.json)实核8PCI busy99%–100%，不推连续利用率；[guard51g3yug_](R209-CPU-DEVICE.json) / 3e89dbd8257a418897e5e46f3726831caf15c0078e45f6958f0d49ead4214eb5 exit0，无GPU API/forward或runtime修改。08:46实观4925/0skip/eval6/bad2、八rank活跃，暂best EMA c4@3200`.9862551189597479`，仍无完整selected。后续158bfb0两technical包已完成，normal在可行收尾前准备，现役冻结不改。

11-03 08:06[真实初态和自然首25](R209-START.json)已证八rank、345全训、5个参数组、各rank优化器初始state为空、whole50c6与formal一致；WORLD8/G64P8/448对、每rank8图，首25日志loss `.6513814926147461`有限、0skip。08:09实观325 success=attempt/0skip/eval0，八rank活跃。当前链路可运行，完整AUROC收益尚待终态；raw .999未达。

独立直接R107 Small/publicfe6/seed20261029/RGB512/FP32/ImageNet/原logistic/G64P8，只用`adamw_amsgrad_policy='adamw-amsgrad-v1'`将五个原生AdamW组设为amsgrad=True。保留原组顺序、LR/WD、age、betas(.9,.999)、eps1e-8、foreach/fused原默认；不叠beta095、hinge、困难样本loss或其他policy。最多24000成功更新/每800内部评估/p4/EMA1600和完整内部AUROC唯一选点保持，显式官方>=.996门沿现R120，历史R107不改。

AMSGrad比较未经bias correction的raw二阶矩max，再由原生实现作bias correction；不声称校正后的分母或实际有效步长单调，也不保证任务收益。[原6必要目标6compile/两fresh](../preparation/amsgrad-20261103/README.md)含独立FP64三步数值和Toy原生状态往返，未当作生产八rank断点恢复。五组AMSGrad kwargs的实测证据为[normal19/19及精确部署](R209-PREPARATION.json)和CPU目标；现有非Lion启动观测读取八rank空state与自然更新，不把字段`all_rank_optimizer_identity_matches_formal`扩称为逐rank逐组AMSGrad kwargs或vmax遥测。

冻结Git `d6fe92f01a174120543e1e82b2006018f7ad5e2c` / tree `dc88747957ffad51d6fb4c0f44acf35976e87e60`，SCI `86272648c18e596c702eb46c131fc55c893897f89df7cf8f1b4cfb538cebf7d9`，actual SITE `37c08d4e197b433d1267cf2ed47886a505f50f6e3934b5cef568f3e2cd5a7ea3`。[精确SHA复用](R209-CPU-REUSE.json)及normal19/19在R207训练期间完成；[窗口guard09vxk32l](R209-CPU-WINDOW.json)/210fa807、[一次activation guardu2cjun4e](R209-CPU-ACTIVATION.json)/da2afd2f、[首成功guardgwtwvxew](R209-CPU-FIRST-SUCCESS.json)/0c8f98b4均exit0，原入口显式8设备direct launch一次exit0，无独立GPU预演。默认12GiB/free16GiB及冻结storage policy93f8a86f保持；CPU原始张量容量估算不当作故障瞬间或设备峰值保证。

[R207完整结果](../R207/R207-RESULT.json)两门false，R208不eligible；同次actualsafe后接续。[实际间隔](R209-HANDOFF-GAP.json)：整轮结束至dispatch104.844秒、最晚设备167.249秒、首25记录237.568秒均<=300；末训练日志至三项173.806/236.210/306.530秒，最后一项超300如实保留。连续DCU利用率未测，旧clock/token/grant保持，必要检查未跳过。接续只串行消费原检查结果，无定时训练/新运行链。

唯一Git/GPU32e3，08:01实际只读核peer idle/activeTurn null，无消息/恢复/派生。ec5准备0–11/保留12–23与训练24–119互斥；owner712175/start372045775在control112–119，八rank712187–712194各11核。当前`work/r137-20261101-a/r209-pod-context.env`按实际start-2/UID/ec5绑定；start-1保留为启动屏障/unknown，不冒成功或重复launch。

[归档guardoey6ry5p](R209-CPU-ARCHIVE.json) / `bc52a291fab437398e99af0e734c9104a579df2be2a01c8d96d16f90b4259ed5` exit0，只读既有小型元数据，无新模型/checkpoint/forward/官方。后续候选尚待准备，不将任何登记或CPU证据写成已启动或已有收益。
