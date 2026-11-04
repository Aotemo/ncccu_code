# R230 fit-only EMA跨视图分布完整阴性

[完整结果](R230-RESULT.json)81e7a8f9：6400成功=attempt、0skip、c8、bad4正常早停；唯一内部selected为EMA c4@3200，AUROC **0.986436939111923**、AP **0.8180667558262131**、error_twice22528。未严格胜R107 .9884405731或R120 .9894773092，两门均false，R231不eligible且未激活；官方未达.996，按政策未运行。目标.995/AP>.9及raw.999未达到。

[配对报告](PAIRED-REPORT.json)只复算既有selected CSV，无模型加载或新增前向。对R107，全dev错序9600→11264，修复1387、新增3051、净增1664；原固定20为6623→7851，修复786、新增2014、净增1228；其余147为2977→3413，修复601、新增1037、净增436。AUROC配对bootstrap2000/PCG64 seed20261103，Δvs R107 −.002003634、95%区间[−.003819758,−.000477970]；vs R120 −.003040370、[−.007109890,−.000055208]。vs R115/R224/R226/R228的区间均跨0。区间按图片重采样，未修正患者聚类、seed、选择及多重比较；不推出临床因果或模型上限。

同1600步不等于同算力。完整fit适配102400主图、204800唯一视图，409600学生/教师encoder-view forward，其中204800教师forward；首至末SSL日志1327.042秒。联合训练组件4529.443秒，其中启动至SSL完成1420.193秒，随后监督至组件结束3109.249秒。教师/center八rank最终一致、无梯度且不持久化，只有学生encoder/final-LN进入监督。[完整SSL](R230-SSL-RESULT.json)a310ec77、[监督交接](R230-SUPERVISED-RECEIPT.json)4955f23e：derived91692404→完整初态e9672ee7，345全训/5组空AdamW、原head-age及监督RNG恢复；G64P8/448对、sampler实际=登记14997478083172343433，首自然25有限。

[唯一终态safe](R230-COMPLETION.json)33c2226b/guardf0482405已一次核实：completed、owner退出、原八rank缺失、锁及设备使用者空、denied0/八卡busy0。结果[归档回执](R230-RESULT-RECEIPT.json)dc4179d0验证原safe和CSV，不重查旧GPU；所选tensor49013bd9、预测b7e9ac91、precision525147ee为历史身份。完整结果及新R232真实首成功同步后才按实际清单清理旧checkpoint；此处尚不声称删除或永久可加载。

[实际首75](R230-START.json)60a79c5a证明公开50c6f55c/教师1a1d51c8与原P8一致、无共享/教师梯度；[来源](R230-PREPARATION.json)19必要目标/5编译、真实Small/P8及冷derived143.844秒、2662精确Git源/18正式CPU，在前轮训练期间完成。冻结4099eed/tree4888dc5c/SCI8cb98a2b/SITE d0385fcd，原owner817444/start377243020，运行未热改或重启。

R230新信号是fresh学生自身EMA教师的居中/锐化stop-gradient跨视图分布，补充R115同时刻统计；三层2048/256及65536原型为临时头，仅现有fit图片/CSV。该1600短窗没有改善主指标，不进入seed27。其结果前已登记的[R232固定9600历史覆盖问题](../../todo/auroc-999/R232-R233-EMA-HORIZON-20261103.md)沿同一目标单独fresh，现已[真实接续](../R232/README.md)；不延长旧VICReg，也不把本轮状态当teacher。两次实际接续间隔分别见[本轮](R230-HANDOFF-GAP.json)与[下一轮](../R232/R232-HANDOFF-GAP.json)，旧clock/token/grant保持。
