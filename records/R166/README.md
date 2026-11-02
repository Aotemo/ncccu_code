# R166：最终 LN 后的原生 patch 均值读出

已完整收口：[结果](R166-RESULT.json)、[原始终态](R166-COMPLETION.json)与[CPU回执](R166-CPU-COMPLETION.json)记录6400 success=attempt/0skip/eval8/bad4，唯一EMA c4@3200 AUROC `.9763103995106509`、AP `.7348542302546986`、error_twice39348。低于R107与R120，两门false；关闭此固定meanpatch配方，R167不eligible且未normalformal/activation/GPU。官方按`.996`政策未跑，freeze.status原null保留；owner同start的Z、八rank缺席、锁/设备使用者空及8PCI busy0，actualsafe=true。不重查旧safe/加载权重或新forward。

归档结果SHA `9a5501c89a9a24bdfade32ce53ccea6f75cef0a634dd1cb9781cd09507dbe6ec`，原终态SHA `dcb197a6244d92e1986954673fae5cefe2fe41d60cbbb0a21aa13c5de91aeea0`；原运行98冻结与旧clock/token/grant/selected依赖均保留。后续独立R168仅改原R107的AdamW beta2，不以本阴性推断所有Transformer、空间信息或临床机制无效。下文为本轮启动时记录。

[真实八rank首成功](R166-START.json)：375 success=attempt、0skip/eval0；177全训/5空AdamW/whole64556b75与formal一致，首自然25 loss `.8336942195892334`有限，WORLD8/G64P8/448pairs。8PCI与训练24–119/96核匹配，owner528676/start363206391及8rank身份匹配；GPU链实际跑通，完整AUROC待结果。

运行Git `98badc9483928e5c1bad25d498999ccdc5f0036a`/tree `cfbdf26df10bf78c5bbde9f4a93d6029f4583801`，SCI36676a44、SITEdbe028c9。[18/18唯一正常formal](R166-PREPARATION.json)51.400秒、精确部署11.590秒、一次activation/direct launch0；[CPU与只读复核后最小修正](../preparation/deit3-meanpatch-20261102/README.md)已实际同步，旧原件保留，无独立GPU预演。

同R164公开8d59b281/原512/原logistic，仅将最终LN后的1024patch平均、排除CLS的末端读出，接原Dropout/Linear/age一次；CLS仍在自注意力序列，fc_norm仍Identity。参数/RNG及初始SHA可相同而函数不同；不是ROI、临床机制或已证收益。R164完整7200/c9/bad4、所选EMA c5@4000 AUROC `.980575346391472`/双门false/官方未跑/actualsafe，不重查旧safe；R165不eligible且未formal/GPU。

[实际交接](R166-HANDOFF-GAP.json)：旧训练完成事件至dispatch322.495秒、最晚设备384.086秒、首自然25记录415.991秒，均>300；整轮ended至三项为289.727/351.317/383.222秒。技术包已在前轮结束前完成，仍因正式准备及完成回执/安全观测/派发衔接偏慢超时；不把round ended起算的dispatch≤300说成用户要求已满足。下一次关键交接直接等待工具完成输出并及时接续，不在此间插入分析/文档。未跳检查、改旧指标或重置clock/token/grant。

唯一Git/GPU owner32e3，准备dca3/根0–22共23核、peer23共1核与训练96核互斥；当前r166-pod-context.env绑定start-2，原98冻结不热改。最多24000成功更新/每800内部dev/patience4，完整所选≥.996才官方5849一次；raw `.999`未达，继续。
