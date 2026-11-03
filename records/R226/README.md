# R226 fit-only 局部梯度目标实际启动

完整1600次适配及八rank监督交接已实证：监督初态db052095、345全训/5组空AdamW、原head/age及监督RNG恢复一致，首25 loss .667283773有限、G64/P8/448对；19:40实观325 success=attempt/0skip/eval0。实际sampler记录与消费均为14997478083172343433，原误登记问题未复现。[监督首成功](R226-SUPERVISED-START.json)及[来源](R226-SUPERVISED-RECEIPT.json)保留，观察者没有创建模型、加载权重或执行前向。完整dev结果尚待产生。

11-03 19:27:03 UTC实际首SSL更新成功，19:28:21实观200次，八rank活跃、loss/梯度有限。首1 loss .315540105、200 loss .027693853，G64/P8、每图154个32px块、单视图fit-only。实际公开模型50c6及encoder/public张量、空优化器、768→432/PixelShuffle4空间头均匹配正式CPU。[实际首成功](R226-START.json)、[公开初态门](R226-SSL-INITIAL-GATE.json)和[准备及各guard](R226-PREPARATION.json)保留，没有重做activation/launch。

冻结Git21dbb113/tree5c975e67、SCI43964fcf、actual SITE1c4e2d58；12必要CPU目标/5编译、真实公开Small-P8及冷derived79.156秒、正式18/18、2585文件精确部署均在R224训练期间完成。启动前重新核完整clock+60秒窗口，一次activation/唯一实际launch均0，无GPU预演。owner798693/start376131227与八rank出生身份已核，4e0c唯一Git/串行GPU监督；准备0–23/24核与训练24–119/96核互斥，实际私有入口`goal-20261103-4e0c/r226-pod-context.env`/ACTUAL-START-2，旧r224 env仅历史。

[R224完整所选](../R224/R224-RESULT.json) .9859528881未过R107/R120双门，R225不eligible且未激活，R226按预定后备问题接续。相同fit图、首视图、mask/学习随机流及1600预算下，以局部归一化梯度替换RGB重建；Sobel/9方向/8px cell、逐cell L2归一化、只hidden HOG cell的MSE。临时decoder丢弃，随后完整R107监督合同保持。没有额外任务图、标签/age用于SSL或旧teacher；HOG不等于真实视盘定位，不能恢复512外信息。依据、固定合同和必要实现边界见[准备记录](../preparation/hog-reconstruction-20261103/README.md)。

完整内部AUROC主判，附AP及相对R107/R120/R115/R224的原固定20/其余147修复和新增错序；仅严格双门后做固定seed27的R227。R227当前仅technical，官方保持≥.996完整selected后一次，当前未运行。

清理R224后的同次实观2250 success=attempt/0skip/c2、八rank同出生身份活跃，暂online/EMA best .9796662456305968/AP .7752608467207747，尚非完整selected。原R2248权重已释放5.941GB逻辑字节，当前R226及三KEEP/CSV/数据缓存保护；准备池预登记的[局部一致性后备](../../todo/auroc-999/R228-R229-FIT-LOCAL-20261103.md)尚未实现、未通过CPU或运行，不改变现役21db。

[接续间隙](R226-HANDOFF-GAP.json)：R224整轮结束至dispatch178.326秒、首SSL272.288秒、自然首25记录281.744秒，均≤300。末监督更新至首SSL369.896秒包含旧轮评估/收尾，不能称GPU空闲。连续利用率未测，旧safe仅复用、clock/token/grant均未改。当前回执只证明新GPU链实际跑通，.995/AP>.9及raw.999均未达。
