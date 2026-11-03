# R221R2 实际跑通及工程恢复

冻结c7326677/tree9d4bbada、SCIecf6e050、actual SITEc833c9ad。正式20/20 CPU和整树精确部署完成；窗口检查、一次activation、direct launch均exit0，未做GPU预演。科学合同与R221/R221R1相同，仅独立fresh身份/order136和存储报错补足四个实际字节值，预算阈值未改。

11-03 16:52:29 UTC记录自然首25成功，loss .5590348839759827有限、WORLD8/G64P8/448对/scale1；16:52:44实观50成功/0跳步，随后17:08实观1700成功/0跳步/c2/八rank活跃，已跨1382旧存储故障点及1601 EMA blend边界（cont4/qchar9_d exit0）。实际八rank/237参数/五空AdamW/初态51592a3d与正式CPU一致。start2/23e51726、guardrznpl7fo/bd48c126 exit0。尚无完整selected或收益结论，raw .999未达。

R1工程结束至本轮dispatch1303.054秒、八卡设备记录1383.764秒、首25记录1435.147秒，明确超过300秒。原因是共享存储容量恢复、精确旧权重清理、必要窄修复与Git同步/正式CPU/部署；[真实间隙](R221R2-HANDOFF-GAP.json)不声称连续利用率或失败调用瞬间未知字节。三份manifest共24旧checkpoint、19,705,864,713逻辑字节已清理，保护对象不变。

唯一Git/GPU owner32e3；7c准备0–11/保留12–23与训练24–119/96核互斥。当前私有r221r2-pod-context.env/start2与source-git-r221r2，旧env仅历史；冻结运行不热改，不重复clock/activation/launch或旧safe观测。后续R222/R223仅在独立准备树完成，不恢复或派生Agent，不新增计算资源。
