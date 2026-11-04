# R240R1：固定fit类别配额对照

在用户新建的独立Pod A fresh启动，未恢复旧R240身份。[正式18/18](R240R1-FORMAL-READY.json)与c4b77a2/treeed24的2959 Git文件精确部署通过；SCI1b1badea、SITE349d1597、部署ed0de39c。96训练核24–119与24准备核0–23互斥，G64/P8、FP32、1600适配后原监督24k/800/patience4均保持。

[公开八rank及自然首成功](R240R1-START.json)SHA75c10e76、CPU81570bdb通过，owner677/start381509155；初始模型50c6、每批8阳性/56阴性，同类正对范围15–111、负视图16–112，loss/gradient有限。首次[unknown](R240R1-START-UNKNOWN.json)仅缺metrics，保留原件；没有再次activation或重启。[一次激活](R240R1-ACTIVATION.json)616074d9/CPUfccac619通过。

[双Pod汇总](../environment/dual-pod-20261104/DUAL-START-SUMMARY.json)当时750/1600，尚无完整dev或selected结果；不作性能改善声明。与R242并行只改变执行先后，原匹配比较和条件seed27仍需完整结果。用户新规则允许通用预训练融合、禁医学预训练及TTA，raw目标0.995/AP > 0.9；不热改本轮冻结合同。
