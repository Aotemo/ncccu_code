# R154训练期间的固定后续准备

唯一Git/GPU owner32e3在aa0f准备分配0–22/23核内完成必要源码、配置、打包和CPU验证，与R154训练24–119/96核及用户另开分析会话CPU23互斥。各执行者自身任务串行；不创建/恢复Agent，不热改R154的04172db冻结、clock、grant或allocation。

R155/order67仅在R154完整工程有效selected同时严格胜R107 `.9884405731067525`及R120 singlebest `.9894773092062407`且actualsafe后，采用同一Swin V2 Small/ImageNet1K/RGB512固定seed20261027。公开fresh初态必须实际测得，不复制R154 seed29初态或加载任何实验checkpoint。不满足则不formal/activation/GPU。

独立R156/order68直接对照R107：仍Small公开fe6b358a fresh、seed20261029、RGB512及原采样/增强/age/G64P8/AdamW/LR/WD/EMA，只替换训练损失。每次WORLD完整64图中8正56负，先计算全部448个`softplus(z_n-z_p)`，沿56负例求各正例均值`l_p`，固定`L=.5*mean_8(l_p)+.5*mean(top2_8(l_p))`。同值按WORLD收集后正例原顺序稳定排序，作为固定subgradient约定；不使用随机数、温度或系数扫描。

依据是[既有内部难例分析](../../records/analysis/20261102-persistent-errors/README.md)的持续错序集中现象，不是图像或临床机制已经获证。全部pairs保留，不用dev名单/分数训练或设权、不改标签/采样、不增加模型forward、teacher或历史权重。用户已确认1=青光眼，阴性不等于所有结构正常。困难样本加权可能放大噪声，半份原均值不保证收益；R100 BCE focal阴性保留。

最小源码范围：现`runtime/bundle/retrain/classification_loss.py`增一个精确版本损失；现`experiment91.py`绑定R155/R156与保存/唯一选点身份；现`tools/validation/check_candidate.py`增加新loss合同的必要正式验收项；对应两SCIENCE及必要loss/candidate测试。现config/engine/distributed_engine已消费`PAIRWISE_LOSS_KINDS`，复用WORLD gather与原DDP归一化，除实查发现缺口不改它们。沿原启动、保存、EMA、评估、监控和两阶段准入入口，不另建平台。

CPU窄验覆盖真实8×56非同分独立解析loss/梯度、全体logit平移与共享bias梯度0、全同分与第2名边界同分、实际WORLD Gloo/DDP各rank一致及参数梯度；旧None/logistic/sigmoid值与梯度保持。配置核public fresh、原G64P8/512/AdamW/LR/EMA/24k-800-p4、已保存loss身份和官方隔离；只测受影响路径，原字节一致通过证据直接复用。稳定排序依据[Torch 2.10 sort](https://docs.pytorch.org/docs/2.10/generated/torch.sort.html)，不能用不保证同值index稳定的[topk](https://docs.pytorch.org/docs/2.10/generated/torch.topk.html)。

完整总体内部dev AUROC为主判，固定难组仅描述，不以AP/局部救门。R156完整工程有效selected未严格胜R107则关闭精确合同；只过R107为有限正向，同时严格过R120才有后续固定seed27资格。最多24000成功、每800内部dev评估、patience4、唯一online/EMA selector，完整selected≥`.996`才官方5849一次，raw `.999`仍是总体目标。

必要最小实现及[真实CPU证据](../../records/preparation/hard-positive-20261102/README.md)已完成：6目标/6源compile、两public fresh、WORLD逐logit与参数梯度通过；原RED及辅助脚本失败保留。源码/配置已实际push/远端回读Git874feff，02:26:35 UTC两技术包、精确SHA和临时调用脚本语法已过；没有新normalformal/grant/activation/GPU或训练收益。前轮训练完成进入freeze时完成唯一正常formal与精确部署，actualsafe后立即一次activation/direct launch，真实8rank初态与首自然成功证明GPU链，不新增独立GPU预演或例行审批。
