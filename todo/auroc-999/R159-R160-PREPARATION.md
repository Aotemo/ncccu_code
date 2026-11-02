# R158期间固定后续

R159/order71仅在R158完整工程有效selected同时严格胜R107 `.9884405731067525`及R120 `.9894773092062407`且actualsafe后，复现同公开14a83735初始化/原logistic/RGB512的固定seed20261027；否则未formal/activation/GPU。根32e3仍唯一Git/GPU，aa0f根0–22/23核准备与peer23/1核及训练24–119/96核互斥，不创建/恢复Agent，不热改R158的c10冻结。

独立R160/order72直接R107/public fe6b358a fresh、seed29、RGB512、原G64P8/采样/增强/age/AdamW/LR/EMA/24k-800-p4，仅替换损失为固定批内primal AUC margin1。全448个raw差值`d=z_n-z_p`，`mu=mean(d)`，`L=mean((d-mu)^2)+relu(1+mu)^2`。等于两类各自总体方差之和，加`relu(1+mean_neg-mean_pos)^2`；两项系数均1、margin固定1，不扫描。

公式由[LibAUC官方AUCMLoss v2](https://docs.libauc.org/api/libauc.losses.html#libauc.losses.auc.AUCMLoss)的条件期望形式在本WORLD完整批次上消去a/b/alpha得到。这里使用批内直接原始目标及原AdamW，**不是完整PESG/min-max训练的复现**，不声称无偏总体风险或论文收益/鲁棒性已在本任务成立。R156固定困难正例加权已阴性，新假设使用所有配对的分布统计；可能仍对离群点敏感，完整整体AUROC是唯一主判，难组不能救门。

最小范围：现classification_loss.py增精确6字段版本kind/原448差值上的3行目标；现experiment91.py绑定R159/R160和原保存/唯一选点；check_candidate.py加新loss合同验收；必要新loss测试、原candidate测试与两SCIENCE。config/engine/distributed_engine已消费PAIRWISE_LOSS_KINDS，复用原WORLD gather和DDP归一化，不改模型/初始化链/EMA/优化器/采样/保存/评估/监控平台。无新增可训练loss状态、teacher、checkpoint、dev名单或模型forward。

先实际RED，再CPU独立解析8×56值/逐logit梯度、均值间隔小于/等于/超过1及方差非零、平移与共享bias梯度0、极值有限、两rank真实WORLD/DDP完整参数梯度；受影响的旧None/logistic/sigmoid/hard-positive值/梯度复核。两fresh身份实际核定，源码/SCI push回读、技术包在当前训练期间完成；旧训练收尾正常formal/部署，actualsafe后立即一次activation/direct launch，真实八rank/首自然成功证明GPU链，不加GPU预演。

R160完整selected未严格胜R107则关闭精确配方，同时严格过R120才有固定seed27资格；不扫margin/系数/LR。官方仍仅完整selected≥.996后5849一次，raw .999未达。必要最小实现及[7目标/5compile、两fresh证据](../../records/preparation/aucmargin-20261102/README.md)已通过，原RED保留；当前待实际Git/技术包，未normalformal/grant/GPU。
