# R177/R178：固定来源复核与排序/逐图联合目标

当前唯一GPU为R176/order88，e9a7f1d冻结；已实际八rank fresh/345全训/5空Adam/自然首25有限。32e3唯一Git/GPU，719a准备root0–11/12核、peer保留12–23/12核，与训练24–119/96核互斥。使用已读brainstorming/writing-plans/executing-plans流程在本任务文件记录设计并直接实施；用户已明确必要最小修改无需审批，不增加设计批准等待或派生Agent。

R177/order89仅在完整工程有效R176的唯一selected严格超过R107 `.9884405731067525` 和R120 `.9894773092062407`、且actualsafe后，复核同Small DINOv3公开c497e64d/原输入512/原logistic/G64P8/LR/EMA/age的固定seed20261027；否则不normalformal/activation/GPU。

独立R178/order90直接R107原Small/fe6b358a公开fresh/seed20261029/512/ImageNet输入/G64P8/原采样增强/AdamW/LR/EMA/age/24k-800-p4，只将目标换为 `L=.5*mean_448(softplus(z_n-z_p))+.5*mean_64(BCEWithLogits(z,y))`，标签为原8正56负；temperature1、pointwise_weight=.5固定。逐图项消费同一WORLD全局logits，每图一次，不重新forward/采样/通信、不按dev名单加权、不加prior修正、loss参数或teacher。它是自定义固定组合，不冒充某论文/LibAUC实现。

假设：纯pairlog对整体logit平移没有监督，而逐图标签项提供绝对分数约束；这可能改变共享骨干的梯度分配，但不证明前述平移自由度是错误根因，也不承诺AUROC或校准收益。R104纯BCE、R107纯pairlog、R132固定raw平方及R156困难正例的结果是相关负先验；联合目标不是已验证收益。BCE基于采样后P8/N56，不能声称恢复真实总体患病率。完整内部AUROC为主判，AP/难组不救门；失败关闭固定组合，不扫系数/temperature/LR/seed，不叠加DINOv3或旧局部策略。

## 实施计划

目标：沿现有PAIRWISE_LOSS_KINDS/WORLD gather/DDP mean实现固定联合目标，原None/logistic/sigmoid/hard-positive/margin分支保持。技术栈为实际Pod PyTorch2.10、Gloo CPU与现有训练链，不新增平台。

- [x] 两项必要loss测试实际RED1fail/1error后GREEN2/2：独立标量softplus/BCE值及解析逐logit/共享bias/age梯度，覆盖8×56、平移后BCE变化、极值有限、旧分支值/梯度；真实两rank WORLD/DDP含一个rank零正例，逐global-logit及共享参数梯度与独立reference一致，无额外world因子。首次传输尚未完成的缺目标exit2另保留，不当RED。
- [x] 最小实现范围：`runtime/bundle/retrain/classification_loss.py`新增精确独立loss版本与目标；`experiment91.py`两候选/seed/互斥/保存选点绑定；`tools/validation/check_candidate.py`仅消费新loss合同；`tests/test_pairwise_bce_mix_loss.py`两目标及现`test_candidate_binding.py`扩两配置；`configs/R177-SCIENCE.json`、`configs/R178-SCIENCE.json`。config/engine/distributed/model/optimizer/EMA/persistence/inference/launch/monitor实现不改，现有tuple调用点直接复用。
- [x] 3实际目标/5源compile、真实两public fresh及身份固定；R177初态实测5f2abba3，R178实测50c与R107一致。[CPU证据](../../records/preparation/pair-bce-mix-20261102/README.md)区分真实无mock元数据/公开字节与Gloo小fixture；旧模型/WORLD八实现按相同SHA复用证据。
- [x] Git313d5ab/tree1dd0d3cb已push/实际回读，两technical包在R176训练期间完成，源/SCI及原任务回执与输出SHA一致，guard0；未normalformal/clock/GPU。
- [ ] 收尾前完成唯一正常CPU/精确部署，actualsafe后按完整结果串行一次activation/direct launch，真实八rank/自然首成功接管GPU可运行证明；不加GPU预演。临近收尾缩短只读观测间隔，文档归档不阻塞接续。

完整R178严格超过两个锚点才有后续固定seed27复核资格；官方仍完整selected≥.996才一次，官方不参与准备/选点/调参。raw .999未达。
