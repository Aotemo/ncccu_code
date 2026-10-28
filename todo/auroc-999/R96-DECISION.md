# R96：在24k/四轮早停合同下将backbone LR减半

2026-10-29科学决定已交A直接实施，非等待批准的候选：以完成的R95为对照，只将`job.backbone_lr`从`0.0001`改为`0.00005`。配置为`configs/R96-SCIENCE.json`；profile=`single-tiny22k-bc48a87f-fixed-9010-24k-earlystop-lr5e5-v1`；candidate=`r96-convnext-tiny-imagenet22k-to-1k-24k-p4-lr5e5-9010`；order6。名称中的lr5e5指5e-5，以配置数值为准。此页写定时R96尚未运行；A复用现有注入/打包/启动链，不等文档批准。

## R94→R95内部事实

规划仅文本读两轮原生freeze及TRAIN_RESULT，不读官方文件、不运行解析脚本或新验证。R95完整14条为c1–c7的online/EMA各一条；TRAIN_RESULT为completed、7周期、5600成功更新、组件exit0、`bad_cycles=4, stopped=true`，reference AUROC为0.9866645153288838。第4–7轮两variant均未超过第3轮EMA峰值，符合每轮max(online,EMA)及连续4轮无提升的已定合同，不是预算未满导致失败。

| 唯一内部选点 | R94：6400固定预算 | R95：24000上限/四轮早停 |
| --- | --- | --- |
| 实际完成成功更新 | 6400 | 5600 |
| 选择位置 | c3 EMA@2400 | c3 EMA@2400 |
| AUROC | 0.9867415781748388 | 0.9866645153288838 |
| AP | 0.8200017145302917 | 0.8193841805300731 |
| error_twice（越低越好） | 22022 | 22150 |

各轮内部AUROC展示到6位小数，选点使用原件精度：

| 周期 | R94 online | R94 EMA | R95 online | R95 EMA |
| --- | --- | --- | --- | --- |
| 1 | 0.973097 | 0.973097 | 0.973097 | 0.973097 |
| 2 | 0.985323 | 0.985323 | 0.985323 | 0.985323 |
| 3 | 0.982686 | 0.986742 | 0.982684 | 0.986665 |
| 4 | 0.977327 | 0.985427 | 0.977746 | 0.985536 |
| 5 | 0.976758 | 0.982403 | 0.977276 | 0.983334 |
| 6 | 0.970528 | 0.978899 | 0.970660 | 0.981346 |
| 7 | 0.973012 | 0.975809 | 0.968865 | 0.979318 |

R95前两轮与R94同值；c4–c7 EMA较R94同期高，但未刷新自身c3，最终选点主指标和AP均略低于R94。c7 online还低于R94同期，不能称后段全面改善。R95 c4 EMA AP为0.8246942558701894，虽高于所选c3 AP，error_twice为24024，仍按原主规则保留c3。不能把辅助online-best字段当最终选择。

R94→R95同时改变最大预算、cosine horizon与早停政策，不能单独归因于“多训练”或早停，也不能推断若不早停跑满24000会怎样。R95实际并未训练到24000；上述比较只描述已完成轮次和各自规则选出的checkpoint。R93→R94同6400预算下LR减半的小幅改善见[R94记录](R94-DECISION.md)，三轮最佳内部位置均为c3 EMA。

## 单一下一步与边界

选择再减半backbone LR，是沿已有LR对照检验更慢的预训练参数适配，而非根据官方结果选参。R93→R94的内部改善很小，R95新曲线没有提高所选峰值；这足以形成一个可执行假设，不足以保证5e-5更好、诊断过拟合或给出最优LR。现有证据未定位head、WD、EMA或增强的独立问题，本轮不同时改变这些轴，也不再增加训练上限。

除backbone LR与新实验身份外，R96完整沿用[R95合同](R95-PLAN.md)及版本化SCIENCE：最大24000成功更新、每800一周期、内部max(online,EMA) AUROC严格刷新历史best才清零、首轮baseline、patience4/min_delta0/min_cycles1；绝对warmup/EMA warmcopy1600、首blend1601、cosine horizon24000/尾倍率0.01、EMA decay0.999；head/age LR1e-3、原参数组/WD、固定fit46256/dev5140、512/FP32、G64/P8、同公开权重/seed/原生fresh初态eba53fee控制值。final LN仍归backbone，AdamW收缩随LR变化，不能声称仅改变卷积梯度项。不加载R95或任何历史训练checkpoint。

R96与R95保持相同最大预算和停止政策，实际停止更新数可因轨迹不同而不同，不要求两轮恰好同为5600。完成后按实际全部周期的既有error_twice/AP/update/EMA规则选唯一checkpoint，再对照内部选择及轨迹；完整官方5849仅一次最终分报，不参与停止、选点或后续调参。单seed/单固定划分不能支持稳定性或显著性结论；raw AUROC未达0.999不称总体完成。

A仅改新SCIENCE、显式candidate/profile/order及必要说明，沿现有消费配置的入口做改动必需的窄验证，不散写新LR常量，不重来源/迁移/模型/native或旧套件，不新增脚本平台或独立零forward GPU预演。现有墙钟（train hard10800、total16200）及集中资源策略保留；实时身份/资源和新Git/tree实际冻结后一次fresh activation/launch。A已核R95安全终态与资源交接，owner原Z状态保留为Z，不称reaped。全部旧运行/clock/token不改；A唯一Git/GPU，准备20%与训练80%互斥并行。

## 原件身份

R95权威根为`/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r95-tiny22k-24k-p4-9010-20261029`，冻结源码为`992c3de8a0793083d1bbd9d759d7e8655e015d44`。A已普通导回工作区`work/exp-v3-8card-review-20261028/r95-update/r95-terminal-export/`，本机未重新解析或哈希。

- R95 `SINGLE-MODEL-FREEZE.json` Pod SHA `13aa989198ca9b4e64ed5e021f6da9b8733eda748878bd5750d19e2ebccc217b`；`evidence/TRAIN_RESULT.json` Pod SHA `dd253b3aa0fdd33bb6a539f01ba596d7f5520dbea1547d543787fdae69f306c1`。
- R94对应freeze Pod SHA `9baf741474085c098170a7dcff60996f9246d60d73e123366fe408dd0ba5a2fb`；TRAIN_RESULT Pod SHA `6ab8146f6c8216f7dd1ed12a3e14ee2e1bb9221166dafdbbda2f58c42b5db614`。原件与精简结论由A统一归档，不为此另造实现提交或等待。
