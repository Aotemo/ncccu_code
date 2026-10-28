# 首轮 R50 ImageNet1K V2 实施规格

状态：`SPEC_READY_WITH_STAGE1_ACTUAL_CONTRACT`。本件是已批准方案的首轮实施合同；规划只阅读文本，未运行代码或独立重验回执。阶段1数据就绪以执行A的 [STAGE1-REPORT.md](../execution-A/STAGE1-REPORT.md) 和真实Pod回执为依据；首轮模型、训练及官方指标仍待执行。步骤见 [PLAN.md](PLAN.md)。

## 1. 已冻结数据原件

服务器根：`/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/technical-prep/resume9010-A-stage1-20261028-attempt1/result`。下表路径均相对此真实根；SHA按执行A报告逐字记录，未在规划端计算。

| 文件/身份 | SHA256 |
|---|---|
| `FREEZE.json` | `90e063dca3ab22225308043ba4b0de379688675ef00606d00b8b36e2135bc0c6` |
| `manifest.json` blob | `7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377` |
| manifest canonical | `3275aaa35d24fe72cf7d867ec8c6e3b0b9ff9f12226a0fb9b0ada516009c9669` |
| fit indices canonical | `82da87177456d5cec5070fee459dcd3c0fe7ff464e1a38904624efa9faa57c62` |
| dev indices canonical | `c466b35ba9c4516a7343a1c9b571c21597e46f9a3c2c53541026bfa96aea1c63` |
| `AGE.json` blob | `16fe022319a48c95aa1167e6efd39ef188fbac7aae75cda0aad048420aecf43e` |
| age contract canonical | `325b47f8898b35e22ec9b1bb061e506e0af4baf47dce4f6c1250024428666d82` |
| `SCHEDULE.json` blob | `5ec732efe02b9bacda45440a00a784d4177e8dbbcc4399b85c47da996b1d50b9` |
| schedule canonical | `1fde0ef536d77f70716c673c88d1483a8f2d328ed2a148f5a4bd4ef5fb1bb840` |
| `EMA-R50.json` blob | `6512e18e7331c7ae0e7016825705df9a85ec875fac8818b6c992c351a47bcda9` |
| EMA-R50 canonical | `0986a28b67ef1f5feec63c2fa9e3b4e0d5efa88f11f0164623a4b6528ef39d49` |

`CONTRACT.json`及其余文件由同根FREEZE列出，执行只从清单取得准确SHA，不猜文件名。阶段1实际回执的本地查看入口为 `../execution-A/stage1/runs/attempt1/evidence/verification/OUTPUT-MANIFEST.json`；verification receipt SHA `d28d9175ba70edfc5a6977b1a39c317f156d45d15327eb74682a80ae83e487bf`。完整manifest和ASSIGNMENTS留在Pod，不为规划下载/解析。

| 数据 | 行数 | Negative | Positive |
|---|---:|---:|---:|
| 原train | 51396 | 49739 | 1657 |
| 新fit | 46256 | 44766 | 1490 |
| 内部dev | 5140 | 4973 | 167 |

split seed20261024，holdout.1、include_valid=false、`content-group-label-age-size-v1`，实际dev比例0.10000778270682543。索引/name/content-group/byte-SHA互斥且完整覆盖已在阶段1检查；后续绑定原件，不重跑整套生成。未发现可靠患者字段；内容组完整不代表患者/双眼独立。

新fit年龄：valid46243、missing13（empty9/outside0_1204）、median57.0、mean56.743204376878666、population_std10.190648068012752。遵循原`age-linear-v1`，有效区间0<age≤120；缺失以fit median代入后标准化，并保留missing_indicator。两个特征为`standardized_age, missing_indicator`，新age_weight为FP32两个零标量、无额外bias，LR1e-3/WD1e-4。dev/官方只应用此合同，不重新拟合统计。

## 2. 公开初始化与实际模型

- 唯一预训练权重：`/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/technical-prep/n08-r50-imagenet/resnet50-11ad3fa6.pth`，SHA `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`。
- 来源记录：`/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/rounds/N08/assets/PUBLIC_ASSET_PROVENANCE.json`。复用既有来源/许可材料；阶段1仅metadata，不是本轮权重strict数值通过。fresh初始化时正常校验此实际文件并strict一次，无需重复下载或旧来源suite。
- 代码从普通R50 reference adapter派生，真实服务器根见PLAN。保留完整1000类ResNet50的320-key strict加载、shape/dtype/finite及加载后相等检查，然后`fc=Identity`，保留原avgpool/2048特征。
- 二分类路径：`avgpool → flatten(2048) → Dropout(0.2) → Linear(2048,1)`，加两维年龄线性残差，输出一条raw logit。无q、无attention、无双头。沿普通工厂head seed20261027、CPU generator normal×.01、head bias0；不接旧训练checkpoint。
- 预期实际训练参数：backbone23508032、head2049、age2，合计162个可训练parameter tensor；这些是准入期望，必须用真实named_parameters及optimizer成员证明。旧head4097/163-tensor空间模型门不能沿用。参数tensor计数不等于含BN buffers的state_dict计数。
- 53个标准BN始终eval，159个`running_mean/running_var/num_batches_tracked`固定为本公开权重值；affine可训练且归backbone LR。每次model.train()保持上述语义，不转换SyncBN。
- EMA是同一训练的候选读出，不参与优化器。正式初始化online/EMA均fresh一致，Adam状态空；禁止从N06 FLAIR、N08/N13/N15训练结果或teacher继承参数/BN。

## 3. 输入、优化与调度

| 项目 | 首轮冻结值 |
|---|---|
| 几何/增强 | 共享prepared512原几何与训练增强，image_size512/prepared_size512；验证identity视图 |
| normalization | RGB/255，mean[.485,.456,.406]、std[.229,.224,.225]；`normalization='imagenet'` |
| 拓扑 | 四rank×16，accumulation1，global64，global positives8/negatives56 |
| precision/seed | FP32，seed20261027；沿用原FP32精度协议并由实际加载/预测回执证明 |
| optimizer | 原AdamW；backbone2e-5，末端linear及age1e-3；clip1、BCE、noMixup |
| WD | 原矩阵参数1e-4，bias/normalization向量0；age独立1e-4。核对真实parameter groups，避免分类头作用域吞入其他参数 |
| S | native sampler已经实测800，等于ceil(44766/56) |
| 科学预算 | 8cycles，6400成功更新，无早停 |
| LR | warmup1600；原cosine尾倍率.01；使用成功进度，保留原下一成功更新的边界语义 |
| EMA | FP32，warmcopy1600，decay.999，首blend1601，integer buffers copy，frozen BN始终copy |

共享512任务几何不等同于原作者224/288分类评估全套变换；不可静默使用`rgb01/flair_rgb01`。原优化器其余默认值由实际torch optimizer/resolved job记录，不新增学习率试探或官方调参。

EMA规则以实际`EMA-R50.json`为准，version为`ema-warmcopy1600-decay0999-frozenbn-v1`。注册、更新和state validation三处一致；不能仅替换version却仍以旧`rule==FROZEN_BN_RULE(1422)`选择BN分支。

| 成功更新游标 | EMA计数/行为 |
|---|---|
| 1600 | update_count1600、copy_count1600、blend_count0；浮点参数copy |
| 1601 | update_count1601、copy_count1600、blend_count1；普通浮点参数`.999*ema_prev+.001*online` |
| 所有时点 | 159BN逐位等于online及公开初始化；integer copy；immutable age不变 |
| skipped | 不推进successful/EMA；正式训练沿用既有skip停止门，不能累计成科学完成 |

## 4. 最小受影响检查与完成门

只做PLAN的三个检查组：①实际fresh/参数组/新合同checkpoint往返；②1600/1601及159BNcopy；③唯一选择与官方入口身份正负例。都在指定Pod CPU隔离执行；不重复阶段1、历史来源/映射或旧模型全套suite，不用官方样本测试入口。可以一runner复用模型和加载结果。

新规则、初始tensor/BN SHA、全部checkpoint路径、官方实际预测与墙钟预算目前均无本轮通过事实，不能从本规格推定。每组需真实exit和逐项回执，部署需实际源文件清单和Pod SHA。既有RGB成功件名字为`RGB01-CPU-RECEIPT-A2.json`，只作历史参考，不证明本ImageNet输入首轮通过。

## 5. 16条内部dev到一个真实对象

1. 完成8cycle/6400成功更新后，检查16条`(cycle1..8, online|ema)`，每条完整5140/167P/4973N、update=800*cycle、finite raw logits、唯一样本顺序/覆盖，不能采样dev。
2. 精确排序：`error_twice升 → AP降 → successful_updates升 → EMA优先`。不按显示AUROC舍入排序；best_online/best_ema保护逻辑必须与本规则一致，使最终winner对应blob可取。
3. 选择后只冻结一个对象。冻结件至少绑定：round/experiment_order；resolved job与source文件清单SHA；公开asset/provenance身份；split manifest blob+canonical/fit+dev indices；age contract；schedule/EMA rule；normalization/geometry/precision；cycle/update/variant；checkpoint实际路径+blob SHA；selected variant实际tensor SHA；内部预测路径+SHA+精确指标；R50固定BN身份。
4. 冻结门交叉检查checkpoint payload、strict reload得到的actual状态及内部预测记录。名字、配置或只验证blob均不足；同blob内online/EMA不同tensor也须识别。缺字段、错variant或任一身份不符即拒绝。
5. 单模型入口移除C1/J0/J1旧候选决策、reference融合向量、dev冠军门与N15空间参数门。保留来源、strict loader、checkpoint完整性及owner守卫；不为绕过旧常量一起删除安全和身份校验。

## 6. 官方一次与目标判定

- 完成冻结后，仅该单模型、该variant、该blob/tensor在identity视图/FP32评估完整官方5849一次。实际核验191P/5658N、唯一全覆盖；原始logit保存完整精度并计算原生指标，不以sigmoid舍入值、融合或`metrics=None`替代。
- 官方读取发生在单模冻结门通过之后；不能用官方在同一训练的16个checkpoint中择优。内部dev不先超过旧冠军也可进入最终官方。
- `error_twice=2*逆序正负样本对数+同分正负样本对数`，`raw_AUROC=(2*P*N-error_twice)/(2*P*N)`。本次完整类别核验后，精确达标条件为`1000*error_twice <= 2*191*5658`。结果同时记录整数、raw AUROC和AP，不能因99.9%显示舍入宣称达标。
- 比较不同完成实验：官方raw AUROC降序，AP降序，较早实验序号优先。内部dev指标单列；官方技术失败不记为模型科学未达，未知不记completed。
- 任何官方恢复只沿同一冻结对象和原attempt事实查证；完整预测已存在则优先只补指标计算，不重新forward。禁止为试分换checkpoint、改变归一化或回到训练更新。

## 7. 执行 A 尚需现场补齐

- [ ] 实际stage2/正式root与源码部署SHA、三组CPU回执，以及真实fresh/optimizer/BN/tensor身份。
- [ ] 当前Pod/四卡/锁/前驱进程/RAM/VRAM准入；阶段1无锁持有者的快照仅为历史观察。
- [ ] 实际6400更新及16次内部dev、官方评估、清理的墙钟预算；新T0/D/H、soft=H−150、token与owner。不得复用旧N15 clock/token或将其1507.87秒当新预算。
- [ ] 训练terminal、唯一checkpoint冻结、官方raw指标与真实全覆盖回执。

未达且正式完成后按既定次序交父代理接Tiny1K，再Tiny22K→1K；两Tiny backbone2e-4/head-age1e-3、末端LN eps分别1e-12/1e-6，内部eps均1e-6。当前不展开Tiny实现。任一官方精确达标且证据齐全即停止新增训练；三项均未达才基于内部dev/训练证据提出新单模研究计划，不保证一定达标。
