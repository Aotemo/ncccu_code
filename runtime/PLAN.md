# R92 ConvNeXt Tiny ImageNet1K 实施规格

状态：`TEXT_SPEC_READY_FOR_A_IMPLEMENTATION`。本件细化已批准的第二个单模型，不新增科学试探或审批门。规划仅阅读文本；未运行模型、解析器、哈希、测试或回执校验。执行步骤见 [NEXT-TINY1K-IMPLEMENTATION.md](NEXT-TINY1K-IMPLEMENTATION.md)。当前准备只生成独立包、窄CPU证据和正式入口模板，不创建正式clock/token，不启动GPU。

## 1. R91短复盘与接续依据

已阅读执行A [R91-REPORT.md](../execution-A/R91-REPORT.md) 和 [FINAL-RECEIPT.json](../execution-A/stage4/evidence/audit/FINAL-RECEIPT.json)：其Pod最终检查20/20、真实owner/ranks/components exit0、原锁释放；这是消费A原件，不是规划独立重验。唯一EMA cycle5/update4000，内部5140 dev AUROC `0.9732399267421321` / AP `0.7510259793479146`；官方完整5849 AUROC `0.9746659041823744` / AP `0.7372105728953852` / error_twice54756，精确目标false。R91证据交付完成，总体0.999目标未完成。

内部曲线中EMA第5周期最佳，之后下降，说明保留16条候选和保护真实最佳blob仍必要；不据此改变已批准周期、LR或EMA。没有读取官方逐样本错误作研究。按既定次序接Tiny1K；旧R91目录、token、冻结、reservation、预测与checkpoint原件保留。其已释放快照允许准备后继，正式启动仍核当时现场资源。

## 2. 唯一候选与公开fresh

候选为 `convnext_tiny`、ImageNet1K、实验序号 **2**；建议绑定版本 `single-tiny1k-6166b761-fixed-9010-v1`。A可沿用内部接口名`experiment91`以减少改动，但绑定内容必须唯一标识R92/Tiny1K；不能凭非空字典启用宽松通道。正式round/root/run_name由A清单给出，不把建议名称当已存在路径。

| 项目 | 固定合同 |
|---|---|
| 发布者/版本 | `facebook/convnext-tiny-224`，revision `6166b7613034066690a621d8bf25ffdf181a34f0` |
| 已有converted文件 | `/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc998-20261025/coordination-huazhong8-20261025/executor-preparation/public-weights/converted/convnext-tiny-tv-6166b761.pt` |
| converted SHA256 | `d1b2c3bdd577641ff1183253c43c8c835be373d809777957ef8f569a4c303f47` |
| 原始来源SHA256 | `1fe15b2eba9e355de14c4e244e6365bf61ffef84da7d3442c0afbb2d8564f477`，由已有provenance绑定 |
| provenance / license SHA | 既有记录分别为 `a26704436878d4bc4654a62c711c8faa4cceb65eb77aeacfc66d7ac9419d7daf` / `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`；准确路径沿已有来源清单/`license_file`解析 |
| 初始化顺序 | native完整1000类工厂 → 实际converted public strict → 替换binary head → 新age；不继承R91/r15/r16训练参数、EMA或Adam |
| 网络 | 原生avgpool → 原末端LayerNorm/flatten（768特征）→ Dropout(.2) → 单Linear(768,1)，加同一新fit的两维age线性残差 |
| LN | **末端eps1e-12；内部LN eps1e-6**。末端eps不在权重tensor中，必须核实际模块与job/provenance，不能只验state_dict |
| 训练结构 | 原生stochastic_depth .1，来自既有Tiny工厂/历史科学配置；无BN、无q、无attention、无双头 |
| head/age初始化 | seed20261027；沿既有Tiny工厂`nn.Linear`原生初始化，不挪用R50的normal×.01；age_weight两个FP32零值、无bias；online=EMA、Adam空 |

来源依据为 [预训练交接](../../continuation-20261027/execution-B-pretrain-inventory-9010/HANDOFF.md)，以及历史 `work/auroc998-20261025/coordination-huazhong8-20261025/r15/SCIENCE_CONFIGS.json` 和 `r16-bundle/retrain/models.py`。复用已有转换/映射证明，不重新下载或跑转换suite。新Pod实际工厂仍须核实际来源、完整strict、形状/dtype/finite、加载后公开tensor对应、实际LN/路径；不能用名义config代替。新初始tensor SHA、实际参数tensor/元素数、完整source map与空Adam证据均由A实际构造产出，本规格不填估计数值。

## 3. 同一数据与训练合同

直接绑定stage1服务器原件根：`/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/technical-prep/resume9010-A-stage1-20261028-attempt1/result`。`FREEZE.json` SHA `90e063dca3ab22225308043ba4b0de379688675ef00606d00b8b36e2135bc0c6`；manifest blob/canonical、fit/dev indices、AGE和SCHEDULE身份完整沿 [FIRST-R50-SPEC.md §1](FIRST-R50-SPEC.md)。不重划、不重新拟合age、不改stage1原件。

| 项目 | R92固定值 |
|---|---|
| fit / dev | fit46256=44766N+1490P；dev5140=4973N+167P；仅原train派生 |
| 年龄 | 原AGE合同SHA `325b47f8898b35e22ec9b1bb061e506e0af4baf47dce4f6c1250024428666d82`；median57、mean56.743204376878666、population_std10.190648068012752；两维原定义不变 |
| 输入 | prepared512/image512、共享原几何/增强；ImageNet mean[.485,.456,.406]/std[.229,.224,.225]；dev/官方identity |
| 拓扑/数值 | 四rank×16、accumulation1、global64、global P8/N56；FP32原协议，seed20261027 |
| 优化器 | 既有AdamW；backbone **2e-4**；末端linear和age **1e-3**；head_lr_scope=last_linear；末端LN仍属backbone |
| 其他优化项 | 矩阵WD1e-4，bias/normalization向量WD0，age WD1e-4；clip1、BCE、noMixup，其他既有默认记录于resolved job |
| 进度 | S800、8cycles、6400成功更新，warmup1600、cosine尾倍率.01、无早停、stop_after_updates6400、无skip |
| EMA | FP32、warmcopy1600、decay.999、首blend1601；整数buffer copy；无R50冻结BN分支 |

Tiny候选单独产生EMA合同（建议version `ema-warmcopy1600-decay0999-v1`），内容为warmcopy_updates1600、decay.999、dtype=torch.float32、floating_buffers=same_as_parameters、integer_buffers=copy，**不带frozen_bn_buffers=copy**。从同一SCHEDULE派生并绑定新SHA；不要修改或冒用`EMA-R50.json`。fresh、runtime、checkpoint、validate_rule、validate_state、update、freeze用同一合同；最终计数update6400/copy1600/blend4800。数学沿现有实现，不重跑R50边界suite。

## 4. 必须改的身份门与复用的行为

| 门 | R92必须绑定/检查 |
|---|---|
| 候选contract | 模型、公开权重/provenance、LN eps、SD.1、LR、no-q、age/split/schedule、新EMA、序号2；拒绝R50绑定、旧8:2和任意未注册candidate |
| fresh/参数 | 实际public strict后tensor与CPU证据一致；四rank同一新初始tensor、FP32、零步、空Adam、age零；参数组以实际named_parameters清单为准。实际0个BN模块/0个BN运行buffer；不能要求53/159/162或head2049、旧4097 |
| initial barrier | 不等待、不读取不存在的`bn-initial-ranks.json`；启动门和训练完成复核两处一起适配。使用真实initial-ranks、parameter-groups、native initial checkpoint及actual EMA核对后释放 |
| 内部dev与最佳保护 | Tiny也执行原精确`error_twice`指标与prediction SHA/actual tensor记录；online/EMA保护按error_twice升、AP降、update升，跨variant最后EMA优先。不能落入旧Tiny浮点AUROC分支 |
| 唯一freeze/strict | 完整8cycles/6400/0skip与16×5140；锁candidate/order2/source/job/split/age/schedule/EMA/input/LN、真实variant/blob/tensor和dev预测。以真实无BN语义替代fixed_bn摘要门；不降低strict/identity完整性 |
| 官方入口 | 复用`single_model91.py`固定单模型入口，CPUfixture拒绝；canonical训练根reservation在实际模型构造/官方数据读取前落盘；先strict actual variant/tensor，再官方完整5849一次。不得通过换output_root/复制freeze再forward |
| owner/退出 | 复用四卡独占串行、前驱exact PID/start_ticks、原instance锁、资源/cleanup/真实OS exit；前驱换为真实R91 owner及train/freeze/official组件和ranks，不能仍检查N15 |

本轮不导入旧融合gate，不复活C1/J0/N15空间q/head门。普通Tiny工厂、optimizer分组、数据/FP32/持久化、精确选择排序、reservation和owner流程尽量原样；仅扩展确被R50硬限制挡住的调用处。不能用删除校验来实现无BN支持。

## 5. 墙钟预算草案与现场冻结

最相关成本原件为 `work/auroc998-20261025/coordination-huazhong8-20261025/r15/evidence/terminal-artifacts-1792975476/r15/outputs/r15-4c-tiny-a25-b64-nomix-fp32-ema/convnext-tiny/complete.json`：Tiny1K FP32、6399成功更新，training3293.714783396106s、evaluation235.5762847061269s、checkpoint74.29952283250168s、process wall3642.6964304861613s。历史为9cycle/旧10277dev，不能视作本轮测量；本轮6400更新、16×5140，保守仍按约一小时量级并给初始化/I/O/负载余量。R91实际train1691.69s不代表Tiny吞吐，**不能照抄trainhard3600（soft3450）**。

| 草案项 | 秒 |
|---|---:|
| total | 12600 |
| binding | 1800 |
| train hard（含初始化、8cycle、16dev、保存） | 7200 |
| freeze hard | 300 |
| official hard（单设备完整5849） | 1200 |
| cleanup及各component soft margin | 150 |

此为供A按当时Pod可用资源/租期冻结的预算草案，不是已创建clock或已验证Tiny耗时。无需新增GPU benchmark；复用已有成本并记录选择依据。正式必须一次固定T0/D/H/token，soft=H−150；不静默延长、不重置、不缩cycle/batch补预算。

wrapper中R91硬编码`D−1200`/`D−750`/freeze300须与新policy一致。最低保留尾窗为：`train_H=min(train_start+train_hard, D−freeze_hard−official_hard−cleanup)`；`freeze_H=min(freeze_start+freeze_hard, D−official_hard−cleanup)`；official仅当`D−cleanup−now >= official_hard`才启动；各phase另扣soft150。A可额外保留overhead，不能少于上述尾窗。正式准入仍须核实际Pod UID/Ready/Python依赖、四卡PCI/占用、RAM/VRAM/磁盘、R91已退场和原锁；CPU准备不获取训练锁。

## 6. 交付、失败与后继

- [ ] A在独立technical-prep合并交付最小源码diff、完整部署清单/Pod实际SHA、实际Tiny工厂/初始状态与参数组证据、窄CPU逐项回执及真实exit、正式wrapper/initial gate/预算模板；新数值身份不得沿用R91。
- [ ] 规划仅核本轮实际小diff与回执文本后由父代理接正式绑定/准入；本件不冒称实现通过。R91终态依赖已由真实交付满足，不再等待旧元数据。
- [ ] 正式完成后保护全部16条内部曲线并冻结一个对象；官方5849=191P/5658N一次，raw AUROC/AP/error_twice与内部结果分列，精确判定`1000*error_twice <= 2*191*5658`。
- [ ] 技术失败/资源unknown/official_unknown保留原件和同token状态，不记科学未达、不重新activate/官方forward；完整官方原始预测已有时只在Pod补对账，不重跑模型。CPU失败只修直接原因，不扩展旧suite。
- [ ] Tiny1K正式成功但目标false，按既定次序接Tiny22K→1K；本件不同时展开其实施。任一精确达标且身份/退出回执齐全即停止新实验。三项均未达才提出有证据的新单模研究计划，不承诺达到0.999。
