# R101：按预定分支复核R95，仅改变训练seed

2026-10-29按[R100预定分支](R100-DECISION.md)执行：已核R100完整内部16条和训练终态，训练正常完成但所选主指标未超过R95，因此停止扫gamma/alpha，转为原R95/BCE基线的固定seed复核。直接对照为R95，唯一训练科学delta为`job.seed: 20261027 → 20261029`；不是从R100 focal继续训练，也不是按结果选择一个表现更好的seed。

精确配置=`configs/R101-SCIENCE.json`；profile=`single-tiny22k-bc48a87f-fixed-9010-24k-earlystop-seed20261029-v1`；candidate=`r101-convnext-tiny-imagenet22k-to-1k-24k-p4-seed20261029-9010`；order11。A已确认同一身份并行准备，此页冻结时R101尚未运行。

## 分支触发的实际证据

规划文本核原生freeze全部16条dev_entries、selected及TRAIN_RESULT。R100训练组件exit0、hard_deadline_met=true、无signals，实际6400成功更新、8完整周期、skip0，c4产生内部峰值，c5–c8连续4周期未刷新，正常early_stopping。06:52原观测中官方原一次进程仍活跃，整轮资源尚未安全交接；训练证据可用于科学准备，但不授权新GPU并发。

| 唯一内部选择 | R95：BCE | R100：focal gamma2 |
| --- | --- | --- |
| 实际成功更新／周期 | 5600／7 | 6400／8 |
| 选择位置 | c3 EMA@2400 | c4 EMA@3200 |
| AUROC | 0.9866645153288838 | 0.9851617898327616 |
| AP | 0.8193841805300731 | 0.8337256490854664 |
| error_twice（越低越好） | 22150 | 24646 |

R100各周期online/EMA内部AUROC如下，显示至6位小数；实际选点和早停仍使用原精度。

| 周期 | online | EMA |
| --- | --- | --- |
| 1 | 0.974609 | 0.974609 |
| 2 | 0.978125 | 0.978125 |
| 3 | 0.983134 | 0.983820 |
| 4 | 0.983666 | 0.985162 |
| 5 | 0.981435 | 0.984602 |
| 6 | 0.978982 | 0.984634 |
| 7 | 0.976086 | 0.984005 |
| 8 | 0.973350 | 0.981142 |

focal的所选AP较高，AUROC/error_twice较差，不能称所有指标均退步；c6 EMA比c5略高，但仍未刷新c4，不能把后段说成严格单调下降。实际停止步数不同，不称等计算量比较。本结果只检验这个固定focal配方，不能否定全部损失设计或推断易负例、错标等根因。依据在官方结果之前确定，不读取官方数值触发分支。

## 复核目的及保留合同

训练seed20261029已在R100结果之前指定。R101用于观察原R95配方对训练随机性的敏感性，补充此前只有seed20261027的证据；不搜索seed、不按新结果挑最高seed。`job.seed`会共同影响native fresh head、样本顺序、增强及模型随机性，不能解释为只改变head初始化。

固定9:1划分及全部清单不变：`data_split.seed=20261024`、fit46256/dev5140、同manifest/FREEZE、同fit-only age统计；不因训练seed变化重划数据。原R95的loss字段缺失，沿现有损失入口直接使用旧BCE，不带R100的focal配置。恢复并保持dropout0.2、backbone1e-4/head1e-3/age独立1e-3、SD0.1、原WD/clip、公开Tiny22K→1K来源与权重、LN1e-6、512/FP32、G64/P8、8rank各8/accum1和原增强算法。相对紧邻R100还包含损失返回BCE，不能把R100→R101称为单变量比较。

最大24000成功更新、每800一周期，完整周期max(online,EMA)内部AUROC严格刷新才清零，首轮baseline，patience4/min_delta0/min_cycles1；warmup与EMA warmcopy绝对1600、首blend1601、cosine horizon24000/尾倍率0.01、EMA decay0.999均不变。公开fresh初始化、零age、空Adam、online=EMA，不加载R95/R100或其他训练checkpoint。

训练seed改变后，整体fresh初态tensor身份须由A通过现有CPU工厂真实生成并记录，然后填入SCIENCE再同步；不得复制旧eba53fee控制值、填空跳过比较或借用旧checkpoint。新hash属于seed带来的派生身份变化，不是额外科学超参。现有正式包和实际8rank初态继续核对新的控制值，无需新身份桥。

结束后分别报告R95与R101各自预定选择规则得到的内部结果、AP、完整周期轨迹及停止步数；不得改为从两次运行中挑最高seed来宣称复现成功。两个seed不足以估计可靠的尾部变异、证明显著性或消除固定dev反复选择的偏差。若差异与此前候选变化同量级，应降低对局部排序的确信，不自动启动更多seed或数值扫描；若接近，也不证明其他配方在各seed上保持排序。后续策略另据实际证据决定，不承诺收益或0.999。

每轮仍在其实际完成周期的online/EMA中按error_twice/AP/update/EMA规则选唯一checkpoint；完整官方5849仅一次最终独立分报，不参与早停、选点或后续调参。

## 最小接续与来源

A负责seed期望及运行名从绑定SCIENCE消费、新实际初态控制、候选身份及改动必需的现有窄验；现有BCE路径已具备，不重focal数值套件、来源、迁移或native验收，不新增模型诊断、平台或独立GPU预演。仍使用现有exp-v3数据/缓存/公开权重，无新资产、旧模型加载或N06来源。准备20%与训练80%按实时有效CPU、当前预算TTL和互斥mask并行，旧allocation和冻结运行不变。

必要CPU证据齐后由A统一一次Git同步，按实际commit/tree/文件SHA冻结部署，核R100原一次官方及整轮资源安全退场后一次fresh启动。不得重发官方、重置旧clock/token或把观察unknown写成安全退出；不增加文档批准门或归档等待。

R100冻结Git=`b8126fdc77fb81a615ea5f4b638275b148bd7345`。原件根`/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r100-tiny22k-focal2-24k-p4-9010-20261029`，已普通导回工作区`work/exp-v3-8card-review-20261028/r100-update/r100-terminal-export/`。以下SHA来自既有Pod回执，本机未重哈希：

- R100 freeze：`e4a0af3f0a17cdc7b20a7332cf2b78d386fb1a600583bbbe0b8120aab3916158`；TRAIN_RESULT：`c9818cf93f13ed07036ddbfe8d5fff0b7c11955ac86af08bd5716762b7b118f7`。
- R100所选tensor：`5a39fbcbc8c9fd37c17924c5986f17a716ee7d9f56273a5fe7073fa856d80745`，仅结果身份，不能作R101初始化。
- R95 freeze：`13aa989198ca9b4e64ed5e021f6da9b8733eda748878bd5750d19e2ebccc217b`；TRAIN_RESULT：`dd253b3aa0fdd33bb6a539f01ba596d7f5520dbea1547d543787fdae69f306c1`；直接基线为`configs/R95-SCIENCE.json`。

规划本次仅新增本文件，冻结交A与必要实现/结果统一集成，不追逐过程状态另作提交。
