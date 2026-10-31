# R133：三处固定 Tri3 低通下采样

根在R132完整结果与actualsafe后依用户持续推进授权采用本轮；fresh授权reference为消息`429ca7f7-cb13-4e51-bb68-32ad1ba5892d`。A `7c17043e-8381-43e9-b185-6e142634ad72`独占Git/Pod/GPU，规划仅一次注册只读审查。当前正在身份接线与正式准备，尚未activation/launch或证明本轮GPU成功；执行后按真实小件接续。

| 字段 | 冻结值 |
| --- | --- |
| order/config | 44 / `configs/R133-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-tri3downsample-v1` |
| candidate | `r133-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-tri3downsample-9010` |
| round | `r133-small22k-seed20261029-scale085-pairlog-tri3downsample-24k-p4-9010-20261101` |
| prep | `exp-v3/technical-prep/r133-20261101-7c17` |

直接R107/seed20261029/公开fe6b358a，native1000 strict后按原生binary head/age初始化，step0/空Adam、原345张量全训。不加载R132/R131/R130或其initial，不拼曲线和历史成本。预期whole保持`50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，本轮normalformal实际重建证明，不另独立公开fresh测量。

唯一方法变化是job/candidate同一`TRI3_RULE`：version `convnext-small-tri3-downsample-v1`、targets `[2,4,6]`、convolution `native2-stride1-pad0`、filter `reflect1-box2s1-box2s2`。三处原LN/2×2 Conv保留对象/Wb/参数名；conv改stride1/pad0后严格ReflectionPad2d(1)→AvgPool2d(2,stride1,pad0)→AvgPool2d(2,stride2,pad0)，ceil_mode=False、普通均值、无divisor override，不融合。无新参数/buffer/RNG，stem4、36 CNBlock、head/age不变；512输入三处128→127→64、64→63→32、32→31→16，末层仍16×16。训练/dev/freeze/official同图。

SCI逐项继承R107，仅身份/note、两处policy、显式`OFFICIAL_GATE_0996`不同。R107原无的可选字段保持缺省，特别是`final_downsample_stride`必须不存在，避免按key存在消费的旧适配覆盖Tri3。无输出正则/LoRA/layerwise/冻结。固定9:1 fit46256/dev5140/splitseed20261024、prepared512/单视图scale[.85,1]/原旋转光度、G64P8/八rank/FP32-noTF32、原pairlog六字段及LR/WD/SD.1/dropout.2/globalclip1、24k成功/每800完整双variant dev/p4/1600warmup和原EMA/唯一selector。优化器/EMA/通信/保存算法不改，实际组数沿原验收，不造4/5硬门。

## 最小执行与证据

CPU-ready基线`e87e90eeed926b3129bc7795a6abdb2e165cb530`已有两项2/2、6源compile、721文件Git blob/mode exact和GREEN SHA匹配。[CPU归档](../../records/preparation/tri3-cpu-20261101/README.md)只证明独立CPU值/梯度、None兼容、random native345和显式toy_factory保存/inference三stage执行，不替代公开初始化、正式outerdescriptor、完整variant、GPU或standalone export。

本轮仅`tests/test_tri3_identity.py`一项正向窄验：旧R13243/新R13344真实SCI→candidate、精确policy/.996/原初态/缺省字段，以及除获准差异外整SCI等于R107。实际RED为未注册新profile导致1FAIL/0ERROR；补最小experiment91注册后，唯一GREEN已实际1/1、0error（0.231秒）、GPU未初始化。SCI SHA `8e57c0ba655a676e9f40061fa71ad14cdc5d46914ac001d1deaf18828f5c1ed2`，`NARROW-GREEN.json` SHA `bf46b1eecdd242ec4148d3b6e706ce4e9ab7ec75088861cf18c7dfafc5736188`、guard SHA `1a2b9f91042258a49d97c0e94fa8682e7ebd7b3cf2bab4f905fb6878ab8b1f1f`/exit0，原件留本轮prep。规划已完成唯一只读注册审查，无Critical/Important或待修Minor，不是额外批准门。不重Tri3两项/旧lossGloo/LoRA/fixture/fullsuite，不加wrapper检查或GPU预演。旧profile禁止非空Tri3；新candidate/runtime job/三处完整实际descriptor一致，descriptor沿原resolved_age_spec，现有inference strict后describe接线复用。

为使normalformal对无状态图的实证充分，只在其原`initial_EMA_exact_and_outside_optimizer`及`CPU_initial_checkpoint_strict_roundtrip_emptyAdam`两个positive check的Tri3分支补实际EMA、默认production factory重建模型的完整descriptor比较及inference返回描述核对；没有新增check、额外forward或独立验收轮。该正式行为尚待本轮唯一normalformal执行，不把identity GREEN代替它。

顺序为窄验→actual commit/nonforce push/readback→实际commit精确部署→当前身份/资源/CPU份额与独立训练grant→normalformal一次→deployment freeze→新鲜Pod快照下唯一activation/正确live launch。normalformal实际覆盖public fresh/正式outerdescriptor/onlineEMA一致和原strict checkpoint路径。GPU初次入口保持uncontracted affinity，由launcher自核绑定；SITE SHA在A12核出，正确`kubectl exec -i`，真marker存在即停，不删除重试、不热改冻结包或旧clock/token/grant。

当前672ddae2准备绑定仍有效，实际effective120时A12/0–11、根12–23保留未用、训练96/24–119互斥；nice19/隐藏GPU，阶段核UID/Ready/affinity/cpuset/quota/TTL。到期只另建future软件绑定。原6h训练/8h整轮/9h grant、Small12GiB/free20/floor16/300s不变，不扩资源/租期、不关实例、不新timer；资源不合格记技术阻断，不降batch/精度/上限绕过。

真实八rank初态接管345全训/实际命名组/空Adam/50c/三处结构/设备与80%整树。保留原log_every25，只读小件和最多2MiB首部，以真实首个已记录成功update的实际游标/finite loss、world8/64P8/448pairs及8rank输入确认；不新增step1事件/collective/forward或逐rank Adam证明，未观察项明确未知。启动后current-small用本轮SITE和真实首成功回执，R132 completion仅历史。

静态额外`2587336704 MAC/image`和最大新增显式单张量per-rank P8 `102242304 B`不是吞吐/峰值/新增峰值实测，不含pool/backward/workspace。R107历史6400时序环境未再核，不能证明本后端完整24k+30双dev+保存余量或100ms。CUDA反射padding反向可能不确定，CPU通过不证明BW严格确定性，不改现有确定性协议或加GPU门。[机制、风险及一手来源](NEXT-PREPARATION-20261101.md)保留高频细节丢失、公开初始函数改变、有限边界与历史负先验，不宣称纯消除aliasing或全网严格平移不变。

## 完整结果分支

完整工程有效selected同时strict>R107 `.9884405731067525`且strict>singlebestR120 `.9894773092062407`才固定seed27，对照R108 `.9872328538178018`。任一未胜即关闭固定三处Tri3整体策略，不扫kernel/padding/位置/stride/LR/seed，不转APS或叠加R132。工程截断不伪作科学阴性，不重置时钟补跑。

完整freeze selected>=`.996`才official5849一次，不足才正常`not_run_by_policy/threshold_not_met`；内部/官方/平台分列，官方不用于训练/年龄/早停/选点/调参，raw`.999`未证不称总体完成。唯一heartbeat `9c70b2bc`继续5分钟轻检及`:00/:30`汇报，准备阶段不重发旧检查或启动。原R1301503依赖及R1/R131/R132/R120/KEEP全部保护，不为新轮清理。
