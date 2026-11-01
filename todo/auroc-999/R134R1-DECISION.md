# R134R1：固定 MixStyle 的独立技术修复轮

根已在R134真实技术失败、actualsafe和完整失败归档后采纳本轮，fresh授权reference `848b1a9c-07ce-4b90-b347-4da79865ff27`。A为唯一文件/Git/Pod/GPU owner，不新增Agent或timer；唯一heartbeat `9c70b2bc`保持。当前正在两项新目标RED与最小修复，未证明本轮GPU成功。

| 字段 | 冻结值 |
| --- | --- |
| order/config | 46 / `configs/R134R1-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-mixstyle13-local8-r1-v1` |
| candidate | `r134r1-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-mixstyle13-local8-9010` |
| round | `r134r1-small22k-seed20261029-scale085-pairlog-mixstyle13-local8-24k-p4-9010-20261101` |
| prep | `exp-v3/technical-prep/r134r1-20261101-7c17` |

科学整件直接R134，只改独立身份/order46与披露失败和修复的note。双`MIXSTYLE_RULE`、显式`OFFICIAL_GATE_0996`和全部R107科学值不变；公开fe6b358a/seed20261029/native1000 strict→原binary/age、step0/空Adam/345全训，whole预期50c由本轮唯一normalformal重建，不加载旧R134或任何旧initial/checkpoint。不继承Tri3/outputreg/LoRA/layerwise/Mixup，final_downsample_stride整键缺省；原512/单视图scale.85–1/旋转光度/9:1名单seed20261024/age/G64P8/FP32-noTF32/pairlog六字段/LR-WD-clip-SD-dropout/24k800p4/warmupEMA1600/唯一selector完全保持。

唯一技术变化在`distributed_engine.prime_reducer`：仅MixStyle非空时给原`age.forward`临时context `{attempted_before_increment:0, rank:ctx.rank, micro:0}`，无trace、保存或真实游标推进；None原调用原样。两次既有dummy forward/backward、capture_rng/clone_state/finally恢复state与全局RNG及清梯度保持。不放宽缺context验证、不关闭MixStyle、不跳过prime、不改数学/policy/optimizer/EMA/save或增加GPU forward。priming context只是无更新预热输入，不算已观察训练attempt0；真实采样仍从原真实attempt游标重建。

只跑两个新增受影响正向目标：真正生产prime_reducer的两rank Gloo小factory回归，核两次原dummy调用、原state/globalRNG/training mode恢复、清梯度/空Adam、None原调用和真实下一attempt context未推进；旧R13445/新R134R1 46的精确注册和SCI仅身份/note差异检查。两者先真实RED再最小GREEN，最终编译受改源；不重旧MixStyle数学/保存、旧lossGloo/LoRA/Tri3/suite，不完整公开模型CPU预演或GPU预演。规划只做该修复/身份的一次只读审查，不是批准门。

两项已真实RED 2FAIL/0ERROR（10.400秒）：两rank各先通过None原路径，非空policy准确因缺context失败；身份只新R1未注册失败。最小修复后GREEN 2/2、0FAIL/0ERROR（10.330秒），4受改Python编译、GPU未初始化/d29 guard0。GREEN SHA `1735aeeaf95ee7c87e94abddd4da0cbb01fdd8a1e9c722375d4032d37d0fa55b`、CPU guard `9351324a8c844d9afe8c629c5f84de94fb72aa10d6d55b73c29a2040fde0da7d`，SCI SHA `875f83957bb2e6b11ce4d6e9ae4bbd7b32df8422e565c3ae9da3d160f464c205`。两rank实际None/非空各2次原dummy后参数零容差、全局RNG、eval mode恢复、grad None/Adam0通过，真实后续caller attempt6仍为6且非空两site `[1,3]`执行。该小factory只保留真实早stage，不是完整公开Small/8rank/GPU证据。规划一次针对静读无Critical/Important或待修Minor，不替代上述实际运行或接下来的独立normalformal。

通过后actual commit/nonforce push/readback→精确Git blob/mode部署→当前UID/Ready和动态资源/CPU互斥池准入与独立新9h grant→本轮唯一normalformal→deploymentfreeze→freshabsence/快照下一次activation与正确direct launch。原R134的18/18不冒本轮执行。CPU pin SITE SHA、正确-i、GPU uncontracted且无预taskset；真marker存在即停，不重入旧轮。复用有效d29 future准备allocation，A12/0–11、root12–23保留，与实核120有效CPU时训练96/24–119互斥，nice19/隐藏GPU；每入口核TTL/UID/Ready。原6h训练/8h整轮/9h grant、Small12GiB/free20/floor16/300s不变，无物理扩资源/续租/关实例或新绑定平台。

启动重新证八rank actualinitial345/实际组/Adam0/50c/双site descriptor/设备和mask，再沿原log25自然首成功核finite WORLD/64P8/448pairs及16site实际trace。priming不记update/trace/step1，不强制gate或挑seed；真实first后才切本轮SITE+first的current入口。未运行或缺字段如实保留unknown/null，不补step1或逐rank Adam结论。

R134技术失败、one-shot、clock/token/grant、全部失败成本和原件永久分列历史；不拼曲线，不把缺游标伪填0或把技术失败写成科学阴性。R1301503及R1/R131/R132/R133/R134/R120/KEEP与共用依赖均保护。[R134失败实证](../../records/R134/README.md)与[同一科学策略的机制/风险](NEXT-PREPARATION-20261101.md)保持：早层统计可能含病灶，local8类别不均、作者CE/跨域/PyTorch采样与当前pairlog/private Python实现非等价；CPU通过不证明GPU峰值、24k+30双dev+save预算或`.999`收益。standalone export与完整owner resume平台不扩。

完整有效selected必须strict>R107 `.9884405731067525`且strict>R120 singlebest `.9894773092062407`才固定seed27对照R108 `.9872328538178018`。任一未胜关闭固定p.5/alpha.1/sites1,3/local8/RNG整体，不扫参数/seed或转Lookahead/叠加关闭分支；工程截断另列、不扩deadline补跑。完整freeze>=`.996`才官方5849一次，内部/官方/平台分列，raw`.999`未证不称总目标完成。
