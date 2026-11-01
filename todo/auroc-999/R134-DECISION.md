# R134：早两 stage 的固定 rank-local8 MixStyle

**本轮已在真实初态门前技术失败，尚无科学结论。** 09:22:53.495411 status failed、owner366816同startZ、八rankabsent，无initial-ranks/ACTUAL_FRESH_INITIAL或记录update。09:23:58.1901767唯一completion为done failed、selected null、TRAIN/ROUND/freeze/official均缺件，status未提供成功/尝试/eval游标，保留null而非补写0；官方未进入，不套threshold_not_met。实际八rank短trace证明原`prime_reducer`的两次既有dummy forward未传MixStyle context：`distributed_engine.py:324 → :122 → age.py:107 → models.py:466 → mixstyle.py:129/:53`，报`ValueError: exact MixStyle policy and explicit attempt context required`。这不是数学或AUROC失败，不关闭MixStyle科学假设。actualsafe true/terminal_idle：owner同startZ非reaped、8rankabsent、锁holders/设备users空/denied0、8expectedPCI seen busy0。只读诊断SHA `7c28afdb014cc1c6c2b254471c4d49c2d27dfdb21c413c04e068a7858cc8678c`，原完整噪声日志留prep；精简失败/安全小件归档另列。原one-shot已消费，保留全部成本/clock/token/grant/markers，不重launch/activation、不热改本轮；最小修复与独立后续合同由根另定。以下准备/待启动为历史阶段。

R133完整结果与actualsafe已核后，根依用户持续串行推进授权采纳本轮；fresh授权reference为消息`f9c93d14-e188-4bdb-84f0-da2824d274e3`。A `7c17043e-8381-43e9-b185-6e142634ad72`为唯一文件/Git/Pod/GPU owner，规划仅一次只读注册审查。当前处于身份接线与正式准备，尚未证明GPU启动或训练成功。

| 字段 | 冻结值 |
| --- | --- |
| order/config | 45 / `configs/R134-SCIENCE.json` |
| profile | `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-mixstyle13-local8-v1` |
| candidate | `r134-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-mixstyle13-local8-9010` |
| round | `r134-small22k-seed20261029-scale085-pairlog-mixstyle13-local8-24k-p4-9010-20261101` |
| prep | `exp-v3/technical-prep/r134-20261101-7c17` |

直接复制R107整件SCIENCE，仅新身份/note、job与candidate同一`MIXSTYLE_RULE`及显式`OFFICIAL_GATE_0996`不同。公开fe6b358a、seed20261029、native1000 strict后原binary head/age初始化，step0/空Adam/345张量全训；预期whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`由本轮唯一normalformal实际重建证明，不另独立public构造。不加载旧checkpoint或initial，不继承Tri3/outputreg/LoRA/layerwise/旧Mixup，`final_downsample_stride`整键保持缺省。

原512/单视图scale[.85,1]/旋转光度/9:1名单与splitseed20261024/age、G64P8/八rank/accum1/FP32-noTF32、pairlog六字段、LR/WD/globalclip1/SD.1/dropout.2、24k成功步/每800双variant dev/p4/1600warmup和EMA/唯一selector全保留。优化器/EMA/保存/通信算法不改，组数以真实原验收为准。

唯一policy版本`convnext-small-local8-mixstyle13-v1`，sites `[1,3]`、probability `.5`、alpha `.1`、epsilon `1e-6`、variance_correction `1`、scope `rank_local8`、salt `mixstyle13-local8-v1`、rng `sha256-first8-big-private-python-v1`。仅训练forward在完整features1/3后执行；FP32分别mean和HW−1方差、sqrt(var+eps)，mu/sigma detach，原未融合表达式混合统计。每site/rank/attempt/micro独立：UTF8冒号join(seed,salt,attempted_before_increment,rank,micro,site)的SHA前8byte大端值建立private Random；先gate random<=.5，激活才依序8次Beta(.1,.1)后shuffle(range8)，允许selfpair。无label/image/age混合、无全局配对或持久RNG，原SD/dropout全局RNG保持。None/eval/EMA/dev/official走原native路径。

## 最小执行与实证边界

实际runGit `15f78f725c6b5aad23da308bb7076801a831e646`/parent `a6e65dedee3a460164ff70b33b5f1e4748c25c9f`/tree `58d8d135893407ef0acb19e122f3b976a81f0d45`已nonforce push/readback。唯一normalformal实际18/18，745源Git blob/mode exact，公开whole50c、345全训/实际5组/空Adam、正式两site描述/onlineEMA一致与默认factory strict重建均通过。正式CPU SHA `1188136e3bba09c6c4c9ac37e55b5c4b82a81918676ec7e3bc0b90826885d599`、deployment `e7d86f27e6eaaf83b6f851d84e0b9b023b11f434978013c872066dd88a9c3dc2`、SITE `c1aa829f7112fdcd619cee9b42b6e1811fa4656187e73686169717f1428ee869`。13项absence/原调用0后唯一activation0与live launch0，正确-i/CPU pin SITE SHA/GPU无预taskset保持。09:22:17.676596第一观测仅owner366816/start352346560及8rank366828–35 argv/start/mask active，当时实际初态/首成功尚未产出，START-PENDING保留，不冒训练成功。后续只读原链自然小件，归档commit与runGit分列；以下待执行措辞为相应历史阶段。

唯一新身份检查已实际RED 1FAIL/0ERROR后GREEN 1/1、0FAIL/0ERROR（1.719秒），GPU未初始化；SCI SHA `94116a481cc3f35aadd7a82410b1e850d17a6c9315d970f49055af88755a5711`，GREEN SHA `e87210d6ed6f1376dacc60d89dcc216af03ce673c533f459d0ac36fadc5e1d25`，guard SHA `55af0dce549da516eefbe43349d36267c2eb5dfaa4a33fb815cc5cf88d29b669`/exit0。规划唯一注册只读审查无Critical/Important；唯一Minor为旧CPU-only注释，已作纯注释修正，不为此重跑identity，最终源码由正常formal编译/部署SHA承接。当前normalformal/真实GPU证据仍待执行，不能用该1/1替代。

CPU-ready基线`a6e65dedee3a460164ff70b33b5f1e4748c25c9f`已有739源码exact/9pin。两项必要CPU目标分两次GREEN：GREEN1第1项通过、第2项fixture缺EMA rule报ERROR；只修fixture两行后GREEN2仅第2项1/1与9源compile通过，生产8SHA保持，不冒同一次最终2/2。独立公式/解析梯度/private RNG、None兼容、native initialize=False345对象名称值/zero buffer，以及小factory真实配置→保存→inference→两site replay和online/EMA eval均有证据；不冒公开fresh、默认production factory、实际DDP八rank或完整Adam/EMA恢复。[CPU边界与风险](NEXT-PREPARATION-20261101.md)及[CPU小件](../../records/preparation/mixstyle-cpu-20261101/README.md)保留。

本轮只新增一项identity正向窄验：旧R13344/新R13445真实SCI→candidate、双policy/.996/whole50c/缺省字段，以及除获准差异外整SCI等于R107。实际RED后最小experiment91精确注册，旧profile仍拒非空MixStyle；runtime期望同一candidate policy，实际两sites完整descriptor沿resolved_age_spec。现有normalformal仅在原positive checks补online/EMA与默认factory strict重建descriptor比较；不新增check/forward/第二formal，不重MixStyle两项/旧Gloo/Tri3/LoRA/suite或GPU预演。

窄验后actual commit/nonforce push/readback→实际Git blob/mode精确部署→当前Pod UID/Ready、CPU配额和资源准入/独立新9h训练grant→normalformal一次→deployment freeze→新鲜快照下唯一activation和正确live launch。SITE SHA在本人CPU guard内pin，GPU初次入口保持uncontracted、由launcher自核绑定，无预taskset，正确`kubectl exec -i`；真marker存在即停，不删除或重入。

本轮独立future CPU binding SHA `d29bbad3d1c88c678355e3a51fca47161430a30d9806a709eac193cf6eb92811`，created1793495312.497302、valid_until1793506112.497302，仅供未来准备/观测；旧672/所有GPU grant不改。实核effective120时prep cap24，A12核0–11/nice19/隐藏GPU，root12–23保留不用；训练96核24–119互斥，每入口重核UID/Ready/配额/TTL。原6h训练/8h整轮/9h grant、Small12GiB/free20/floor16/300s不变，无新资源、物理续租、关实例、timer或Agent。

真实八rank初态接管345全训/实际命名组/Adam0/50c/两site descriptor/设备与80%整树；首自然已记录成功update沿原log25，核finite loss/world8/64P8/448pairs及rank_inputs。对实际caller-owned临时MixStyle trace核各rank两site、attempt-beforeincrement/rank/micro及固定key/采样字段；两gate全关合法，不强制开启/换seed，不新增event/collective/forward/step1或逐rank Adam证明。真实first回执出现后才切current-small定位R134；R133 completion仅历史。

局部统计可能含病灶/标签语义；local8的rank0/4各4P4N，其余六rank纯负，保持原sampler，不补gather或调配对。作者跨域CE及Torch采样不等于本pairlog/private Python采样，不预测收益或`.999`。每site单个FP32特征张量50331648/25165824 B和两site同时激活的75497472逐元素算术计数不是GPU峰值/总FLOP/满24k+30双dev+save时耗证明；不靠早停担保、不改预算/精度/batch。历史R14/R15像素label-age Mixup、R112 fixedBN、R122 photometric等非等价负先验保持；通用standalone export仍不支持/未验证，不扩平台。

## 完整结果分支

完整工程有效selected同时strict>R107 `.9884405731067525`且strict>singlebestR120 `.9894773092062407`才固定seed27，对照R108 `.9872328538178018`。任一未胜即关闭精确p/alpha/sites/local8/RNG整体，不扫概率/alpha/site/归一化/配对/LR/seed，不叠加关闭分支或转Lookahead；工程截断不伪科学阴性、不扩deadline补跑。

完整freeze selected>=`.996`才official5849一次；内部/官方/平台分列，官方不参与训练/年龄/早停/选点/调参，raw`.999`未证不称总目标完成。唯一heartbeat `9c70b2bc`继续五分钟轻检及`:00/:30`公开汇报，准备中不重派旧检查或启动。R1301503全部依赖、R1/R131/R132/R133/R120/KEEP均保护，不为新轮清理。
