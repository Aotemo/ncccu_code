# R136 固定 raw Conv/Linear GC：完整结果与启动实证

2026-11-01 20:07:47.397894 +08:00唯一completion确认done/TRAIN/ROUND completed，6400成功=尝试/0跳步/eval8，连续4轮无提升正常早停，component exit0/signals[]/errornull。唯一选中EMA c4@3200，raw AUROC `.987366509691255`、AP `.82175064566623`、error_twice20984，tensor `928e2d106a379637aeabff34ffcef8e4d8776e07ede7f5f6706290cd5f5c885f`。strict>R107 `.9884405731067525`和strict>R120 `.9894773092062407`均为false，固定raw Conv/Linear GC关闭/no seed27；不扫位置、范围、均值组、系数、LR或seed，也不外推所有GC方案均无效。

freeze.status原值null；ROUND官方为not_run_by_policy/threshold_not_met/evaluatedfalse/metricsnull，observer official_readfalse，未追加官方前向。actualsafe=true/terminal_idle：owner390921原start355888853现Z非reaped，8rank390933–390940 absent；global同inode锁存在且可读、holders[]，deviceusers[]/scanDenied0，8expectedPCI全部seen/busy0。只复用此次completion/HB/CPU及小freeze元数据归档，未hash/load大blob或重复安全扫描，selected及全部依赖/成本继续保护。

终态原件见[R136-RESULT.json](R136-RESULT.json)、[R136-COMPLETION.json](R136-COMPLETION.json)、[R136-CPU-COMPLETION.json](R136-CPU-COMPLETION.json)。6400实际训练成本与选中3200游标分列；暂无新训练候选授权，raw .999未达。

以下为启动历史：2026-11-01 19:13:48.709331 +08:00实际观测running25成功=尝试/0跳步/eval0。首自然log25 loss/global_loss `.6515139937400818`、WORLD8/G64/P8/448pairs，八rank均记录固定`raw-convlinear-weight-gc-v1` policy及registered_target_count113。真实初态八rank345全训/实际5组/初始Adam0/原whole `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，完整113个实际Conv2d/Linear直接weight名称/shape/非output轴描述同formal。初态空Adam不指后续仍为空；未另观察step1/逐rankAdam，也未另做GPU梯度clone/hash或forward，注册范围字段不冒额外数学探针。

冻结runGit `f38adab66219bbe2e3a79467b1710015c9ace120`，tree `5c70fc24d696ba23f4a49bc0c7021fb6088f367c`；SCI SHA `098768cc3eeb0e02f9fa742daa72b2fc076d461787538ad1b56caefaeb49817f`，SITE SHA `36015944f7519353f228318f1d40951105ac982523ed134578e9ab7118d6baf9`。唯一normalformal19/19（90.007秒）及804源Git blob/mode exact、13项absence→唯一activation0/CPUguard0→live launch0实证分阶段保留，归档Git另列不热改运行包。没有独立public控制构造或GPU预演。

owner390921/start355888853、8rank390933–390940 argv/birth/mask/8PCI匹配；实时有效120核，训练96核24–119（control112–119/ranks24–111）、prep24核0–23互斥，A用0–11、root12–23保留不用。新9c1b CPU软件allocation有效至1793540480.1967466，旧74bd已过期且未修改。只读current入口`work/r136-20261101-a/pod-context.env`绑定本轮SITE与真实START1，复用既有通用small-check/observer和唯一heartbeat，不新增监控或timer。

必要窄证据分列：算法RED2FAIL/0ERROR，身份RED1FAIL/0ERROR；GREEN1首算法PASS、第2 FAIL，DIAG1确认toy norm.bias公共平移理论零方向的FP32约1e-9归约差被原Adam epsilon放大。只给测试factory固定input gain以辨识该方向，生产与容差未改；仅第2目标GREEN2 1/1（9.372秒），身份GREEN1 1/1（3.413秒），首目标不调用forward保留原PASS，最终7源compile。没有“同版同一次最终3/3”声明；原失败/诊断和一次无Pod执行的kex运输失败全部保留。测试仅小合成/两rank CPU Gloo，原50c及113个真实Small目标由本轮normalformal和实际八rank初态承接。

小件为[R136-START.json](R136-START.json)、[R136-PREPARATION.json](R136-PREPARATION.json)、[R136-CPU-CHECKS.json](R136-CPU-CHECKS.json)、[R136-CPU-START.json](R136-CPU-START.json)、[R136-THIN-START.txt](R136-THIN-START.txt)。完整原件位于`exp-v3/technical-prep/r136-20261101-7c17`，未为归档新观察、读大blob或查旧safe。

唯一科学合同见[R136决定](../../todo/auroc-999/R136-DECISION.md)：直接RGB3 R107 seed29/fresh；仅raw梯度GC后走原globalclip1/AdamW/LR/EMA，其他参数缩放和Adam矩不承诺不变。完整selected同时严格胜R107 `.9884405731067525`和R120 `.9894773092062407`才固定seed27；否则关闭这一固定范围/位置GC，不扫参数或seed。完整freeze≥.996才官方5849一次，raw.999未证不称总体完成。R130全部/R120/KEEP/R134R1/R135 selected与原始数据/public/prepared512及全部失败成本继续保护。
