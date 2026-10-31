# 设备迁移接续审计与单一后续草案

用户最新明确指令为：**“不需要续训，直接重头开始训练 R130 即可”。** 现按原 R130 科学方案准备独立新轮 fresh，公开初始化、step0、空 Adam，不加载原1503步或 initial。原 one-shot/clock 不复用，原 R130 继续保留 interrupted/1503/无最终 selector。新入口已连接成功，当前 Pod 与原绑定五项身份一致，依赖、资源和一次小件实查已完成；下文保留此前缺文件与认证失败的历史。总体 raw AUROC 0.999 尚未达到。

本页独立接续 [停止交接](DEVICE-HANDOFF-20261101.md)，不改 R130 历史结果、决定页或冻结产物。下述新 fresh 已由用户明确授权，沿现有链完成必要身份配置与冻结后直接启动，不新增用户批准门。

## 当前角色与初次实查范围

当前 Paseo server 为 `srv_KqW7XmJdXxGJ`，本轮新建配对为：

| 角色 | agentId | 范围 |
| --- | --- | --- |
| 根协调 | `b2a18d8d-628c-4c7c-9cf7-9345e736e0ac` | 用户沟通、接续安排 |
| 规划 | `6b8ad6f4-d29a-4647-8fcf-d986715575f0` | 文本只读审计和单一方案 |
| 执行 Sol / A | `7c17043e-8381-43e9-b185-6e142634ad72` | 唯一 Git 写入、Pod 操作及潜在 GPU owner |

`list_hosts`、`list_agents` 和两位新 Agent 的 `get_agent_status` 已实查；初次核验时规划与 A 均为 running。无 B、无派生 Agent；旧 Agent 不恢复。本机仅文本、Git、连接来源检查与展示，没有本机或登录节点 Python、JSON 解析、哈希、模型加载或测试。以下保留 `d8915e5e1525f441e3755d3a961fe76572d3623e` 中的初次连接快照，后续补充见下一节。

| 连接来源 | 本次实际结果 |
| --- | --- |
| Paseo 主机与终端 | self 如上，注册远端 `hosts=[]`；`list_terminals(all=true)` 为 0 |
| Paseo workspace | 父目录 `/home/hao/Desktop/challenge` 与本仓库；父目录 `other/` 为空 |
| SSH 配置 | `~/.ssh/config` 缺失；`~/.ssh/` 仅有 known_hosts 和 old，无密钥文件；系统配置只有通用默认，include 目录为空 |
| 环境与认证 agent | `NCCCU_SSH_CONFIG/IDENTITY/PORT/TARGET`、`SSH_AUTH_SOCK/SSH_AGENT_PID` 均空；`ssh-add -l` 无 agent 可连接 |
| Kubernetes 本地入口 | `KUBECONFIG` 空、`~/.kube` 缺失，本机未发现 kubectl；项目要求仍经既有 SSH 连接动态发现 |
| 迁移小原件 | 此设备没有交接所引 `../work/r130-20261101-a/stop/`；仓库中的历史 RESULT/README 可文本读取，不能冒作这次实查原件 |

初次没有从 known_hosts、入站 SSH 环境或历史地址猜目标，没有尝试旧地址。根当时一次性向用户索取当前 SSH 目标/已有 config 的绝对路径及认证方式；不要求把密码、私钥或 token 发进聊天/Git。连通后 namespace、Pod、UID、容器和 Python 应由工具发现，无须用户沿用旧常量。

该初次快照时未执行 `pod_cpu.sh discover/interpreters/budget/probe/run` 或任何 Pod 命令，未创建 technical-prep、软件 CPU allocation/TTL、训练 grant、clock 或 token。没有部署、训练、GPU 前向、官方评估、大 checkpoint hash/load 或已通过 fixture 重跑。实时安全、8 卡类型/显存/PCI/NUMA、CPU affinity/cpuset/quota、内存/磁盘、进程/设备使用者/锁、依赖与资产可见性全部未核；旧 safe 仅在原观察时成立。

## 用户补充连接后的实查

本节首次记录时间为 `2026-10-31T18:26:05Z`，纠正账号后记录时间为 `2026-10-31T18:28:18Z`（北京时间 2026-11-01 02:28:18），均为本机记录时间，不用于推算原 Pod deadline。执行前已实际 fetch，干净 `main=origin/main=d8915e5e1525f441e3755d3a961fe76572d3623e`。

用户随后提供 `other/ssh` 和认证文件位置；在用户指定范围自主定位到实际 `other/ssh.md` 与 regular/600 的密钥文件。此时文件已存在，不覆盖上节初次目录为空的观察。A 只读取连接配置文本，私钥仅供 SSH 认证；没有 cat、回显、复制私钥，配置与密钥均不入 Git 或传至远端。

| 次序 | 实际命令/阶段 | 结果与边界 |
| --- | --- | --- |
| 1 | 以配置中的真实目标/端口/账号和给定认证文件运行 `pod_cpu.sh discover` | exit255，`Host key verification failed`；默认 `StrictHostKeyChecking=ask`，未进入认证或 kubectl |
| 2 | 本次独立临时 SSH 配置仅用 `accept-new` 接受首次新主机密钥，仍拒绝已有密钥冲突；再次 discover | 服务器 ED25519 记入 known_hosts；到达认证后 `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)`，exit255 |
| 3 | 唯一一次定向 SSH debug，目标与账号不变，远端命令仅 `true` | 已知主机 ED25519 匹配；explicit RSA 公钥被 `Offering public key`，服务器未 `Server accepts key`，继而 `No more authentication methods` 与 Permission denied，exit255 |
| 4 | 用户明确确认初始用户名给错；根已更正用户连接文件，A 同步更正本次临时配置，只换为用户确认的账号，再次 discover | host/port/密钥不变；仍 Permission denied，exit255，未进入 kubectl/Pod |
| 5 | 对纠正后的账号做一次定向 SSH debug，远端命令仅 `true` | 当前账号身份符合用户确认；主机密钥匹配，同一 explicit RSA 公钥被 offer，但服务器仍未接受；无签名阶段，exit255 |

可确认网络/SSH 握手到达服务器、用户确认账号与当前入口下提供的公钥仍未获接受。没有出现 key 格式/加载/passphrase 错误，但认证未到签名阶段，**不能据此声称已验证私钥可解密**。初始账号与历史资产目录字符串不同，当时仅列为需确认；随后用户明确纠正，A 据此定向尝试，不是猜旧用户名。纠正后仍拒绝的具体原因未证，不能断言是服务器公钥配置、账号状态或某一具体权限错误。根已向用户索取匹配该账号/入口的有效密钥或认证配置路径；A 不试其他账号、密钥或泛化认证参数，不绕过认证。

该阶段原始连接日志仅留本机 `../work/device-resume-20261101-a/connection/` 的 `DISCOVER.txt`、`DISCOVER-2.txt`、`AUTH-DEBUG.txt`、`DISCOVER-3.txt`、`AUTH-DEBUG-CORRECTED-USER.txt`，不当作 Pod CPU 回执，不入 Git。当时 `discover` 未成功，没有 kubectl/Pod 命令成功执行；资源和原小件全部 unknown，没有远端 prep。以下新入口实查接续这些历史事实，不覆盖失败原件。

## 新入口成功与实际 Pod 小件核验

用户另行指定当前入口与目录中另一份匹配密钥；根更正仓库外连接文件，A 用独立临时配置定向连接成功，未再尝试旧入口。新主机仍采用 accept-new 并拒绝已存密钥冲突；认证材料与机器连接配置不入 Git。部分成功连接的 SSH stderr 出现额外 RSA 主机密钥更新签名警告，日志保留；ED25519 主机密钥匹配，既定认证及命令成功，未修改校验或算法绕过。

`discover` exit0 得唯一 Running/Ready=True Pod；`interpreters`、Pod 内 bootstrap 和 guard 实查结果如下。以下只绑定本次观察，不是以后执行的默认值。

| 项目 | 实际结果 |
| --- | --- |
| namespace / Pod | `acfwvv9ejn` / `vcjob-2115385083737251842-acfwvv9ejn-44186-worker-0` |
| UID / container | `51fe3daf-ba0d-46e8-919c-99caec599c99` / `vcjob-2115385083737251842-acfwvv9ejn-44186` |
| Python | `/usr/local/python3.12/bin/python3.12`，3.12.7；原 R130 site 五项身份全匹配，无改绑 |
| 实际依赖导入 | torch2.10.0、torchvision0.25.0、numpy1.26.4、Pillow12.3.0、timm1.0.27；GPU 未初始化，无模型/前向 |
| CPU 实核 | affinity/cpuset 0–127，cpu.max=12000000/100000，effective120；版本化 policy 为1/5准备、4/5训练 |
| 新软件份额 | 独立 prep `technical-prep/device-resume-20261101-sol-a`；TTL3600秒、总上限24核；A实际0–11共12核，根12–23为完全未用保留槽；96核训练仅计划24–119，不建旧轮 grant、不侵占其他池 |
| 8卡 | lspci BW3000 `[1d94:6320]`，KFD `beimeizhou`；各VRAM68702699520B；PCI依次09/36/55/77/85/b5/d5/f5:00.0，NUMA依次0/3/2/1/4/7/6/5；与原site PCI/显存一致 |
| 观察时占用 | 各busy0、VRAM used2207744B；device_users=[]、scanDenied0、project live=[]；原instance lock holders=[]，未打开/获取该锁；原owner326810/start349511910仍Z匹配，不称reaped |
| 内存/磁盘 | cgroup memory max506806140928B/current123449794560B；exp-v3 free54034169856B；具体数值只对应此次观察 |

一次 metadata 观察于 `2026-10-31T18:41:21.606972+00:00`，原始 tokenfree [Pod 回执](../records/environment/20261101-device-resume/DEVICE-R130-OBSERVATION.json) 44523B、Pod SHA `76abe8a04e84d0470c92a8cc61b30f39ab466003ad0b535a829679a019c35f50`。原件与完整任务回执保存在上述 exp-v3 独立 prep，本机仅传输和文本展示。

- 原 clock、train-phase、output contract 均在：implementation expired=true；round total/train hard/train soft/contract absolute expired=false；`contract absolute==soft==hard-150` 为true，`contract absolute==round total` 为false。未打印 private clock/token，也未加载 payload 内 deadline。
- 原 GPU grant 小件 SHA 与 site pin 相符、观察时 valid=true、owner 与原site相符，但不属于新执行A。旧prep本人entry未定位，保持未核，不用旧allocation执行；本次新A份额已实核。原 dispatch/pod-admission/controller-start/owner/train-phase/terminal/done 全在，terminal仍timeout/exit130、done.failed，状态仍 interrupted/owner_stop_requested、1503=attempt/skip0/eval1。TRAIN/ROUND/freeze/official 均不存在，双门未判。
- last仍为原806c文件、792176842B，index广告SHA08ac与原记录相同，sidecar/index相等。previous/best、initial（396088443B）与best_ema/best_online/snapshot-800均存在且sidecar一致；index不含成功游标，1503来自实际status，不冒称payload恢复。所有大blob均未hash/load。
- 实际公开Small来源、provenance、train identity、固定manifest、data根、prepared512 cache根、原source manifest/runtime binding均可见。R120 freeze及其selected checkpoint、KEEP三份权威路径权重均regular可见。以上为stat/小件证据，不是大blob字节、cache全覆盖或逐字段模型加载验证；全部原件保留。

新份额 SHA `6fec3e855f823d83ef43104a56f49fc03738ab857d84652f90d14ffc7016c630`；policy SHA `725b2b90d6932a4e80d1f60993f024209e95a263172a37562629f1f0e8d8c4bf`；原guard SHA `734f937a83374d11f05ebcbd5ba64e4d330ee612cf74ade784779a9a7ad4814d`。依赖任务 `tasks/7c17043e-8381-43e9-b185-6e142634ad72-mr2domi3/CPU-RECEIPT.json` SHA `e2b50718b3c46212ea0e7f265b9dad015349b061c91c48ad0306a70102c0ea3a`；metadata任务 `tasks/7c17043e-8381-43e9-b185-6e142634ad72-ifm_gkb9/CPU-RECEIPT.json` SHA `669bda62e4b268786fd335fc14f731cf3291b6dbf8aff721c51413c91f9feadd`；均guard exit0，后者peak runnable1/12、affinity violations[]、signals_sent=false、tree exited。未复跑原3算法fixture或formal，未启动旧轮/GPU/官方；用户随后已授权独立fresh，按下述固定合同推进。

## R130 不能按原链续接的具体原因

起始 Git 为干净 `main=origin/main=3a0090124598209432e23c34edfc6d667aeb0a17`。规划只读 Git 比较确认该基准与 R130 runGit `c8c610dd5ee42e9de99473a83c15ae14b8189178` 的 `runtime/`、`tools/`、`configs/` 无差异；根独立核对相关运行路径亦无差异。以下为冻结实现的文本证据，不是新测试或实时运行回执：

| 原实现 | 决定性约束 |
| --- | --- |
| `runtime/launch_once.py:19`、`:47` | dispatch/controller-start/pod-admission 任一已存在即拒绝；原 implementation deadline 须未过期；准入仅消费一次 |
| `runtime/run_training.py:133`、`:185`、`:267` | owner 入口只有 token；独占写 owner/controller-start/train-phase-start/done；清除 RESUME，新算训练 deadline，训练 argv 不传 `--resume`；无已终止 owner 恢复分支 |
| `tools/activation/activate.py:68`、`:298`、`:314` | 原 formal 目录/clock/失败 marker 存在即拒绝，只准 resume=false；再次 activation 不能成为旧轮接续 |
| `runtime/run_round.py:27` | 冻结源码、site、clock、输出路径均绑定 SHA；热改旧链或时钟会破坏身份 |
| `runtime/bundle/parallel_fresh_train.py:83`、`:119`、`:139` | 底层虽接受 `--resume`，但要求原绝对 deadline 未过期且完整 contract 相同；直接调用会写 config/environment、删除 complete 等状态，不能绕过原 owner 合同 |
| `runtime/bundle/retrain/site_binding.py:88`、`:265` | UID/Ready/主机/Python/准入新鲜度以及原 GPU allocation owner/身份/TTL 均须匹配；新 prep allocation 不续旧 grant |

上述 one-shot/owner 结构限制已足以判定原入口不可续训，**即使将来发现 deadline/grant 尚有效，结论也不反转**。当前不能从本机时间或旧停止时间写“已实核过期”。不删除 marker、不重复 launch/activation、不重置 clock/token/lease、不绕过 owner，不为恢复旧轮新增桥或平台。

## Checkpoint 证据的边界

历史 [R130 RESULT](../records/R130/R130-RESULT.json) 记录 `interrupted/owner_stop_requested`、1503 success=attempt、0 skip、eval1。未产生最终 selector、TRAIN_RESULT、ROUND_RESULT、freeze 或 official；R107/R120 两门仍未判定，不能记 false、科学阴性或 `threshold_not_met`。

历史 last 为 `checkpoint-806c3832c05e487d88c938245ac1d3e2.pt`，提交 index 广告 SHA `08ac68114696b7bd698fb8e70658f7cfbeecf994088a023a115b7aa0591eb377`、792176842 B。精确 exp-v3 相对路径保留在停止交接及 RESULT。**index/sidecar 仅提供文件名/SHA 等提交记录，没有 1503 游标字段；1503/eval1/0skip 来自原 STOP/RESULT/status 的保存描述。** 初审时未复验远端元数据，随后已按上节实核 index/sidecar/stat；始终未 hash/load 大 blob，不能冒称 payload 完整可恢复。

`distributed_engine.py:523` 的保存合同包含 model/Adam/scaler、8 rank RNG、成功/尝试/评估游标、early_stop、online/EMA best/last、coverage/timing、schedule/fresh_guard/absolute_deadline；`:578` 先保存再记录 interrupted。`:240`、`:288` 的底层恢复要求完整身份、deadline、initial 和优化器组映射一致，再恢复 Adam/RNG/游标/EMA。这些源码能力不等于此 checkpoint 的逐字段加载实证。`persistence.py:337` 的 `load(last)` 可能回退 previous 并写 recovery，当前不调用它探测，更不能把回退冒称恢复 1503。

保留原 last/previous/best/initial、best_ema/best_online/snapshot-800、R130 全部 source/manifest/cache、R120 selected/source 及 KEEP-MODELS。没有清理或移动原件。

## 本次既定有界核验范围

1. A 用当前连接运行现有 `pod_cpu.sh discover`，核 namespace/Pod/UID/Ready/container；多个候选按 R130 真实 binding 核对，不自动选第一个或改绑旧运行。再以 `interpreters` 核绝对 Python 路径。
2. 在已核 Pod 新建独立 exp-v3 technical-prep，只为未来准备阶段建立新软件 allocation/TTL 与 CPU guard。按实时 affinity/cpuset/quota 和版本化 RESOURCE-POLICY 算 `floor(20%)` 准备与 `floor(80%)` 训练互斥池，不沿用旧核数；仅 A 可执行，nice19、隐藏 CUDA/HIP/ROCR，不改旧 allocation/grant/clock。随后核依赖、8 卡实际资源、内存/磁盘/进程/设备使用者/锁，观测不完整记 unknown。
3. 一次读取必要小件，脱敏报告原 round-clock implementation/total、训练 hard/soft、contract absolute 的未过期布尔及旧 grant/prep entry TTL。正确关系为 `contract.absolute_deadline_at == train-phase-start.soft_deadline_at == train hard - 150`，不是等于 round 总 deadline；不打印 private clock/token。payload 内 deadline 未 load 则仍未证。
4. 核原 dispatch/pod-admission/controller-start/owner/train-phase-start/train-component-terminal/done 的存在与必要 terminal 状态；只核 index/sidecar/stat、initial/previous/best/protected 与真实 source/manifest/cache 依赖的可见性，不盲扫描/搬移。完整证据写新 technical-prep，仅导回必要小脱敏件。

这组核验补齐设备与保存现状，不试旧启动，不因元数据一致而加载大 blob 来重复裁定原入口。只有日后另获明确 checkpoint 继承合同，才评估一次精确 last 验证；本次不实施继承。

## 已授权单一后续方案：独立 fresh R130R1

**用户已明确授权重头训练R130，按既定单一方案执行。** 协调已冻结 [R130R1独立合同](auroc-999/R130R1-DECISION.md)：order41、`configs/R130R1-SCIENCE.json`、profile的`layerdecay08-r1-v1`后缀、candidate前缀`r130r1`、独立round及prep `technical-prep/r130r1-20261101-7c17`，不能复用 R130 order40 假装原轮续训。目的仅为完成被用户中断的固定 0.8 逐层 LR 问题；同 seed20261029、Small 公开 fe6b358a 与原原生 head 初始化，从 step0/空 Adam 开始，不使用 R130 第1503步或 initial checkpoint 作为初始化。旧 1503 步和成本保留，不拼接曲线或补作新轮前1503步。

全部原 R130 科学值固定：9:1 fit46256/dev5140；单 Small/GAP+age、prepared512/scale[.85,1]、原旋转/光度、G64/P8 八 rank、FP32/noTF32、pairwise logistic；原 layerdecay .8 分组映射、backbone1e-4/head-age1e-3、WD1e-4/SD.1/drop.2、一次 globalclip1、从 step1 全层更新；最多24000次成功更新、每800双 variant 内部 dev、连续4轮 AUROC 无提升早停、原1600 warmup 与 EMA/唯一 selector。独立新轮使用原6h训练/8h整轮/9hgrant预算，不延长旧 R130 时钟。

完整工程有效 selected internal AUROC 须同时严格胜 R107 `.9884405731067525` 与 singlebest R120 `.9894773092062407`，才具备固定 seed27 科学复核资格；用户最新持续推进授权已覆盖通过两门后的固定第二seed独立复核，由根协调串行派发。未同时胜即关闭本固定方案，不 factor/LR/分组/seed 补考，AP/partial/official 不救主判，随后有限评议一个独立方向。完整 freeze selected>=.996 才官方5849一次，完整 selected 不足才 `not_run_by_policy/threshold_not_met`；工程中断不套此标签。总体 raw .999 目标保留。

最小实现仅新增科学身份及必要 `experiment91.py` 映射（当前 `LAYERWISE_LR_SMALL_BINDING` 固定 candidate/order40，仅改配置文件名不足），复用既有 layerwise、保存和直接训练链。源码变更前汇报具体目的；新代码实际 push/readback 后精确部署。只核新增身份/实际迁移影响的窄绑定及现有必要 formal，不重跑已通过的3项算法 fixture，不做独立 GPU 预演；真实训练8 rank 初态/组/空 Adam 接管 GPU 验收。

仍由 A 唯一串行 GPU，训练80%整树与准备20%互斥；Small12GiB、启动 free20GiB/运行 floor16GiB、300s atomic/prune/failfast 与全部资产保护保持。不新建计算资源、不关闭实例、不续租、不恢复旧 timer/heartbeat；新观测安排由根按用户最新指令处理。新轮吞吐、完整训练容量、满24k耗时、500MB包体或100ms均未证。

当前连接已通、用户已授权独立fresh；A沿既有直接链完成新身份/必要窄检查/实际Git同步冻结和准入后直接启动，不再等待额外文档或用户批准。不继承原checkpoint，不擅自更改固定方案；后续安排由根按完整结果与用户最新持续推进指令协调。根已实际创建唯一active heartbeat `9c70b2bc`（`*/5`，Asia/Shanghai，无maxRuns/expiry），含每5分钟完成小检测、`:00/:30`报告；不恢复旧 `1c2c8c3b`，A不另建timer。

新增唯一身份窄验 `tests/test_layerwise_lr_r1_identity.py` 已在新prep narrow-source、本人有效guard中执行：先行1项失败精确为新profile尚未接受，随后仅补精确新身份映射和现有formal分类，实际1/1 PASS，旧40/新41接受与科学值全等、公开fresh/seed/无initial及两profile的正式layerwise路径同时覆盖。RED CPU回执 `tasks/7c17043e-8381-43e9-b185-6e142634ad72-d7avud6_/CPU-RECEIPT.json` SHA `cf34f1ad2468a2edabfc737ddf18eb69fae42e0b9c3e8bbf0d383296b157a188`；GREEN `tasks/7c17043e-8381-43e9-b185-6e142634ad72-3tckykfx/CPU-RECEIPT.json` SHA `56b1e755a4ac59f285cedfe26c02cf66faa8ba314e3b669235dc1d09b6aa9c98`，均在本次audit prep下。原算法与保存fixture未重跑；formal将从已同步commit新建，仅登记测试身份不冒作正式初态或训练通过。
