# 8 卡适配与 exp-v3 接续

更新：2026-10-28。本页是当前任务记录，不是机器默认配置。历史回执及旧冻结运行不改写。

## 授权、角色与当前阶段

用户要求从 GitHub `Aotemo/ncccu_code` 接续，新建规划和执行 Agent，不恢复旧 Agent；先识别当前 8 卡环境并适配代码，再进行后续实验；本轮及后续源码同步 GitHub，其他数据保存 `/public/home/acfwvv9ejn/SothisAI/exp-v3`。用户随后明确“批准，解除旧限制并按此顺序推进”，再授权“可以多开一个执行 agent 来并行推进”；现为 1 规划 + 2 执行。旧暂停已被覆盖，但旧运行、旧 token/clock、定时任务与 heartbeat 不恢复。

Paseo `list_hosts/list_agents/get_agent_status` 已实时核实以下角色；执行向规划确认接单。表中身份仅属于本次 campaign `exp-v3-8card-20261028`，后续操作仍需工具复核。

| 角色 | serverId | agentId | 职责 |
| --- | --- | --- | --- |
| 规划 | `srv_3NMM8gzO88V8` | `1e254e66-e9c4-450c-9bad-b42fca2990dc` | 科学合同、最小适配、验收和逐轮决策；仅编辑 AGENTS/docs/todo |
| 执行 A | `srv_3NMM8gzO88V8` | `19c0bc67-1a8c-462a-afb6-f6b80d090986` | 八卡主线 runtime/configs/tools/真实证据（排除 B 专属范围）；唯一 Git 写入者与唯一 GPU owner |
| 执行 B | `srv_3NMM8gzO88V8` | `5d11d2c3-54b6-4037-ad23-a38f3849238f` | 预训练来源子门已交付；现负责`tools/migration/**`、`tests/test_asset_relocation.py`、`records/migration/**`与迁移目标写入，其中B1文件委派下列子Agent；无Git写/GPU权 |
| A1，父级A | `srv_3NMM8gzO88V8` | `ce267453-f28c-42f4-9096-f6cb51dbf335` | 指定activation/启动绑定/CPU树文本编码，精确白名单见下；无Pod/Git写/GPU/再派生权 |
| B1，父级B | `srv_3NMM8gzO88V8` | `0a198f7e-6cca-4280-9d17-2dffde4adde7` | 仅`tools/migration/cache_rebind.py`、`tests/test_cache_rebind.py`文本编码；无Pod/Git写/GPU/再派生权 |
| 协调会话 | `srv_3NMM8gzO88V8` | `9b7c9567-b59e-43d2-9e35-576d79613f5d` | 用户进度、必要阻碍处理与具体同步结果审阅 |

最新用户明确“不关闭当前实例了，执行agent可开子agent并行，加速迁移构造”。当前实例继续使用，不暂停、不关闭或申请资源。本次A/B各自行按执行profile创建1名文本子Agent，覆盖此前不额外扩员限制，但不再派生。规划已通过Paseo实时核父级及两名子Agent，二者running、无待批权限、parent分别为A/B，workspace相同。

A1精确独占：`tools/activation/{activate.py,activate_audit.py,site_preflight.py}`、`runtime/{run_training.py,run_round.py,r92_initial_gate.py,launch_once.py,component_gate.py}`、`runtime/bundle/{parallel_fresh_train.py,parallel_evaluate.py}`、`runtime/bundle/retrain/{execution_policy.py,site_binding.py}`、`tests/{test_activation_binding.py,test_runtime_cpu_binding.py}`。A本体不共写这些文件，集中`tools/activation/prepare.py`、`tools/validation/check_candidate.py`、configs和production factory集成。B本体不共写B1两个文件，保留其余迁移实现和回执；A不写迁移目标。子Agent只本机文本编码/Git只读，不运行Python/解析/测试、不发Pod命令，父级在自身12核内统一部署验收。全体不建计算资源/定时任务/heartbeat，不唤醒旧owner；B只复制实际依赖，不删除历史原件。

A1上述14文件交付冻结且父A集成29项已通过后，仅增加两个文本文件白名单：`tools/validation/gpu_admission.py`、`tests/test_gpu_admission_contract.py`。原14件不热改、不另开Agent；新两文件由父A做必要Pod CPU合同窄验后纳入具体Git变更，实际GPU仍等同步及冻结。

A/B 使用独立 exp-v3 technical-prep。用户最新要求“重新分配 cpu 使用额度，执行 agent 可使用总 cpu 数的 20%”：两位执行合计≤实时有效CPU的20%，初始平分、允许并行，规划可重分配但总和不越限。旧2线程/重CPU串行限制已被覆盖，历史回执保留原值。有效CPU取当前Pod affinity/cpuset/cgroup quota可用上限，总预算向下取整；本次新Pod已实核并确认12/12份额，数值仅适用于下述精确绑定和TTL，不能成为环境默认。具体通道合同见[EXECUTION](../../docs/EXECUTION.md)。

用户随后明确纠正：20%只用于执行Agent准备/编码计算，模型训练使用整体设备CPU80%；代表实际训练资源的GPU准入整树同用80%，包含owner/rank/loader/control/helper。该最新语义覆盖此前协调/规划将20%扩大到GPU的误读。未启动阶段另建绑定，准备与训练并发时mask互斥、总和不超实时有效CPU；历史20af、attempt1、旧allocation及失败回执保留原样，不把旧限制成功当新目标。

最终进一步澄清：训练80%与下一轮计划/CPU验证20%同时运行，比例从集中版本化资源policy读取并经运行env注入，代码不散写比例；核数/总额/mask全部取实时环境。只核两个池互斥、总量不超有效CPU，允许prep活动持续，不再要求全体prep/peer退场。A合最小资源policy、ENV及TERM修复一次窄验/同步/新冻结/准入后直接fresh R93；规划在训练开始后统筹20%池的下一轮计划/CPU验证，沿既有执行文件边界进行，GPU仍仅A串行。

已核原blob SHA引用真实回执，身份未变时避免重复哈希。B先验公开完整模型与来源，A固定生产工厂源码SHA后再做必要集成验收；不能把旧工厂拒绝当作新工厂通过。跨Pod的未执行阶段须新建明确资产续验绑定，pin旧source-chain SHA并核当前路径/身份（包括已记录mtime/ctime），保留来源历史和当前执行身份，不能直接松掉UID约束或重写旧回执。

当前为 **首次GPU准入失败后的根因核实与最小修复阶段**。A生产18/18、启动绑定29/29、准入合同CPU11/11、部署关联3/3以及B来源/完整迁移已通过。实现已同步GitHub `20af511b0177c0f1576c8c630baaf1865d8001f9`，tree=`85fc6860bec2b5827a990e864ca444e0472550b4`、唯一parent=`cb939c8e5078a9c74a26158a78c4aa033009042a`，ref/commit/tree独立回读相符；Pod实际部署回执已生成。A已另建GPU owner CPU绑定，B/B1退场，旧12/12回执不改。首次0forward/0update技术准入实际执行49.281秒后失败，CPU helper越界和owner退出收集缺陷尚待修复；R93未启动。细节见[首次准入记录](GPU-ADMISSION-1.md)。

关键路径为新80%训练资源语义、必要helper适配及owner有界回收的最小修复，一次具体diff/必要Pod CPU窄验/新Git同步及实际SHA冻结后，进入独立准入和fresh R93。A父级负责`runtime/bundle/retrain/site_binding.py`、`tools/activation/prepare_gpu_site.py`、`tests/test_runtime_cpu_binding.py`、`tests/test_gpu_site_inputs.py`及直接训练fraction断言；A1仍只改`tools/validation/gpu_admission.py`和`tests/test_gpu_admission_contract.py`退出收集，互不共写。已通过且未受影响的来源/迁移/生产检查不重跑整套；不新增门、平台、benchmark或归档润色，不改科学合同。

已报告的局部CPU进展：A 的 `site_binding` 5项标准库检查通过、exit0，回执为 `exp-v3/technical-prep/adaptation-cpu-20261028/green-site-1/RECEIPT.json`，Pod SHA `58783f90648cc214f5322ed22391726fe3a57fc15bb85858bbd73aee529ae031`。4/8rank全局样本及增强种子重组边界检查已有通过回报；首次旧拓扑API检查失败是改动前证据，不能充当适配后通过。这些检查在旧Pod、旧2线程约束下完成，现均已退出；不能替代新预算通道和新环境检查。其余适配/生产工厂检查仍待集成回执。

新Pod的验收失败、修复与后续通过记录（以下失败发生在成功回执之前）：

- A固定快照21项检查首次15项通过、6项失败（8个subfail/1个error），source unchanged=true。通过范围为预算5、拓扑3、site5、dev16条selector及官方canonical预约；候选4项、launch旧hostname断言、component_gate错误PID仍运行待修。A新prep下`red-current-1/RECEIPT.json` Pod SHA `ae31d6068875cb332fe2c9cbd8f2a68022f7831389a5235d0acee90ea9d90e80`，guard回执SHA `3dd1672955ba109f72b29b76b4c7cb1e375a3ed7af17017785119bfcfef150ec`，childexit1是测试真实失败，不是通过。
- B新review-only首次9/10，真实同长度原位改写+utime未被拒绝；人工stale/缺字段和跨UID绑定等其他项通过，整轮回执仍false。B新prep下`REVIEW-TESTS.json` Pod SHA `fedb168ecc1d862b1085cc5817a70f70ebf76c9fc0f18cf01a14edcc08cbb178`，guard回执SHA `48d900472c65a478386f555e15b3a0946863b412327fbc2537f574e7ab7de532`。尚未生成真实来源stage，未做本次模型load；只诊断小fixture实际内容与before/write/utime/fsync的stat，不用sleep、改长度或skip让测试通过。
- 若诊断证明当前文件系统可观测stat不变而bytes已变，stat一致不能当作当前内容证明。此时对缺证的实际资产进行必要内容重哈希是新证据下的窄验，不受“避免无变化重复hash”限制；不重放完整source-chain或扫描全部历史。A继续独立主线修复，B继续专属资产护栏，不串行阻断双方工作。
- B单次FS诊断已实证上述问题：同一3B文件内容从`{}\n`改为`[]\n`，内容SHA变化，但write、显式utime、fsync/fstat后完整7项stat相同。`FILE-IDENTITY-DIAGNOSTIC.json` Pod SHA `c0e1cbf845747a08e5964e31834c2c6a618eb44a8c2161077a69ee50213221fd`，guard回执SHA `91596df558d4a216748700530cc8d1256aecbebe1292dc346d3b1869be9e5362`。诊断未读权重，FS调查到此结束。
- 规划/A/协调确认`evidence.kind=fresh_pod_sha256`：新stage和strict使用点分别核必要11件资产当前SHA，记录新完整7stat，保留旧chain/pin；跨Pod的device/inode可不同，新绑定和使用点身份须一致。`inherited_file`在stat匹配后仍核当前内容，如实记`weight_blob_hashes_recomputed=true`。两阶段内容证明替代stat-only假设，不扩来源语义/官方/历史扫描；9/10失败和诊断保留，修复后成功见下表。

后续B完整通过，原件见`records/pretraining/20261028-tiny22k/pod-51fe3daf/`：

| 回执 | 真实结果与Pod SHA |
| --- | --- |
| `REVIEW-TESTS-2.json` | 10/10；`bb801bd7752eb4a910d9a4b047efcade58835bb11eee7dfc29ebca7a92980587` |
| `STAGE-BINDING.json` | 新资产续验绑定通过；`7277013030069484cca9eac7d5ebc674225a54751edfa890338ed4780bd04a5d` |
| `STRICT-LOAD.json` | 全路径exit0；`9d74f10363d622b21e760811a6cfc323c42c63436c2d19ebb36e4a8d848c94f6` |
| `TENSOR-MAPPING.json` | 完整映射；`7741247a9be209399de643a591c94e05303254128abf3578a9dccb1be26db8c2` |
| `CPU-STRICT-RECEIPT.json` | wall12.873秒、runnable peak12、mask12–23、无affinity violations、task_tree_exited=true；`b1890ba1ca1e68653004ea2084220857462bfddec2bc3fbe33727ba9d97c8b8f` |

stage/strict各以`fresh_pod_sha256`核11对象228873307B，原device161与新device132分别保留。182原始/182TV严格加载missing/unexpected空；23LN1e-6、18GELU none、BN/buffers0、avgpool768、180backbone保留及fresh无age binary演示通过，无forward/GPU。B全部必要原件已交回，不覆盖两次真实失败；不再重跑来源模型。A生产binary+age/参数组/emptyAdam/EMA与迁移新路径仍是独立待验门。

A随后在新exp-v3小批资产上完成生产CPU18/18，guard exit0，无图片/官方语义/forward/GPU/clock：完整public1000 strict、182FP32 finite、23LN1e-6、18GELU none、无BN、fresh binary+age/agezero、183参数tensor（backbone27820128、head769、age2）、空Adam、online=EMA、初态checkpoint严格回读、八rank软件initial门及LR0/1599/1600/1601/6400、EMA1600copy/4800blend均通过。初态tensor SHA`eba53feeafdf20bc86855a40648bc453358928d148078c6ae78b65f059846cc7`。

真实原件根为A新prep下`production-cpu-1`：`CPU-RECEIPT.json` SHA`4dd5dfee07bcc662f2d8e4aca53c5a3a596421da101ed7233cfd33b115143d19`；`CPU-PREP-RECEIPT.json` SHA`5e811c0e517fd844ab0d74ae3289e97c23755d6f58c8bc1d68dadf4400d53b33`；guard tasks/...-ztn98nzp回执SHA`546c5a2f45fe62772a0a974f52b6023f16496db654cb4103acf4bc28167ac492`。包明确`scope=cpu_technical_prep`、uncommitted snapshot/base cb939，pin小批回执2025c819...；不是Git同步或正式初始化。A1已交14文件冻结，后续16项新binding及受影响site/candidate/once共29项、Pod语法仍待真实结果，不能用此18项代替启动链路通过。

随后A1集成29项已全exit0、source_unchanged=true：A新prep下`green-integrated-binding-1/RECEIPT.json` SHA`588e7589acf6772f9d7dab63029d3bb3ae0b3bb6ff0b600b00fb05dff53f6d38`，guard SHA`4f38f6b0529d7812b5625e4e8b033aad29fb9fef7ed80ec9e0b4c72d9bfeec28`。新GPU准入工具单独待CPU合同验收；此前已通过的来源/生产/启动检查不为新增工具重跑整套。准备采用0update/0forward/no-data的8rank native初始化+FP32 collective，300秒上限；不走旧100/250步benchmark，不声称已测forward/backward或吞吐。

新增GPU准入工具CPU11/11、AST/source unchanged、exit0已通过：A新prep下`green-gpu-contracts-1/RECEIPT.json` SHA`0bbba8fa08c841613bd3322d870567177c56b8b6b83dd7f1a8e34b9e82b1979d`；environment tasks/`19c0bc67-1a8c-462a-afb6-f6b80d090986-58mfcgrq/CPU-RECEIPT.json` SHA`08a0bcd3cc98886390dcf0cc590dba3bace6b99ab168cab862b761a4e3e408f3`。父A新增`tools/activation/prepare_gpu_site.py`与`tests/test_gpu_site_inputs.py`（3项），A1的`tools/validation/gpu_admission.py`与`tests/test_gpu_admission_contract.py`（8项），全量具体diff必须包含这4件新增文件。未实际GPU，原18+29不重复；生产snapshot与当前工具源码各按其回执保存身份。

## 环境变化与当前预算（按发生顺序）

1. B修复版strict待执行时，旧Pod `vcjob-2115367159301828609-acfwvv9ejn-96064-worker-0`、UID `f9fe6762-7127-479b-8e80-09ce2487d754` 在入口查询返回NotFound，`STRICT-LOAD-2`未进入Pod，未生成成功回执。A/B确认无未结束CPU/GPU命令，旧CPU窗口释放。
2. B此前唯一重新discover返回exit0、当前范围无Pod行；随后A一次discover查询到新既有候选 `vcjob-2115385083737251842-acfwvv9ejn-44186-worker-0`、UID `51fe3daf-ba0d-46e8-919c-99caec599c99`，Pending/Ready=False。该候选不是Agent创建的资源。
3. A单次有界Ready尝试在SSH控制面前exit255、`Network is unreachable`，未执行kubectl wait，因此当时Ready为unknown；未重复轮询。
4. 协调随后用同一可信连接只读查询成功exit0：context `context-cluster1`，namespace `acfwvv9ejn`，上述新候选UID不变，Running/Ready=True，container `vcjob-2115385083737251842-acfwvv9ejn-44186`。这是最新Ready事实，不将旧Pending或网络失败当最新状态；实际解释器/依赖、CPU约束、8卡/资产仍待A核验。

随后A在新Pod实算预算并回报资源，规划已按用户20%授权确认本次实际分配（见下表），无需再次例行许可。A统一共享资源发现，B不重复探测；不新建资源、定时探测、守护进程或本机/登录节点计算。新CPU阶段使用新独立prep/资产绑定，旧回执始终保留旧UID。

| 当前新Pod/CPU阶段绑定 | 真实回报与规划确认 |
| --- | --- |
| 身份 | namespace `acfwvv9ejn`；Pod `vcjob-2115385083737251842-acfwvv9ejn-44186-worker-0`；UID `51fe3daf-ba0d-46e8-919c-99caec599c99`；container `vcjob-2115385083737251842-acfwvv9ejn-44186`；node `10.5.15.38` |
| 资源 | requests/limits120CPU/472Gi/8DCU；实际affinity和cpuset0–127，cpu.max=12000000/100000，effective120，总20%=24 |
| A份额/新prep | 12核，mask `[0,1,2,3,4,5,6,7,8,9,10,11]`；`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/adaptation-cpu-20261028-51fe3daf-a` |
| B份额/新prep | 12核，mask `[12,13,14,15,16,17,18,19,20,21,22,23]`；`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/pretraining-cpu-20261028-51fe3daf-b` |
| 唯一启用allocation | `/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/environment-20261028-51fe3daf/CPU-ALLOCATION-2.json`；Pod SHA `25fe957398e81748c170d929b7b662b24270d1012e7f2940ec28ed315d9385ad` |
| guard | 同环境根下 `tools/environment/cpu_budget.py`；Pod SHA `b7ea98435b2a2ae07fa2ee146cfc5f35f86a1c8c346c475c594c03841805894f` |
| TTL原始事实 | created_at=`1793183303.3697946`，valid_until=`1793194103.3697946`；由Pod每次核有效期，过期停止当次，不自动续时或混用旧分配 |
| 新Pod8卡 | 8×BW65520MiB，PCI顺序09/36/55/77/85/b5/d5/f5:00.0；报告时busy0、used约2MiB；这是只读资源证据，尚无GPU计算 |
| 存储 | exp-v3报告时可用356391059456 bytes；未据此宣称数据迁移完成 |
| 解释器/依赖 | Python路径`/usr/local/python3.12/bin/python3.12`、版本3.12.7；metadata版本torch2.10.0+das.opt1.dtk2604.2607011947.g91a580、torchvision0.25.0+das.opt1.dtk2604.torch2100.2607021704.g7ffc50、numpy1.26.4、Pillow12.3.0；A已实际import torch/TV/numpy/PIL/timm，运行时版本分别2.10.0/0.25.0/1.26.4/12.3.0/1.0.27，GPU未初始化 |

前一个 `CPU-ALLOCATION.json` 因传输完成前读到旧guard、缺SHA/依赖字段，明确未启用且不覆盖。只有上述`-2`文件和精确SHA可用于本次阶段。A/B先经新UID/Ready核验、pin allocation/guard、各自实际import probe，再在互斥份额中并行CPU续验；这不放行Git发布、GPU或正式R93。

A的首次新通道probe已真实exit0：mask0–11、torch intraop12/interop1、Python3.12.7、gpu_initialized=false。回执 `/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/environment-20261028-51fe3daf/tasks/19c0bc67-1a8c-462a-afb6-f6b80d090986-y2revj_d/CPU-RECEIPT.json`，Pod SHA `12ed4daa35a9a3afdc294bffe4d5371680e8e5161c8b73d4165c18a5454bd2e6`。该命令只实际import torch，不能将其余metadata版本视作import通过；后续补充证据见下。

后续新增两份真实通道回执，均在同一environment根的`tasks/`下：

- B自身probe：`5d11d2c3-54b6-4037-ad23-a38f3849238f-rp72z_hn/CPU-RECEIPT.json`，Pod SHA `1362213c5156b38987879dea047df7b8854f8c6f9928d76409d45aa1d8e211af`；exit0，actual affinity12–23、intra12/interop1、Python3.12.7、torch实际import、GPU未初始化。
- A补依赖实际import：`19c0bc67-1a8c-462a-afb6-f6b80d090986-4zepnc81/CPU-RECEIPT.json`，Pod SHA `3aa17b58e5533c5bcb90326498540631daaa784a69ebaff84b5b0c9d3aa32367`；exit0，torch/TV/numpy/PIL/timm均实际import，mask0–11、nice19、intra12/interop1、GPU未初始化，无模型/forward。timm ConvNeXt实现SHA `a2ef5eb22cc1039dadcddcbf7b243034e98a553630349483302ed21d09fdb50f`，TV ConvNeXt实现SHA `5c00c412bc79e5c650f887dfe0d21aa8028903a3bbeae1a95c3b0fdfbdaf55c2`；timm activation文件及GELU.forward SHA与旧诊断一致。这些导入与实现身份不代替完整模型strict或生产工厂验收。

A/B已确认并验收预训练续验schema：`schema_version=1`、`stage=pretraining-strict`，`execution`恰为`namespace,pod,uid,container,python`，`cpu_allocation`恰为`path,sha256,executor_id,threads,mask`。mask来自通道JSON数组，actual_affinity/torch统计另记；pin旧source_receipt与新stage的path/SHA。`fresh_pod_sha256`当前内容证明与完整身份已通过，不填造历史字段。固定cb939c8来源snapshot与A生产工厂分别记录，不能互相替代。

## 旧Pod已有环境证据（不再作为当前准入）

以下为旧Pod仍可用时，新执行Agent在可信SSH配置、开启主机密钥校验的既有连接上，通过kubectl和Pod内只读工具取得的回报。规划只审阅文本，不在本机重算或测试；这些旧身份不替代当前新Pod准入，也不据此分配新CPU份额。

| 项目 | 已核事实 |
| --- | --- |
| Pod | namespace `acfwvv9ejn`；`vcjob-2115367159301828609-acfwvv9ejn-96064-worker-0`；UID `f9fe6762-7127-479b-8e80-09ce2487d754`；Running/Ready=True |
| 容器 | `vcjob-2115367159301828609-acfwvv9ejn-96064`；镜像 `image.ac.com:5000/dcu/acfwvv9ejn/base/ncccu:v1.0.0` |
| 加速卡 | 8 × BW，厂商 C-3000；每卡 65520 MiB；报告时 used=2207744 bytes、busy=0，无 KFD PID 或 Pod 内设备句柄占用 |
| PCI（card 顺序） | `09:00.0,36:00.0,55:00.0,77:00.0,85:00.0,b5:00.0,d5:00.0,f5:00.0` |
| NUMA（同顺序） | `0,3,2,1,4,7,6,5`；异卡报告 1 hop HSW，权重矩阵全零，不能据此推断实际带宽 |
| CPU/内存 | affinity 0–127，CPU quota=120 核；memory.max=506806140928 bytes |
| 解释器 | `/usr/local/python3.12/bin/python3.12`，Python 3.12.7 |
| 依赖 | torch `2.10.0+das.opt1.dtk2604.2607011947.g91a580`；torchvision `0.25.0+das.opt1.dtk2604.torch2100.2607021704.g7ffc50`；numpy 1.26.4；Pillow 12.3.0 |
| CPU probe | nice=19，最多 2 线程，CUDA/HIP/ROCR 可见性清空，gpu_initialized=false |
| exp-v3 | 已存在于共享挂载；报告时可用 356393156608 bytes，约 332 GiB；迁移前重新核容量 |

真实回执根：`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/environment-discovery-20261028-2n61dzfw`。

- `ENVIRONMENT.json`：执行报告 Pod SHA `6e87e4599e7791c57829ce68d8352c1b751524df96bec1075fd36199656d6718`。
- `ASSET-SUPPLEMENT.json`：执行报告 Pod SHA `fba468ad2433f403aa877227e88cf69e9784a3bd1f8c3475451a041aca686d70`。
- 旧 stage1 冻结 18 文件 SHA 匹配；train.csv 51396 行，SHA `ec81a23ba7758eccdd09dba027ce1a50998d81c18ed30ab6d2d608b3397f67d7` 与原合同一致。
- 旧数据根、prepared512 缓存与 Tiny1K、Tiny22K→1K 权重存在。源路径是待盘点迁移来源，尚未成为新正式绑定。
- Tiny22K→1K 转换权重 SHA `1ec1dd0b7045237eafe1c8f1becd58456cc811326c97ec416ae8b2e5ede860ed`；provenance SHA `40c14c8bd780d4feb88ae96d7e9def1ef78499daf93ae4412eff78502780233f`。
- 原 blob SHA `8cb65dc650a7fb25eaadbff3eb69559d02f83cdc5afdfb3785d1889cdc3e3518` 匹配；来源 manifest/model-api/config/LICENSE/download-receipt/NOTICE 匹配。B 已从历史原件找回缺失 README，传至独立 prep 的 `recovered-source/README.md`，Pod SHA 精确匹配 `5829109fcf28ec11af3f28fe4477a8319a4aaf39aa67a3ab7b4499e8d1720d4d`，不修改旧资产目录。
- B 报告来源链与历史 converter 身份通过，`SOURCE-CHAIN.json` Pod SHA 为 `0d46797900fdef0a479f68845ace7a1ce4ad83165cccfd22a7b4e91bacb75414`；独立准备根为 `/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/pretraining-cpu-20261028-execution-b-5d11d2c`。原始182键、转换182键严格载入/逐键一致性及LN等非张量结构正在CPU验收；生产binary+age工厂需绑定A适配后的实际源码SHA另行集成核验，不能用旧工厂或独立工厂结果替代。
- B 首轮 strict 在 GELU 非张量配置检查失败，`STRICT-LOAD.json` Pod SHA `ab29731f11f38f4a37c32c45c1ded10910a9b8cb283419db872851152d005dc9`；原始/TV strict、182键映射和23个LN1e-6已执行到，但整体验收未通过。当前只窄查实际激活类/approximate/实现身份，区分识别漏项与真实架构差异；不放宽匹配、不改生产GELU、不覆盖失败回执，不以张量一致性代替非张量语义。
- GELU 窄诊断已定位为验证器 `isinstance(nn.GELU)` 漏识别 timm 自定义类，非 tanh 差异。B 报告18个 source 实例均为 `timm.layers.activations.GELU`，forward仅调用 `F.gelu(input)`；18个TV实例均为 `nn.GELU(approximate='none')`。`ACTIVATION-DIAGNOSTIC.json` Pod SHA `11a913b723384184fd9ab187cd1d1e0c6585514f1d00fc3e5e1b9c89f24ce405`；timm forward SHA `375052b2c2b44c570274dee239db60f55afda25d413c98c827561f63a0ca5a77`。诊断未读权重、未forward。只修检查器的明确位置/精确类型/实现身份识别并保留unknown/tanh拒绝；生产GELU不改，修复后完整strict成功回执仍待取得。
- 修复后的8项来源窄测试已在旧Pod通过，`TESTS-2.json` Pod SHA `032841cd319bab2f097ba4f6fad091365a6e300558a2445a35ad7e3cb8651efc`，0 failure/0 error；随后strict复验因NotFound未执行。该测试回执只覆盖其记录的当时源码SHA，后续20%预算接口及跨UID续验绑定文本仍待窄验，不能据旧8项通过宣称新改动已验。
- B当时原样导回来源链、首次strict失败、激活诊断及8项测试四回执，未在本机或登录节点解析/重哈希；当时9件文件不是最终清单。后续增加stage/file identity工具、新Pod失败/成功原件与CPUstrict回执，完整交付见[来源阶段记录](../../records/pretraining/20261028-tiny22k/README.md)和[续验记录](../../records/pretraining/20261028-tiny22k/REVIEW-FOLLOWUP.md)，A须纳入全部untracked实际内容。

Git：本地 main 为 `cb939c8`，执行报告 GitHub 连接器读到远端相同 commit；终端 fetch 的 HTTPS 认证未解决。连接器存在写工具不等于实际同步成功。协调收到对“未知变更集写 main”请求的自动审批拒绝；应先形成具体文件清单、完整差异和 CPU 真实回执，再由协调处理可审阅同步动作，不能绕过拒绝或把尚未推送记为已同步。

协调随后核明连接器`create_tree`可批量inline content及`sha:null`删除，返回tree须匹配本地已审阅暂存write-tree，A仍唯一Git写。另实证cb939 API作者日期显示UTC而实际Git对象原时区+0800，unsigned payload为空，因此不依赖API字段重建原对象。只在实际对象SHA匹配后导入本地ref；若唯一remote commit、代码tree和部署SHA齐全而local HEAD暂不能无损取回，则如实记差异并后置元数据归位，不造第二实现commit、不假称fetch通过、不乱reset。具体diff/CPU→远端父租约/非强制提交与复读→Pod SHA冻结顺序不变；协调未写远端，此处未声称新同步已经完成。

## 执行顺序与最小文件任务

当前入口已去除旧R92生成配置/证据共9件：`runtime/{TRAIN_CONFIG.json,ASSETS.json,CLOCK_TEMPLATE.json,FRESH-SPEC.template.json,RUNTIME_BINDING.template.json,SOURCE-MANIFEST.reference.json}`及`runtime/evidence/{CPU_PARAMETER_GROUPS.json,FRESH_EMA_CONTRACT.json,MODEL_CPU_ADMISSION.template.json}`。历史在Git/records/R92追溯，旧服务器未改。新`configs/R93-SCIENCE.json`与`tools/activation/prepare.py`按显式site/迁移回执生成当前身份；PLAN为R93、CLOCK_POLICY.round_id由site注入。目录说明不将旧模板当新准入，具体同步差异须包含删除项及新增文件。

1. 只读识别与资产核验。`tools/environment/` 增补显式输入、无默认主机/Pod/解释器的清单工具；输出实际 UID/Ready、依赖、PCI/资源/占用、资产文件和空间回执。官方文件只做必要存储身份盘点，不读取官方标签用于适配决策。
2. 代码适配。沿现有 `runtime/` 演进，以显式绑定替换旧主机和四卡常量；保留所有科学与一次性控制校验。具体边界见下表。
3. 在实时核验的Pod独立 `technical-prep` 做必要CPU检查，nice19、隐藏GPU，A/B合计≤实时有效CPU的20%，按显式份额和互斥mask允许并行。所有解析、哈希、模型载入、模型分析和测试都在Pod；不取训练锁、不发训练信号。
4. 执行提供完整具体差异、待同步文件列表和 CPU 回执，统一提交并成功同步 GitHub，核远端实际 commit；从该版本按实际文件 SHA、配置身份与资产清单冻结新部署到 `exp-v3`。**同步及部署核验前不得进行任何新代码 GPU 试跑、准入或训练。**
5. 唯一执行 owner 串行完成本轮选定的有界 8 rank 准入：0update、0forward、no-data，保存 rank/PCI/CPU/初始化/FP32 collective/退出证据，不声称已验梯度或吞吐；技术产物不可用作正式初始化或候选选点。正式 R93 另取 fresh 零状态，不继承技术验收的优化器、EMA、clock 或 token。
6. 规划根据资产与验收回执冻结 R93 候选合同；按一次性新绑定启动 6400 成功更新，内部 dev 选唯一 checkpoint，完整官方 5849 最终只评一次。结果核实后再决定下一候选。

| 源码范围（执行负责） | 必要改动 | 核心验收 |
| --- | --- | --- |
| `tools/environment/` | 实时资源/资产清单、真实回执和显式参数；无固定环境默认值 | UID/Ready/容器、解释器、8 卡和占用、资产身份齐全；无 GPU 初始化 |
| `runtime/n08_topology.py`、`bundle/retrain/execution_policy.py`、`distributed.py` | 使用实际 world/devices；复用已有 8 rank 支持，不重写全局采样算法 | reference32；8×8×1=64；全局 P8；4 rank/8 rank 重组样本流一致 |
| `runtime/r92_initial_gate.py`、`launch_once.py`、`run_round.py`、`run_training.py`、`component_gate.py` | 显式运行身份和动态 rank 覆盖、PCI/设备准入、解释器、路径、结果名与单实例锁；旧历史运行不触碰 | 缺 rank、重复 PCI、过期身份、错误源码/配置、重复 launch 拒绝；成功与失败均真实退场 |
| `tools/activation/` 与相关 observer/validation 调用链 | 新绑定/clock/token 从实核清单产生，路径落 exp-v3；不复用旧 R92 专用入口 | 仅新运行一次 activation；拒绝覆盖、重置或改绑；未知观测只记 unknown |
| `bundle/retrain/experiment91.py`、模型来源校验、fresh/precision 相关合同 | 将候选来源、元数据根与 LN 身份显式配置；去掉仅允许 physical16/32 的旧约束，按解析拓扑校验 | 固定 split/age/schedule；候选精确来源和模型结构；FP32；无旧 world4 resume 文案或宽松 fallback |
| `runtime/*.json`、`configs/` | 新模板/本轮配置区分科学值与环境绑定；清单逐文件 SHA；旧值由 Git 历史追溯 | 不是把旧 UID、exp-v2、四 PCI 机械替换；不默改科学字段，不接受 N06 训练依赖 |
| `tools/validation/`、受影响目录检查 | A 添加上述行为的窄 CPU 检查与可核回执；B 专属来源检查按其文件边界实现 | 见 [ACCEPTANCE.md](ACCEPTANCE.md)；不把文本审阅当已通过 |

## 存储与迁移清单

正式新部署、数据副本、预训练、缓存、日志、预测、checkpoint、技术准备及验收输出统一落 `exp-v3`。源码在 GitHub，服务器冻结源码只是运行产物。旧 `exp-v2` 和数据目录先作只读来源；本阶段不盲删、不全量搬迁。

| 对象 | 迁移/绑定要求 |
| --- | --- |
| 固定数据与 prepared512 | 列真实文件清单、容量/硬链接/符号链接/挂载和当前依赖；核预计峰值及余量后按需复制到 exp-v3；不能只改路径或在未知容量下复制整棵历史目录 |
| stage1 的 18 件冻结元数据 | 保留原 bytes 与原 SHA；新路径引用单独建立 relocation/资产绑定，明确物理路径变化与科学身份不变，不能改旧 FREEZE 冒充原件 |
| 当前候选公开权重与来源原件 | 仅迁入实际待用候选的完整来源链、转换权重与精确 SHA；旧 Tiny1K 只在本次兼容性窄检查实际需要时纳入 |
| N06、R17、R8-C25 | 仅按权威保留清单核存在与身份；不作为 R93 初始化。迁移和清理不能破坏唯一对象；N06 继续研究隔离 |
| 旧历史运行/预测/checkpoint/缓存 | 只读核真实依赖；历史清理另按原清单接续，不因本次八卡适配扩大删除范围 |

正式运行前要证明新绑定所引用对象确已到位且 SHA 正确；存储未就绪不得仅凭旧路径存在通过。资产缺失或容量不足只阻断相关迁移/正式准入，不阻断独立文本适配和已具备条件的 CPU 检查。完整产物留 exp-v3，仅必要小回执和精简结论入 Git。

A迁移前inventory已真实exit0，回执位于A新prep的`asset-inventory-1/INVENTORY.json`，SHA`3c60a944cdd43bbbd241cd87bbca1bd9fdb0650630f3f709ea4c38365fa422a2`，逐件`FILES.jsonl`由其中`file_inventory_sha256`固定。raw63103文件/40254109282B，cache171745文件/34939102849B，均无symlink；stage1共19文件/23776876B，公开original7文件/114437877B、converted3文件/114424744B、恢复README15750B。official manifest仅stat2182932B，三个保留checkpoint仅stat，不作训练依赖。报告可用356297736192B。

A已向B交`MIGRATION-DEPENDENCIES-1.json`（同A新prep根），Pod SHA`88bfb9d67a92894131dd2ed1408225951d7fd5fe0720aff5468a06e3cf318dc9`：18冻结件+FREEZE共19件、train_identity字节身份与cache分组/首条policy已核。B只读消费`tools/environment/inspect_migration.py`及`configs/EXP-V3-ASSET-SOURCES.json`，不重复盘点、不共写A文件。正式目的布局如下，根均为`/public/home/acfwvv9ejn/SothisAI/exp-v3`：

| 来源清单项 | 根下目的路径与要求 |
| --- | --- |
| `raw_glaucoma_dataset.path` | `data/glaucoma_first`；原始文件bytes不变 |
| `train_identity` | `assets/train_identity.json`；13825613B，SHA`07f8e8765b66d671303d613afa1839c0685d0f38edec63fce0d0ba0400f62c66` |
| `fixed_9010_stage1_18_objects.path` | `assets/fixed-9010-stage1`；18件+FREEZE原bytes/SHA不改，canonical/source_identity保留，物理路径另记 |
| 公开original/converted及恢复README | `assets/public/tiny22k-bc48a87f/{original,converted}`；README进入original，转换文件名及`../original/LICENSE`引用保留 |
| sealed官方manifest | `assets/official-valid-sealed.json`；仅存储复制/哈希，不解析标签调参 |
| `prepared512_pillow123.path` | `shared-assets/prepared512-v2-pillow12.3.0`；只迁实际train/sealed与valid/sealed，arrays字节不变，新索引另立 |

旧cache两split namespace均为`19b2e1abff37369711cf59110dd52a0fb4894dd35037ee4b3eb758f8bd201a48`，train/sealed102792文件、valid/sealed11698文件（每样本npy+json）。policy为version=`rgb-conservative-border-prepared-v2`、prepared_size512、pillow12.3.0、dataset_root=原rawroot；新policy只替换新dataset_root，namespace沿原排序JSON哈希规则，key保持relative_path+NUL+source_sha哈希。新source_file记录新raw实物inode/size/mtime_ns/ctime_ns，并验raw内容SHA与原source_sha、当前arraySHA与原arraySHA一致，不能仅凭stat。旧builder.lock/clock/progress和verification不作新阶段证据。B先交小批资产供A生产验收，再并行推进bulk复制与索引代码；最终RELOCATION记录绝对目标、逐件清单SHA、两split计数/字节一致、source改动false及当前UID/allocation/guard，A缺目标即拒绝，无旧路径fallback。

小批复制已真实exit0：起拷`2026-10-28T11:07:36.271601Z`（北京19:07:36），32文件268663792B，wall6.118秒。B迁移prep下`SMALL-COPY.json` Pod SHA`2025c819bcd56390e0c6d8ae9eba4fbefcf27496b6d378ae84aecf0283fd8d66`，逐件`SMALL-COPY.files.jsonl` SHA由小回执manifest字段固定；CPU回执SHA`69fdaed75367051169386334eca473293449e1f373d6c91650eab2eae7196719`。上述stage1原19件、train_identity、公开原/转换/README及opaque官方manifest已到位，源流SHA与独立目标当前SHA一致；源和FREEZE不改。小批任务已退出，已通知A消费实际目标做生产验收；B继续raw/cache，bulk与新索引尚未通过，不能称全迁移完成。

bulk真实进度：`BULK-COPY-2`于`2026-10-28T11:14:34.853677Z`（北京19:14:34）起拷；`11:15:31.250895Z`已6144文件/3885813260B，计划120348文件/75104986850B，量指源流SHA及目标独立当前SHA已核的复制对象。B12内11copy线程，A12独立。首次bulk因启动回执缺passed字段在0B退出，失败回执保留，第二次不覆盖。B按初段进度估bulk另需15–25分钟、索引另5–15分钟，后者未实测，两者均非开训承诺；不增加benchmark。B1两文件已冻结，父B待bulk退场后在自身12核跑6项必要窄验及两split完整重绑，未完成不得记通过。

后续bulk已于`2026-10-28T11:32:44.926183Z`完成，120348文件/75104986850B，完整阶段wall1154.566秒，源流SHA与独立目标SHA一致、source_modified_by_tool=false；原件见`records/migration/20261028/BULK-COPY-2.json`，逐件清单SHA为`ac23aac59391f4d459b4c7b753bc7695a36f8c54735d4ee414bb21ad0322b05c`。6项cache窄验已通过；新索引从11:36:28Z开始，B于11:41:05Z报告train11776/51396，读取新raw与新NPY作使用点当前内容证明，不重复制或读取旧raw/NPY。该时点估计余18–25分钟，覆盖此前索引5–15分钟估计；仍非完成/开训承诺，终态以两split和RELOCATION回执为准。

迁移最终于`2026-10-28T12:01:47.253812Z`取得`passed=true/phase=complete`，完整复制120380文件/75373650642B，train51396与valid5849索引全过，共57245新JSON、reuse0；新namespace=`429cbdbda7d0827c564d279fcfcef12d5bf3784f127c7226841d9b5948fee960`。NPY bytes/key与原FREEZE/canonical/source_identity均不改，最终可用280355667968B。3复制窄项、6缓存窄项及两个split完整重绑通过，全部原件见`records/migration/20261028/`；大manifest与数据留exp-v3，不force-stage被忽略的manifest。

最终迁移回执为B迁移prep下`RELOCATION.json`，Pod SHA=`be516bfec741dd9c7f5cc4b41694be562c9e1fceb195fa1e41e535b715f9273e`；最后guard为environment prep下`tasks/5d11d2c3-54b6-4037-ad23-a38f3849238f-nc1vx8nk/CPU-RECEIPT.json`，Pod SHA=`f603ad360051f111335d52e4839c831b0d838d7c71d46d507a38745ca5922b24`，exit0、task_tree_exited=true、affinity_violations空、runnable预算通过。规划文本核对原件后接受迁移子门及B退场；此结论不替代A的Git同步、部署和GPU验收。B无在跑lease或待改文件，后续仅回应必要整合问题。

正式前租期预查询：当前Pod UID/Running/Ready仍匹配，startTime=`2026-10-28T10:08:03Z`，Pod未声明activeDeadlineSeconds；真实上级Volcano Job UID=`ad1c7f20-6d09-46a7-b739-782881ca5b09`，读取被RBAC Forbidden。上级租期保持`unknown_not_an_unlimited_lease`，不越权、不解释为无限、不新增门禁；正式前仍取新Pod元数据和观测时间，已有声明期限须满足冻结预算。

bulk后续只读进度（未起新任务/benchmark）：`2026-10-28T11:25:36Z`已71168/120348文件、45159523554/75104986850B，源流SHA及独立目标SHA复制继续在B12内。raw部分已过，正在cache npy，尚无终态；缓存6项窄验与两split新JSON索引待当前lease自然退出后执行。工具已冻结/部署，最终RELOCATION schema已与A对齐；metadata脚本一次SSH传输reset未影响现有copy会话，只在后续phase前补传，不重启复制或扩大预算。

## 后续阶段CPU预算

CPU准备A+B及其子Agent计算合计≤`floor(effective_cpu / 5)`；模型训练及代表训练资源的GPU准入整树≤`floor(4 * effective_cpu / 5)`。若实时有效CPU仍120，则准备总24、训练总96。B已退场，旧allocation-2与首轮GPU绑定保留；未启动阶段按最新语义另建绑定，不热改活跃任务。准备若并发，其mask须与训练互斥且合计不超设备有效CPU；子Agent现仍仅文本，不另占Pod预算。

A提出在实时120核成立时为训练owner分配96：8rank各11核共88、control8，loader1、torch intra/inter1，helper亦受整树80%约束；这只是待实核的新阶段资源方案，不能把helper全128视作合规。不建设通用调度。GPU设备可见性按独立入口处理，不继承technical-prep隐藏GPU环境。G64/P8、全局样本/增强身份、800×8成功更新、LR/EMA及官方隔离不变；不以旧20%性能或全量CPU表现担保新阶段速度。

## 科学合同与 R93 候选

固定科学合同见[实验任务](../auroc-999/README.md)。8 卡适配保持 9:1、fit46256/dev5140、全局 G64/P8、512、FP32、成功更新 S800×8=6400、warmup/EMA warmcopy1600、EMA decay0.999、backbone LR2e-4、head/age LR1e-3、weight_decay1e-4，以及既定 16 条 dev 唯一选点顺序。不要按卡数线性放大学习率，也不要按每 rank 的数据长度重算周期。

R93首候选为公开ConvNeXt Tiny ImageNet22K→1K，**来源链、原始/转换模型strict及A生产CPU18项已通过；启动链路和正式部署仍待验**。保持native avgpool/768/dropout0.2/linear+fit-only age、内部LN1e-6、SD0.1；末端LN按来源为**1e-6**，与R92的**1e-12**不同。不能沿用错误LN来宣称“只变来源”，也不能把将来差异全归因于预训练。候选revision/来源、严格映射和生产fresh结构分别由实际CPU回执固定。

R91/R92 已完成且未达目标，不重跑其官方前向。技术验收不形成新的科学基线，跨 4/8 rank 的浮点归约及 rank RNG 可能不同；可验收全局样本流、更新语义和初始化一致性，不承诺完整训练 bitwise 相同。每轮先核内部结果与工程有效性再冻结唯一 checkpoint；官方结果仅作预定最终报告，不用于下一轮调参、难例挖掘或模型选择。若 R93 无效，先修工程问题，不将其作为科学结论；若有效但未达标，以内部 dev 的预定汇总及合规候选依据形成下一轮书面合同，不依据官方样本或官方排名挑方案。

## 阶段待办

- [x] 新规划、执行 A 与新增执行 B 身份均已核实；A/B 已接单，唯一 Git/GPU owner 仍为 A。
- [x] 当前 Pod、8 卡/PCI/NUMA、空闲占用、CPU 依赖与 exp-v3 挂载已有真实发现回执。
- [x] train.csv 与 stage1 18 文件身份匹配，Tiny22K 原 blob/转换权重和已找到来源原件 SHA 匹配。
- [x] B 补齐精确 README 原件，来源链和历史 converter 身份已有 Pod 通过回执。
- [x] B保留GELU首轮失败/诊断及旧8项通过；新Pod10项回归、stage、完整strict/映射已通过。
- [x] 新既有Pod UID/Ready、Python版本、依赖metadata、实际CPU/8卡已有回报；规划确认精确allocation-2及A/B12/12份额、新独立prep。
- [x] 新预算通道A/B各自probe通过；A补齐实际依赖import与相关源码身份，双方在各自12核份额并行受影响窄验。
- [x] 新资产续验stage和受影响stat/跨UID10项通过，完整strict全路径通过且CPU进程树退出。
- [x] A/B各新建1名文本子Agent，父级身份/文件边界已交接；继续当前实例及A/B12/12。
- [x] A生产binary+age/参数组/空Adam/EMA等18项真实通过；技术初态不复用为正式初始化。
- [x] A启动绑定/一次性控制及受影响29项真实exit0/sourceunchanged，记录固定源码身份。
- [x] 新GPU准入工具CPU合同11/11、AST/source unchanged、exit0，4件新增文件纳入待审diff；实际GPU尚未执行。
- [x] B小批资产32文件268663792B交付，实际起拷/耗时/逐件SHA回执齐，A可读目标集成。
- [ ] B raw/cache实际复制和新索引身份验收，补bulk真实进度/耗时。
- [ ] 完成最小代码适配与 exp-v3 实际迁移清单、容量和对象身份核验。
- [ ] 必要 Pod CPU 检查通过并留真实回执；规划审阅具体差异。
- [ ] 执行统一 GitHub 同步成功，核远端 commit 和冻结部署逐文件 SHA。
- [ ] 唯一 owner 有界 8 rank GPU 准入通过，所有技术进程退出。
- [ ] R93 精确候选合同与新正式一次性绑定齐全，串行执行并回收真实终态。
- [ ] 内部/官方分报；只有真实 raw AUROC 达到 0.999 才能标总体目标完成。
