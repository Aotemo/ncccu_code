# 设备迁移接续审计与单一后续草案

用户已迁移设备并明确解除暂停，继续挑战赛；本轮先核现有连接、计算 Pod 和 R130 第 1503 步 checkpoint 能否按原合同续接。**源码审计结论：原 one-shot 启动链没有已终止 owner 的合法恢复入口，不能按原合同直接续训。** 当前连接信息缺失，Pod、资源、资产、原 deadline/grant 的实时状态均为 `unknown`；不能据此声称它们不存在、不可达或已过期。总体 raw AUROC 0.999 尚未达到。

本页独立接续 [停止交接](DEVICE-HANDOFF-20261101.md)，不改 R130 历史结果、决定页或冻结产物。下述新 fresh 仅为具体方案，尚未授权启动。

## 当前角色与实查范围

当前 Paseo server 为 `srv_KqW7XmJdXxGJ`，本轮新建配对为：

| 角色 | agentId | 范围 |
| --- | --- | --- |
| 根协调 | `b2a18d8d-628c-4c7c-9cf7-9345e736e0ac` | 用户沟通、接续安排 |
| 规划 | `6b8ad6f4-d29a-4647-8fcf-d986715575f0` | 文本只读审计和单一方案 |
| 执行 Sol / A | `7c17043e-8381-43e9-b185-6e142634ad72` | 唯一 Git 写入、Pod 操作及潜在 GPU owner |

`list_hosts`、`list_agents` 和两位新 Agent 的 `get_agent_status` 已实查；规划与 A 为 running。无 B、无派生 Agent；旧 Agent 不恢复。本机仅文本、Git、连接来源检查与展示，没有本机或登录节点 Python、JSON 解析、哈希、模型加载或测试。

| 连接来源 | 本次实际结果 |
| --- | --- |
| Paseo 主机与终端 | self 如上，注册远端 `hosts=[]`；`list_terminals(all=true)` 为 0 |
| Paseo workspace | 父目录 `/home/hao/Desktop/challenge` 与本仓库；父目录 `other/` 为空 |
| SSH 配置 | `~/.ssh/config` 缺失；`~/.ssh/` 仅有 known_hosts 和 old，无密钥文件；系统配置只有通用默认，include 目录为空 |
| 环境与认证 agent | `NCCCU_SSH_CONFIG/IDENTITY/PORT/TARGET`、`SSH_AUTH_SOCK/SSH_AGENT_PID` 均空；`ssh-add -l` 无 agent 可连接 |
| Kubernetes 本地入口 | `KUBECONFIG` 空、`~/.kube` 缺失，本机未发现 kubectl；项目要求仍经既有 SSH 连接动态发现 |
| 迁移小原件 | 此设备没有交接所引 `../work/r130-20261101-a/stop/`；仓库中的历史 RESULT/README 可文本读取，不能冒作这次实查原件 |

没有从 known_hosts、入站 SSH 环境或历史地址猜目标，没有尝试旧地址。根已一次性向用户索取当前 SSH 目标/已有 config 的绝对路径及认证方式；不要求把密码、私钥或 token 发进聊天/Git。连通后 namespace、Pod、UID、容器和 Python 应由工具发现，无须用户沿用旧常量。

目前未执行 `pod_cpu.sh discover/interpreters/budget/probe/run` 或任何 Pod 命令，未创建 technical-prep、软件 CPU allocation/TTL、训练 grant、clock 或 token。没有部署、训练、GPU 前向、官方评估、大 checkpoint hash/load 或已通过 fixture 重跑。实时安全、8 卡类型/显存/PCI/NUMA、CPU affinity/cpuset/quota、内存/磁盘、进程/设备使用者/锁、依赖与资产可见性全部未核；旧 safe 仅在原观察时成立。

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

历史 last 为 `checkpoint-806c3832c05e487d88c938245ac1d3e2.pt`，提交 index 广告 SHA `08ac68114696b7bd698fb8e70658f7cfbeecf994088a023a115b7aa0591eb377`、792176842 B。精确 exp-v3 相对路径保留在停止交接及 RESULT。**index/sidecar 仅提供文件名/SHA 等提交记录，没有 1503 游标字段；1503/eval1/0skip 来自原 STOP/RESULT/status 的保存描述。** 此次尚未对远端 index/sidecar/stat 复验，也未 hash/load 大 blob，不能冒称 payload 完整可恢复。

`distributed_engine.py:523` 的保存合同包含 model/Adam/scaler、8 rank RNG、成功/尝试/评估游标、early_stop、online/EMA best/last、coverage/timing、schedule/fresh_guard/absolute_deadline；`:578` 先保存再记录 interrupted。`:240`、`:288` 的底层恢复要求完整身份、deadline、initial 和优化器组映射一致，再恢复 Adam/RNG/游标/EMA。这些源码能力不等于此 checkpoint 的逐字段加载实证。`persistence.py:337` 的 `load(last)` 可能回退 previous 并写 recovery，当前不调用它探测，更不能把回退冒称恢复 1503。

保留原 last/previous/best/initial、best_ema/best_online/snapshot-800、R130 全部 source/manifest/cache、R120 selected/source 及 KEEP-MODELS。没有清理或移动原件。

## 连接补齐后的有界核验

1. A 用当前连接运行现有 `pod_cpu.sh discover`，核 namespace/Pod/UID/Ready/container；多个候选按 R130 真实 binding 核对，不自动选第一个或改绑旧运行。再以 `interpreters` 核绝对 Python 路径。
2. 在已核 Pod 新建独立 exp-v3 technical-prep，只为未来准备阶段建立新软件 allocation/TTL 与 CPU guard。按实时 affinity/cpuset/quota 和版本化 RESOURCE-POLICY 算 `floor(20%)` 准备与 `floor(80%)` 训练互斥池，不沿用旧核数；仅 A 可执行，nice19、隐藏 CUDA/HIP/ROCR，不改旧 allocation/grant/clock。随后核依赖、8 卡实际资源、内存/磁盘/进程/设备使用者/锁，观测不完整记 unknown。
3. 一次读取必要小件，脱敏报告原 round-clock implementation/total、训练 hard/soft、contract absolute 的未过期布尔及旧 grant/prep entry TTL。正确关系为 `contract.absolute_deadline_at == train-phase-start.soft_deadline_at == train hard - 150`，不是等于 round 总 deadline；不打印 private clock/token。payload 内 deadline 未 load 则仍未证。
4. 核原 dispatch/pod-admission/controller-start/owner/train-phase-start/train-component-terminal/done 的存在与必要 terminal 状态；只核 index/sidecar/stat、initial/previous/best/protected 与真实 source/manifest/cache 依赖的可见性，不盲扫描/搬移。完整证据写新 technical-prep，仅导回必要小脱敏件。

这组核验补齐设备与保存现状，不试旧启动，不因元数据一致而加载大 blob 来重复裁定原入口。只有日后另获明确 checkpoint 继承合同，才评估一次精确 last 验证；本次不实施继承。

## 单一后续方案草案：独立 fresh R130R1

**需另有明确新 fresh 授权，当前不启动。** 暂名 R130R1，正式 order/config/candidate/round 身份由协调者冻结，不能复用 R130 order40 假装原轮续训。目的仅为完成被用户中断的固定 0.8 逐层 LR 问题；同 seed20261029、Small 公开 fe6b358a 与原原生 head 初始化，从 step0/空 Adam 开始，不使用 R130 第1503步或 initial checkpoint 作为初始化。旧 1503 步和成本保留，不拼接曲线或补作新轮前1503步。

全部原 R130 科学值固定：9:1 fit46256/dev5140；单 Small/GAP+age、prepared512/scale[.85,1]、原旋转/光度、G64/P8 八 rank、FP32/noTF32、pairwise logistic；原 layerdecay .8 分组映射、backbone1e-4/head-age1e-3、WD1e-4/SD.1/drop.2、一次 globalclip1、从 step1 全层更新；最多24000次成功更新、每800双 variant 内部 dev、连续4轮 AUROC 无提升早停、原1600 warmup 与 EMA/唯一 selector。独立新轮使用原6h训练/8h整轮/9hgrant预算，不延长旧 R130 时钟。

完整工程有效 selected internal AUROC 须同时严格胜 R107 `.9884405731067525` 与 singlebest R120 `.9894773092062407`，才具备固定 seed27 科学复核资格；草案不自动授权第二轮。未同时胜即关闭本固定方案，不 factor/LR/分组/seed 补考，AP/partial/official 不救主判。完整 freeze selected>=.996 才官方5849一次，完整 selected 不足才 `not_run_by_policy/threshold_not_met`；工程中断不套此标签。总体 raw .999 目标保留。

最小实现仅新增科学身份及必要 `experiment91.py` 映射（当前 `LAYERWISE_LR_SMALL_BINDING` 固定 candidate/order40，仅改配置文件名不足），复用既有 layerwise、保存和直接训练链。源码变更前汇报具体目的；新代码实际 push/readback 后精确部署。只核新增身份/实际迁移影响的窄绑定及现有必要 formal，不重跑已通过的3项算法 fixture，不做独立 GPU 预演；真实训练8 rank 初态/组/空 Adam 接管 GPU 验收。

仍由 A 唯一串行 GPU，训练80%整树与准备20%互斥；Small12GiB、启动 free20GiB/运行 floor16GiB、300s atomic/prune/failfast 与全部资产保护保持。不新建计算资源、不关闭实例、不续租、不恢复 timer/heartbeat。新设备吞吐、剩余容量、满24k耗时、500MB包体或100ms均未证。

当前只落此独立审计和 todo 页首接续，由根处理连接信息与后续合同授权；不自行 fresh、继承 checkpoint、seed27 或改方法。
