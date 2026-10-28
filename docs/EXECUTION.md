# 动态发现执行环境

项目不指定永久主机、Pod 或 Python 路径。环境来自工具实时查询；旧回执用于核对已经绑定的运行，不能替代发现步骤。

## 发现与绑定

1. 使用 Paseo `list_hosts` 查询当前 server，再用 `list_agents`、`get_agent_status`/`get_agent_activity` 核实 `todo/` 当前授权的规划、执行 owner、CPU 执行者及连接上下文。保留 `{serverId, agentId}` 配对；不要唤醒旧Agent或创建计算资源。本次用户允许A/B各自行按执行profile新建1名文本子Agent；子Agent无Pod命令/Git写/GPU/再派生权，父级统一部署验收。当前实例继续使用，分工见[本轮任务](../todo/exp-v3-8card/README.md)。
2. 通过已有连接工具或 SSH 配置取得实际连接目标。执行 [pod_cpu.sh](../tools/environment/pod_cpu.sh) 的 `discover`，使用 kubectl 当前上下文查询 Pod 的 namespace、名称、UID、运行/Ready 状态和容器名。多个候选须按当前任务的真实绑定核对，不能自动选第一个。
3. 使用 `interpreters` 在选定且 UID/Ready 核对通过的 Pod 内查询解释器绝对路径；然后执行 `probe`，在该 Pod CPU 内核 Python 版本、torch/torchvision/numpy/Pillow 和未初始化 GPU 的事实。多容器 Pod 必须显式指定查询到的容器。
4. 保存实际发现结果，按任务绑定核对实际卡数、类型/显存、PCI/NUMA、CPU affinity/quota、内存和设备占用，再核数据、缓存、公开权重、工作根与源码清单。卡的逻辑序号与物理PCI需实核，不能由旧绑定或仅凭卡数推断。发现新环境不授权改绑活跃任务；已有运行继续使用原冻结副本。非源码产物写入用户指定的 `exp-v3`，旧资产按真实清单核容量与依赖后受控迁入，禁止盲搬。
5. 用 `run` 执行已传输到各执行者独立 `technical-prep` 的 Python 工具。每次执行重新查询 UID/Ready，隐藏 GPU、nice 19；按最新用户授权，两位执行合计最多使用实时有效 CPU 的 20%，初始平分、允许并行。规划按实核 affinity/cpuset/cgroup quota 分配份额及互斥 mask，不沿用旧核数或默认份额；身份、预算、文件或依赖不符立即停止。无本机或登录节点回退。

环境变量只提供当次工具返回的连接信息。可选 `NCCCU_SSH_CONFIG` 指向已有 SSH 配置，或使用 `NCCCU_SSH_IDENTITY`/`NCCCU_SSH_PORT`；不要将凭据或机器专用配置提交到 Git。`NCCCU_CONTAINER` 仅使用当次查询到的容器名。

CPU新额度覆盖旧2线程/重CPU串行限制，既有2线程回执仍保留原事实。通道显式读取 `NCCCU_CPU_ALLOCATION`（Pod内JSON绝对路径）、`NCCCU_CPU_ALLOCATION_SHA256`（Pod实核的小allocation文件SHA）、`NCCCU_EXECUTOR_ID` 和 `NCCCU_CPU_GUARD`（部署的Pod检查器绝对路径）；没有默认分配。allocation绑定 namespace/Pod/UID/container/Python 五项实核身份、实时 affinity/cpuset/quota、`max_total_threads=floor(effective_cpu/5)`、`allocations[executorId]={threads,mask}` 及 `created_at/valid_until`。所有mask必须为可用CPU集合的互斥子集，长度等于份额，总和不得超过20%上限；预算为0或无当前Ready环境时不派发CPU任务。

通道在Pod内验预算后设置进程affinity，导出 `NCCCU_CPU_THREADS`、`NCCCU_CPU_MASK`（固定为JSON整数数组）及等份额的OMP/MKL/OPENBLAS线程限制，子进程继承mask。执行工具只消费自己的份额，torch intraop按份额、interop=1，不另行占用一份总20%或无预算增开workers。采用现有通道、互斥affinity及线程限制即可，不建设额外调度服务或守护进程。CPU额度变更不改变batch、采样、LR、EMA或其他训练科学参数。

CPU份额可在现有任务自然退场、实际lease释放后，为后续未启动CPU或GPU阶段重新实核并显式分配；不热改活跃任务或侵占其他份额。B迁移已完成、任务树真实退出并释放份额，规划已确认A可为尚未执行GPU阶段按实时资源另建owner总20%绑定；若仍有效120核则最多24核，但未生成实核新绑定前不声称已启用。旧12/12 allocation和回执保留。文本子Agent不占另一份Pod额度，验证由父级执行；若以后让子Agent进入Pod，必须先在父子合计20%内划明确mask。

以下变量均应填入实时工具输出；例子不含任何机器默认值：

```bash
bash tools/environment/pod_cpu.sh discover "$SSH_TARGET"
bash tools/environment/pod_cpu.sh interpreters "$SSH_TARGET" "$NAMESPACE" "$POD" "$POD_UID"
bash tools/environment/pod_cpu.sh probe "$SSH_TARGET" "$NAMESPACE" "$POD" "$POD_UID" "$PYTHON"
bash tools/environment/pod_cpu.sh run "$SSH_TARGET" "$NAMESPACE" "$POD" "$POD_UID" "$PYTHON" "$REMOTE_SCRIPT" --help
```

此入口只提供 CPU 准备与只读检查的执行通道。它不获取训练实例锁、不发训练信号、不启动 GPU 训练，也不产生新的训练 token/clock。完整发现、CPU验收及日志保存在用户指定的 `exp-v3/technical-prep`；只导回必要精简结论与小回执，本机只传输和展示，不解析、重算SHA或测试。

原Pod失效时停止使用其UID，保留历史回执。由A共享一次只读发现结果；若出现新的既有任务Pod，需先核UID/Ready、解释器依赖、实际CPU/8卡与资产可见性，再为未执行CPU阶段建立新独立prep、份额和资产续验绑定。新绑定显式pin旧来源回执SHA及当下路径/文件身份，不改旧回执，不将旧UID检查视作新环境通过，不自动改绑任何活跃运行。

## 从 CPU 验收到 GPU 准入

先完成代码适配及必要 Pod CPU 检查，再由唯一 Git 写入者汇集具体文件差异和真实回执，成功提交并同步 GitHub、回读远端 commit，按 Pod 实算逐文件 SHA/配置/资产身份冻结部署到 exp-v3。之后唯一 GPU owner 才能进行有界 GPU 准入和正式实验；未推送的新代码不得先试跑。具体标准见[八卡验收](../todo/exp-v3-8card/ACCEPTANCE.md)。

CPU执行者无GPU派发权，技术验收状态不能成为正式训练初始化；正式训练使用新fresh绑定。任何旧运行的clock/token/冻结副本都不能因迁移、验收或观测失败而重置、改写或重启。观测不完整记unknown，不据此认定设备空闲。

GPU阶段的CPU使用仍受执行者合计20%上限约束，不能只在technical-prep限额而让`allocate_cpus`重新占全部affinity/quota。B真实退场后，可为尚未执行的GPU阶段生成新owner CPU绑定，将当时总额度分给唯一A；已有12/12分配不热改。owner、rank、loader和控制进程的CPU集合均受新A份额约束，子进程继承，rank内线程和workers不额外越额。实际mask、rank分片、剩余控制核与性能预算写入新阶段回执；模型batch、采样/增强身份、成功更新和LR/EMA不因CPU份额改变。正式GPU使用独立显式GPU入口，不继承CPU准备的隐藏设备环境。

## 历史源码与可迁移入口

原R92的实际冻结包和按轮工具身份由Git历史、`records/R92/`及历史服务器原件追溯；R92已完成，不再接续旧观察/启动入口。当前`runtime/`及activation/validation沿同一路径适配，不再从旧R92生成配置、clock/fresh/source模板或旧CPU evidence获取新准入身份。`configs/R93-SCIENCE.json`与`tools/activation/prepare.py`按当次显式site和真实资产回执生成新配置/身份，实际CPU回执必须绑定其源码SHA；未适配历史工具仍不能作为新环境入口。

新工具使用发现结果作为显式参数或由通道注入的 `NCCCU_POD_*`/`NCCCU_PYTHON`，并读取当次实际资产/部署清单。新部署由当前授权的唯一执行 owner 从已同步 commit 创建绑定、核文件 SHA 和资源准入。历史科学参数、checkpoint 身份与已有运行的冻结源码不由环境发现步骤改写。

## 保留模型参数导出

将本仓库的相关源码和权威保留清单传入独立 `technical-prep`，在核验环境后调用：

```bash
bash tools/environment/pod_cpu.sh run "$SSH_TARGET" "$NAMESPACE" "$POD" "$POD_UID" "$PYTHON" \
  "$REMOTE_SOURCE/tools/models/export_parameters.py" \
  --catalog "$REMOTE_SOURCE/records/storage/KEEP-MODELS.json" \
  --source-root "$REMOTE_SOURCE" --output "$NEW_PARAMETERS_OUTPUT"
```

只导回三个 `parameters.json` 和 `PARAMETERS-RECEIPT.json`。工具校验完整 checkpoint SHA、原侧车、更新数、年龄合同及 EMA 状态并记录张量摘要；不构建模型、不前向、不复评官方、不复制权重。导出失败保留真实错误，不填造参数。
