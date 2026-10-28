# 8 卡最小验收标准

2026-10-29 R95更新：下文6400/8周期/16条为初次适配及R93/R94历史合同。R95已由用户明确为最多24000/800×最多30、内部dev AUROC连续4轮无提升早停；合法早停也是completed，按实际已完成全部轮次选唯一checkpoint。当前合同与对应墙钟见[R95计划](../auroc-999/R95-PLAN.md)，不为R95重跑初次适配或新增独立GPU预演。

**2026-10-29最新用户覆盖规则：下列独立GPU预演和初次适配检查不再是R94 LR-only的重复前置。** 删除重复hardcoded LR断言，停止新增拒绝case/重9或其他已过检查；沿用已有实际CPU证据，取消独立零forward GPU预演并如实记未运行。真实训练8rank初态/参数组/空Adam/资源校验接管；必要Git冻结、实时身份/独占资源、一次activation/官方一次不变。不新增桥/平台/文档门，先启动R94；随后20%池覆盖不依赖结果的代码适配、配置注入、干净打包和现有CPU验证。

所有条目是待取得的证据标准，不代表已执行。A/B 按文件边界保存具体命令、输入身份、真实 exit code、时间、Pod UID、依赖和产物 SHA，A 统一集成；规划只审阅回执。B 无 Git 写入/GPU 权。每次执行前实时复核 UID/Ready。旧通过项只在未改变且实际身份相同的范围内复用，不机械重跑全套。

本次新增A1/B1只在父级划定文件内文本编码，无本机Python/解析/测试、Pod命令、Git写/GPU或再派生权。父A/B分别在自身12核内完成验收，不新增预算平台/检查范围。B来源fullstrict、A生产18项、启动集成29项、准入工具CPU11项及部署关联3项分别有真实通过回执；B完整迁移和两split索引亦已通过并真实退场。各门彼此不替代；首次GPU准入因helper线程mask越界和退出收集缺陷失败，详见[失败记录](GPU-ADMISSION-1.md)。科学与真实退出标准保留，训练CPU比例按用户最新纠正为80%；新版本/新阶段须实际冻结，旧allocation与attempt1不热改。

## CPU 阶段

地点为已核实既有Pod的独立 `exp-v3/technical-prep`，nice19，隐藏CUDA/HIP/ROCR，确认GPU未初始化。最新用户授权A/B合计使用实时有效CPU的20%，允许并行、初始平分，由规划按affinity/cpuset/cgroup quota调整份额，实际总预算向下取整。旧2线程及串行限制已被覆盖，历史回执不改；未知/失效Pod或未实配预算不执行，不共写prep。不取训练锁、不发训练信号。CPU子进程继承本执行者互斥mask及线程上限；采样分片采用顺序模拟，无需在CPU起8个训练rank。

CPU预算新增窄验范围：显式allocation的namespace/Pod/UID/container/Python、有效期和live资源约束匹配；总份额≤floor(effective_cpu/5)，每份threads等于mask长度，各mask为当前可用集合的互斥子集；缺预算、旧UID、越额、冲突mask拒绝。入口导出 `NCCCU_CPU_THREADS` 和固定JSON整数数组 `NCCCU_CPU_MASK`，执行工具核实际affinity及线程限制一致，不自行另分总20%，不使用固定120/24/12。无需建设额外调度系统；该额度变更不改变训练科学参数。

跨UID只续验尚未执行的CPU阶段：新阶段绑定显式pin旧source-chain SHA并核当前路径/stat（包括已有mtime/ctime），保留来源历史和当前执行身份。不得删掉UID校验或重写旧回执；只补预算/身份护栏及受影响strict路径，不重做未变来源链或无依据重复已核大对象哈希。若新诊断发现stat不能反映实际内容变化，则必须为缺少当前内容证明的对象取得必要新内容SHA；不能把“避免重复hash”变成不允许补证的门禁，也不能用sleep/改长度/skip掩盖真实同长度改写漏检。

本次FS诊断已证实stat不变而内容变化，因此来源续验采用`fresh_pod_sha256`：生成新stage和strict使用点分别核必要11件资产的当前内容SHA，匹配已pin旧来源链的科学内容身份，新完整7stat绑定当前对象，诚实记录重哈希。原`unchanged_shared_object`不可作为本次内容证明；新Pod的device/inode变化由新身份记录，不改写旧回执。只复验受影响回归和strict，不扩到完整来源语义或历史对象。

| 条目 | 通过条件 | 失败条件 |
| --- | --- | --- |
| 环境与路径 | 显式 UID/Ready/容器/解释器，资产与写入根来自实际清单；新产物位于 exp-v3 | 旧固定 UID/PCI/Python、隐式回退、本机或登录节点计算 |
| 配置与拓扑 | reference physical32，world8，per-rank8，accum1，全局64/P8，设备唯一；physical_batch 解析前后身份正确 | 用 rank16 得到 G128，按卡数改 LR，仍要求旧4rank或误将 reference32改8 |
| 全局样本流 | 用同一真实 fit 清单/种子/update，将 4×16 和 8×8 分片重组，对照原两个 reference32 拼接序列；覆盖周期边界与 warmup 边界 | 丢样本、跨 rank 非预期重复、改变全局正例总数8、改变固定划分；不强制各 rank 恰好1正例 |
| 更新时钟 | 核0/1599/1600/1601/6400边界；仅成功 optimizer step 增计数；cycle800、total6400、LR和EMA使用成功更新 | 以 microbatch、world、attempt 或墙钟替代成功更新；FP32 skip 被静默接受 |
| 模型来源与结构 | 精确原 blob/转换/provenance/许可证身份；strict-load与映射/维度一致；候选末端LN正确；fresh head/age、空Adam、零更新、online/EMA初始一致 | 缺来源链、宽松missing/unexpected key、N06/旧训练权重混入、沿用Tiny1K末端LN或旧candidate标识 |
| 归一化与精度 | Tiny无BN/无buffers；FP32、无autocast/scaler、TF32关闭；image/prepared512、ImageNet normalize、age仅fit | 适配设备时换AMP/BF16、改变图像处理或年龄统计 |
| 内部选点 | 完整8周期×online/EMA共16条，按error_twice升/AP降/update升/EMA优先；线上/EMA均可被选 | 缺条目、重复条目、非有限值或错误update被接受；按官方改选 |
| 一次性控制 | 在独立fixture验证重复绑定/launch/official reservation拒绝，错误UID/SHA/资源身份拒绝；不用真实R91/R92路径 | 覆盖已有token/clock、吞掉未知状态、重新官方前向、fixture可成为正式checkpoint |
| 必要语法/目录/链接 | 仅检查受影响文件，Python解析与链接/配置检查在Pod产生真实回执 | 本机执行测试，或把未执行写通过 |

采样重组检查证明样本流和更新语义，不证明随机增强、dropout和浮点归约跨world的全轨迹完全相同。CPU初始模型摘要应绑定精确来源与代码；不得用随机 fixture 代替真实候选载入门。

迁移只验实际依赖：源/目标清单与容量、纯复制当前bytes/SHA、18冻结件+FREEZE原科学身份、train_identity、公开来源相对引用、raw与prepared arrays字节一致、新cache policy/namespace及source_file身份。缓存数组不重建，索引另立；当前FS的stat不能证明bytes，必须有当前内容SHA。官方只做必要存储复制/哈希及隔离索引，不解析标签用于适配或选点，不读取数组进行模型前向。B先小批资产交A生产验收，再将bulk copy与索引编码重叠；不用归档润色和历史清理阻塞开训。

## 同步与冻结闸门

CPU 阶段通过后，执行提供待同步文件全列表与内容差异、真实回执给规划和协调。执行是唯一提交者；当前 main 写入动作须针对具体可审阅变更处理，不能重试已被拒绝的未知变更集请求。成功同步后回读远端 commit，确认部署文件来自该版本；在 Pod 计算实际逐文件 SHA、配置/资产身份，再冻结于 exp-v3。代码、配置或身份变更需要更新提交与受影响的窄验收，不能热改已绑定运行。

远端同步、Pod CPU 回执或冻结身份任一不成立，GPU 阶段不开始。工具存在、创建了本地 commit、或有提交意图都不是同步回执。

连接器同步可按已审阅暂存tree批量inline内容/删除，返回tree须与本地write-tree一致；A基于实核父commit作唯一非强制有租约更新，再复读远端commit/tree。若API元数据丢失原作者时区，不能伪造Git对象或乱reset；只有实际对象SHA匹配才导入本地ref。远端唯一commit、代码tree与Pod部署SHA证据齐全时，本地HEAD暂不能无损取回可如实记录并后置，不增设重建Git元数据的开训前置，也不虚报fetch成功。

## 初次八卡适配的有界 GPU 准入（非R94 LR-only前置）

只由当前唯一执行 owner，在上述闸门全部通过后串行进行。准入前再次查 UID/Ready、8 PCI、实际 ordinal 映射、CPU quota/affinity、显存、设备句柄和任务剩余时间。未完整核实占用为 unknown，不按空闲放行；不杀未知进程。

用户最新纠正后的资源标准：模型训练及代表其实际资源的GPU准入整树（owner/rank/loader/control/helper）使用实时`floor(4 * effective_cpu / 5)`，实际仍120核则最多96。准备计算仍合计20%；若并发，双方mask互斥、合计不超有效CPU。只为未启动阶段另建绑定，旧12/12及20af/attempt1/失败回执不热改。按新80%合同核实际分片、helper/线程/worker，不把旧20%gate通过作为目标；helper全128仍拒绝。仅修资源语义及已有失败所需退出/helper路径，不增新门或平台；科学batch/采样/6400成功更新/LR/EMA/官方隔离不变。

最终语义为两个池同时运行：训练80%、下一轮计划/CPU验证20%。策略由集中版本化配置提供、运行env注入，各入口读取同一策略；实际总额、核数、mask从实时环境取得。验收仅补此次改动必要的配置消费、池互斥和总量检查，允许20%验证活动持续，不再检查“所有prep/peer已退出”作为训练启动条件。已有必要科学/退出/同步标准不变，不新增过程性RED、全套测试或文档批准门。

技术验收使用独立一次性 `technical-smoke` 身份和目录，单次上限 **300 秒、最多 2 次全局成功更新**。只取 fit 数据，不跑 dev/官方，不选 checkpoint，不把技术优化器/EMA/权重作为正式初始化。若现有 native 启动门加 collective/初始化证明足以覆盖改动，可明确记录减为0更新；不能为省改动默认套用旧100/250步benchmark。不得扩大预算或重置本次身份来掩盖失败。

本轮选定0update、0forward、no-data的8rank native初始化和FP32 collective；保留同fresh身份/空优化器零时钟、8PCI覆盖、训练资源CPU80%及真实退出证据，仍限300秒。此范围只验设备初始化与通信，不宣称完成forward/backward或吞吐测试。新工具及窄合同检查先入本次具体Git变更，成功同步及冻结之后才实际派发。

必须取得8个rank覆盖、互异物理PCI、正确world/local batch、可用且互不冲突的CPU masks、同一fresh模型身份/空优化器/零时钟、FP32有限值与集体通信证明；若执行更新，需有一致成功计数、有限梯度/参数与正确EMA计数。技术结束记录所有rank与owner的真实退出状态、占用释放；未实核退场不得启动正式R93。此阶段不能声称完成吞吐基准或整轮时限验证。

## 正式 R93

来源/模型、资产迁移、代码同步、冻结部署及GPU准入全部通过后，规划将待核候选固定为精确R93合同，执行生成全新正式绑定，fresh开始；禁止复用旧R92和technical-smoke状态。实际预算根据当前Pod租期和已知性能保守设定，未知性能如实记载；不给旧clock续期。

完成条件：6400成功更新、8周期、0 skip、完整16条内部dev、按预定规则冻结唯一checkpoint，随后完整官方5849一次最终比较。实际exit、checkpoint/tensor/source/配置身份、官方reservation和结果回执均齐全。内部与官方raw AUROC分别报告，未达0.999不得标总体完成。
