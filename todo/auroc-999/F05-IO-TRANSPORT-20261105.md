# F05：新任务的连接与 I/O 降载

2026-11-05 用户经 c607 要求降低登录节点连接数及共享 NFS 元数据压力；只用于未启动的新任务，不改运行或冻结证据。模型汤、五折和安全 GPU 接续优先，本项在准备池并行推进。当前旧式整树核验启动于本要求之前，不重复派发；后续不用 NFS 整树克隆/核验。

## SSH

本执行 owner 的本机 SSH 入口使用跨进程共享的两个槽位，最多两个在途 SSH/kubectl，等待有界；连接复用 `ControlMaster`/`ControlPersist=600` 和 server-alive，控制连接失效时在本机隔离路径新建，不向 Pod 中训练进程发信号。只有显式标注的真正只读、幂等命令允许有限指数退避重试；activation、launch、写回执、导出或任何状态不明的调用均不自动重试，超时先按原链只读对账。最小范围为 `tools/environment/ssh_control.sh`、其 Pod 内隔离测试以及本执行 owner 私有 PATH 包装；不改旧活跃客户端及其上下文。

SSH 包装的 4 项隔离检查已在 Pod A 通过：并发五次实际最多两条、失效 master 新建、非幂等不重试、只读三次有限重试；CPU 回执 SHA `af432b5397d3cf80d56edecd9769069444ed4382efdca58b336fe8f36febda64`。仍须将通过版本 push 后接入本机私有 PATH 并做限时真实连接验证；此前在途连接不受影响。

真实首次连接发现 `ControlMaster=yes` 与 `-M` 叠加进入询问模式，虽回退到独立连接成功却未实际复用；改为 `-Nf` 后第二次真实 SSH 输出 `master session id: 2`，复用成功。修正版 4 项隔离检查回执 SHA `043537abc4ef815bd55cdc2f1f08b86bbc98ed30864518549cb40cf079ef072c`；旧 control socket 留待自身到期，不发送关闭信号。

## 已推送源码的本地快照

每次使用指定已推送 commit 的 `git archive` 单文件和 GitHub 原始 ref/commit/完整 tree 回读；在 Pod 内核 tar SHA，将其解包到该 Pod 唯一 `/tmp` 私有新目录，只允许常规文件/目录、拒绝链接和路径穿越，再以原 `freeze_deployment.verify_tree` 对整棵本地树逐文件核 Git blob、mode、文件集合和 SHA。冻结结果、tar 身份与环境/Pod UID 回执仍存 NFS。不得把同名但未核验的本地树当部署；Pod 或本地盘变化须重新核。必要范围拟为 `tools/activation/freeze_deployment.py`、`tools/activation/local_snapshot.py`、`tests/test_local_snapshot.py` 及现有针对路径/篡改/树的窄测试；先核现有 formal package、site binding 与输出路径限制，完整树校验强度不降。

调用链核对发现 `tools/activation/activate.py::verify_deployment` 也要求源码位于 work root；因此本地源码的正式部署兼容范围增加该现有函数及对应窄测试。仅以显式、NFS 留存并绑定当前 Pod UID、commit/tree、tar SHA 和非 NFS 本地挂载的完整树回执放行 `/tmp/technical-prep/<UID>/.../source-git`；无回执仍执行原 work-root 限制。`prepare.py` 原要求路径含 `technical-prep`，本路径满足，不为绕过证据降低检查。

## 中间 checkpoint 本地暂存

拟对新轮显式启用本地持久化选项：将 `last`、`previous`、过程保护快照和初态写该 Pod 的唯一 `/tmp`，正常终态后只把固定 selector 的 selected checkpoint 与必要 sidecar/index/回执以复制、校验 SHA、fsync、原子提交发布至 NFS。selected 发布未成功时不声明完整或清理本地源；在相同 Pod 的正常恢复和事务语义须保留，Pod 丢失使本地恢复点丢失的风险明确记录。未选、活跃中间文件不触碰 NFS 或其他轮；现有未配置本地选项的行为不变。最小范围暂定 `runtime/bundle/retrain/persistence.py`、`distributed_engine.py`、必要的 `runtime/` 终态归档和相关测试；经调用链核实、登记精确范围并通过中断/损坏/复制失败和选点隔离测试后才允许新轮启用，不把设计写成已部署。

## 缓存和错峰

历史 512 缓存回执为 51,396 个数组、`31,286,003,712 B`，不包含完整 sidecar/目录开销，也不代表当前容量或完整性；先用既有回执和限时具体目录盘点当前占用、两 Pod 本地 `df`，再评估完整只读复制、SHA 清单、源 fingerprint 及 NFS 剩余 stat 的成本/收益，不直接复制。P3 扫描、整树校验、checkpoint 导出不与关键 activation/评估同时运行；若场景冲突先暂停新批工作，不干预在途训练或声称未知为完成。
