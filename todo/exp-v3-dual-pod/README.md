# 两独立8卡Pod接续（2026-11-04）

**10:49 UTC 自然交接完成。** [监督双Pod汇总](../../records/environment/dual-pod-20261104/DUAL-SUPERVISED-SUMMARY.json)21dffac4/CPUcb4c2414：A525成功/0skip，B1350成功/0skip，仍无完整selected。两适配均1600/attempt1600/0skip/0dev，标签只来自fit、age不参与适配。原生监督八rank、345全训/5空Adam、head-age不变/factory RNG恢复、G64P8/448及资源池身份已各自然首25通过：A model3197c153、观察b5ea0574/guard44185282；B model85cf288d、观察f53355ee/guard2ec421df。仅复用quota监督metadata helper接受已登记SupMin版本，调用原`ssl_phase_completed`和`verify_records`核真实产物，没有新forward、模型加载、activation或clock重置。唯一4e继续控制两Pod；当前冻结c4b77a2保持，最新规则已06ace3c发布/API，新增F01只是准备，不把融合结果写成完成。

**10:33 UTC 已真实并行运行，完整dev结果仍待。** A R240R1原生八rank/公开初态及首成功`75c10e76`、guard`81570bdb`通过，owner677/start381509155；B R242对应`1be111a1`/guard`35f4aa71`、owner432/start381475114通过。[两Pod小回执](../../records/environment/dual-pod-20261104/DUAL-START-SUMMARY.json)CPU导出`740cb1de`：当时A750/1600、B1250/1600，均有限loss/gradient，分别为同类SupCon及SupMin目标，每批8阳性/56阴性。两边18/18、345全训/空Adam、c4b77a2/treeed24的2959文件bytes/mode、各31560秒整窗、一次activation/direct launch均通过。实际SCI A1b1badea/Bcef33b16，SITE349d1597/525ac08a，部署ed0de39c/878895df。各训练96核24–119与准备24核0–23互斥，模型仍FP32/G64P8；真实首更新后不加GPU预演。

用户最新规则已直接改raw目标为0.995，AP > 0.9保留；通用公开预训练可用，医学/眼底预训练禁止，允许内部dev选择成员/权重融合，禁止TTA。两现役冻结包、clock/token/grant及版本化官方门均保持。[F01融合准备](../auroc-999/F01-GENERAL-ENSEMBLE-20261104.md)先核真实合规依赖，暂保护R120/R238所选EMA及最小加载依赖，尚无融合结果或官方forward。c617清理已c4b77a2同步/API，其CPU树退出，原A12/12 allocation继续复用。新Pod准备窗口超过5分钟（NFS实际慢/旧handoff UID拒绝/源码同步与正式验收等），每个Pod Ready至首更新的实际时长列在上述JSON；连续设备空闲与利用率未测，不称满足五分钟。原完整源码展开最终成功；NFS慢期间尝试的链接暂存因tar对已有非空目录unlink失败，未用于任何正式包或GPU，原件保留。

以下为准备期原始记录，早期“未启动/待同步”不覆盖上述实核现状。

用户09:19新要求重新发现已创建的两个8卡Pod，并明确选择两个独立8卡实验并行，保留每轮G64/P8科学合同；这覆盖旧单GPU串行资源规则，不合并成16-rank，不申请新资源。唯一源码/Git写入者与两个Pod的GPU执行owner仍为4e。R238已完整7200/c9/bad4/0skip、唯一EMA c5@4000 AUROC.9851461364/AP.8147244364、完整结果6b43e762及唯一safe4f40fa9b已落地但尚未同步；旧safe不再重查。

旧R240在原UID51fe实际派发，首观测仅owner/8rank活跃、公开gate/首更新尚缺；后续查询原Pod NotFound。旧R240当前运行/结束状态unknown，不写成完整阴性，不恢复或改绑旧clock/token。新环境只使用fresh独立身份：拟PodA做R240R1（R240相同固定fit配额控制），PodB做已登记R242（同配额下仅改变多数标签正对）。用户新并行授权改变执行先后，晋级与固定seed27仍必须等两者完整结果及实际匹配控制；不提前跑R241/R243。

实时发现两个Pod均Running/Ready，Python均`/usr/local/python3.12/bin/python3.12`，同`ncccu:v1.0.0`镜像标签。两边原生probe均实核Python3.12.7/torch2.10.0/torchvision0.25.0/numpy1.26.4/Pillow12.3.0。09:53两当前UID资产续验及native CPU构建通过：A small66535567/complete020df177/nativefc5a1d96/CPU45a0d759，B small360af932/complete7924a47c/native8beb1cf1/CPU12b7cd0e。每边23个必要小文件和公开Small新SHA、51396 train/5849 valid缓存receipt SHA/原图fingerprint/array大小已核；缓存大字节证明明确继承旧pin，生产每次读取array仍核SHA，不冒充重拷或全量大字节重哈希。正式18/18和冻结部署在两Pod各自12核池并行执行中，GPU尚未声称已启动。

| 槽 | Pod / UID | Node | 实测资源 |
| --- | --- | --- | --- |
| A | vcjob-2117909060596142082-acfwvv9ejn-44303-worker-0 / 04203d31-96b5-415d-b126-a30f3ac6107f | 10.5.15.26 | affinity/cpuset0–127、quota120CPU、8 DCU、472Gi |
| B | vcjob-2117909275134578690-acfwvv9ejn-18257-worker-0 / 37e3fb34-c233-4294-b517-e98978cbd016 | 10.5.15.35 | affinity/cpuset0–127、quota120CPU、8 DCU、472Gi |

两Pod在不同Node，PCI字符串相同不代表同一物理卡；实际/dev/dri各8 render节点，共16卡。NFS同源`scnet.hx:/mnt/public/home/acfwvv9ejn`，原exp-v3/数据/缓存/源码可见。实测policy SHA0b1a5f95、guard05f3b6b7；不能沿用旧725b policy pin或旧UID资产验收。GPU96核池按每个Pod实时有效120核的80%重新绑定，旧运行文件均不热改。

c617由用户另行授权清理，仍不接管GPU或Git。其已建立PodA fresh双owner准备分配：总24核、4e为0–11/12核、c617为12–23/12核，SHA `9fca3bbf3682a1f647167a9cdb08199b372c920ad8df3c5aebe487fd1a3efeb9`，TTL1793791920.1212575。4e复用`/home/hao/Desktop/challenge/work/storage-cleanup-20261104-c617/owner-cpu-context.env`，不在A另建独占额度。B由4e沿原budget入口建立同样双owner份额；root若两Pod同时准备最多12+12=24核，c617先只用A的12核，全局最多36/240CPU，仍低于两设备合计20%=48且各Pod与96训练核池互斥。B新份额未生成前不在那里跑Python计算。

文件分工：4e仅维护本页、todo/README页首、轮次/新准备与运行记录、必要源码及Git；c617仅`todo/storage-retention/CLEANUP-20261104-c617.md`、`records/storage/20261104-redundant-c617/**`及本人独立technical-prep，不共写4e文件。c617盘点保护所有原始数据、assets、shared-assets、三KEEP及R120提交/模型依赖，R238未同步完整结果和旧R240全部root，以及所有`exp-v3/technical-prep/goal-20261103-4e0c`、`fit-label-contrastive-20261104-4e0c`、`fit-label-quota-20261104-4e0c`、`fit-supmin-20261104-4e0c`、本次dual-pod新准备及两个fresh round root。未收到4e精确依赖释放回执前，不删除R238或旧R240及上述准备。其他冗余按用户授权真实清单处理，不重查旧actualsafe，不改冻结/clock/token/一次性标记。

09:53准备范围：仅私有`technical-prep/dual-pod-20261104-4e0c/revalidate-shared-assets.py`读取已pin旧来源，当前UID实哈希必要小文件/公开Small，扫描现有缓存receipt SHA、原图fingerprint和array大小，明确继承旧大字节证明、生产每次读取仍核array SHA；不重拷/重建缓存、不读取官方标签语义。两新UID各出独立续验回执和原native guard CPU构建；复用原prepare/check_candidate/freeze链，不新增训练平台。B已实核12/12分配f48cb4b7、TTL1793806639.2465532与A96训练核池同样互斥。

10:00首轮技术准备因旧公开Small handoff UID被拒（A fe93572e、B b27e6eee），无activation/clock/GPU，原件保留。新UID公开绑定窄检查A cf6a5f59/B94e837bd通过，original新SHA+已验converted/provenance、旧strict结构证据明确复用；新增两SCIENCE仅更新public_asset_receipt，源码/全部科学值保持。新正式包使用env2独立ready，原失败ready不覆盖。两边统一使用已测125ceea8的运行文件，默认SupCon不变的原数学/梯度证据和完整quota流/P8复用，不混入未通过实现。私有formal-fit-supmin-next.py仅为已有prepare/check/freeze链补当前资产参数，不建立新训练链。用户清理执行者已交付50个冗余CPU fixture真实删除/保护不变及49.70GB free，其指定两文本范围由4e精确纳入同次Git同步。

准备时长复盘：实际Pod Ready至自然首更新A3731.701秒、B3343.652秒；这不是连续设备空闲测量。除NFS慢、旧UID公开交接拒绝、必要CPU/发布/部署外，也包含执行者的绑定决策、入口编排及上下文整理时间，不能全部归因于存储。后续准备在当前GPU运行的20%池完成，复用已通过的源身份，避免再次等待整套源码重复展开。上文345全训/空Adam指监督模型CPU初态，当前原生SSL初态已证，监督原生交接另核；不把阶段前置验证冒充后续自然成功。

后续执行提示（不派发新GPU）：最新小回执与用户规则已292956194b903d7e68f81d832fdf6fac8d834179 push/API回读。两现役仍c4b77a2，A context为`work/storage-cleanup-20261104-c617/owner-cpu-context.env`，B为`work/goal-20261103-4e0c/dual-pod-b-context.env`；不得再用旧r240/r238 context或51feUID。当前已通过的监督首回执为各ready-env2/START-OBSERVATION-supervised-1.json，后续沿原observe_round_once作progress/completion，不再次验证旧safe。A准备TTL1793791920.1212575、B1793806639.2465532；CPU过期仅为未来未启动阶段新建分配，当前GPU grant/clock不热改。接续前先完成F01真实checkpoint/selected/precision通用源核验及21点CPU分析的必要准备；R242完整结果若早于R240R1，只记匹配控制pending，不提前执行R243。P1/P2仍候选，不能把旧51fe technical包或未提交census草稿直接当作下一轮可跑证据。三KEEP、R120提交和R120/R238 selected临时依赖继续保护。
