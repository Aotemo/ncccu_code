# 两独立8卡Pod接续（2026-11-04）

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
