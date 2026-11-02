# R170：原 Base 输入合同下的公开 CLIP 初始化

**本轮技术失败，未形成完整所选结果。** [唯一终态观测](R170-FAILED-OBSERVATION.json)在6144 success=attempt/0skip/eval7/bad3记录`round storage limit reached`，train状态interrupted/owner_stop_requested、整轮failed，TRAIN_RESULT/freeze/official缺件、selected为null。[失败汇总](R170-FAILED.json)将两锚点比较保留null，不把暂best EMA `.9855170013883353`当完整成绩或科学阴性。actualsafe=true：同start owner为Z非reaped、八rank missing、锁holders和设备users空、denied0、八PCI busy0；不重启旧轮、不重置clock/token/grant，原产物保留。R171无完整工程有效结果而不eligible，已安全串行接续[独立R172](../R172/README.md)。

[终态存储元数据](R170-STORAGE-AUDIT.json)未加载或hash checkpoint：当时目录11,951,892,576字节、原上限12,884,901,888字节、文件系统free42,273,341,440字节，事后未超限，不能重建故障瞬间。最大已有blob约1,401,945,034字节；原`CheckpointStore.save`在写入/提交新blob后才prune，须核原子保存的临时峰值。同尺寸R116既有16GiB独立例外未随本新round沿用，后续仅在未启动的新fresh身份中最小修复，不修改R170失败现场或当前R172冻结包。以下是此前真实启动证据。

[真实八rank首成功](R170-START.json)证明75 success=attempt/0skip/eval0，345全训/5空Adam/whole5d2ebdcf与正常formal匹配，8rank身份/8PCI/训练96核24–119及控制112–119匹配。首自然25 loss `.7022261023521423`有限，WORLD8/G64P8/448pairs；首成功guard `f3476cedbf7ba47e417ed4dc79178a9416cf269e24aab619d8449484263694a9`，非CPU替代或独立GPU预演。

运行Git `58bc7616af41df1e9b84f7e412319bee6de7cc7f` / tree `90a0c8e1da9380c3e1fcbe9ec01ffdd515f98940`、SCI `5a91dada7391b7e1aea7e085dd0624cb0fdc90138c9c70ee96b61c620981a01b`、SITEddf4cf3b。[正常formal](R170-PREPARATION.json)18/18、35.186秒和精确部署1.021秒已在前轮训练期间完成，随后唯一activation/direct launch0；无重跑formal或旧clock/token/grant重置。

沿原R116 native Base/512/seed29/原logistic/默认AdamW beta2=.999/age/LR/EMA/24k-800-p4，仅换公开768f3849 CLIP LAION→12K→1K初始化。仍用原比赛ImageNet均值/标准差；发布者384/squash/CLIP normalization只是来源属性，不称端到端发布配方复现。完整344strict和两fresh/P8/配置证据见[准备](../preparation/baseclip-20261102/README.md)，CPU不证明本任务AUROC或DCU峰值/吞吐。当前收益待完整内部所选，官方仍≥.996才一次，raw .999未达。

[交接实际间隔](R170-HANDOFF-GAP.json)：R168训练完成至dispatch134.865秒、最晚设备198.222秒、首自然25记录268.499秒，三项均≤300；整轮ended至三项67.256/130.613/200.890秒。分别报告事件间隔，未采集连续DCU利用率或证明其最大连续空闲上界。提前formal减少了终态后的等待，没有跳校验或改历史回执。

R168完整阴性/actualsafe已归档，不重扫旧safe；R169无资格且未formal/GPU。根32e3唯一Git/GPU，peer仅CPU23/dca3；根准备0–22/23核与训练24–119/96核互斥，当前r170-pod-context.env绑定start-2。后续R171条件seed27/R172固定局部视野仅在独立准备目录推进，不热改R170。
