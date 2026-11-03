# R215：原生 SigLIP2 公开 MAP 图像塔

13:23[9个冗余权重清理已完成](../storage/closed-r215-20261103/README.md)，逻辑12,723,697,075 B；0skip/0error，sidecar、预测、归档结果、当前R217依赖和三KEEP保护核验通过。R215所选现不可再称可加载。旧safe不重查，初次纯文本适配校验失败及独立bound2修复回执保留。

13:20已用原封存四锚点/20组定义只读核验本轮唯一selected CSV和precision各一次，无模型/图片/官方forward。[固定组摘要](../analysis/20261102-persistent-errors/R215-FIXED20-SUMMARY.json) SHAbdac668d / [guard](../analysis/20261102-persistent-errors/R215-FIXED20-CPU-GUARD.json)thwnmquk/8b5e52dc exit0：20难正例错序对质量13326，其余147正例11014，总24340，较R107分别多6703/8037，守恒于error_twice48680。两组均恶化；这是已选点的描述性分解，不证明临床原因，也不追加来源/池化/学习率扫描。无后续模型forward依赖，按用户存储规则清理冗余权重前先核真实manifest；预测、precision和精简结论保留。

已[完整结束](R215-RESULT.json)：10400 success=attempt/13次内部dev/连续4轮无提升/0skip，唯一EMA c9@7200内部AUROC .9706920363977455、AP .6894472108312087、error_twice48680。严格R107/R120双门false，R216不eligible，官方按阈值未跑，raw .999未达。tensor008ead18/blob4dd567b6/CSV1925aa06/precisionafef2605，freeze.status=null原样保留。[唯一actualsafe](R215-COMPLETION.json)/guardwagpf140/e49b7d64已核owner inactive、八rank退出、设备无holder/user，旧safe不重查；[归档guard](R215-CPU-FINAL-ARCHIVE.json)zl_gm8x_/c283bfdc exit0。随后[R217实际首成功](../R217/README.md)已证，整轮结束至首25记录242.089秒，末训练日志至首25为277.981秒，均<=300。难组描述与冗余权重清理尚待；以下均为此前启动时点记录，不再作为当前running事实。

当前唯一GPU轮，11:44[真实8rank初态与自然首25](R215-START.json)通过；11:49有效观测800 success=attempt/0skip/eval1/bad0，暂内部AUROC`.9534745108616469`、AP`.5772599571038547`，尚无完整所选结论，raw .999未达。11:52[单次原sysfs快照](R215-DEVICE-POINT.json)八绑定卡均100% busy，非连续利用率。165全训/五组/每rank空optimizer初态/whole13e6317f与实际CPU一致；首25 loss`.6560595631599426`有限、WORLD8/G64P8/448pairs、各rank8primary/8view符合合同。generic非Lion字段不代表独立逐rank实测AdamW kwargs/moment数量。

来源为固定`vit_base_patch16_siglip_512.v2_webli`自然WebLI公开图像塔，完整162张量strict/exact/FP32后保留公开MAP latent/q/kv/proj/LN/MLP及所有位置参数，原head Identity，追加Dropout.2+Linear768→1，age一次。原R107 512几何/增强/采样/448 logistic/AdamW-LR-WD/globalclip/EMA1600/.999/24k-800-p4保持；仅使用匹配来源的现成unit(.5/.5)归一化。它是架构/公开预训练/归一化组合假设，非临床机制或收益证明，不称来源crop/bicubic预处理复现，不重开fit-only池化扫描。完整selected>=.996才官方5849一次。

冻结source`c9bbde52240c458c577cd2c582f327bd31e18dec`/tree`4180bcf3803aa98b9d8fb842acf00452a8f90b31`；SCI`687032413916fbf64689071f30c0bc1e850aa94bd677736d8d0910f35aa9081e`，actual SITE`d155e13a60f52a056d566418f67bb194b6b32c05766eb734a9317c50e3da7988`。manifest5a8273e8/cpu-prep947d6455/deployment0e46ef8b/reuse02ec623a；精确本轮24GiB/storagea5329e3c，默认12GiB/free16GiB保持。

[实际formal CPU原件](R215-CPU-PREPARE.json)c4814972为completed/exit0/19全部通过，8受影响目标/13唯一compile按SHA复用。运输外层状态分列：[完整归档](../preparation/siglip2-20261103/READY-VERIFICATION.json)保留首个末端LN遗漏、第二个fresh来源kind遗漏、第三个不完整Git树部署拒绝及最后私有打印缺json的exit1。补齐同提交2221文件后原部署工具exit0，原post-deployment检查已写真实PREPARATION-RESULT，打印异常保留；归档0sfy8_5e/c0e686a5仅校验这些真实文件，不把它叫新模型验收。模型/检查/部署未因打印失败重复运行，未放宽Git全树校验。旧两partial stage/grant无clock/activation/GPU且未改；当前唯一阶段ready-freshfix。

[窗口](R215-CPU-WINDOW.json)h2fqm0o6/cf673bfb、[一次activation](R215-CPU-ACTIVATION.json)vn042amf/2cfa0e4c和显式八设备direct launch一次均0；原四调用同SSH串行、非0立即退出，无新启动平台或GPU预演。start1 unknown保留，start2/vgknnzml/cffd6ab0是真实首成功。owner739329/start373352136；八rank739354–739361及各自start_ticks见初态原件，训练24–119/96核、rank24–111各11核/control112–119；准备d837/root0–11与保留peer12–23互斥，11:22实核peer closed未恢复/派生。当前r215-pod-context.env/start2，旧env不作当前入口。

R213[完整所选](../R213/R213-RESULT.json)双门false/R214不eligible，actualsafe后才接续。[实际间隔](R215-HANDOFF-GAP.json)整轮结束至dispatch1026.439秒/设备1102.118秒/首25记录1155.284秒（19分15秒），全部超300；末训练日志至首25为1179.237秒。两处必要接入遗漏、独立副本复制和不完整树补传造成延误，未跳过校验或改旧clock/token/grant，不把时点差当连续DCU利用率测量。后续完整后备CPU/部署须在当前训练期间完成，完整树可压缩传输，避免再次临近结束才发现接入缺口。
