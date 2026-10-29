# R109：Pairwise 的 P4 曝光对照

直接匹配 R107 seed20261029，仅正例配额8→4；64主图/单视图、4正60负240对，rank正数[2,0,0,0,2,0,0,0]。Small公开fresh、原loss/scale085/LR/age/SD.1/dropout.2/固定9:1保持；800/24000/绝对1600由新SCIENCE绑定，未重写历史SCHEDULE/FREEZE。

6400成功=attempt、skip0，完整8轮/16内部条目、连续4轮未刷新后正常早停。唯一 c4 EMA@3200：内部 AUROC .9855326547789199，AP .8333775688375845，error_twice24030。对 R107 所选点 AUROC 下降 .0029079183、AP 上升 .0059448800，本次P4未支持主目标，不以AP/局部周期救结论。

既定固定20难正组错序质量6334→9120，其余147组3266→2895，整体9600→12015；净增2415。固定组含早期结果定义、固定dev和单seed偏差，仅描述性分解，不推机制或泛化。远程聚合 technical-prep/r109-error-decomposition-20261029-b/SUMMARY.json，Pod SHA fb1fe0528db16f64749019c3bfaa6129e8468901d00921a070c02f2574b20fd4。

官方原一次5849：AUROC .9886312111470762、AP .8484972461588262，独立分报，未用于科学取值。TRAIN/freeze/official均exit0、hard deadline通过、signals空；22:01:57实核round complete、ranks/components inactive、设备idle/无users/原锁释放。owner138041为Z，不称reaped。

实际Git fb64a822735fc3547d344f0beeb44e40a4e3d5c0；权威root为 exp-v3/rounds/r109-small22k-seed20261029-scale085-pairlog-pos4-24k-p4-9010-20261029。prepare与selected原路径确为b64-p4，不采用通知陈旧p8模板。

原件SHA：SINGLE-MODEL-FREEZE 6f3071726fe6d86855054b4151cf1556cb460fd02bed52e331dc2bc4c827ec52；TRAIN ae3f53e7f00a7da8ecdb1d8ba198792bc1f99994520f1cac8e3a6e6b61b6adcc；ROUND a59f1a622e215ad2250b4d5888d803666048daedbddd4d8427d0391604151501。原官方SHA a2073e72b59b8adb0ee775cea8ce2a1b9a3cfb71066f44e7ff4180bddd829330保留远程。终态观测SHA39d83bda88207b43d9427fca7722bb014725dc6e3d7da375cf1888545597b64f、CPUguard e1e84352e5e19fccfecbe689298e52649d426930f1eb21eff8b6471c29d80f87留exp-v3及work；不新前向/官方重试，不以设备idle代替实际进程身份。

训练累计2717.1838s、dev146.3778s、27次保存135.7970s、训练组件wall3222.9628s；原生training/successful-update .42455997s，仅本轮观察，非benchmark或稳定吞吐。实际输出时钟/配额、初态/资源、官方隔离见原件；原始预测、图像及checkpoint不入Git。
