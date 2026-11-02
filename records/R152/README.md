# R152 全局RGB768完整结果

2026-11-02 01:49:07.604424 UTC[完整终态](R152-COMPLETION.json)：10400次成功和attempt相同、0skip、13次评估、连续4次无提升早停。[唯一内部选点](R152-RESULT.json)为EMA c9@7200，AUROC `.9875928818012477`、AP `.8381138566395393`、error_twice20608；权重tensor `31ca2b3fa6624848edf5e49b0993984e2a5717cb38e2cf1884afdd234b4738b8`。低于R107 `.9884405731067525`及R120 singlebest `.9894773092062407`，固定全局768合同关闭，R153 seed27无资格，未formal/activation/GPU。不扩尺寸或LR扫描。

官方因内部选点低于`.996`按政策未运行，metrics为null。原始终态已确认owner同birth的Z状态、八rank退出、锁持有者和设备users为空、8PCI busy0，actualsafe true；[实际CPU终态回执](R152-CPU-COMPLETION.json) exit0。安全证据不重复扫描，旧clock/token/grant及selected依赖保留。metadata归档没有加载权重或新forward。后续R154独立fresh已接续，raw `.999`仍未达。

以下为本轮原始启动证据；其中“当前”仅指启动时刻。

23:08:18.425436 UTC[真实首成功](R152-START.json)：running75success=attempt/0skip/eval0；345全训/5组空AdamW/whole`50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`与formal一致。首自然log25 finite loss`.6362380981445312`、WORLD8/G64P8/448pairs；owner468369/start360171470及八rank468381–468388的birth/argv/mask/8PCI匹配。实际保存job/已准入asset均为768，[几何证明](R152-ACTUAL-GEOMETRY.json)与源码精确部署、真实初态/自然成功组合，不冒作新增GPU tensor hook或独立GPU预演。

冻结runGit`b58f4a99f2b0f43e581f150e2a134e5ee80be396`/tree`2d6f614d01adeffdd263be043e8b3cca54324159`，SCI`013208252c6992566ef4ed8821113ef1840830e1ae2b8599ca371e1ecd32fbd9`，SITE`ed5e46c07e38b144196bef1d8237d49a2d6ff37e4eb3ba26e239d5fe2eff1ec7`。三必要CPU目标/7源compile、51396行只读cache和实际public fresh按精确SHA复用；[正常formal](R152-PREPARATION.json)唯一19/19、95.570秒，[实际部署](R152-DEPLOYMENT.json)6.028秒，一次activation/direct launch0。正式训练、启动、保存、评估、监控与验证链路沿原入口，无新平台。

[实际接续间隔](R152-HANDOFF-GAP.json)分列：从前轮训练完成事件至dispatch219.934秒/至最晚八卡设备277.074秒（均<=300）/至首自然25记录337.815秒（>300）；以前轮整轮ended_at计分别177.838/234.978/295.718秒（均<=300）。没有连续利用率测量，不以较短基点隐藏较长事件间隔。normalformal在前轮收尾启动，actualsafe后只activation/direct launch，未跳必要校验或重置旧clock/grant。

当前唯一Git/Pod/GPU owner32e3，训练原24–119/96核冻结保持；后续CPU准备新分配b7811b4e，[原件](R152-PREP-ALLOCATION.json)由实时affinity/cpuset/quota实核effective120，cap24、本执行者0–22/23核、槽23保留不用，至2026-11-02 05:11:20 UTC。新旧准备任务仍全局串行，旧00202581及R152 GPU allocation不改。当前入口`work/r137-20261101-a/r152-pod-context.env`绑定START2/新CPU分配；只为后续准备换绑定，不热改训练。

本轮直接R107/public fresh、单RGB全局768，无ROI/六通道叠加；原单Small/FP32/G64P8/AdamW/LR/WD/age/EMA/24000成功/800评估/patience4及唯一selector保持。新工程窗口为train27900/total31500/binding600/official2400秒，原9h allocation上限不变，启动后不续或重置。内部完整freeze>=.996才官方5849一次；当前尚无完整selected/official，raw .999未达，继续训练与20%池后续准备。
