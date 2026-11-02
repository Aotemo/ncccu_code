# R156期间的固定后续准备

唯一Git/GPU owner32e3在aa0f的0–22/23核准备池内执行，peer23/1核与训练24–119/96核互斥。不新建/恢复Agent、不热改R156的874feff冻结或clock/grant。

R157/order69仅在R156完整工程有效selected同时严格胜R107 `.9884405731067525`及R120 singlebest `.9894773092062407`且actualsafe后，复现原固定困难正例loss的seed20261027。公开fe6b358a fresh、RGB512和原G64P8/AdamW/LR/EMA/24k-800-p4不变；不满足则不formal/activation/GPU。

独立R158/order70候选直接对照R107，只换同架构Small的公开ImageNet22K→1K **384微调来源**，比赛输入仍原RGB512，原logistic loss/seed29/采样/增强/age/优化合同保持，不叠R156 loss。不是重开已关闭的全局768尺寸方案。依据仅是同架构公开高分辨率微调来源可独立检验，不能称已证明青光眼纹理/定位根因或保证提升。[作者发布模型卡](https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k_384)给出Apache2、ImageNet22K→1K与384来源；不可变revision、LFS、许可、原1000类严格加载/映射及公开fresh初态仍须Pod实核，不通过就不采用，不填假SHA。

最小范围：现`tools/pretraining/small_source.py`与`small_strict.py`仅增一个固定来源选项，复用原relay/下载/许可/LFS/344张量转换/严格加载链；现`runtime/bundle/retrain/experiment91.py`绑定两固定SCIENCE及保存/选点身份；`tools/validation/check_candidate.py`仅把R157纳入原hard-loss验收tuple；相应SCIENCE和candidate必要测试。models/config/engine/EMA/inference未改，不新建训练、启动、保存、评估或监控平台。

CPU只验实际受影响的来源身份、同架构公开1000类与转换后数值/语义、原512 fresh/保存身份、R157旧loss与新seed、R158只换public来源，其余已通过字节直接复用。代码和配置实际push/回读后，当前训练期间构建两技术包；完整结果决定分支，收尾期间唯一正常formal/精确部署，actualsafe后立即一次activation/direct launch，由真实八rank/首成功证明GPU链，不另做GPU预演。

完整内部总体AUROC仍为主；R158未严格胜R107则关闭精确来源合同，同时过R120才有固定seed27资格，不扫公开checkpoint或输入尺寸。官方仅完整selected≥`.996`后5849一次，raw `.999`未达。[必要实现/真实CPU证据](../../records/preparation/small384-20261102/README.md)已通过：1配置目标/5compile、旧6loss目标SHA复用、公开来源344严格链及两fresh/新512保存冷加载。当前待实际Git同步/回读和技术包，尚未normalformal/grant/GPU。
