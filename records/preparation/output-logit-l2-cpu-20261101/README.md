# 固定 raw-logit 平方项的 CPU 技术准备

本次只授权结果无关的代码与CPU准备，活跃R131继续使用冻结`7aa3232b7a2e47dedf68839e19deb29ce4ec6751`。未采纳或启动新GPU实验，没有新order/profile/SCIENCE/正式root/grant，未跑normalformal。固定科学草案、原始论文边界及双门/seed27/官方隔离见[NEXT](../../../todo/auroc-999/NEXT-PREPARATION-20261101.md)。

生产变化仅为固定`global-raw-logit-square01-v1`总目标、非空policy配置/保存身份及原pairwise分支接线。原六字段pairlog及一次forward/WORLD gather保持；None直接返回原函数并从job_contract省略新key。当前正式profile强制None，单卡入口明确拒绝非空policy。模型、优化器、EMA、selector、保存算法均未改变。

两项必要新测试先真实RED两FAIL/0error（生产helper缺失），然后最小实现。首次GREEN-1在SSH kex关闭exit255，未到认证/Pod命令，未执行测试；同入口后续GREEN-2是唯一实际GREEN，2/2、0fail/error、8.547秒，6个实际受改Python compile通过。未重旧pairwise/LoRA/fixture套件，也未进行GPU预演。

- [RED-CPU-CHECKS](RED-CPU-CHECKS.json)：SHA `3dea3280d1e9bf0ec2a01a249eb100831d8cfc2a6b7667b3a42f8c38a75ec265`。
- [CPU-CHECKS](CPU-CHECKS.json)：SHA `415f2bd2a233fd9a5e1f4c1cf8906f7e0f05ddd01e1ef73c5888f223c01ad4e9`，记录执行源码SHA及真实2/2、compile和资源边界。
- [CPU-GUARD](CPU-GUARD.json)：SHA `1886f4744c5cc87baca020a275503a24eb45929932dd14b31bffbc2c6e7aef30`，exit0、当前Pod UID/Ready入口、A12核0–11/nice19，隐藏GPU且未初始化。f5软件份额到期前实核；根12–23保留未使用，与训练96核24–119互斥，旧grant不改。
- [GLOO-RESULTS](GLOO-RESULTS.json)：从原执行stdout提取的两rank实际结果，含源stdout SHA。每rank全64raw logits/448pair、local positives分别0/8，总loss均`.4426952600479126`；共享weight/bias/age梯度逐分量与独立全局reference匹配，bias实际`.02895835041999817`，无额外WORLD倍率。

第一项还以手算标量公式验证0.1平方项、公共bias与age链式梯度；None旧值/梯度零容差、Python/NumPy/Torch RNG与canonical job identity不变。第二项由真实load_config/job_contract/CheckpointStore保存加载policy，再在两rank调用生产总目标；这是小型保存身份与消费证据，不是完整Small/Adam恢复或八rank训练验收。所有Python、解析、哈希与测试均在核验Pod内执行，本机只文本/Git/传输。

原件目录为`exp-v3/technical-prep/output-logit-l2-cpu-20261101-7c17`；本机原执行日志在`work/output-logit-l2-cpu-20261101-a/{RED,GREEN-1,GREEN-2}.txt`（工作区外过程材料，不入Git）。原R1301503、R1/R131冻结产物及R120/KEEP资产全部保留；未读/hash/load任何旧大blob。此归档commit与R131运行commit分列；后续干净远端Git源码校验小件留独立prep，避免自引用归档循环。
