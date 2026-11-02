# 固定原生 AdamW beta2 对照准备

[CPU-CHECKS.json](CPU-CHECKS.json)记录3个受影响目标、6源编译通过：独立三步参数/一二阶矩公式与保存后下一步、None/省略原生默认组/RNG/更新一致，以及R153–R168配置绑定含缺失/错误policy拒绝。先观察原_optimizer忽略新policy的真实RED，最终guard `02edd6ca73647c7ebebcfbab87f08fcf44cdff1061cce1d5978c1f45a49f2691`。engine整文件确有构造分支改动；逐顶层节点核除_optimizer外相同，原loss/测试/gather与distributed_engine/EMA/model/inference未改，因此复用原6loss/WORLD目标，不声称新整engine未改或新独立WORLD实跑。

[两public fresh](PUBLIC-FRESH-SUMMARY.json)：R167原生meanpatch seed27为8108ef45/177全训；R168原R107 Small/fe6b/seed29为50c6f55c/345全训，均5空原生AdamW组、0buffer且完整公开源strict/exact。R168五组实测betas(.9,.95)、eps1e-8/amsgradFalse，原LR/WD与age不变。只传betas给原生[torch.optim.AdamW](https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html)，其余默认及算法/state schema保留；不是Lion/PESG或新增优化器平台。

R167仅在完整R166双门通过且safe后复核固定seed27，否则独立R168只改beta2平滑系数，回原512/logistic/G64P8/age/clip/EMA/24k-800-p4，不叠加DeiT/meanpatch。检验优化时间尺度，不预设梯度噪声/临床机制或收益；完整内部AUROC主判、AP/难组不救，阴性不扫beta/LR/seed。旧Small时间情景为参考，不是实测GPU上界；12GiB及当前时钟不扩张，无新GPU预演/资源。

当前R166的98badc9冻结不改；dca3准备根0–22/23核、peer23/1核与训练24–119/96核互斥，至13:17:09 UTC。CPU通过不等于后续已启动，实际Git同步、技术包、唯一normalformal/精确部署及安全后一次启动仍由根32e3完成。
