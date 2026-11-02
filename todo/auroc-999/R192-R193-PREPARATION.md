# R191 后续固定分支实施计划

使用 writing-plans / executing-plans，由唯一owner32e3原位完成；用户已授权最小修改和同步，无例行审批或新Agent。目标是在R191训练期间完成下一串行fresh所需准备，不改现役a77冻结。

22:20已完成[实际4目标/3编译与两public fresh](../../records/preparation/pair-sigmoid-lr-20261102/README.md)，最终guard x9eqkxst/942fe6d6…1c1 exit0。R192 whole6a261e48/SCI689c3b57，R193原whole50c6/SCI0985f576；一次私有helper语法失败已保留，修正后实际fresh通过。当前同步源码/最终SCI，尚未technical包/formal/clock/GPU；R191原a77八rank仍活跃，22:20最新1500成功/0skip/eval1。peer22:13实核idle/无activeTurn，仍保留12–23不唤醒。

固定合同：R192/order105仅在完整有效R191唯一selected严格超过R107 `.9884405731067525`与R120 `.9894773092062407`且actualsafe后，复核同Base公开c6a5fb7d/CLS/512/原logistic的固定seed20261027。未满足仅技术准备。R193/order106独立直接R120 Small/fe6b/seed20261029/512/pair-sigmoid，只把backbone_lr `1e-4→5e-5`；head/age仍`1e-3`，原AdamW/WD/采样/增强/G64P8/EMA/绝对warmup1600/24k-800-p4/唯一内部selector保持。仅检验较慢骨干适配这一固定配方，不先声称收益，不扫描LR或用官方选参。R96同数值是Tiny+BCE，R130是逐层衰减，均不代替此Small+sigmoid的直接比较。

范围仅`configs/R192-SCIENCE.json`、`configs/R193-SCIENCE.json`、`runtime/bundle/retrain/experiment91.py`登记两身份及精确合同，`tests/test_candidate_binding.py`覆盖候选/保存job/来源/LR实际消费；沿现有prepare/fresh/check_candidate/CPU/Git/冻结/启动/保存/评估/观察链。R192仅新精确round16GiB例外及原storage函数检查；R193默认12GiB，所有旧例外/free16GiB不改。无模型、loss、optimizer或scheduler实现改动。

- [x] Pod内由R191/R120原SCI生成两独立草案，候选test实际RED确认尚未登记；原JSON/源码不热改活跃包。
- [x] 最小登记后两个必要目标：两候选/config/job_contract身份与R193真实5空Adam组、backbone/head/age参数归属及LR；原apply_lr在0/799/1599/1600/23999消费，核每组原公式，旧R120配置仍1e-4。精确新round存储函数目标复用原fixture。仅编译实际改动源码。
- [ ] 两actual public fresh核初态及5空组，原不变模型P8/EMA/cold/loss/WORLD按SHA复用；写真实whole/SHA/CPU回执后同步Git并远端回读，实际技术打包。
- [ ] 可行收尾前完成唯一后备正常CPU/精确部署；完整R191决定R192资格，actualsafe后沿现入口一次窗口/activation/direct launch，真实八rank初态/首自然更新证明GPU链可跑。不因文档或重复验收等待。

准备使用已实核20%互斥池，训练80%不热改；当前5de准备root0–11/保留peer12–23，训练24–119。官方5849隔离，完整内部selected≥`.996`才最终一次；raw `.999`未达。
