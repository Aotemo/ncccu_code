# R194/R195：固定较早余弦衰减

[实际CPU证据](CPU-CHECKS.json)共6目标/6个受影响源码编译通过：原生五组真实LR的独立公式/边界/重复成功计数；None/省略保留旧曲线、state及RNG；保存job重建后Toy+native AdamW状态往返及后续参数/state exact；新两候选/config/保存job/来源绑定；原R192/R193身份和半骨干LR消费者保持。两个预期RED（未消费日程/未登记配置及候选）原样保留。没有新P8/EMA/cold/WORLD前向，原模型/loss/EMA/DDP/保存/推理源码逐件SHA不变，按作用域复用旧证据。不是新的生产八rank恢复或DCU峰值/吞吐证明。

R194/order107只在完整有效R193唯一内部selected严格胜R107 `.9884405731067525`与R120 `.9894773092062407`且actualsafe后，复核同半骨干LR/Small/fe6b/sigmoid的固定seed27；否则技术准备。[实际fresh](R194-PUBLIC-FRESH-INITIAL.json)whole `91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67`、345全训/5空Adam，SCI `514cab3827a0b95b6a81a6331473085d9203ac055d7941669a021d747368dd20`。

独立R195/order108直接原R120/seed29/512/G64P8/原sigmoid/原AdamW与backbone1e-4/head-age1e-3，仅新增`lr_decay_updates=8000`。下一成功更新的计数s<1600沿原warmup；之后原余弦分母为6399，于s=7999（第8000次更新使用的LR）到尾倍率.01，往后保持.01。总上限仍24000、每800评估/patience4、EMA warmcopy1600与唯一内部selector不变；五组都消费此因子，原生decoupled decay随LR变化，不声称只改梯度项。[实际fresh](R195-PUBLIC-FRESH-INITIAL.json)whole仍 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`、345全训/5空Adam，实际optimizer日程元数据8000，SCI `57c1d89a61ef9e9e4306b680961c4e5cac7be5c13e6c3569deeb3a9e693a4841`。这是固定优化假设，不是对原早停的bug判定、临床机制或收益证明，不叠R193/不扫LR。

最小运行改动：原_optimizer仅在非None时从job设置日程元数据，原_apply_lr读该跨度调用未变_lr_factor；参数组和AdamW构造、DDP调用均未改。非张量日程随保存job/job_contract固定，由原构造器重建；不能仅凭相同模型tensor SHA识别日程。None不加optimizer属性，配置load/contract省略None。新原CPU producer的日程边界块已静态核对/编译，真实check_candidate整体只在后续normal准备执行一次，尚不称其运行通过。

必要guard在53ab准备root0–11/12核、nice19/隐藏GPU执行：数学/config/native-state三目标99qyoiy_/`158ec11a78370f131c9f690d0af625257a20713d4c423c5ca54779f7c1ea8645`；新旧候选三目标i9jyxmf5/`efaca619237d4fbefc964f00359a6bb751e2518d663a4956a7545fe11dc6c93b`；两actual fresh 1h2cfznh/`1cc8c687778a8c008c5d06889409ca616d4c9b4c552712a25a0769af91c744f8`均exit0。最终p9m1h9xx/`81ace41b96a3efcd78f1a68047d1bf4f6679dc648bc7f5a0b3a18fbe06a650fd` exit0；原件 `exp-v3/technical-prep/cosine8k-20261102-32e3`，0图像forward/0官方/0GPU/0新clock。

两轮默认12GiB/free16GiB不变；当前R191/a77与更早既定R192/R193/2a冻结及顺序保持。下一步仅实际Git同步/回读与技术包，不把这份CPU记录等同formal、启动或收益。32e3唯一Git/GPU，无新Agent或资源，raw .999未达。
