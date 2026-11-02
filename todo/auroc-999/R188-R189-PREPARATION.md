# R184R1 之后：条件均值复核与固定原生 CLS 读出

21:16当前唯一GPU已为R184R1/cd7246a，最新2925成功/0skip/eval3/bad0，暂EMA内部AUROC `.9750773939753712`，尚无完整selected。R186已完整阴性/safe，R187不eligible；R188/R189仍原0f3两technical包、未normalformal/clock/GPU，完整R184R1双门与safe决定既定分支。下一单一ViT-B公开来源只作[独立可行性准备](VIT-BASE-SOURCE-AUDIT-20261102.md)，不改本队列或现冻结。下文为原准备历史。

20:26两technical包已按实际推送/远端回读的0f3b19e6/tree458e9c7f完成，[回执](../../records/preparation/vit-cls-20261102/TECHNICAL-PACKAGES.json)均guard0。仅扩现prepare辅助脚本两SCI/4目标5编译/12GiB路径并保留R187/R184R1执行文本；它们的cd冻结包/接续顺序及当前R186不改。新两轮尚无normalformal/clock/GPU；R186最新4575成功/0skip/eval5/bad0，暂EMA .983750576466211，完整结果未出。

20:20实际4个必要CPU目标/5唯一源码编译及两public fresh已通过，[原件](../../records/preparation/vit-cls-20261102/README.md)保留真实数值RED。两个CLS模型目标25.617s、共享非持久回归6.131s、新身份目标3.592s均guard0；R188 whole8c62435a/SCIb21417ee，R189 wholee347757c/SCI11cf4bc7，均真实noBN EMA首更新、0实图/0GPU。None/原avg兼容与保存冷图已测，source模型不添参数/forward；零SHA不进Git。当前Git同步与技术包尚待完成，不改变R186及既定R187/R184R1接续，无newnormalformal/clock/GPU。

仅在R186运行期间的既有20%准备池预备后续，原R187/R184R1顺序与cd7246a技术包、当前R186 f9冻结均保持。没有新Agent、GPU预演或新资源；32e3唯一Git/GPU。当前空闲准备池用于一个独立、最小的末端聚合假设，不按中途R186分数改本轮。

R188/order101：只有完整有效R184R1唯一所选内部AUROC严格过R107/R120且actualsafe后，复核原meanpatch/同3bf4720a公开源/固定seed20261027；旧R185已不eligible，不能复用其身份。否则R188仅技术准备。

R189/order102：独立fresh固定seed20261029，直接R184R1原native ViT-S/DINOv3/512/原pairlog/G64P8/AdamW/LR/EMA/age/24k-800-p4。仅将原生final LN后的聚合从1024 patch均值改为第0个CLS token；仍保留全部5个prefix和1024 patch参与Transformer自注意力。完整原生162公开张量先strict载入，保持现avg构造后仅改global_pool属性，不能重建LN/参数或改变RNG消耗；外部Dropout.2/Linear384→1/age仍各一次。先完成R184R1及有资格的R188并安全收口；总体raw .999已经达到则无需该后备。

依据只作为可检验假设：官方[DINOv3训练源码](https://github.com/facebookresearch/dinov3/blob/main/dinov3/train/ssl_meta_arch.py)将norm后CLS用于DINO全局head，patch用于iBOT；[官方骨干](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py)显式返回norm CLS。官方另有CLS+patch mean拼接分类头，本轮只检验固定CLS单读出，不声称复现其完整分类配方。没有临床机制或收益证明，不扫池化系数、位置、输入、LR或seed。

精确必要范围：config.py增加一个固定vit_dino3_pool_policy及None省略/校验；models.py现有工厂在严格加载后只设原生pool属性并在describe中验证/保存非张量身份；experiment91.py现有科学合同登记R188/R189精确id/order/policy并绑定保存job；configs/R188-SCIENCE.json、R189-SCIENCE.json；tests/test_deit3_model.py补原生CLS独立oracle/default兼容及P8/真实EMA/冷图两目标，tests/test_candidate_binding.py补两身份目标。inference现已有job→factory→describe链，优先原样复用；engine/WORLD/loss/EMA/persistence/r92 gate/启动/监控均不复制或改写。若现有共享非持久buffer回归受模型描述改动影响，仅重跑该必要目标；既有不变P8不重跑。

旧None/省略保留原avg函数和值/RNG/张量/描述；新policy进入保存job与冷构图，即使全tensor SHA相同也不能误认同函数。新增两身份先CPU窄验及实际公开fresh、最终SHA再Git同步，不提交初始化占位。默认12GiB/free16GiB与官方所选≥.996后5849一次不变；CPU存储不是DCU峰值或吞吐保证。当前只是后续规划，尚未实现/训练或改变任何既定GPU合同。
