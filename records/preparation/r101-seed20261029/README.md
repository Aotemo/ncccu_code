# R101固定新seed初态控制与配置验收

直接R95/BCE复核，仅job.seed20261027→预固定20261029；splitseed20261024、固定fit/dev清单、age统计、dropout0.2/LR及24k/800/p4/绝对1600不变。SCIENCE不带R100 focal字段，旧BCE入口仍保持缺失loss/None语义。不是seed搜索或挑最好seed。

CPU生产入口原configure→_seed(job.seed,cpu)→create_model(initialize=True)抽为同一个create_cpu_fresh_model helper，正式check_candidate.main仍调用该路径并要求实测整体digest与pinnedSCIENCE完全相同。技术control-only阶段使用新seed草稿，以实际公开权重严格加载后的fresh binary+零age、空Adam和初始EMA实算控制，未加载训练checkpoint、未model forward/optimizer step、无GPU/clock/正式CPU-PREP或activation。空控制草稿从不用于正式准入。

实际新整体模型控制为 `18313b8b6d240b080eef7af214b41306b83a347135a2b7f5958addafc79877dd`，与旧eba控制不同；EMA相同、age零、Adam状态0。Pod生成 `SCIENCE-CONTROLLED.json` 后普通字节填入 `configs/R101-SCIENCE.json`，SHA `2f01bb94c6ae5e8eaa4ff8933e76778e9075a34c592be540ad2202ff8045a35e`。`INITIAL-CONTROL.json`原Pod SHA `8230b7f50a49804589ce3519341ff5d28b82e4c3dadf6c2fd05884d3f9c8ed0a`，含head/整体tensor、public strict证据、实际p和seed/运行名，不保存模型权重。

现有10项candidate/config窄验在实时UID/Ready及validation5 TTL/live资源核对后的A mask0..11、nice19、隐藏三GPU环境中通过，0 failure/error，5.705秒；source在测试期间SHA不变，torch intra12/inter1。seed期望从pinnedSCIENCE读，prepare运行名使用实际job.seed，不沿旧seed名称。未重损失微型数值、旧来源/迁移/native/全套或独立GPU预演。

独立prep `/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/r101-seed20261029-20261029-a`。CHECK原Pod SHA `00007c31ca8b8e98175b4bc31081191bd7b596d6bccedcc0b95be2bdf9b783e8`；同任务只读确认R100安全终态，guard原SHA `def0583be62441108e77a97b3cfd130a621898802bd5e3867ba309942ea40b67`。所有文件仅普通bytes导回，未本机解析或重哈希。

本控制阶段不是正式CPU admission；真实Git身份的新正式CPU包仍执行原完整18门，重新构造的实测模型必须匹配本次新控制，随后实际GPU8rank初态继续验证。没有继承桥或放宽初态门。
