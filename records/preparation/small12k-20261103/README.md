# R218/R219：Small12k 来源和绑定 CPU 证据

[CPU-CHECKS.json](CPU-CHECKS.json) SHA `98ae50cd1c8cfe4b6e2c171c24be526543454e16750cd761a378d81686fde15b`：7个受影响目标、5源码编译及两完整public fresh通过；[最终guard](CPU-FINAL-GUARD.json) tt0qtfwv/b890d1c3 exit0。原件`exp-v3/technical-prep/small12k-source-screen-20261103-32e3`，d837/root0–11/nice19/隐藏GPU，与当前R217训练24–119互斥。

固定`timm/convnext_small.in12k_ft_in1k_384` / revision e9f96b3e，public/non-gated/Apache2；fixed API/config/card的Git blob逐字节匹配，完整许可SHAcfc7749b。原bin201017809 B/SHA0a127fb2已Pod完整核验。实际生产download新profile先RED（未登记，exit2），最小扩展后GREEN；旧small224 API/CLI默认、旧常量和旧源分支保持。没有用mock替代下载或来源证明。

[来源handoff](PUBLIC-HANDOFF.json) SHA67b56ac1、[实际provenance](PUBLIC-PROVENANCE.json) SHA772dde6c；完整344全1000/342骨干+2原头逐张量strict/exact/FP32、冷读取及41LN/36精确GELU-SD/40Conv/无BNbuffer通过。新增profile额外核前置norm/head.pre_logits Identity，未减旧检查。converted SHA553fd294；图像forward为0，不把映射与语义检查称成新timm→TV前向数值一致。

真实新来源provenance消费先因ImageNet12k未支持而RED，最小固定版本/许可接入后GREEN。原生产create_model函数文本与旧6fb逐字相同，forward及data/loss/gather/optimizer/EMA/保存/评估/入口源码按身份复用。新候选绑定与受影响的旧R216/R217目标两项GREEN；首版漏核条件EMA别名的原bin SHA，实际失败保留，补一项精确pin后通过，不软化测试。

R218仅完整R217严格胜两固定基线且actualsafe后的同原fe6/long EMA独立seed27；[实际fresh](R218-PUBLIC-FRESH-INITIAL.json) whole91507a65/SCIcbdd1629。R219直接原R107/seed29原job，只有公开初始化换为固定ImageNet12k→1k；[实际fresh](R219-PUBLIC-FRESH-INITIAL.json) whole568263ab/SCIca352c2e。均345全训、49,455,459参数、0buffer、5空原生AdamW组、0图像forward；metadata实际EMA分别1600/.9999与1600/.999。来源384/squash不改比赛512预处理，不叠EMA或其他策略，不作纯分辨率因果解释。

原P8/冷加载仅按未变函数/源码身份复用，未来正式CPU仍由原入口绑定实际阶段。容量为原Small浮点情景，不是DCU峰值或磁盘峰值保证。未读官方/训练checkpoint或新增GPU预演，未证明临床机制、fit上限或AUROC收益；当前R217冻结6fb不改。

13:20实际源码已提交并推送 `b8b8bbe6a21e60620a858c299172e1de9b5247fc` / tree `de0a717672a61a1c2a541759f68f921d44ea140d`，GitHub API回读与受影响文件/SCI SHA一致。[两技术包](TECHNICAL-PACKAGES.json) SHA320efb7e已生成，R218/R219 guard分别vkfzglw5/5a533017和drgdsdci/87649cf7均exit0；绑定guard1h4uh6kp/d51e6180。完整Git源树已传输，逐文件部署核验仍由后续原normal/precise-deployment执行，不能把技术包当成正式可启动。未创建新formal clock或GPU任务；后备normal临近当前轮收尾执行，避免提前耗尽完整授权窗口。

13:35[完整源树传输核验](TRANSFER-VERIFICATION.json) SHAa359a5eb / [guard](CPU-TRANSFER.json)u9t5mhon/08a1bb38 exit0：直接复用现有freeze_deployment.verify_commit/verify_tree，2337文件清单/全部Git blob内容和执行权限与b8远端完整tree一致。提前排除R215曾遇到的漏文件传输问题，无模型或数据检查重跑，不产生formal allocation/clock/GPU。后续正式包/manifest/源码之间的部署绑定仍由原normal入口完成，不能以本次源树核验替代。
