# 准备助手重复容量判断最小修复

静态核对发现现有private prepare适配助手的正常路径虽已读冻结STORAGE_POLICY并核其与Git配置相同，随后又按SCI编号手写12/16/32GiB预期。R198–R201新增编号未在默认12名单，落到32预期，因而会在正常CPU/部署后错误拒绝；此前technical-only已提前返回，未执行该段。尚未实际运行这些新轮normal，没有把反事实拒绝写成正式阶段失败。

[精确补丁](HELPER-FIX.patch)只删除这2行重复判断、0新增行。冻结policy等于Git配置的检查及读取同一round_overrides/default值保持；原runtime/run_round.py预算函数和版本化存储policy完全未改。旧helper SHA `aedc61d4860fb819df367c20ffd30565ac42f262a2dcf647f677aaeb5b93a32d`、新候选 `7fc6b3ddf803ed12922df5250ce0f1c7b27bb289752043d1f9c2de9048ee1aa3`，实际完整原件在`exp-v3/technical-prep/prepare-storage-20261103-32e3`，不新建维护一套训练/启动平台。

[原段RED](CHECKS-RED.json)实际执行原助手存储段，四个新轮因重复预期被拒，R197/16/32旧规则通过。[修后GREEN](CHECKS-GREEN.json)七例全部通过，原预算函数对超limit及free15GiB均拒；1份完整helper编译通过。四新例读取各自真实technical包，R197读取现役冻结policy；16/32两例为同一真实policy上的函数fixture，不是重跑旧实验或实际写入大文件。[汇总](CPU-CHECKS.json)逐件核CPU guard/OUTPUT SHA与整树退出/无违规，核源码只删两行、当前入口尚为旧SHA、四新轮无formal/gpu-inputs/真实round。verify guard_guxcdrd / `72fa9398379fdc356c18a341de454c849949bcda343f0c0729c79b4333689a96` exit0；均在53ab/root0–11/nice19/隐藏GPU，与R197训练池互斥。

首两次窄检查因自身语法拼写和runtime导入路径失败，原8xcxzf6s/a935a871与eyg_egtk/a099cadb回执保留；修检查入口后shg7o73x/b72b946f才是真实RED，5irt3adg/43d74123为GREEN。没有新模型/forward/正式准入/allocation/clock/GPU，未改当前R197。补丁和回执先Git同步，入口安装随后单列；既有5/4或3/3模型/候选证据不重跑，下一normal仍沿原入口执行一次完整producer。
