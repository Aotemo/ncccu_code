# R154 Swin V2 Small实际八卡启动

2026-11-02 01:51:51.963063 UTC[真实首成功](R154-START.json)：running50 success=attempt、0skip、尚未评估。402个全训参数张量、5组空AdamW、48个静态位置buffer、无BN，whole `4ff58b9c91fb82b9577bcd7de7711d3eb07e24bd05b8dcacd94c54c117b58726`与formal一致。首自然25 loss `.6772598624229431`，WORLD8/G64P8、全部448pairs、每rank8图；owner478602/start361157965及8rank478614–478621的birth、argv、96核mask与8PCI身份匹配。没有新增GPU预演。

冻结runGit `04172db46eccc884f5d4e242c1235d066c8657af`、tree `c3e626dc38fc8c528b271ae718b6ebecc7986462`，SCI `290f67afb7263b04e71431b2dec95c65467bf5c95edda030a0f51048fd0b06f9`、formal SITE `96547e8539661e849954590f802cdf9a7d0a5b77be2623afeb11acb1d6fd802e`。原3目标/12compile及附加实际512 CPU priming按字节复用；[唯一正常formal](R154-PREPARATION.json)21/21、74.490秒，[精确部署](R154-DEPLOYMENT.json)13.838秒，随后一次activation/direct launch exit0。保存job及准入输入几何见[实际几何身份](R154-ACTUAL-GEOMETRY.json)。

[接续间隔](R154-HANDOFF-GAP.json)分列：R152训练完成事件至dispatch251.165秒、至最晚八卡设备309.649秒、至首自然25记录350.845秒，后两项超过300秒；从R152整轮ended_at计分别196.835、255.319、296.515秒。没有连续利用率测量，不能称连续空闲不超过5分钟已证。后续交接在必要校验通过和actualsafe后立即执行，文档与归档留在准备池，不新增GPU预演。

唯一Git/GPU owner32e3。实时有效120核，训练冻结24–119共96核；[本轮准备绑定](R154-PREP-ALLOCATION.json)aa0f下，根准备0–22共23核、用户另开分析会话CPU23共1核，互斥且合计24核，有效至2026-11-02 07:33:47 UTC。各执行者自身任务串行，允许互斥准备池并发，不热改旧allocation、活跃运行或clock/grant。

本轮直接对照R107：独立public ImageNet1K fresh Swin V2 Small、RGB512，原FP32/G64P8/AdamW/LR/WD/age/增强、最多24000成功、每800评估、patience4及唯一内部selector保持。R117 V1 Base为负先验，来源与宽度也不同，不作纯V1→V2因果解释。完整工程有效selected同时严格胜R107及R120才有固定seed27资格；官方仍仅完整内部selected≥`.996`后5849一次。启动证据不代表实验完成或优化有效，raw `.999`未达，继续训练。
