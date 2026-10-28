# R91 完成记录

普通 avgpool ResNet50，公开 ImageNet1K V2 fresh，线性头与 fit-only 年龄分支，固定 9:1。

6400 成功更新、8 周期、0 skip；内部规则选 cycle5 EMA / update4000。内部 dev AUROC `0.9732399267421321`，AP `0.7510259793479146`；官方完整 5849 一次 AUROC `0.9746659041823744`，AP `0.7372105728953852`。目标 0.999 未达到。

服务器 source digest 为 `76cb37a6cd69238d6269c02e019c79a3349b58fad017f9a71eca867dba624b01`。[最终回执](FINAL-RECEIPT.json)保留完成状态和指标；最终对账 20/20、真实 exit0。未重复训练或测试。

本模型不在用户确认的三个永久保留对象中，checkpoint 与过程产物属于清理范围。以下仅记录历史正式根，不表示其中原件仍存在：

`/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/rounds/R91-R50-IMAGENETV2-9010`
