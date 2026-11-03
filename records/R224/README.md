# R224 fit-only 遮挡空间重建

11-03 18:25 UTC，实际八rank公开初态及自然首25次适配更新通过。首1 loss1.66234958、25 loss1.10275948，loss与梯度均有限，G64/P8、每图154个32像素块遮挡、单视图fit-only。公开模型50c6、encoder/public张量、空优化器及临时空间头768→3072/PixelShuffle32均与正式CPU匹配。当前是SSL阶段；1600次适配后的监督derived初态、首监督成功和完整内部dev尚待实证，没有收益或达标结论。

冻结Git `9c82ac0ded19446b9e1cdeb6c229fa0dee2f29e1` / tree26818e2a，SCI79afb5e1、actual SITE0f4eed58。7必要目标、6编译和[真实公开Small/P8及冷derived](../preparation/masked-reconstruction-20261103/README.md)通过，正式包18/18 CPU和整树Git字节/模式部署通过。一次activation及唯一实际launch均exit0，无独立GPU预演。[正式准备](R224-PREPARATION.json)、[实际首成功](R224-START.json)、[公开初态门](R224-SSL-INITIAL-GATE.json)及相应CPU guard保留；登记不替代后续监督/完整结果证明。

只用现有fit图，保持1600适配预算，以masked RGB空间回归补充R115全局VICReg；随后保持原R107监督合同，丢弃临时decoder、重建空AdamW/EMA。不是延长VICReg，也不能补回512输入外的细节。完整内部AUROC为主，AP、原固定20/其余147的修复和新增错序并列；仅严格超过R107 .9884405731067525与R120 .9894773092062407后才运行固定seed27的R225，单次高分不升级基线。官方依现有≥.996完整selected门一次，尚未读取。

owner790052/start375759290与八rank出生身份、实际设备证明已核。本会话4e0c为唯一Git/串行GPU监督，准备0–23/24核、训练24–119/96核互斥；当前私有入口`goal-20261103-4e0c/r224-pod-context.env`，原部署/clock/token/grant均冻结。R221R2旧safe不重查、R222不eligible、R223暂缓，无新Agent或计算资源。

前轮17:55:17结束至新dispatch1692.424秒、首SSL成功1796.130秒、自然25记录1805.525秒，超过300秒。[真实间隙](R224-HANDOFF-GAP.json)记录新模型必要验证、源传输及可避免的归档/传输错误；不能称为连续DCU利用率。第一次传输遗漏stdin，launcher和dispatch均未出现；修正`kubectl exec -i`后仅调用一次实际launcher，使用新的launch-only Pod快照，保留原激活时钟。一次导出脚本未显式传入active env而在读取身份前失败，改为读取该ready下已验的SITE后成功；不改训练或重跑模型检查。
