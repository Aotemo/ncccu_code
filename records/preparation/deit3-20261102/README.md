# 原生 DeiT III Small 下一轮准备

R164/order76独立直接比较R107：自然ImageNet22K→1K公开DeiT III Small初始化，原512输入、单视图logistic、G64P8、age及AdamW/LR/EMA/24k-800-p4合同。公开384与比赛512分列，原生bicubic/antialias重采样patch位置，保留CLS/LayerScale/norm。不是医学预训练、teacher或旧训练checkpoint，不将模型与初始化整体比较解释成纯像素效应或临床机制。R162完整阴性，R163固定seed27无资格且不formal/GPU。

[CPU-CHECKS.json](CPU-CHECKS.json)记录最终5个受影响目标、8源编译及原6个loss/WORLD目标按未改SHA复用；最终guard `99ff13b087da645eed6eb1411d38b824897759b404e334146514d3bae678da9c`。真实CPU合成P8×3×512反传全部177张量梯度有限；原工厂同seed参数/RNG/512输出对照、5空AdamW组、age一次、EMA及真实CheckpointStore→inference冷图通过。初始化回执补齐kind/来源后重新实核完整public fresh，张量仍64556b75。先前缺初始化格式的失败、仅配置窄重试及来源打印/拒绝覆盖失败均保留。

[PUBLIC-HANDOFF.json](PUBLIC-HANDOFF.json)及[PROVENANCE.json](PROVENANCE.json)绑定8d59b281固定公开来源、原bin实际88140b2a/88885197B、完整176张量及1000头strict/exact/FP32有限，零转换、来源核验零forward。timm1.0.27版本及两个原生定义文件SHA固定，不称pin整个transitive包。[PUBLIC-FRESH-SUMMARY.json](PUBLIC-FRESH-SUMMARY.json)保留两fresh/空优化器身份。原Pod proxy503和raw relay均有原件，权重/数据不进Git。

CPU保存的tensor storage合计2848580704B，既有Small完整24k×2时间情景24154.892秒，仅为资源参考；均不是DeiT3实测DCU峰值、workspace或吞吐上界。12GiB默认预算、20GiB运行可用下限、16GiB设备下限及现有时钟不扩张，无独立GPU预演，后续以唯一真实八rank初态/自然首更新证明GPU链。aa0f根0–22/23核、peer23/1核与训练24–119/96核互斥；新运行沿现有正常formal、精确Git部署及一次activation/launch。此处CPU准备通过不等于已执行新GPU。
