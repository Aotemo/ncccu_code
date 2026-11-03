# 未来准入授权准备窗口：CPU核验

[实际核验](POLICY-CHECK.json) SHA `be1fffbb5bf4aa79e1e7eccccdc4fb541d316d038405955fb3a7ef73d5e9d3c5`；[guard](CPU-POLICY-GUARD.json)j5ymtrey/9dde8d2d exit0，现有4个GPUSiteInputsTest方法全部通过（0.004s）。原件`exp-v3/technical-prep/gpu-allocation-headroom-20261103-32e3`；d837/root0–11/nice19/隐藏GPU，与训练24–119互斥。

唯一生产差异为配置`gpu_allocation_max_seconds`32400→43200；新policy SHA `0b1a5f9509251c383c0b0103419d0ab673f92dc2a8269bfa5971d3c5123ea063`，新旧字典其他字段完全相同。时钟仍31500秒、预检60秒，准备余量840→11640秒（3小时14分）；不延长训练或增加计算资源，不改当前R217及旧grant/clock/token。

现有实际consumer按实时120核构造的新旧pool完全一致：准备24/训练96，rank各11+control8，复用同native guard artifact63ccf6b6并通过原验证函数；未重新编译native guard或重跑模型。CLOCK、两SCI与既有生产Python SHA均一致，原Small12k7目标/5compile/两public fresh按原98ae50cd身份复用，原回执不修改。

该核验尚不是新的正式包/GPU授权或已部署实测。后续需先同步新Git，再在独立未启动ready-headroom阶段由原prepare_gpu_site消费43200秒、原normal及precise-deployment绑定真实文件；不热改原b8 technical目录、旧GPU绑定或当前CPU pool。未新增平台、GPU预演、模型forward或官方读取。
