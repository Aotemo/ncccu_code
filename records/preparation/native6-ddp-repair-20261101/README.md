# R150 DDP 启动输入修复

真实八rank均在`prime_reducer`失败：模型stem要求6通道，原预热固定构造3通道。CPU配置/模型forward检查没有覆盖这个DDP专用预热，原20/20不等于真实GPU跑通。R150没有有效初态、成功更新或科学selected；[原始失败](../../R150/R150-FAILURE.json)保留，safe观测不重复。

最小生产修改仅`distributed_engine.py`导入既定六通道规则，并据实际job选择6或原3通道。原两次预热、反向、state/梯度/RNG恢复和后续训练均保持。新增一个现有candidate test目标，真实Small factory、CPU32画布、world8上下文直接调用生产prime函数，3/6两分支检查两次前后向、state/RNG恢复及梯度清空；这是CPU启动路径回归，不冒充真实DDP通信或512 GPU证据。

RED实际复现6/3错误；GREEN单目标通过，9.283秒，GPU未初始化。[CPU-CHECKS](CPU-CHECKS.json) SHA`a37cd9be15dd9be2cc51afcd306f86d721da8001b986fcbb1ac18bdd28ebcb98`合计覆盖4目标：原3目标按六runtime精确SHA和测试模块除新增方法外AST相同复用。9源编译中本次2源，其余7源复用；未重复已通过范围。原partial exit1和首次未完成传输造成的ImportError均留在Pod，不计为有效RED或PASS。

R150R1为独立fresh工程重试，沿相同R150 SCIENCE `4ef14336`/order62和公开fresh初态`ea92356b`，不是新科学变量。新round/clock/grant/activation必须独立，原R150冻结、token、失败、全部依赖不改。继续使用既有prepare/formal/deploy/activation/direct launch入口；实际八rank初态和自然首成功是GPU跑通依据，无独立GPU预演。R148完整双门false，R149无资格，未formal或GPU。
