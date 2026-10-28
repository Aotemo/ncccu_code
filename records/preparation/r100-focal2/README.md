# R100 focal微型CPU与配置接线验收

对照R95，唯一科学delta为隐式BCE→binary_focal gamma2/alphaNone/float32/mean/modulation_detachedfalse/version binary-focal-v1。dropout0.2、backbone1e-4/head1e-3/age1e-3/seed20261027/24k/800/p4/1600全沿R95。不使用R17/N06或其他旧模型权重；只实现显式配方。

实时UID/Ready及validation4 TTL/live CPU核对后，A mask0..11、nice19、隐藏CUDA/HIP/ROCR内统一执行13项：B的3个微型值/梯度/有限性方法和10个现有受影响candidate/config集成方法。13/13、0 failure/error、exit0，测试5.774秒，torch intra12/inter1，GPU未初始化、无模型load/forward。AST解析和8件源码SHA前后相同。只做微型loss张量运算，不是模型/GPU预演或数据前向。

默认None直接原BCE（FP32及FP64值/梯度，保持旧dtype/mean）；gamma0与BCE等价。gamma2对独立sigmoid/pt/log概率参考核值/梯度，rtol2e-6/atol1e-7，并与detach参考梯度有明确差异；±10000/±1000/0下loss/grad有限。两训练循环都调用同一classification_loss(logits,targets,job.get('loss'))。旧N05 BCE监控不改，模块顶层仅math、Torch在loss实际调用时延迟导入。config可选loss字段不向旧BCE默认job加入loss=None，保留旧job identity；新非None值由共享validator验证、由pinnedSCIENCE精确绑定。

独立prep `/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/r100-focal2-20261029-a`。CHECK原Pod SHA `f50b2d9177d5690575f2cd909443aad71d8a0d4610d7d1964c4044f3eb47dc44`；guard原SHA `fd316f4a2b8311a70e93653afa96bfe7fb9c43eb0eb6ad1ca1890d0431511ccf`。源码逐件SHA见该原件；同任务随后只读确认R99真实终态和安全交接，没有重复官方前向。本机仅普通bytes传输/文本/Git，不解析或重哈希。

新实际Git身份的正式CPU初态包和真实训练初态仍沿现有入口独立生成一次，不以本13项冒称已GPU开训，不新增来源/native/旧模型全套或独立GPU预演。
