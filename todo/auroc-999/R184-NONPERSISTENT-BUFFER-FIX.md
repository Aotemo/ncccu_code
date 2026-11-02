# R184 非持久 RoPE 缓冲区：工程失败与最小修复

R184/order96 在八rank初态门、首成功更新之前工程失败：`native immutable model buffers differ`。19:20:13 UTC 一次终态观测确认 done=failed、selected=null、0成功/0尝试、owner同birth已Z、八rank消失、锁与设备使用者空、8PCI busy0，actualsafe。旧clock/token/grant、部署4dc51803与失败原件不改，不重启；R185条件seed27不eligible，不将本轮作科学阴性。独立R186沿既定串行队列接续。

根因来自当前源码/已有CPU原生图证据：ViT-S DINOv3 的 `network.rope.periods` 为 nonpersistent buffer；实际 `named_buffers()` 有1件且有固定SHA，`state_dict()`不保存。前一最小修复仅补了 verify_records 对实际模型buffer数量的识别；生产 maybe_release 随后把checkpoint中的持久buffer子集与全部模型buffer的1件/SHA比较，故失败。原normal CPU未覆盖此完整调用点。不能删去原生buffer、注册为持久或降低实际rank/冷图核验来通过。

后续只修当前不可缺少路径：

- runtime/bundle/retrain/single91.py：区分checkpoint持久buffer摘要与原生模型完整normalization描述；原ViT checkpoint仍0件持久buffer，RoPE仍由实际rank描述及冷构图固定SHA校验。其他模型沿原摘要。
- runtime/r92_initial_gate.py：生产初态门使用同一持久buffer合同，保留所有rank/model/optimizer/FP32/实际normalization校验。
- runtime/bundle/retrain/ema.py：源码追踪另发现当前 noBN 分支将任何 live buffer 拒绝，原CPU只构造EMA未执行真实 update；仅为原生ViT的固定nonpersistent RoPE校验在线/EMA描述并放行既有参数更新，其他模型分支不变。须实际验证warm-copy、blend与下一步state往返，不能只证明构造。
- tests/test_deit3_model.py 或现有对应必要测试：补实际生产初态与freeze/冷加载受影响路径的CPU回归，包含RoPE被篡改会失败；不重跑P8/旧全套，不加GPU预演。

先在现有5de准备池root0–11、nice19、隐藏GPU取得真实窄验，随后只为独立新fresh身份登记技术复验。原R184、R185身份不复用。尚未实现或验证成功，尚无新技术复验合同、formal/clock/GPU；具体后续顺序在R186运行期间落实。
