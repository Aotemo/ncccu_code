# R98配置与身份窄验

R98直接对照R95，仅job.head_lr从0.001改为0.002；backbone1e-4、独立age1e-3和24k/800/p4/1600等合同不变。现有candidate绑定9项在实时UID/Ready核对后的Pod验证池A mask0..11执行，nice19、隐藏CUDA/HIP/ROCR；9/9通过、exit0，实际测试2.557秒，配置检查wall2.973秒，前后源码SHA一致。没有新模型load、前向、GPU预演或评估。

独立prep `/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/r98-head2e3-20261029-a`。`CONFIGURATION-CHECKS.json` Pod SHA `4240a26884fca0ef8e1c4dbab1570aa2507e3792a818cd7accc54f2058bbe533`。配置SHA `a8d7d6782fcc07095233ad7c742e77be0c3d65f4c4b4447325c2967497d0bd6a`，experiment91源码 `3e9ff8a88abe186494586c9e1d99ba5679fb3045682d4bb236274fe38d8ac4cd`，测试源码 `4cc35937658ddb7b9cb69b647f2845606df0daa1f31bacfd9ef529f9a769eb3d`。

同一预算任务随后只读R97既有终态并核安全交接，未重评或重复官方。CPU原件SHA `1c1df2765771f1c6e3b39defa1575e20724c60f3e472f27dd931d5d3f9fefeb5`；Pod UID `51fe3daf-ba0d-46e8-919c-99caec599c99`，allocation4经入口实核TTL及资源。JSON均普通字节导回，本机未解析或重哈希。新Git部署和真实fresh训练初态仍由随后运行原生链证明，不以这9项冒称GPU已启动。
