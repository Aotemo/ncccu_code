# R216/R217 CPU 准备

[CPU-CHECKS.json](CPU-CHECKS.json) SHA `02e4af5288d23781e9149918d498bc4f93c86098b3ef5c8754bd6b9167944f7f`：5个受影响目标、4源码编译通过。原件位于`exp-v3/technical-prep/long-ema-20261103-32e3`；最终guard4soj0gxm / `063ae9af59c81381646420fe8213ae2eb3cbfeea646230c7bf2e443b341c59e4` exit0。d837/root0–11/nice19/隐藏GPU，与R215训练24–119互斥；当前冻结c9bbde5不改。

实际生产EMA在成功更新1600前逐步复制，1601/1602与独立标量递推一致；保存EMA状态后1603完全一致，旧R8规则与构造RNG保持。此fixture是标量模型及状态张量往返，不冒称新生产CheckpointStore或八rank恢复。候选绑定核R216同SigLIP2独立seed27、R217与R107的job/source/structure/split/schedule逐项相同，错误EMA/seed/LR及未登记策略被拒绝。存储目标核精确R21624GiB，原R21524、R213/R214/default12和free16GiB保持。

两actual public fresh各0图像forward、GPU未初始化、5空原生AdamW组。R216 whole`1f4076be05cc2ed21091ddd7397b559229b4949ef502b1e99ef59249105c1610`，165全训；R217 whole`50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，345全训；均无buffer。实际metadata消费的EMA decay分别.999与.9999、warmcopy均1600。公开源162/344张量实载；原模型/P8/冷加载窄验按相同源码身份复用，未重复前向。

只增加experiment91精确候选和EMA metadata绑定；EMA数学、模型、data、loss/WORLD、optimizer、保存/评估与GPU入口保持原源码。R216仅完整R215严格胜R107与R120且actualsafe后eligible；R217直接原R107，只固定EMA衰减.9999，不扫系数/窗口或叠加其他改法。EMA研究仅一般动机，未证明临床机制、fit表现或AUROC收益。实际RED/绿色回执均保留；尚无新normal/clock/GPU。
