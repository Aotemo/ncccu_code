# R103：预定seed20261029的Small新鲜初态

仅相对R102将训练seed20261027改为预先固定的20261029；固定splitseed、清单、公开Small资产、BCE、LR、dropout/SD、24000/800/p4及工程预算全部沿用。不是seed搜索或旧checkpoint续训。

现有生产CPU工厂生成整体tensor SHA `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，不同于R102；零age、空Adam、online与EMA初态一致，参数组与Small结构匹配。实际控制见[INITIAL-CONTROL.json](INITIAL-CONTROL.json)。本阶段无forward、训练更新、GPU初始化或正式clock；不是正式CPU-PREP或GPU通过证明。

10项受影响既有配置检查真实通过，源码在检查期间未变，A份额mask0..11/intra12/interop1、nice19、隐藏GPU；见[CONFIGURATION-CHECKS.json](CONFIGURATION-CHECKS.json)与[CPU-RECEIPT.json](CPU-RECEIPT.json)。原Pod检查SHA `b8bade54bd6651f83a5cf4ae07e53690294a7e26322e97826a159f8d5f9562f3`，guard SHA `61bcd82cc7fe28a39772e5a5e22f85c95e2563c9af65e2bc79092c3190d6bd8b`。受控SCIENCE SHA `530907ebc665536446446d63fbef39a3628ed7f3c9739daeb9ae79e1c5adaa05`。

Pod原件根为`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/r103-small22k-seed20261029-20261029-a`。只普通字节导回，未在本机解析、重哈希或测试。实际新Git身份下的原生正式CPU包、部署和唯一fresh启动由随后运行回执证明，不以本页登记代替。
