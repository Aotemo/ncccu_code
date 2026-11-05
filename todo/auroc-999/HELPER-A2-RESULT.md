# A2 CPU 准备状态（helper/cpu-prep-20261105）

独立 64 例 fit 盲评已完成：R120 Grad-CAM 按固定规则含余量覆盖 **2/64**，低于预登记的 ≥63/64 门；旧双线索对照为 **51/64**。固定 A2 方法关闭并归档，不调整规则或重新抽样。粗框属于非专家技术参考，不能作医学判断。

本人受 guard 约束的 A/B 12 核 probe 分别通过，回执 `e42f51609c891b564db0be8304e6d6fd904b6fd9c2974ca87898a3d5024fd4c5` / `92eff8568aaef562b1cf24f4d3dfce07768436ad1a83a1b956de8a8e76ef79cc`，均仅显示 CPU、GPU 未初始化。A 源码传输超时后的远端 tar 曾处于 D 状态，原次与目标目录保留、最终状态 unknown；B 已复制已验证的基础树并叠加助手代码用于合成检查。

必要文件范围：新增 `tools/diagnostics/fit_disc_attention.py` 与 `tests/test_fit_disc_attention.py`；为复用既有盲包与对照，窄改 `tools/diagnostics/fit_disc_consensus.py`，增加上批 64 例排除、A2 seed 和准确的居中图像矩形记录，保留原入口默认行为。已依次封存无预测包和粗框，再分别执行原双线索对照及 R120 显著图。

B 准备池在经验证的 `00eb266` 树复制件上叠加助手代码后，旧测试快照 **6/6** 合成检查通过，guard 回执 `590ad0b77a6d14360a4dff208e69237c42ff3021574074ad041392872e2728ec`。实际 `fit_disc_attention.py` SHA `ea9f5a363f0a3e696198f96a52a27ed47047b2be729377a598b418647060efaa`，`fit_disc_consensus.py` SHA `50d95a5bc6e141fdb6b409a23e5dd8cf4350c0da90c916be395e8d826c695da3`，当时测试文件 SHA `5cee24e024828c41cb747ad7ebce4ca65de812606b53656de4bef6a993408ecd`。该批未覆盖后来新增的盲包篡改检查；当时真实研究尚未执行。

R265/B 首成功后，在相同基础树的独立新版测试快照上，**7/7** 合成检查通过，含盲包成员、映射及逐张图像身份绑定与篡改拒绝；guard 回执 `470be1f1d0978e0de17ad10356b8ecae630592baf2377b628eea90220b7acf44`，新版测试文件 SHA `568c81abef30d6d99425a4226d7ef75f3715c00cebc0c406283f6301f097f722`，上述两份实际源码 SHA 不变。当时未读取真实图像或加载 R120。

随后按 08ac 指定的 **af1f942** B 已验证三折源码树作为只读基线，复制到助手独立 Pod-local 目录，仅叠加 A2 两源码和测试，保留基线 F04 入口；集成窄测 **7/7**、guard exit 0，回执 `60acd02286cd4680953f63520b97d931e1498726f73406d6bde6eb4f655340fb`。两份 A2 源码实际 SHA 仍为上述 `ea9f5a36…` / `50d95a5b…`；基线 F04 入口 SHA `d1e077931a0ed967dd4f2a6547ed3d9cd8470b2816926016f025265f7ffb1203`，测试 SHA `298473d4a767d83608700dd8c30832c71c76babc7199888d447841168f3e1221`。当时未读取真实图像、未加载模型。

08ac 后续只读核查报告：同一登录节点已未见该 namespace 的 `kubectl` 客户端；Pod 内原 tar 进程尚未复核，因此该次 A 传输的 Pod 端终态仍是 unknown，原目标保留，不据客户端消失断言完成。

## main 盲封及对照

08ac 已将 A2 四文件合并、push/API `f49b7bf04011e12c0120299951786616dd66efba`，B 全树证明 `da372f637af9e6018b69795d15791331f0eb64c08695e4b7f2626f62e596cc66`。在该准确树上 A2 与原 consensus 合成窄测 **11/11**、guard `bb992add4c8e4af2ebe44e8105f44079517632a181bab0166012269c7c30505b`。

新独立 fit 盲名单 SHA `79846deb3f0d24607888d98e77c7a9367084d4e6d0f4b9d52ed8c49c6f7e263e`，无预测包 SHA `a418bf0f0d670370f66c598c62d3ad3b80c08d898ba2add28ea3372aac417df5`，64/64 图像可用；packet-ready `c1172328cbf63f8456236410e1a8de78a182fa84c89d4db20130880328e274d8`、guard `7a3e9a6d6099c2834cbaf26a11fa15f6fba6d74d5550ff424dde6b2d6a751147`。显示副本保持联系图几何，其原图 SHA 在 Pod 核对、显示清单 `44a141fda55ce3a53c7f4c46df1ea5143502777e7a356e252c544204bf5b408f`、guard `b14f6bec8e43939278c524e1e11b987c9cb15afa75a5d48a80912c0175c954d9`。全部 64 项非专家技术粗框先于任何预测封存，seal `9145eb07e21dc2e1b99eb5c67e10201cef3466e575edad59a1b08b857add468d`、guard `88387a0de2ec4de907908fdb3e13ddfedec1b1f31ceeb6e7d148c6b95dade64d`，其中 2 项 unknown；按事前规则最多 62/64，固定 ≥63/64 门已不可能通过。

封存后，原双线索对照同批完整包含 **54/64**、含规定余量 **51/64**、算法 unknown/回退 9、人工 unknown 2、可比较个例的中心误差中位 `10.8729 px`；summary `ce3c63e76a47663d7b4ec35c2673a49f1c66b9f88db3ac2530e355db65e57359`、guard `cf2750a54f0cbf30a5678731febee6e77c36768300f6b79c18176e47d5027050`。

R120 Grad-CAM 单次有界 CPU guard 调用完成，完整包含 **4/64**、含规定余量 **2/64**、算法 unknown/回退 **57/64**、人工 unknown 2；可比较个例的中心误差中位 `51.0840 px`，两种方法的中位数各自只包含可比较个例。64 次前向，隐藏设备、GPU 未初始化，未读 dev/官方；guard exit 0、回执 `1cf08ce273ed52f32bf358f4e4c6875a9300402fed92032c1923ef323176d103`，summary `ba31b44e660859ec3feed72fff283bb6372b0545fe2bfa936c9bbe800ff20e16`，样本级结果只留 Pod 独立产物目录。预登记门未通过；固定方法关闭，不起草 A2 训练合同。精简记录见 [归档](../../records/analysis/20261105-a2/README.md)。
