# R105：既有seed27的尺度配对配置

直接对照R102，仅新增`training_scale=[0.85,1.0]`；与R104仅训练seed及其派生初态控制不同。Small/seed20261027工厂控制91507a65来自既有公开fresh生产工厂，由随后实际新Git原生正式包自然匹配；本阶段不构造模型或独立control-only，也不加载旧checkpoint。

CPU7 A份额完成2项现有必要配置检查，技术包实际注入seed27、尺度范围、旧BCE及原job_contract；源码在检查期间未变，无模型、训练更新、GPU初始化或正式CPU-PREP。未重跑未变的像素、来源、native或旧套件。

[CONFIGURATION-CHECKS.json](CONFIGURATION-CHECKS.json) Pod SHA `d982c458a52d910d81c57d10bc0e7f49af817572c0350399374382fd0112e13a`；[CPU-RECEIPT.json](CPU-RECEIPT.json) SHA `005a6d4fa7a051cdc7d08a0cddd89021a838c7ca3abad2d19991d42c0a2a1ade`。实际mask0..11、nice19、隐藏GPU，SCIENCE SHA `06941934ad34ed484be1c62d097bbf7087c77ccc1ba1653210aa62759248f0b1`。

原件根`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/r105-small22k-scale085-seed27-20261029-a`。旧CPU6到期后仅新建CPU7，不改旧allocation或R104 GPUgrant。小原件仅普通字节传回；未本机解析、重哈希或测试。实际新正式包/部署/唯一启动结果由其运行原件另证，不以本页登记代替。
