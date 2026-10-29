# R102 Small22K：新生产初态与必要配置验收

对照预登记R95/BCE/seed20261027，成对更换为Small架构与固定fe6b358a公开checkpoint。其余R95科学字段保留；P4不执行，显式SD0.1，不使用TorchVision Small默认0.4。新来源与full1000 strict见[Small回执](../../pretraining/20261029-small22k/SMALL-HANDOFF.json)，本阶段不重跑该来源链或Tiny检查。

本次CPU5 A0..11、nice19、隐藏CUDA/HIP/ROCR，在UID51fe3daf实时核验的Pod执行。相同生产configure→seed→factory首次生成Small fresh binary+age：公开344张量strict/exact/FP32后新建head、age为零、Adam状态为空、EMA与model初态相同；参数元素分组backbone49454688/head769/age2。

实际新整体tensor SHA为`91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67`，不复制Tiny控制。该值已写入实际Pod生成的SCIENCE，SHA为`484c74d9fc929a861f00d8009517b6ba419e2250754aeecce21de95be2878875`，普通字节传回配置；没有本机解析或重哈希。

既有受影响candidate配置10项真实GREEN（5.868秒）；相应源码AST通过且测试期间SHA不变。原始[CONFIGURATION-CHECKS.json](CONFIGURATION-CHECKS.json) SHA为`355893692fa708573738282efeddf33623a6f38f6f631a8171266a0aa42c26b8`，[INITIAL-CONTROL.json](INITIAL-CONTROL.json) SHA为`44e23c726a6b803eaabf7d39f73b54f7b8a800e385290fffeaeb9427bd9ccd36`。守护原件[CPU-RECEIPT.json](CPU-RECEIPT.json) exit0，SHA为`473212a52a072533022b13737e4f128396dc495c7a373753c02a0e2e08a6dafb`。

本阶段没有forward、优化器更新、数据/官方语义读取、GPU初始化或训练clock，也未生成formal CPU-PREP；该新控制在实际Git同步后仍由原生正式CPU包与真实8rank初态核对。工程预算train21600/total28800与GPUgrant32400是未执行新绑定的配置，不是吞吐或云资源租期保证。
