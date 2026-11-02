# 固定自然图像 DINOv3 ViT-S 来源准备

18:00 后续生产适配已经完成 CPU 窄验：[模型证据](MODEL-CPU-CHECKS.json)固定 4 个实际通过目标和 11 个唯一源码编译，复用相同源码的模型2/候选1/存储1，不重复旧 suite。原生 factory/RNG/独立 prefix 均值/age 一次、P8×512 全165有限梯度/5空组/EMA及真实 CheckpointStore→inference 冷图通过；非persistent FP32 RoPE 由同 native factory 重建。当前只 pin 所列四个 timm 文件，未宣称整个依赖包或新 DCU 峰值/吞吐。

[R183 fresh](R183-PUBLIC-FRESH-INITIAL.json)为同 Large DINO seed27、whole c51c4d8b/SCI1f116cd6/345全训；[R184 fresh](R184-PUBLIC-FRESH-INITIAL.json)为固定 native ViT-S/seed29/512、whole e347757c/SCIa231b7d0/165全训/1个静态buffer，均5空Adam/0图像前向/0GPU。R184保持原R107数据/采样/loss/优化/LR/EMA/age与24k-800-p4，仅架构及匹配自然图像初始化改变；R183须完整R182严格过两锚点且safe才有GPU资格。新增R183精确32GiB，R184默认12GiB，原free16GiB不改。

原缺构造/候选/存储的RED及一次 fresh 准备脚本语法错误均保留原 guard/output；修正后两实际 public fresh 通过。最终归档 guard sphvx9bc、SHA `e1bb84992b45a70bd9f969d84497dc291006399e491769cb241670801b6447ca` exit0，已核原件 SHA、退出、任务树和CPU预算。现仅受影响代码/配置/CPU可运行，不等于已正常formal或GPU；精确Git冻结/打包接续待执行。下文“仅来源/尚无factory”等均为此前阶段原始记录，当前R182冻结2a128733不改。

公共非gated `timm/vit_small_patch16_dinov3.lvd1689m@3bf4720a82ec2066db88137180ff1f83a675cef0`，86,407,250 B、LFS b7e7b7d1…8e7a2，完整DINOv3许可SHA25d122eb…4999e。原件在exp-v3，未进Git；原生完整162张量strict/exact/FP32有限通过，无转换、图像或forward。实际原生256来源/384维/1CLS+4register/avg/final LN eps1e-5、一个非persistent FP32 rope.periods[16]语义已核；来源属性不表示比赛改256。

只给既有deit3_source.prepare增显式vitdino3 profile及models库常量/字节检查函数；没有生产训练factory、新job白名单或forward适配。旧三参数API真实完整176/full1000 strict通过，provenance内容与原DeiT3相同；API/CLI默认仍deit3，未把回执字节或CLI实际执行说成相同。两真实来源检查替代重复字典断言，unit新增/执行0；源码已由实际导入和两路径执行解析，未另列py_compile计数。

来源原件provenance6193971f…aed8、handoffc899c713…7fb2。SOURCE-CHECKS.json保留两条真实成功和原API不接受新profile的RED、首次factory报告读取不存在属性的失败及修正后证据，均核guard/output SHA、exit、任务树退出、CPU mask和线程预算；归档guardnonkv6zc/38bdef44…2cf exit0。完整源/库身份只覆盖所列四文件，不声称pin整个timm transitive包。

当前仅来源准备；模型fresh、合成P8/冷图、完整候选、Git部署与GPU均未就绪。当前R180及后续R181/R182冻结源码不改；后续精确范围见[R183/R184准备](../../../todo/auroc-999/R183-R184-PREPARATION.md)。历史ViT/Small DINOv3阴性保留，无临床机制或AUROC收益结论；医学权重/N06不用，官方隔离，raw .999未达。
