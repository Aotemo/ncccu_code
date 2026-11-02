# 固定自然图像 DINOv3 ViT-S 来源准备

公共非gated `timm/vit_small_patch16_dinov3.lvd1689m@3bf4720a82ec2066db88137180ff1f83a675cef0`，86,407,250 B、LFS b7e7b7d1…8e7a2，完整DINOv3许可SHA25d122eb…4999e。原件在exp-v3，未进Git；原生完整162张量strict/exact/FP32有限通过，无转换、图像或forward。实际原生256来源/384维/1CLS+4register/avg/final LN eps1e-5、一个非persistent FP32 rope.periods[16]语义已核；来源属性不表示比赛改256。

只给既有deit3_source.prepare增显式vitdino3 profile及models库常量/字节检查函数；没有生产训练factory、新job白名单或forward适配。旧三参数API真实完整176/full1000 strict通过，provenance内容与原DeiT3相同；API/CLI默认仍deit3，未把回执字节或CLI实际执行说成相同。两真实来源检查替代重复字典断言，unit新增/执行0；源码已由实际导入和两路径执行解析，未另列py_compile计数。

来源原件provenance6193971f…aed8、handoffc899c713…7fb2。SOURCE-CHECKS.json保留两条真实成功和原API不接受新profile的RED、首次factory报告读取不存在属性的失败及修正后证据，均核guard/output SHA、exit、任务树退出、CPU mask和线程预算；归档guardnonkv6zc/38bdef44…2cf exit0。完整源/库身份只覆盖所列四文件，不声称pin整个timm transitive包。

当前仅来源准备；模型fresh、合成P8/冷图、完整候选、Git部署与GPU均未就绪。当前R180及后续R181/R182冻结源码不改；后续精确范围见[R183/R184准备](../../../todo/auroc-999/R183-R184-PREPARATION.md)。历史ViT/Small DINOv3阴性保留，无临床机制或AUROC收益结论；医学权重/N06不用，官方隔离，raw .999未达。
