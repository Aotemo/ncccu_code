# CAFormer-S36 固定公开来源

固定 `timm/caformer_s36.sail_in1k_384@695903aebe60ba2a96868cedd9129ea256a1f96e`，ImageNet-1K、public/non-gated/Apache2，157,357,817 B/LFS `424a3a02…ed592`。完整395/full1000原件[实际 strict](PUBLIC-STRICT.json)逐张量相等/FP32有限，tensor061e1227、无转换/图像前向/GPU。source384是来源属性；尚无二元fresh或512训练证明。

复用现deit3_source.prepare新增caformer显式profile，[新来源实际调用](SOURCE-ACCEPT.json)和[原三参数默认DeiT3路径](DEFAULT-COMPAT.json)都通过，后者完整176/full1000/provenance语义与原件相等，API/CLI默认保持；未称新handoff字节相同。CAFormer provenance1c870b75、handoff97e59fd5、上游64c534ab完整Apache许可c71d239d均实际核到；[库身份](LIBRARY-PIN.json)仅所列5个timm1.0.27文件，不代表整个依赖包。

原生完整分类头包含avg/LN512、公开512→2048 MLP/SquaredReLU/LN2048及2048→1000，不能当成简单512线性头。source检查保留完整395张量，未替换二元头或测fresh身份。源原件与模型只在exp-v3，不进Git。

[原始CPU证据](SOURCE-CHECKS.json)保留一次审计脚本语法错误、修正后的随机factory、完整原件strict、库pin、新profile及默认兼容的真实SHA/output/exit；归档guardino2brl3/04955537 exit0，任务树退出且CPU预算符合。准备root0–11/12核与训练24–119/96核互斥，peer12–23保留，无新GPU或clock。当前R182/2a128733及R183/R184的ab43f542冻结均不改；后续二元适配见[R185/R186计划](../../../todo/auroc-999/R185-R186-PREPARATION.md)。
