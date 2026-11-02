# CAFormer-S36 固定公开来源

18:49 后续二元适配已取得[3个受影响目标和9个源码编译的真实CPU证据](MODEL-CPU-CHECKS.json)。原native factory/保留MLP对象/RNG/手工及原生前向/单drop和age目标在原69n0jyve任务通过；同任务P8停在优化器头命名定位，整任务exit1原件保留。仅为CAFormer给既有_optimizer增加fc2定位后，单独复测P8目标通过：396全训/5空Adam、公开fc1走backbone而末层走head、全梯度有限、EMA图及真实CheckpointStore→inference冷图一致；未重跑已通过的首目标。原其他模型_optimizer路径AST与ab43基线相同，ViT三个实际构造/描述函数正文相同，复用原P8证据，不称engine整文件未变。

[R185实际public fresh](R185-PUBLIC-FRESH-INITIAL.json) whole8c62435a/SCI039715e4，165全训与1个静态RoPE；[R186](R186-PUBLIC-FRESH-INITIAL.json) wholed2369b38/SCI37befe80，396全训/37,250,153元素/无buffer，均5空Adam、0图像forward/0GPU。R186完整395公开加载后只换最终fc2，保留公开MLP；构造drop_rate0，仅内层head_drop=.2，外层drop0，age一次。CPU saved storage11,627,686,508 B不是DCU峰值或上界；默认12GiB及满24k历史时间情景只是规划参考，不靠早停或新GPU预演兜底。

最终归档guardg56kpfse/50aaa7bf exit0，已核guard/output SHA、退出和预算。当前仅代码/配置/CPU准备完成，Git冻结及技术打包待执行，无新formal/clock/GPU；下方“仅来源”均为此前阶段原始记录。现役R182与R183/R184的ab43冻结不改。

固定 `timm/caformer_s36.sail_in1k_384@695903aebe60ba2a96868cedd9129ea256a1f96e`，ImageNet-1K、public/non-gated/Apache2，157,357,817 B/LFS `424a3a02…ed592`。完整395/full1000原件[实际 strict](PUBLIC-STRICT.json)逐张量相等/FP32有限，tensor061e1227、无转换/图像前向/GPU。source384是来源属性；尚无二元fresh或512训练证明。

复用现deit3_source.prepare新增caformer显式profile，[新来源实际调用](SOURCE-ACCEPT.json)和[原三参数默认DeiT3路径](DEFAULT-COMPAT.json)都通过，后者完整176/full1000/provenance语义与原件相等，API/CLI默认保持；未称新handoff字节相同。CAFormer provenance1c870b75、handoff97e59fd5、上游64c534ab完整Apache许可c71d239d均实际核到；[库身份](LIBRARY-PIN.json)仅所列5个timm1.0.27文件，不代表整个依赖包。

原生完整分类头包含avg/LN512、公开512→2048 MLP/SquaredReLU/LN2048及2048→1000，不能当成简单512线性头。source检查保留完整395张量，未替换二元头或测fresh身份。源原件与模型只在exp-v3，不进Git。

[原始CPU证据](SOURCE-CHECKS.json)保留一次审计脚本语法错误、修正后的随机factory、完整原件strict、库pin、新profile及默认兼容的真实SHA/output/exit；归档guardino2brl3/04955537 exit0，任务树退出且CPU预算符合。准备root0–11/12核与训练24–119/96核互斥，peer12–23保留，无新GPU或clock。当前R182/2a128733及R183/R184的ab43f542冻结均不改；后续二元适配见[R185/R186计划](../../../todo/auroc-999/R185-R186-PREPARATION.md)。
