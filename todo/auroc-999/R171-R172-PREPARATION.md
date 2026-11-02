# R171/R172 后续假设：共享骨干的固定重叠局部视野

当前R170冻结不改；本页是R170之后的固定准备，不影响当前接续。R171/order83只在完整R170内部所选严格超过R107 `.9884405731067525` 和R120 `.9894773092062407` 时复核同CLIP Base/原输入合同的seed27，否则不formal/GPU。独立R172/order84回原R107 Small/fe6b/seed29/原512/logistic/G64P8/AdamW/age/LR/EMA/24k-800-p4，不叠CLIP、beta095或旧ROI。

R172只改一个图像级读出。对同一最终增强/归一化后的512张量，原共享network处理全图一次；四个固定288窗口按TL/TR/BL/BR、起点(0,0)/(224,0)/(0,224)/(224,224)堆成B×4后处理一次。四窗有64像素重叠，不重采样；原网络/原Dropout/原Linear共享。图像logit = `.5*全图logit + .5*max(四个局部logit)`，然后原年龄残差一次。`torch.max(dim=1)`按固定局部顺序选第一个同值最大项，梯度只回该项，依据[实际所用PyTorch2.10文档](https://docs.pytorch.org/docs/2.10/generated/torch.max.html)。不新增参数、buffer、模型、teacher或RNG初始化，不改loss/WORLD/448个图像级pair，不把局部窗复制为独立有标签样本。

这是相同像素下改变局部视野与聚合的可证伪假设，不增加原图信息或声称恢复native分辨率、定位真实视盘、发现临床根因。反光/边缘伪影也可能被max放大，盘状结构可能跨窗；全图项保留上下文，但不保证收益。此前R124内容注意力、R143/R144亮区裁剪、R146/R150组合及R152大图阴性均保留，不用其局部改善救总体。本固定合同阴性后不扫窗大小、位置、权重或max/mean公式。

单个外层模型调用内部两次共享network计算：B张512及4B张288，图像级批仍64/P8。原日志primary/view_inputs只表示外层输入张量数，不能当320局部视野的独立观测；新policy/实际CPU hook另记录内部调用。输入像素面积约为原2.265625倍，不能据此宣称实际FLOPs、DCU显存或吞吐等比。先用受影响CPU P8与存储/时间情景核预算，再在既有资源、时钟和GPU准入中证明自然八rank/首成功，不独立GPU预演或新资源。

最小文件范围：`config.py`可选版本化policy/None省略与互斥；`models.py`原BinaryModel.forward可选聚合及描述校验；`inference.py`保存job冷图校验；`experiment91.py`身份、policy与保存/选点绑定；新`tests/test_quadrant_mil.py`和现`test_candidate_binding.py`、两SCIENCE。data/cache、训练/启动/保存/评估/监控平台、optimizer、loss、gather和EMA实现不改。双调用不等于两个独立模型或旧checkpoint融合。

- [x] 在独立technical-prep先用实际helper/模型路径观察必要RED，再最小实现。独立值与x/w/b梯度oracle覆盖非同分及同分但零空间特征不同的窗口，确认实际tie梯度而非仅参考index。
- [x] 实际public fresh/None兼容/RNG/原参数对象与345张量、5空组、P8有限反传、EMA图及真实CheckpointStore→inference冷图验证；原loss/WORLD按相关源码完全相同复用。3目标/6编译、原CORE资源超限与修正后guard0原件见[实际CPU证据](../../records/preparation/quadrant-mil-20261102/README.md)。
- [x] 模型/局部计算预算、源码Git1859670/tree216951f同步回读、两技术包及独立后备R172正常CPU18/18/精确部署在R170训练期间完成；尚无新grant/clock/GPU，完整R170决定唯一分支，actualsafe后一次串行GPU接续。

窗口288、顺序、同分规则和.5均一次固定，不用dev样本名单/逐例分数/官方设值。完整内部AUROC是主判；完整selected≥.996才官方一次，raw .999未达。
