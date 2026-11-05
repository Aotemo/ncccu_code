# 上游预训练数据说明草稿

本说明按用户提供的比赛规则编制，供主办方核验；当前只完成下表已存在模型的上游来源核对，并非对未来 F05 全体产物或历史 fit-only 适配的完成证明。F04 官方结果不用于本次审计或新方案选择。

**共同上游 A：** `timm/convnext_small.fb_in22k_ft_in1k`，revision `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b`。官方[模型卡](https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k)披露主预训练数据集为 ImageNet-22k，随后在 ImageNet-1k 上微调；[ConvNeXt 原作者仓库与论文](https://github.com/facebookresearch/ConvNeXt)提供方法和来源。仓库登记的固定版本原始权重 SHA-256 为 `c912f45cd738512978a900cc8aefade65bc44e507321f0144378a2d98aa5e914`，本任务转换后权重 SHA-256 为 `7aaea5ce6fe6db953874d9792d3f1711a5cc0193a10d740e023108043918c610`，转换 provenance SHA-256 为 `8ea3e957e2381a40e2f8fc5feae5df31d0d8b33faf3bdd3fa5749e64ae98368b`。2026-11-05 在 Pod 内实读以上三个文件并核对 SHA；小型审计回执 `2020f13a11e743b9ecc6ed6010f55440a11fceb6462c64ccabeca4d9f6aef34a`。

| 轮次 | 当前作用 | 上游与主数据集 | 当前证据及限制 |
| --- | --- | --- | --- |
| R120 | F04；R120 汤候选 | A；ImageNet-22k → 1k | SCIENCE SHA 及当前公开资产核实；F04 selected 已严格加载 |
| R134R1 | F04；R134R1 汤候选 | A；ImageNet-22k → 1k | SCIENCE SHA 及当前公开资产核实；F04 selected 已严格加载 |
| R250 | F04；R201 汤候选 | A；ImageNet-22k → 1k | SCIENCE SHA 及当前公开资产核实；F04 selected 已严格加载 |
| R249 | F04 | A；ImageNet-22k → 1k | SCIENCE SHA 及当前公开资产核实；F04 selected 已严格加载 |
| R135 | F04 | A；ImageNet-22k → 1k | SCIENCE SHA 及当前公开资产核实；F04 selected 已严格加载；任务内 RGB+XY 输入改动不是外部预训练 |
| R254 | R120 汤候选 | A；ImageNet-22k → 1k | SCIENCE SHA 核实；selected 文件存在，严格加载待核 |
| R256 | R120 汤候选 | A；ImageNet-22k → 1k | SCIENCE SHA 核实；selected 文件存在，严格加载待核；不改变其 F04 排除记录 |
| R258 | R120 汤候选 | A；ImageNet-22k → 1k | SCIENCE SHA 核实；selected 文件存在，严格加载待核 |
| R262 | R120 汤候选 | A；ImageNet-22k → 1k | SCIENCE SHA 核实；selected 文件存在，严格加载待核 |
| R253 | R134R1 汤候选 | A；ImageNet-22k → 1k | SCIENCE SHA 核实；selected 文件存在，严格加载待核 |
| R257 | R134R1 汤候选 | A；ImageNet-22k → 1k | SCIENCE SHA 核实；selected 文件存在，严格加载待核 |
| R260 | R134R1 汤候选 | A；ImageNet-22k → 1k | SCIENCE SHA 核实；selected 文件存在，严格加载待核 |
| R261 | R134R1 汤候选 | A；ImageNet-22k → 1k | SCIENCE SHA 核实；selected 文件存在，严格加载待核 |
| R252 | R201 汤候选 | A；ImageNet-22k → 1k | SCIENCE SHA 核实；selected 文件存在，严格加载待核 |
| R259 | R201 汤候选 | A；ImageNet-22k → 1k | SCIENCE SHA 核实；selected 文件存在，严格加载待核 |
| R263R1 | F05 候选 | 版本化配置声明 A；ImageNet-22k → 1k | fresh 接续沿用 R263 科学配置；实际 selected 文件 SHA、部署加载与最终候选身份待逐项核对 |
| R264 | F05 候选 | 版本化配置声明 A；ImageNet-22k → 1k | 完整训练记录存在；实际 selected 文件 SHA、部署加载与最终候选身份待逐项核对 |
| R270 | 768 方向候选 | 版本化配置声明 A；ImageNet-22k → 1k | 768 配方与 fresh 初态已有准备证据；唯一 selected、实际权重 SHA、部署加载和最终身份待终态后核对 |
| R271 | 新 seed 候选 | 版本化配置声明 A；ImageNet-22k → 1k | B 公开源绑定回执及完整训练记录存在；实际 selected 文件 SHA、部署加载与最终候选身份待逐项核对 |

2026-11-05 只读核对上游[模型卡](https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k)：其披露 ImageNet-22k 预训练及 ImageNet-1k 微调，与 A 一致。上述新增行中的“配置声明”不是对每个最终部署权重的独立字节或加载证明；须以该轮冻结配置、实际 selected、任务数据及严格加载回执完成核验。

三份已记录的均匀汤均继承上表各父模型的 A 来源；下表只引用已有产物回执，**不以父模型来源登记代替最终汤权重和其 dev 评估的核验**。

| 汤及实际回执 | 记录的全部父模型 | fp32 平均状态文件 SHA-256 | 待补 |
| --- | --- | --- | --- |
| [R120 均匀汤](R120-UNIFORM-SOUP.json) | R120、R254、R256、R258、R262；其中 R256 为 online selected，其余为 EMA | `1943b980ee5a0ade77bce57595642a3514251fbc473bf7f088ec0d587ae6a472` | 实际部署 fp16 文件身份、全部父源与加载证据、完整 dev 及最终候选门 |
| [R134R1 均匀汤](R134R1-UNIFORM-SOUP.json) | R134R1、R253、R257、R260、R261，均为 EMA selected | `c2263433672864078384c40d74d2dfbf2d787a8cd36e536bdad2fe3b642135eb` | 实际部署 fp16 文件身份、全部父源与加载证据、完整 dev 及最终候选门 |
| [R201 均匀汤](R201-UNIFORM-SOUP.json) | R250、R252、R259，均为 EMA selected | `cb949231fb12fb76ec591347f9423c9f13d33d0717c74391cc522d98f457f2cc` | 实际部署 fp16 文件身份、全部父源与加载证据、完整 dev 及最终候选门 |

已有独立合成加载回执所记录的 fp16 模型文件 SHA 分别为 [R120](R120-UNIFORM-STANDALONE.json) `28f3fc555c4d81c02ac2e34f4c06bbcfbeede7d18af96d06068fa1d40d5a24e7`、[R134R1](R134R1-UNIFORM-STANDALONE.json) `3a9984fff418624f4817fbecdf6251f239f61a1c706fb869572c0c6ba37a6ed3`、[R201](R201-UNIFORM-STANDALONE.json) `fef3d8d4930f0dcb9c6e7886019490de0e72ea67eefc867584ddd36d26023c8c`；最终部署时仍须重新绑定实际文件和元数据，不能将历史合成回执写成完整 dev 证据。

三折 R265–R267、未来清洗与蒸馏模型的配置拟沿 A 来源；实际最终 selected/导出 SHA、训练数据和来源须逐产物核实。蒸馏另须绑定仅由本任务合规三折模型、各自折外 fit 分数及折训练子集标定组成的教师血缘；实际教师尚未产生，不预标通过。原五折已由三折取代，不能作为未来教师。F05 最终候选清单尚未封存，其他可能入围的历史模型也须逐一补审。N06 使用 FLAIR 眼底预训练，按用户规则排除，既不是候选也不是教师或任何训练信号。

## 历史 fit-only 适配核查

对已完成的 R115、R224、R226、R228、R230、R232、R234、R236R2、R238、R240R1 和 R242，逐轮核对真实阶段回执、固定 46,256 行比赛 fit / 5,140 行 dev 清单、合规公开 Small 初始化、原站点数据根及冻结 `ssl_stage.py`/`data.py` 文件 SHA：11/11 回执记载仅使用 fit 索引、适配阶段 dev 评估为 0，均无发现外部医学数据输入；明细为[小型审计回执](HISTORIC-FIT-SSL-AUDIT.json)，其 SHA-256 为 `5bd9a3ae3e84b35131cfe8e8364aa0c001793937310803d5d129f66b5fcdd9e7`。其中有的后期表征适配使用比赛 fit 标签，不能一概称无监督。

这一结论由已保存回执、源码和绑定路径支持；未获得完整操作系统网络和文件访问追踪，不能声称对环境中所有可能访问作了绝对证明。未运行或缺乏完整证据的尝试不因此自动获得通过。
