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

汤模型的上游为其实际纳入的全部父模型；五折、清洗、蒸馏及 768 模型尚未创建，其真实上游 SHA、训练数据和任务内教师血缘均**待核**，不能提前标为合规。F05 最终候选清单尚未封存，其他可能入围的历史模型也须逐一补审。N06 使用 FLAIR 眼底预训练，按用户规则排除，既不是候选也不是教师或任何训练信号。

## 历史 fit-only 适配核查

对已完成的 R115、R224、R226、R228、R230、R232、R234、R236R2、R238、R240R1 和 R242，逐轮核对真实阶段回执、固定 46,256 行比赛 fit / 5,140 行 dev 清单、合规公开 Small 初始化、原站点数据根及冻结 `ssl_stage.py`/`data.py` 文件 SHA：11/11 回执记载仅使用 fit 索引、适配阶段 dev 评估为 0，均无发现外部医学数据输入；明细为[小型审计回执](HISTORIC-FIT-SSL-AUDIT.json)，其 SHA-256 为 `5bd9a3ae3e84b35131cfe8e8364aa0c001793937310803d5d129f66b5fcdd9e7`。其中有的后期表征适配使用比赛 fit 标签，不能一概称无监督。

这一结论由已保存回执、源码和绑定路径支持；未获得完整操作系统网络和文件访问追踪，不能声称对环境中所有可能访问作了绝对证明。未运行或缺乏完整证据的尝试不因此自动获得通过。
