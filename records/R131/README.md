# R131：固定pointwise-r8完整结果

2026-11-01 05:56:58.071936北京时间完整终态：completed/early_stopping，11200success=attempt/0skip/eval14/bad4，唯一online c10@8000内部AUROC **.9809751099048635**、AP `.7859642705348402`、error_twice31600。未胜R107 `.9884405731067525`或singlebestR120 `.9894773092062407`，两门false、关闭精确r8/no seed27；不扫rank/alpha/位置/WD/LR/seed，不推所有LoRA普遍无效。AP或partial不救主判，原1503中断不拼曲线。

[R131-RESULT](R131-RESULT.json)、[COMPLETION](COMPLETION-R131.json)及[终态CPU回执](TERMINAL-CPU-RECEIPT.json)来自唯一已核completion；SHA分别由Pod归档回执固定，completion为`41d41e92c694d2d300c2283baa87affcd4f259cf34e453f3100fc6cab5f7b2e1`、CPU为`48b144c0d7a82a531a19ae1b9621cf4428090cef669613740e3b0a86f38768a4`。TRAIN/ROUND/done complete、train-component exit0/signals[]/errornull。完整selected不足`.996`，official正常not_run_by_policy/threshold_not_met/evaluatedfalse/metricsnull，未新增官方前向。

同次actualsafe=true/terminal_idle：owner342435/start350805912现Z仅Z非reaped，8rank342447–54 absent，锁present/readable/holders[]、device_users[]、scanDenied0，8expectedPCI全见且busy均0。未为归档再查旧safe，未hash/load旧大blob；freeze报告的selected tensor SHA `e570a4fddc33c2d16b76379e6778962130d22380425f6ee147894856e0d9ec09`只是原件广告，不冒重载实证。全部R130/R1/R131及R120/KEEP依赖保留。根已采纳独立[R132合同](../../todo/auroc-999/R132-DECISION.md)，以下启动/partial原文为历史阶段。

运行Git `7aa3232b7a2e47dedf68839e19deb29ce4ec6751`，parent `6f31002ab63d43031a3524e8d10fe648a934d411`，tree `0a8d04b9e951f54379ede80c01474a7e69baba10`；SCIENCE SHA `45995c2fe6783199c8df561accd5da623c6726c2276eef491c2d429ba0b7a321`。本轮为order42，直接R107/seed29/public fresh、无R1层倍率；原R1301503和R130R1完整11200结果分别保留，不加载其checkpoint或initial，不拼曲线。

唯一新增身份窄验1/1通过；本轮normalformal一次21/21，[准备回执](PREPARATION-RESULT.json)对应692源码Git blob/mode exact，deployment SHA `3bf7d2c70c05363df838907d0ef2b5064ee1bd30bd3cb8783136f0d07c9eee0c`、CPU formal SHA `6140d7d8a233d67142cb6d375e3ba25aed3d2f9413bc24af658ff4bffbb60ba1`。公开fresh新whole9a0665/共同345为50c、native public1000 strict、489总/342冻/147可训/4组/空Adam；旧两toy和原算法fixture未重跑，无独立GPU预演。

新R131根的13项启动前marker均[实际缺失](PRELAUNCH-ABSENCE.txt)，随后唯一activation0/正确live launch0；preflight SHA `22c64c5edf3222cd66cdb0571c7d03368896e6b226c7f817f240e2f57796d9d3`、activation audit SHA `3e50ad1fc115cd089a7ea5fb10c541d6ad440e491062b8a4ce723c23e1f6bc20`。实际copied site SHA `f397887e2d345a8b6de534a95b6142d34a02a342e3eafa34f7b7007a6fc0ed25`在同次A12 guard核出，GPU段只消费该pin，未在本机/登录节点计算。新9h软件grant SHA `be7806473183b520d9e5a4bf51d3f4de0427899a10f2ce7785cf917034b94b50`，实际effective120、训练96核24–119，与准备0–23池互斥；A只用0–11，根12–23保留未用，不新物理资源/续租或改旧grant/clock。

05:06首次小件见八rank初态记录、仍waiting_start_barrier，原`START-PENDING-R131-1.json`保留在本轮prep，不把spawn冒作成功。随后[真实启动观测](START-OBSERVATION-R131-2.json)于2026-11-01 05:07:33.619216北京时间证实running225success=attempt/0skip/eval0：owner342435/start350805912与8rank342447–342454全部argv/start active匹配，实际control112–119、rank24–111及8PCI/显存与本轮site一致。

真实ACTUAL_FRESH_INITIAL completed SHA `e8bd4df3e7278f7959f1204a0dc1ff900f94dff6dcc0d9df441fe22d409a46c9`，8rank初态Adam0、489总/342冻/147训/4组。自然step1的8rank WORLDfinite、147个实际Adam state/observed step[1]，冻基座grad/Adam均absent且实际SHA与初态同为 `3af1b12cda45c131053933eefdeddac6462f0d970760ba0bfac06b497ee0a40e`。此为真实首步证据；初态Adam0不代表225步后仍空。START SHA `6c031d368715732137b8e779650cb67b1cc308a965d654515235914a023221f2`、[观察CPU原件](START-CPU-RECEIPT.json) SHA `55cd903188b6577387c526e1de46457b25ae81c427601fc392cfae78c093d67f`；observer未load blob/模型或forward。

目前尚无本轮完整AUROC/selector/official或终态，不能判两门或overall`.999`。活动冻结包保持本runGit，后续文档归档commit分列。根唯一heartbeat沿当前[R131合同](../../todo/auroc-999/R131-DECISION.md)继续观察；existing small-check现用本轮SITE_BINDING+本轮first-success作locator，R1completion仅历史，不当新轮状态来源。原始产物在`exp-v3/technical-prep/r131-20261101-7c17`及独立R131 round，全部旧保护资产保持。
