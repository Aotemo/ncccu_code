# R186：原生 CAFormer-S36 单模型

[真实首成功](R186-START.json)已证八rank、396全训/5空Adam、whole d2369b38与CPU一致；自然首25次成功=尝试、loss `.6897131204605103`有限、WORLD8/G64P8/448对，实际八PCI与96核训练mask一致。19:25观测175 success=attempt/0skip/eval0，无完整selector/官方结果，raw .999未达。

运行冻结Gitf9ea149d4ab528b49b5595b5e93e3418b29a95ee/tree86a38dd7、SCI37befe80a2e553a58a8aed2dc204d8e7687716e3f2d620b325f1bebe52e8f93f、SITEeda57a7c。[受影响3目标/9编译及两public fresh](../preparation/caformer-20261102/README.md)按相同源码复用；[正常CPU18/18与精确部署](R186-PREPARATION.json)、一次activation/direct launch均0，未独立GPU预演。默认12GiB与free16GiB保护不改。

原R107的seed29/512/ImageNet输入/原logistic/G64P8/采样增强/默认AdamW/LR/EMA/age/固定9:1/24k-800-p4；仅原生CAFormer-S36及匹配自然ImageNet1K公开695903ae初始化。保留native avg/LN512与公开512→2048 MLP、原激活/LN2048；仅最后fc2换为2048→1，内层Dropout.2/age各一次。来源384不表示比赛输入改变，不叠加ViT、ROI或MIL，不声称临床机制或AUROC收益。

R182完整阴性且safe，R183不eligible；R184初态工程失败且safe，R185不eligible。此[接续事件](R186-HANDOFF-GAP.json)自R184整轮结束至dispatch209.286秒、最晚设备271.495秒、首25记录328.330秒，最后一项超过300秒；自最后interrupted日志分别216.740/278.949/335.784秒。突发初态失败后完成本轮正常CPU/精确部署，再一次CPU激活与原生设备启动，未跳必要校验、改旧身份或测量连续DCU利用率。

唯一Git/GPU owner32e3；准备5de56a8f root0–11/12核、peer12–23保留，训练24–119/96核互斥；19:18实核既有peer idle、未唤醒/派生。当前r186-pod-context.env绑定ready/start-4。三轮小件归档最终guard khwnyx16/exit0；两次归档辅助脚本手写SHA断言失败csde8apm与_46o6dpg原件保留，未影响已启动训练，最终改消费实际CPU/部署记录的源SCI身份，未追加运行准入门。
