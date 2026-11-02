# R172：共享骨干的全图与固定重叠局部视野

[完整结果](R172-RESULT.json)：6400 success=attempt/0 skip，8次评估、连续4次未提升，正常早停。唯一EMA c4@3200内部AUROC **.9882081804619195**、AP **.8264419710103553**、error_twice19586；低于R107与R120，两门false，固定配方关闭、R173无资格且未formal/activation/GPU。不扫窗口、聚合权重或seed。官方不足.996按政策未跑；[同次终态](R172-COMPLETION.json)证实completed/组件exit0/无signals、owner原身份现Z、八rank缺席、锁/设备使用者空、8卡busy0，actualsafe。原权重和冻结依赖保留，归档不新forward或重查safe；raw .999未达。

[真实首成功](R172-START.json)为75 success=attempt/0skip/eval0，八rank实际345全训/5空Adam、whole50c和正常formal一致，8PCI/训练24–119共96核匹配。首自然记录25 loss `.7223862409591675`有限，WORLD8/G64P8/448个图像pair；[guard](R172-CPU-FIRST-SUCCESS.json)exit0。运行Git`1859670356f1068ce57726b6e072485959154d4b`/tree`216951f23b3bb81e2c3a26357dc90a750c210d55`，SCI`c7cc4bc4008386c0b7f1d70a1b969dc61e4132a8ceb9a8fb1334d6146da93230`，SITE8a076e08。[正常formal18/18及精确部署](R172-PREPARATION.json)已在R170训练期间完成，[一次activation](R172-CPU-ACTIVATION.json)/direct launch0，无重复formal或旧clock/token/grant重置。

直接R107 Small/public fe6/seed29/512/原pair-logistic/default AdamW/age，仅同一个network先处理全图B×512、再处理image-major四个重叠288角窗4B×288，`.5*global + .5*max(local4)`，原age一次；参数和初始化张量身份不变，函数由保存policy区分。[保存与准入配置](R172-ACTUAL-GEOMETRY.json)保留policy，CPU实际hook和P8/coldexact见[准备证据](../preparation/quadrant-mil-20261102/README.md)。GPU日志primary/view_inputs为外层图像输入64，不把它称为内部320视野的实测hook；本轮不额外加GPU预演/forward。初始化无额外RNG不代表训练Dropout/SD轨迹相同。

前轮R170为存储保护技术失败、无完整selector，未把暂best判负。[事件间隔](R172-HANDOFF-GAP.json)从前轮最后interrupted metrics到新dispatch101.017秒、最晚设备166.113秒、首自然25记录245.130秒，均≤300；failed整轮ended到三项分别91.816/156.912/235.929秒。分别记录失败事件/整轮结束，未采集连续利用率，不声明连续最大空闲上界已测。准备dca3根0–22/23核与peer23/1核、训练24–119/96核互斥；当前唯一Git/GPU owner32e3，当前入口`work/r137-20261101-a/r172-pod-context.env`绑定实际start-4。

后续独立[R174](../R174/README.md)已在本轮actualsafe后串行接续。NFS准备锁等待延误了终态观测，后续原事件至dispatch537.365秒/设备597.897秒/首25记录664.967秒，未满足5分钟；[本地CPU互斥锁最小修复](../preparation/cpu-local-lease-20261102/README.md)保留身份、额度和整树检查，旧回执/锁/运行包未改。这里只报告分离事件间隔，不称连续利用率已测。
