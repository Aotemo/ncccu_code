# R92 Tiny ImageNet1K：已完成，未达0.999

固定9:1，6400成功更新/8周期/16条完整内部dev/0skip，无早停。唯一cycle5 online@4000由预定error_twice/AP/update/EMA排序选出，冻结后完整官方5849仅评一次。

| 用途 | 样本(P/N) | raw AUROC | AP | error_twice |
|---|---:|---:|---:|---:|
| 内部dev选中项 | 5140 (167/4973) | 0.9829293755140032 | 0.8089328436574713 | 28354 |
| 官方最终一次 | 5849 (191/5658) | 0.9833687740474035 | 0.790262001203195 | 35946 |

Pod从已有预测复算精确AUROC=354235/360226；error_twice35946超过目标允许2161，target=false。官方不进入训练、年龄或选点。

公开facebook Tiny1K revision6166b761，768/native avgpool/Dropout.2/nn.Linear原生头，末端LN1e-12、内部1e-6、SD.1、0BN/0buffers；fit46256/dev5140、S800/total6400/warmup和EMA copy1600，backbone2e-4/head-age1e-3，512/G64P8/FP32。native四rank fresh183 tensors/空Adam/零age通过，最终EMA1600copy/4800blend。与R91的骨干/LN/头初始化差异已记录，不能将全部差异归因于来源。

- source digest：`57ef83209692eb79e682e04e2b3fe7a060225856f673505ab67396080216d51e`；原bytes已由协调方迁入Git `7280b190077347f97ce5a6f2e3fa5d54da54b885`，运行未热改。
- 选中online tensor：`922291c4a84700460f74cff13b7faa29620f30a4986afc825992371d71991b8d`；blob SHA：`8d130247f1b84c5cc10a053c079c7b522d442c03dee5cf64c89b039099bc4230`。
- 官方预测SHA：`5dff2e566c8d0fd9042edda04459a2b7657c0a653d186f53036a0091086d52c9`。
- 本轮token：`f1d4b2f895a84a15a3d81634b5153cee`，仅历史对账，禁止再activation/launch/official forward。

Pod终态audit真实exit0、22/22通过；owner430002/start320053441实际OSexit0，四rank及train/freeze/official组件全exit0；原锁inode18030447997释放、无活跃owned ranks/device users、四卡空闲。上海2026-10-28 15:38:18.327589首次T0，16:40:54.557027 done；组件训练3622.706921339035秒、冻结7.572642803192139秒、官方61.249107122421265秒。

[真实最终回执](FINAL-RECEIPT.json) Pod SHA `6c4e8264dc743f90feecd401baf60f22ccf6e0835aead3b246758e1c9a18cb1f`；[16条dev汇总](DEV-METRICS.csv)；[真实时间表](TIMING.json)。全部解析/校验在指定Pod，本地只传输和查看。29项CPU准备检查未重跑。

用户仅永久保留N06/R17/R8-C25三个历史模型；R92所有checkpoint/fixture无保留例外，本记录只留实验结论与必要小回执。R92已安全退场，未达到整体目标。后续实验已按用户要求暂停，未经新的明确恢复指令不接续R93。R92产物清理尚待执行，见[清理状态](../storage/CLEANUP-STATUS.md)。
