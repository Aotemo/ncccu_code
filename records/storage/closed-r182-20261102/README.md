# R182 完整归档后的冗余权重清理

来源归档commit af21d54665de088f3e18d72173a9806a3e716b6e已push并实际远端回读；R182完整6400/c8/bad4、唯一EMA c4 AUROC .9868174369138257、两门false/官方未跑/actualsafe先封存。R186已真实八rank成功运行后，仅清理原R182闭合目录内8个checkpoint/initial blobs；[精确清单汇总](PLAN-SUMMARY.json)/[实际删除回执](DELETE-RECEIPT.json)保留，不重查旧safe、不加载/复hash权重或新forward。

逻辑23,554,303,873 B、原stat allocated23,554,306,048 B；共享FS实测free27,131,904,000→50,686,066,688 B，差23,554,162,688 B，三种量分列，不称后端物理回收保证。8/8 unlink、无跳过/错误；保护stat及小SHA前后一致。KEEP N06/R17/R8-C25、原始数据、共用cache、待用公开权重及活跃R186整树均保护；无训练信号、无训练实例锁、未删sidecar/index/one-shot/clock/token。

继续复用已有PLAN/DELETE/FINALIZE工具，必要适配只改闭合轮标签/归档commit与实际结果字段；根5de/CPU0–11/nice19/隐藏GPU。plan guard hvd8kwfx、delete 5tsn7jq1均exit0，[原件](CPU-PLAN-GUARD.json)与[删除CPU](CPU-DELETE-GUARD.json)保留。最终窄核guard wu1ju0pr exit0，SHA97e461d419cfcd31f25120710894334d4cadb04335f312869f58c7340311af12；原件在exp-v3/technical-prep/cpu-renewal-20261102-1651-32e3/tasks/32e3b079-81b0-4f66-9604-b32491599f44-wu1ju0pr/CPU-RECEIPT.json。逐件unlink journal、保护快照及明细清单仍在exp-v3，未知/其他目录保持。
