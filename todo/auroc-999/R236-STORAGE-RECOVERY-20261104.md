# R236 系列激活存储观测恢复

R236及独立R236R1均在`admit_now`存储条件失败，尚无正式clock/root/GPU派发。首轮前未复核checkpoint增长后容量；R234完整结果已e65373b/376fd3c同步回读，随后精确9项6.733GB删除dce7265e及终检e297ecd4通过。删除后及04:45只读诊断多次free=27165458432，标准库/同环境子进程/实际源码/同UID和路径一致，可通过原严格20 GiB条件；第二次失败瞬间真实free未记录，根因尚未证实，不将后验读数冒充失败瞬间证据，不以新身份盲目连续补考。

必要最小实现仅`tools/activation/activate.py`的既有存储检查：将一次实际free读取封存为小dict，严格free>21474836480保持，错误包含当次bytes/threshold/path，成功回执随既有handoff记录。增加`tests/test_activation_storage.py`两必要目标，独立验证25 GiB通过、0/20 GiB边界拒绝且错误包含实际值；原一次性激活、设备/锁/源/clock及训练数值代码不改。先缺行为RED再GREEN，全部Python/检查在当前Pod CPU3独占24核池；源码提交/push/API回读和实际文件核对后才建立新独立R236R2正式包。

新私有范围限`technical-prep/activation-storage-20261104-4e0c`内CPU检查/小metadata及既有prepare/prepare_gpu_site/check_candidate/freeze_deployment包装；复用逐字相同runtime/SCI、已通过模型/P8/数据证据，不追加GPU预演。R236R2仍R236 SCI266beb7f/order151/fresh公开Small/1600强光度，原监督及官方门不改；原两失败ready、audit、failure marker、clock空状态及成本均保留。新实际入口只调用一次；若仍拒绝，根据它实际报告的读数处理，不能删标记重试或降低存储门槛。R234唯一actualsafe不重查，当前接续超300秒如实记录。

c800已push/API回读、2822全树及新正式18/18/部署50f33058通过03224975，实际R236R2八rank首75通过8244ae79。新成功存储当次26,719,813,632已进入既有handoff，第二次历史失败根因仍unknown；[实测接续](../../records/R236R2/README.md)终态→首1为2121.741秒。小metadata仅`formal-storage-recovery.py`和`export-r236r2-start[-corrected].py`，复用现有链和原证据，不调用模型或GPU API。发布前自审发现旧metadata模板将R2入口错误写成终态前准备，另存corrected导出改为False、原R236为True，并明确2次失败+R2一次activation/入口；旧私有稿保留，未修改运行或重做前向。
