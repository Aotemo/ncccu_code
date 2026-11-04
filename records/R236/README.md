# R236 正式激活前存储失败

R234完整阴性且唯一actualsafe94f21ed6后，04:33启动已预填原四步入口。窗口检查5730d999通过，31560秒要求仍有41793.870秒；正式preflight通过后，一次activation调用在`activate.py:admit_now`的存储检查报`ValueError: insufficient storage`，见[原始错误](ACTIVATION-ERROR.txt)和[实际退出](ACTIVATION-EXIT.json)。激活guardb6d9a817 exit1，原四步脚本随即停止，尚未执行GPU launcher；没有训练或科学阴性结果，不用旧18/18冒充恢复轮实测。错误文本为原私有ACTIVATION.log逐字副本，不提交其余原始日志。

标准激活要求工作盘free大于20 GiB。当前轮checkpoint增长后的空间余量未提前复核，这是本次准备遗漏；不降低阈值、删除一次性标记或重试原activation。错误发生在正式clock/root创建前，原formal root/control/dispatch未出现；将另存小metadata回执核对全部标记。原ready、SITE c4cb7dd9、部署24129974、audit和日志均保留。GPU接续将超过5分钟，结束→新dispatch/首1/25的真实间隔在恢复后报告，不声称连续利用率。

恢复使用新独立`r236r1-small-fitntxentcolor1600-seed29-pairlog-24k-p4-9010-20261104`和`r236r1-20261104-4e0c-ready`。同一已发布c57e3ea/tree31f31a21、R236 SCI266beb7f/order151、fresh公开初始化和1600强光度合同保持；既有prepare重新生成技术/正式绑定，完成真实18项CPU和精确部署后，释放已归档且无后续forward依赖的R234 checkpoint，再沿原入口一次激活和派发。私有`strong-color-20261104-4e0c/formal-strong-color-recovery.py`仅替换新round/输出参数并复用prepare/prepare_gpu_site/check_candidate/freeze_deployment，无新训练、保存、评估或监控链。R237资格仍待完整恢复轮严格双门，原失败不当种子结果。
