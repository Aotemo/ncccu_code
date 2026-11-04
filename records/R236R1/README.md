# R236R1 存储激活失败，未派发GPU

新独立身份沿c57e3ea/SCI266beb7f，实际18/18 CPU、公开50c6f55c、345全训/空Adam及2792文件精确部署通过（guardb1b9e25e，SITE4bc36dbe、部署8d931605）。窗口检查85de39d4后，唯一activation guarde250d131 exit1，再次在原20 GiB存储门报[insufficient storage](ACTIVATION-ERROR.txt)；[退出](ACTIVATION-EXIT.json)为epoch1793767378.2226708。原formal-cpu/ACTIVATION-FAILURE存在，正式clock/root/CHAIN-RESULT均未出现，GPU launcher未执行，不重试原activation或改旧标记。

此前R234精确9 checkpoint删除后free=27165458432，04:43终检仍相同；[04:45只读诊断](STORAGE-DIAGNOSIS.json)/guard4862a009再次核实际源a453747f、同UID/路径、标准库、三次statvfs/shutil及同环境子进程，均27165458432且可过原严格20 GiB条件。失败瞬间实际free没有记录，当前读数不能证明当时值或断言根因已修复。后续仅为现有存储门增加一次实际读数及错误/成功回执，见[恢复范围](../../todo/auroc-999/R236-STORAGE-RECOVERY-20261104.md)；新独立R236R2须新Git/部署，科学合同不变，旧两失败保留。当前没有新训练结果。
