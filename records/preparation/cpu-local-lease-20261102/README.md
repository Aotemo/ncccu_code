# CPU 准备任务的 Pod 本地互斥锁

旧guard在共享NFS上的flock多次停在`nlmclnt_call`/RPC等待，LOCK_NB仍未返回，尚未创建任务目录或启动子进程；新NFS目录亦复现。当前Pod `/tmp` 是本地overlay。只将现有互斥锁改为按真实Pod UID+executor派生的本地文件，同一执行者跨准备绑定共用；保持原O_NOFOLLOW/0600/LOCK_NB、继承descriptor、整树退出、身份/期限/资源比例/mask/线程检查，回执新增实际lease_path。旧allocation/锁文件及冻结运行包保留。

Git`aaf475a9ab5cac0bee35c869adc47ad252a0ee93`已push/实际远端回读。实际guard SHA `f7eebb84650da9b8225f5369fc55d06004825938a0be3a44b3678ffaeed71941`，test `22bd3aa517b98d53cfc9ce1b81d563b68f925bd9c6d3c0bedc26cde662d1775c`。[CPU原件](CPU-RECEIPT.json)SHA `96be3799a953981b49af251a697fc140335bf030e046b573bb7d95abaa500efd`：真实两子进程分别证明占用时拒绝/释放后取得，原run_task政策转发目标也PASS，两源编译通过，exit0/无越界/任务树退出；不新模型、forward或GPU预演。[新绑定](CPU-ALLOCATION.json)SHA `719ab7ffe645ba47eddf3788781c2171f8d360802c9fd36dbbd04d32b60d06fd`，当前有效120CPU，准备24按原工具平分为root0–11与保留peer12–23，后者已closed且未恢复；训练仍24–119/96核。新guard仅用于未启动的准备工作，不改旧dca3/8cf或GPU绑定。

原件在`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/cpu-local-lease-20261102-32e3/`。cont15/19/21在精确PID/start/argv、无子进程核对后仅终止本人阻塞入口；cont20及R174 window自行结束，条件清退未命中已退出PID便中止，无信号。现场保留在原prep，不虚构guard通过回执。R172→R174的真实间隔超5分钟，详见[R174](../../R174/README.md)；本修复不宣称解决所有NFS读写延迟或量化长期吞吐。
