# 激活存储读数窄检查

为解释R236/R236R1相同存储错误与后验可用容量的差异，仅改现`activate.py`：一次读取free，原严格free>21474836480保持，失败携带bytes/threshold/path，成功写入既有handoff.storage。训练、SCI、selector、设备/锁/一次性检查和clock逻辑不改；这是观测补足，尚不认定第二次失败根因解决。

[CPU检查](CPU-CHECKS.json)：缺helper两目标RED495a4794；最小实现后两边界目标/两编译及15未改函数/类、admit除存储读数/证据外AST相同通过b0de1b1a。25 GiB通过，0/严格20 GiB边界拒绝，错误包含实际值；实际工具SHA3b05f127、测试7a098513。一般文件系统正例探针路径为SothisAI父目录，按原样保留；R236R1实际work_root三次及同环境子进程诊断在该轮记录，不用父目录探针冒充激活瞬间读数。CPU3独占24核/nice19/隐藏GPU，无模型/权重/前向或GPU API。

首个大文本源码传输在CPU目标前被中止，随后窄测试直接针对明确的冻结旧activate.py，GREEN只用两个变更文件；Kubernetes API曾EOF，恢复后原CPU guard完成实际RED/GREEN，不将传输unknown写成测试通过。新发布源全树核对与R236R2正式包/实际启动仍待；原两个失败attempt不重试。
