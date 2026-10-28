# 本次仓库整理的实际状态

记录日期：2026-10-28。本文件是工作事实记录，不是测试通过回执。

## 已完成

- 分离根 README 的项目导航、AGENTS 的协作/执行规则，以及 `todo/` 的当前任务、暂停状态和接续。
- 增加 `models/` 三模型索引，通过 SSH 只读传回服务器此前已保存的 15 份参数/来源 JSON 原件；未下载或复制权重。
- 增加目录说明和真实源码入口地图，未改动已验证的训练数值实现，未移动训练内部模块或改变活跃冻结部署。
- 新 CPU 通道真实执行了 `discover`、`interpreters`、`probe`，核到当时 Pod Ready/UID、实际 Python、依赖与 `gpu_initialized=false`。实际探测输出保存在 [ENVIRONMENT-PROBE.json](ENVIRONMENT-PROBE.json)。

## 中断及未执行检查

随后向独立 `technical-prep` 暂存源码时，传输进程实际返回 137；查询原 Pod 得到 NotFound。没有自动改绑其他 Pod，没有创建计算资源，也没有回退本机或登录节点执行项目代码。

- `tools/models/export_parameters.py` 未在 Pod 执行，未生成 checkpoint 内新增参数导出或成功回执。
- `tools/repository/check_structure.py` 未在 Pod 执行；文档链接/参数一致性/语法检查无通过回执。
- 参数原件传输后的 SHA 复核及三个模型的重新加载/EMA 张量核验未执行。先前存储工作核过模型 SHA，不能替代这些新增检查。
- 新代码仅完成文本审阅；没有执行模型构建、前向、官方评估、训练锁操作或训练信号。

这些检查需要经授权且可用的合适计算环境。当前后续实验暂停，缺少环境不构成恢复或新建资源授权。待办见 [todo/repository-layout/](../../todo/repository-layout/README.md)；存储清理另见 [CLEANUP-STATUS.md](../storage/CLEANUP-STATUS.md)。
