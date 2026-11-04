# 本机 shell 包装失败传播检查

检查 13 个实际使用或已备的本机 shell 包装：均通过 Pod 内 Bash 语法和 `set -euo pipefail` 检查；R248/R251 跨批接续脚本与准备副本、`pod_cpu.sh` 与已发布源码身份一致。没有修改活跃训练或冻结部署。

发现私有取回包装的远端 `sh -c` 未在 `cd` 失败后中止，可能继续从错误目录打包；在本机私有包装加入 `set -eu`、`exec tar`。同时把一次性 Pod 列表采集改为严格失败即停、仅完整两 Pod 列表才传输；此前不完整列表已被删除守卫拒绝，直接缺行原因 unknown。

Pod 准备池故障注入复现旧目录切换问题并验证修复；SSH 失败与 tar 成功时整体仍失败、不完整列表不传、CPU 门失败不进入 launch。小型[回执](RECEIPT.json) SHA `488c839b5306a82f947e83180be049c80537c3f93ba3877f0422ee65c0446994`，CPU 回执 `9bc593201cfa0a305b976505bd3c00e42cf6db4629897df7b3ee3efef619357a`。首次小包传 Pod A timeout 的目标文件保持 unknown，新路径转 Pod B 成功；未重试原路径。
