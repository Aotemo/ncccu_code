# 动态发现执行环境

项目不指定永久主机、Pod 或 Python 路径。环境来自工具实时查询；旧回执用于核对已经绑定的运行，不能替代发现步骤。

## 发现与绑定

1. 使用 Paseo `list_hosts` 查询当前 server，再用 `list_agents`、`get_agent_status`/`get_agent_activity` 核实本任务的既有规划、执行 owner 和连接上下文。保留 `{serverId, agentId}` 配对；不要新建 Agent 或资源。
2. 通过已有连接工具或 SSH 配置取得实际连接目标。执行 [pod_cpu.sh](../tools/environment/pod_cpu.sh) 的 `discover`，使用 kubectl 当前上下文查询 Pod 的 namespace、名称、UID、运行/Ready 状态和容器名。多个候选须按当前任务的真实绑定核对，不能自动选第一个。
3. 使用 `interpreters` 在选定且 UID/Ready 核对通过的 Pod 内查询解释器绝对路径；然后执行 `probe`，在该 Pod CPU 内核 Python 版本、torch/torchvision/numpy/Pillow 和未初始化 GPU 的事实。多容器 Pod 必须显式指定查询到的容器。
4. 保存实际发现结果，按任务绑定核对数据、缓存、公开权重、工作根与源码清单。发现新环境不授权迁移或改绑活跃任务；已有运行继续使用原冻结副本。
5. 用 `run` 执行已传输到独立 `technical-prep` 的 Python 工具。每次执行重新查询 UID/Ready，隐藏 GPU、nice 19、最多 2 线程；身份、文件或依赖不符立即停止。无本机或登录节点回退。

环境变量只提供当次工具返回的连接信息。可选 `NCCCU_SSH_CONFIG` 指向已有 SSH 配置，或使用 `NCCCU_SSH_IDENTITY`/`NCCCU_SSH_PORT`；不要将凭据或机器专用配置提交到 Git。`NCCCU_CONTAINER` 仅使用当次查询到的容器名。

以下变量均应填入实时工具输出；例子不含任何机器默认值：

```bash
bash tools/environment/pod_cpu.sh discover "$SSH_TARGET"
bash tools/environment/pod_cpu.sh interpreters "$SSH_TARGET" "$NAMESPACE" "$POD" "$POD_UID"
bash tools/environment/pod_cpu.sh probe "$SSH_TARGET" "$NAMESPACE" "$POD" "$POD_UID" "$PYTHON"
bash tools/environment/pod_cpu.sh run "$SSH_TARGET" "$NAMESPACE" "$POD" "$POD_UID" "$PYTHON" "$REMOTE_SCRIPT" --help
```

此入口只提供 CPU 准备与只读检查的执行通道。它不获取训练实例锁、不发训练信号、不启动 GPU 训练，也不产生新的训练 token/clock。发现原始输出可存 `todo/.local/`；正式检查保存真实回执。

## 历史源码与可迁移入口

`runtime/` 的原 R92 包以及 `tools/activation`、`tools/observer`、`tools/data-contract`、`tools/validation`、部分 `tools/storage`/`tools/repository` 脚本保留当时绑定值和 SHA 身份，是历史按轮工具。R92 已完成终态核实，不再接续原观察/启动入口；这些脚本不作为未来环境的通用运行入口。

新工具使用发现结果作为显式参数或由通道注入的 `NCCCU_POD_*`/`NCCCU_PYTHON`，并读取当次实际资产/部署清单。新部署由现有执行 owner 从已同步 commit 创建绑定、核文件 SHA 和资源准入。历史科学参数、checkpoint 身份与已有运行的冻结源码不由环境发现步骤改写。

## 保留模型参数导出

将本仓库的相关源码和权威保留清单传入独立 `technical-prep`，在核验环境后调用：

```bash
bash tools/environment/pod_cpu.sh run "$SSH_TARGET" "$NAMESPACE" "$POD" "$POD_UID" "$PYTHON" \
  "$REMOTE_SOURCE/tools/models/export_parameters.py" \
  --catalog "$REMOTE_SOURCE/records/storage/KEEP-MODELS.json" \
  --source-root "$REMOTE_SOURCE" --output "$NEW_PARAMETERS_OUTPUT"
```

只导回三个 `parameters.json` 和 `PARAMETERS-RECEIPT.json`。工具校验完整 checkpoint SHA、原侧车、更新数、年龄合同及 EMA 状态并记录张量摘要；不构建模型、不前向、不复评官方、不复制权重。导出失败保留真实错误，不填造参数。
