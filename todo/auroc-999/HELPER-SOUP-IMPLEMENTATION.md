# Helper 三汤 dev 入口准备

复用 `tools/diagnostics/f04_gpu_full_dev.py` 的八 rank 监督、Pod 身份、锁、容量与释放检查；增加显式的内部 dev 汤模式，按固定配置 `r120/r134r1/r201` 的 3/3/2 设备分片，每组先后各读一次 fp32/fp16 存储权重，计算均采用 fp32。使用 `tests/test_f05_soup_gpu.py` 做 CPU 合成隔离检查。

必要时新增 `tools/diagnostics/f05_soup_dev.py` 作为现有入口的小型输入/结果辅助：核对三组事前固定的血缘与 standalone 回执、两种存储权重及模型元数据 SHA，封存输入，并归约完整内部 dev 与替换预览；不新建进程监督、部署、训练或官方评估链。原 F04 五成员和官方路径保留原语义，汤模式明确拒绝官方参数。F04 的 fp16 原分片若缺逐文件绑定证据，则该精度替换预览标 unknown，不能将当前文件的事后 SHA 冒充事前绑定。

本项为代码准备，不代表 CPU 测试、精确部署、GPU 推理或新指标已完成；实际变更和回执交 08ac 审阅合并。

## 交接前缺项与最短接续

1. **代码尚未验收：** helper 中三汤输入、分片和归约草稿仍未在 Pod 执行合成测试；必须先在独立准备池运行 `tests/test_f05_soup_gpu.py`，并对修改的 F04 监督入口运行既有 `tests/test_f04_gpu_full_dev.py` 回归。任何失败先修复、记录真实回执，不能把静态审阅当作通过。现役训练及已冻结 F04 副本不改。
2. **实际三汤资产尚未完成本入口绑定：** 三个 `SOUP.json` 与 `STANDALONE.json` 虽已有事前固定 SHA 和历史合成加载回执，新 `prepare-soup` 仍须在目标 Pod 以完整已推送源码核实这六份回执、每组 fp32/fp16 的模型/metadata、原 F04 输入及历史 source/site；输出新的 fit/dev 内部评估输入回执。`prepare-soup` 只做 CPU 资产核验，不前向、不读官方、不认领 GPU。真实权重路径和当次 Pod/CPU 身份必须实核，不能把现有历史回执当作本入口的通过证明。
3. **fp16 替换预览存在明确缺口：** 当前草稿始终将原 F04 fp16 分片的替换预览标为 unknown；是否存在能把原 8 个 packet 与冻结的内部评估回执独立绑定的可用证据尚未在 Pod 核查。若找到可核验证据，须在读取新汤结果前补齐并测试固定的分片绑定及 15 个 fp16 替换组合；若无法核验，按合同如实保留 unknown，不能仅以事后计算的文件 SHA 冒充原始绑定。fp32 预览和三汤本身的两种存储精度指标仍按合同提供。
4. **最短 GPU 接续：** 让三折正式准备和接续优先；三汤代码窄测通过后由 08ac 审阅并合并 main、push/API 回读及目标 Pod 完整树校验，再以实际三汤/原 F04 身份生成 CPU 输入。待某 Pod 的授权前轮实际安全终态、现场容量及现有 80/20 资源门满足后，08ac 沿现有 supervisor 显式 `--soup-mode` 使用 3/3/2 八 rank，分别单视图评估 fp32/fp16 存储；不要求独立 GPU 预演，也不把研究预览作为 F05 选择。任一检查未知或失败即保留回执、不自动重派；原 F04 官方 claim 不参与。

`prepare-soup` 接受 `--source-root`、`--remote-metadata`、`--git-commit`、`--f04-inputs`、三次 `--soup-receipt` 与新 `--output`；后三个汤回执按版本化配置核验。具体 GPU owner 入口仍承接原完整 site/safe/window/ready 参数；汤模式不重筛 F04 成员，也不读取或消耗 F04 官方路径。
