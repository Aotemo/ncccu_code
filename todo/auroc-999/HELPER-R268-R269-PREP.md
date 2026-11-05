# R268/R269 清洗与蒸馏训练接线：CPU 准备记录

本记录只描述 helper 分支准备与合成验证。真实三折 OOF、目标表、正式 CPU 18 项、部署和 GPU 训练尚未执行；轮次、seed 与 Pod 资产仍由 08ac 在正式准备前复核。科学依据为主线 `R268-R269-F05-CLEAN-DISTILL-20261105.md` 与 F05 三折合同。

## 实现和边界

- 四份 `configs/templates/R268-{a,b}-SCIENCE.json.in`、`R269-{a,b}-SCIENCE.json.in` 分别绑定对应 Pod 的已登记公开资产回执。模板中唯一有意未填的是 `__OOF_SUMMARY_SHA256__`，占位符被校验器拒绝，不能作为正式 SCIENCE；若 v1 写入失败/状态 unknown，必须使用新 v2 路径与新身份，不重置 v1。
- 除轮次、fresh seed、目标模式与公开资产 Pod 回执外，模板沿用 R120；SCIENCE 校验固定除 seed 外完整 R120 job 的 canonical SHA `5fd5f1eefccf76efbf5dad25d1b2e4c9211e57b896b302eb3fb6a48bc3b767c7`、9:1、公开权重/来源、24000/800/1600 更新、P4 selector 和初态。
- `tools/activation/prepare.py` 在现有隔离 CPU prepare 中加载已发布的 `tools/diagnostics/f05_oof_targets.py`，独立核每折 selected、训练集完整标定、held-out 数据血缘，重算标定、清洗及蒸馏目标，并核原 manifest、完整 fit 覆盖、输出字节与摘要。成功后将小型 `evidence/F05-OOF-VERIFICATION.json` 写入待部署包，`TRAIN_CONFIG` 绑定该回执、SUMMARY 与对应目标的实际 SHA；不把样本级文件写入 Git。
- `distributed_engine.py` 各 rank 在正式 fresh WORLD8 训练开始前重核 CPU 回执、SUMMARY、单一目标文件 SHA 和完整 fit 原始索引/标签；不在 rank 重载三折 checkpoint。原有 WORLD logits/标签顺序不变，清洗按正负权重乘积和归一化，蒸馏为原 pairwise 加 `0.25 × BCEWithLogits`、温度 1。发生目标缺项则各 rank 同步失败。`pipeline.py` 和普通 `train.py`（含 worker）拒绝 F05 目标，避免走不消费辅助目标的旧入口。
- 08ac 的 `tools/activation/prepare.py` 在主工作区还有独立旧草稿，合并须逐项保护；当前 helper 改动没有覆盖主工作区或冻结部署。OOF 目标 consumer 据 08ac 已于 main `91ea87d` 入库，真实 sidecar 修订为 `d3ceb81`；合并后应以最终 main 完整树重新验证兼容性。

## CPU 证据

B Pod 12–23、显式隐藏设备、nice 19 的独立源码副本上：

- 配方、模板、合成 prepare 独立重算、`config.load_config`→科学约束、SHA/覆盖篡改和旧入口拒绝：**10/10**；WORLD 两 rank 合成清洗/蒸馏和原损失、G64/P8 顺序与 DDP 梯度：**4/4**。guard 回执 SHA `613025baacffd2b5ae0d0a13492b196fc24a4ea30472c5f77a9c838d85cf214b`。这不等于真实 OOF 或正式 18 项。
- 源码 SHA：`config.py` `5db254ef4c562be53fe82ad59d2b16d2fa1f40bad5ff9607dfdaff147fe25c75`；`data.py` `251082aef23f2025780c8641417e65ea48eb72a6cef5eab93d8fd5c0b5caed52`；`distributed_engine.py` `8a855826c5da8d3ece88a4d05abba292c4f27f761bb4a6f8160ea78289cca1fd`；`experiment91.py` `c9e8e35bd67acb896985949aeb6dfc888dbc08521ab15c4ae13bf739eb9ac193`；`pipeline.py` `49f93a32ec3a3926ed9c79539ed683370a80c532e7d0bf6c39782c2cb34e915b`；`train.py` `5edb7b03c36636beed16a3fcf26b2a0df9cde52194cebe1e3528f6e18f128bf9`；`prepare.py` `6d5fda7890d432cc01f18a157f3de14e12fdeebeb95a1a664720554ba2f85f80`。
- 公开权重 `7aaea5ce6fe6db953874d9792d3f1711a5cc0193a10d740e023108043918c610` 仅读一次后内存复用；先重现 R120 seed `20261029` 初态 `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`，再得到 R268 seed `20261201` 初态 `6e2e8f4b20c211d3ed446187ddc9ea4803e1d560ed1741b00381213431ae7001` 和 R269 seed `20261202` 初态 `b14f7c368ab70e6591c25ea6d9ab4c67b563876bb10dcd2050d75127e7d0da2c`。guard 回执 SHA `0c58da23c83fc1e5190e2fa659f75d3bf896f19dd26475ff5fe726d2821df001`；最大 8 MiB 读取约 0.0049 秒、观察期间 I/O 压力为 0。该 CPU 构模测量不替代正式逐 rank 初态回执。

## 真实 OOF 到位后的操作清单（仅 08ac 执行部署）

1. 审阅并合入本分支必要文件，保护主工作区未完成草稿；push/API 后，验证最终 main 完整树及受影响集成测试。确认三折各自 actualsafe、selected checkpoint/sidecar 与完整 fold-train/heldout 前向及 rank 证据均已封存。
2. 现场确认 `work_root/technical-prep/takeover-20261104-08ac/F05-OOF-TARGETS-v1` 不存在，通过已合入的 OOF 目标生成器一次生成；保存 SUMMARY 与所有引用的真实字节 SHA。unknown 或失败保留原件，改用新 v2 身份和模板路径，不重试覆盖。
3. 选两 Pod 的对应模板，仅填实际 SUMMARY SHA（如改 v2，则同步更新相对目录）；形成正式 SCIENCE 并提交/推送，按既有链生成新的 site/package。prepare 从已核 main 独立重算三折标定和两个目标，自动生成固定验证回执与两个目标 SHA。若任一源缺失、重算不一致或完整 fit 覆盖不符，停止准备。
4. 逐 Pod 确认 R268/R269 尚无旧实际轮次、clock、token、claim，seed 未用，核当前 Pod UID、对应公共资产和资源拓扑；完成正式 CPU 18/18、部署/容量/设备检查。该项不可由上面的合成回执代替。
5. 由唯一 GPU owner 按互斥资源与各 Pod 安全终态执行既有接续；逐 rank 严格初态及首成功回执后再视为训练可运行。无任何一次官方样本读取、TTA 或医学预训练来源。

## 与 main b62 的最小整合审阅

main `b62bdddbb2b552f0d384058de49af85d2c9c1d84` 中，上述七个接线源码文件及 `tests/test_f05_clean_distill.py` 与本助手原始基线 `a004c08` 相同；因此待审入范围可直接依据 helper commit `bb220a3`：七个接线源码、两个测试、四个 SCIENCE 模板及本说明。`classification_loss.py` 和 OOF target consumer 在 b62 已与 helper 相同，无需覆盖；现有 OOF GPU 入口由 08ac 维护。

08ac 主工作区的未提交本地 checkpoint/cache 草稿也修改 `tools/activation/prepare.py`：其局部存储准备与本提交的 F05 目标验证分别位于不同辅助函数和调用点。整合时须逐项保留实际采用的两侧改动；当前 **10/10 + DDP 4/4 不覆盖两功能同包**，如采用该草稿，须对最终合并树补同包合成和受影响 I/O 窄测。不能用 b62 或 helper 单侧证据代替最终树检查。

正式 CPU prepare 对每个新轮次各自独立核三折 selected 字节及父回执、重算完整训练子集标定和两个目标表，再将结果固定到该轮 `evidence/F05-OOF-VERIFICATION.json`；之后每个 rank 只核回执、SUMMARY、所需目标表和完整 fit 对齐，不重复读取三折 checkpoint。两次正式 prepare 的真实读取时长和共享存储压力尚未测量，不应假定可忽略。真实 SUMMARY、两轮历史身份及 seed、最终树和正式 18/18 均未在本记录中声明通过。
