# 三折 OOF 接续缺口与接口审阅

状态：**只读源码审阅及实施准备**；R265–R267 最终 selected、真实标定和 held-out 分数均未在本审阅中读取或验证。以下是拟接线方案，不代表任何 GPU 前向或资产已具备运行资格。原始 fit/dev manifest SHA `7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377`，三折封存 SHA `eb880273832b81be61830a069d5ae9ba8dc2e52be4af2f911f7a6b1ec9e7f6d8`；科学要求以 [三折合同](F05-OOF-CLEAN-DISTILL-768-20261105.md) 为准。

## 当前入口与缺口

- `tools/diagnostics/f05_oof.py` 建立并核验 fit-only 三折；`f05_oof_assets.py` 和 `runtime/bundle/retrain/f05_folds.py` 建立并复核每折训练元数据。训练 manifest 的 `audit_indices` 有意为空：held-out 是独立封存角色，不应解除训练入口的 audit 禁令来生成 OOF。
- `tools/diagnostics/f04_gpu_full_dev.py` 已有八设备 owner/worker、proof barrier、父进程监护、超时和各 rank 包；`worker_batch_records` 验证 index、标签和有限分数，`canonical_member` 复用 `runtime/bundle/retrain/parallel_evaluation.py` 的确定性分片/无重复聚合，`bind_member_runtime` 绑定原冻结 runtime。当前 `build_plan` 固定五个 F04 成员，`evaluation_population` 只允许完整内部 dev 或已有官方路径，`aggregate_packets` 也固定完整 dev；不能通过改名或伪造 manifest 直接当作 held-out 入口。
- `paired_evidence._selected` 与 `ensemble_dev.py` 的现有候选发现按同一原 manifest 进行 selected/dev 绑定；折训练使用派生 manifest，须显式核原始 manifest、折派生 manifest 和独立封存角色之间的关系，不能把不匹配绕过或把原 dev 当 held-out。
- selected 的 online/EMA、update、tensor SHA、checkpoint blob/sidecar、完成回执、冻结 runtime/科学配置、fold 派生 manifest/age、cache 与 site 必须逐折实核。不能用某次扫描看到的临时最优或事后自报的模型 SHA 代替实际终态选点；尚缺 selected、完整回执或严格加载证明即 unknown，不重新选点。

## 拟按既有链实施的顺序

1. 三轮分别完成且安全终态后，由 08ac 核对应唯一 selected 及原 runtime；核 `FOLDS-private.json` 与原 manifest 的文件 SHA、`verify_roles`、训练 fit 与 audit 互斥/组隔离、原 dev 不进入两类输出，且每个 fit 行只在一个 audit。固定每折 age 统计来自该折训练子集。保存只含身份和聚合的准备回执。
2. 每折先封存两个互斥 population：全部该折 held-out，以及从该折训练 fit 按类抽出的标定行。合同规定每类最多 1024、seed `20261121 + fold`；具体抽样遍历顺序及样本权重的全局归一化须在读任何新 OOF/标定分数前固定并保留，否则 `LogisticRegression(C=1000)` 的正则化可能受权重整体缩放影响。
3. 由唯一 GPU owner 对既有 `f04_gpu_full_dev.py` 明确添加受封存证据约束的 **fit-only OOF 模式**：单 selected 的八 rank 分片，依序处理上述互斥 population；复用原 barrier、设备/CPU 准入、历史 runtime、严格 selected 加载、单视图和 packet/聚合，不启动另一个监督平台。每个 held-out 行对所属 selected 只前向一次；标定行是该折训练子集、另一次单视图前向。每折输出唯一 identity/claim，超时或通信中断记 unknown，先只读核输出和实际进程，禁止自动重派。此模式拒绝官方/dev population、TTA、错误变体及非 fold 训练 site。
4. 每个私有 packet/完整汇总绑定 fold/round、base manifest 与 sealed folds SHA、派生训练 manifest 和 age SHA、selected blob/tensor/variant/update、模型严格加载证据、角色和预封样本列表 SHA、rank proof、单视图标识及每行原 manifest index/标签/logit；核每行所属角色、rank 分片、顺序、无重复/缺项/非有限后再对该折输出 receipt SHA。逐样本分数、标定样本、目标表和可疑名单只留 exp-v3，不进 Git。
5. CPU 下只用该折训练子集的标定标签/分数拟合合同中的加权一维 `LogisticRegression(C=1000, solver='lbfgs', max_iter=1000, tol=1e-8, random_state=20261121)`；核两类与收敛、系数/截距及有限输出。将映射应用到所属 held-out raw logit，再检查三折完整覆盖原 fit 恰一次。`tools/diagnostics/f05_oof_targets.py` 拟将核验后的数据生成 clean/distill 两份 `f05-fit-oof-targets-v1`，来源列表含每折 selected 与标定 SHA；按 [helper 接口合同](HELPER-CLEAN-DISTILL.md) 重算 confident-joint、0.25/1 权重和软标签。
6. 08ac 负责在 `runtime/bundle/retrain/experiment91.py`、`pipeline.py`、`distributed_engine.py` 及 `tools/activation/prepare.py` 接入：先独立核完整模型/预测/标定回执和目标文件 bytes，再将实际验证所得 `verified_sources` 与 `calibrated_oof` 交 `data.validate_f05_target_table`。当前表校验本身不证明自报 SHA 对应真实文件；不做这一步不能训练。`f05_train_batch_loss` 在原 WORLD 顺序 gather logits/labels/目标并给出固定损失；接入时须保留原训练循环对 `global_targets` 的统计，避免对 logits 重复 gather 或改变 DDP 梯度缩放。

## 拟覆盖的隔离验证

- 合成三折与两个受封 population：少一行、重复、错误 rank/fold、组重叠、dev 混入、标签/模型/variant/SHA 不匹配、非有限分数、旧五折和残缺 packet 均拒绝；三折 held-out 的并集恰为 fit，标定只能来自相应 train。
- 受控合成标定：每类抽样固定、类比例权重和收敛回执；单类、未收敛、非有限、缺标定源/selected 源拒绝；soft target 与映射逐行一致、clean 阈值/矩阵/权重可独立重算。
- 已完成的双 rank G64/P8 数学检查只覆盖目标输入后的 WORLD 顺序和梯度，不证明真实 OOF 的产生、血缘或主循环接线。待实现相关入口后在核验的 Pod 12–23 准备池、隐藏设备运行新窄测；真实前向需 main 合并、完整树及 owner 的现场 GPU 安全门。

## 实际文件边界

助手拟新增 `tools/diagnostics/f05_oof_targets.py`、`tests/test_f05_oof_targets.py` 和本说明；不修改 08ac 负责的 `f05_oof.py`、`f05_oof_assets.py`、`f05_folds.py`、`experiment91.py`、`pipeline.py` 或 activation 文件。复用 `classification_loss.py`/`data.py` 已交付接口；若实际八 rank OOF 模式需要改 `f04_gpu_full_dev.py` 或其测试，由 08ac 审阅协调。最终数据格式和写入策略须与后续 GPU 回执精确对齐；本审阅不声称现有接口已实现该模式。
