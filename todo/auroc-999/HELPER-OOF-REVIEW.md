# 三折 OOF 接续缺口与接口审阅

状态：**只读源码审阅及实施准备**；R265–R267 最终 selected、真实标定和 held-out 分数均未在本审阅中读取或验证。以下是拟接线方案，不代表任何 GPU 前向或资产已具备运行资格。原始 fit/dev manifest SHA `7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377`，三折封存 SHA `eb880273832b81be61830a069d5ae9ba8dc2e52be4af2f911f7a6b1ec9e7f6d8`；科学要求以 [三折合同](F05-OOF-CLEAN-DISTILL-768-20261105.md) 为准。

## 当前入口与缺口

- `tools/diagnostics/f05_oof.py` 建立并核验 fit-only 三折；`f05_oof_assets.py` 和 `runtime/bundle/retrain/f05_folds.py` 建立并复核每折训练元数据。训练 manifest 的 `audit_indices` 有意为空：held-out 是独立封存角色，不应解除训练入口的 audit 禁令来生成 OOF。
- `tools/diagnostics/f04_gpu_full_dev.py` 已有八设备 owner/worker、proof barrier、父进程监护、超时和各 rank 包；`worker_batch_records` 验证 index、标签和有限分数，`canonical_member` 复用 `runtime/bundle/retrain/parallel_evaluation.py` 的确定性分片/无重复聚合，`bind_member_runtime` 绑定原冻结 runtime。当前 `build_plan` 固定五个 F04 成员，`evaluation_population` 只允许完整内部 dev 或已有官方路径，`aggregate_packets` 也固定完整 dev；不能通过改名或伪造 manifest 直接当作 held-out 入口。
- `paired_evidence._selected` 与 `ensemble_dev.py` 的现有候选发现按同一原 manifest 进行 selected/dev 绑定；折训练使用派生 manifest，须显式核原始 manifest、折派生 manifest 和独立封存角色之间的关系，不能把不匹配绕过或把原 dev 当 held-out。
- selected 的 online/EMA、update、tensor SHA、checkpoint blob/sidecar、完成回执、冻结 runtime/科学配置、fold 派生 manifest/age、cache 与 site 必须逐折实核。不能用某次扫描看到的临时最优或事后自报的模型 SHA 代替实际终态选点；尚缺 selected、完整回执或严格加载证明即 unknown，不重新选点。

## 拟按既有链实施的顺序

1. 三轮分别完成且安全终态后，由 08ac 核对应唯一 selected 及原 runtime；核 `FOLDS-private.json` 与原 manifest 的文件 SHA、`verify_roles`、训练 fit 与 audit 互斥/组隔离、原 dev 不进入两类输出，且每个 fit 行只在一个 audit。固定每折 age 统计来自该折训练子集。保存只含身份和聚合的准备回执。
2. 每折先封存两个互斥 population：全部该折 held-out，以及该折**完整训练 fit** 作为标定行。2026-11-05 c607 在任何 OOF 分数前裁定完整训练子集替代原合同每类最多 1024 的抽样；标定权重均为 1，均值为 1、总和等于训练样本数。原分层抽样时的类比例修正不是反比例类平衡，全量时不另做类平衡。训练索引和排序须独立绑定封存折角色。
3. 由唯一 GPU owner 对既有 `f04_gpu_full_dev.py` 明确添加受封存证据约束的 **fit-only OOF 模式**：单 selected 的八 rank 分片，依序处理上述互斥 population；复用原 barrier、设备/CPU 准入、历史 runtime、严格 selected 加载、单视图和 packet/聚合，不启动另一个监督平台。每个 held-out 行对所属 selected 只前向一次；标定行是该折训练子集、另一次单视图前向。每折输出唯一 identity/claim，超时或通信中断记 unknown，先只读核输出和实际进程，禁止自动重派。此模式拒绝官方/dev population、TTA、错误变体及非 fold 训练 site。
4. 每个私有 packet/完整汇总绑定 fold/round、base manifest 与 sealed folds SHA、派生训练 manifest 和 age SHA、selected blob/tensor/variant/update、模型严格加载证据、角色和预封样本列表 SHA、rank proof、单视图标识及每行原 manifest index/标签/logit；核每行所属角色、rank 分片、顺序、无重复/缺项/非有限后再对该折输出 receipt SHA。逐样本分数、标定样本、目标表和可疑名单只留 exp-v3，不进 Git。
5. CPU 下只用该折完整训练子集的标定标签/分数，以逐行权重 1 拟合合同中的一维 `LogisticRegression(C=1000, solver='lbfgs', max_iter=1000, tol=1e-8, random_state=20261121)`；核完整训练索引、两类与收敛、系数/截距及有限输出。将映射应用到所属 held-out raw logit，再检查三折完整覆盖原 fit 恰一次。`tools/diagnostics/f05_oof_targets.py` 拟将核验后的数据生成 clean/distill 两份 `f05-fit-oof-targets-v1`，来源列表含每折 selected 与标定 SHA；按 [helper 接口合同](HELPER-CLEAN-DISTILL.md) 重算 confident-joint、0.25/1 权重和软标签。
6. 08ac 负责在 `runtime/bundle/retrain/experiment91.py`、`pipeline.py`、`distributed_engine.py` 及 `tools/activation/prepare.py` 接入：先独立核完整模型/预测/标定回执和目标文件 bytes，再将实际验证所得 `verified_sources` 与 `calibrated_oof` 交 `data.validate_f05_target_table`。当前表校验本身不证明自报 SHA 对应真实文件；不做这一步不能训练。`f05_train_batch_loss` 在原 WORLD 顺序 gather logits/labels/目标并给出固定损失；接入时须保留原训练循环对 `global_targets` 的统计，避免对 logits 重复 gather 或改变 DDP 梯度缩放。

## 拟覆盖的隔离验证

- 合成三折与两个受封 population：少一行、重复、错误 rank/fold、组重叠、dev 混入、标签/模型/variant/SHA 不匹配、非有限分数、旧五折和残缺 packet 均拒绝；三折 held-out 的并集恰为 fit，标定只能来自相应 train。
- 受控合成标定：每折完整训练索引、逐行权重 1 和收敛回执；单类、未收敛、非有限、缺标定源/selected 源拒绝；soft target 与映射逐行一致、clean 阈值/矩阵/权重可独立重算。
- 已完成的双 rank G64/P8 数学检查只覆盖目标输入后的 WORLD 顺序和梯度，不证明真实 OOF 的产生、血缘或主循环接线。待实现相关入口后在核验的 Pod 12–23 准备池、隐藏设备运行新窄测；真实前向需 main 合并、完整树及 owner 的现场 GPU 安全门。

## 实际文件边界

助手拟新增 `tools/diagnostics/f05_oof_targets.py`、`tests/test_f05_oof_targets.py` 和本说明；不修改 08ac 负责的 `f05_oof.py`、`f05_oof_assets.py`、`f05_folds.py`、`experiment91.py`、`pipeline.py` 或 activation 文件。复用 `classification_loss.py`/`data.py` 已交付接口；若实际八 rank OOF 模式需要改 `f04_gpu_full_dev.py` 或其测试，由 08ac 审阅协调。最终数据格式和写入策略须与后续 GPU 回执精确对齐；本审阅不声称现有接口已实现该模式。

## 提议的私有 producer／consumer 格式 v1

此节在真实分数前登记，供 08ac 的八 rank producer 与助手 CPU consumer 合成对接；producer 合并前不得执行真实数据。所有 `{path,sha256}` 是独立文件的绝对路径与该文件原始字节 SHA-256，路径限制在已核的同一 `work_root`；新输出在 `work_root/technical-prep` 的新建独立目录，拒绝链接与覆盖。任何预测、索引、标定逐例数据都只留 exp-v3。文件一旦写入即封存，失败或未知身份不自动重派或覆盖。

**角色索引摘要：** train 取 sealed fold 的原 `fit_indices`，heldout 取原 `audit_indices`，均必须为唯一升序整数列表；SHA 使用现有 `retrain.persistence.canonical_digest`，即 `sha256(json.dumps(indices, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8'))`，无换行。两者分别必须与 sealed fold 的 `fit_indices_sha256`、`audit_indices_sha256` 精确一致。分数 JSON 文件 SHA 则计算文件原始字节，不等同角色索引摘要。

**两个 score 文件：** 各为单个 UTF-8 JSON 对象，字段精确为 `format='f05-fold-scores-v1'`、`fold`（0–2）、`role`（`train` 或 `heldout`）、`base_manifest_sha256`、`folds_sha256`、`selected_model_sha256`（实际 selected checkpoint blob SHA）、`selected_tensor_sha256`、`selected_variant`（`online` 或 `ema`）、`role_indices_sha256`、`rows`。`rows` 按该角色升序逐行且恰一次，每行仅含整数 `index`、原 manifest 整数 `label`、有限原始 `logit`；不得含概率、dev/官方行、缺项或重复。建议用 `json.dumps(..., sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)` 加单个换行并独占创建，consumer 以实际原始文件 SHA 验证。

**每折父回执：** `format='f05-fold-inference-receipt-v1'`、`status='completed_fit_only_selected_logits'`、`fold`、`round_id`（完整真实轮次）、`base_manifest_sha256`、`folds_sha256`；`training_manifest`、`site`、`science`、`freeze`、`completion`、`checkpoint`、`sidecar` 均是 `{path,sha256}`；`age_contract_sha256` 取该折封存 age 合同 SHA；`selected` 为 `{variant,tensor_sha256,successful_updates}`，checkpoint 文件 SHA 即上述 `selected_model_sha256`。`producer_inputs`、`worker_spec`、`release`、`source_tree_proof` 是 `{path,sha256}`；`rank_proofs`、`train_packets`、`heldout_packets` 均为按 rank 0–7 升序的八个 `{path,sha256}`，沿现有 worker 包字段（rank/device_rank/member/input/proof、batch/offset、原 index/name/label/logit、error、TTA/official 标志）并增加 `fold`、`role`。`scores` 恰含 `train`、`heldout` 两项，各为 `{path,sha256,role_indices_sha256,count}`；`forward_counts` 对两项分别等于对应角色 count，`world_size=8`、`batch_size=8`、`model_mode='eval'`、`view='identity'`、`TTA=false`、`dev_read=false`、`official_read=false`。父回执的自身文件 SHA 由外部固定输入指定，不能来自其自身字段。

**独立验真：** consumer 不把父回执的 SHA 声明本身当成证明。逐项实读并核所有上述文件字节；从原 manifest 和三折封存重新执行 `f05_folds.verify_bound_fold`／`verify_roles`，比对派生 manifest、site/science、完整 train/heldout、group 与 fit-only age；从原 round 的 `SINGLE-MODEL-FREEZE.json`、`TRAIN_RESULT.json`、checkpoint 实际 blob 与 sidecar、原 runtime 严格加载/八 rank proof 及发布 tree 核相同 fold、终态、selected variant/update/tensor，避免将另一个 fold 的自报值当来源。核 producer inputs 与 worker spec/release 相互绑定、八 rank proof/packet 与 producer 输入相连，并用 `parallel_evaluation.batch_assignments`/`canonical_predictions` 校验两个角色的完整 rank 分片、顺序、有限 logit 与标签，再逐行比对两个 score 文件。实现若暂无法独立验证某一证据就拒绝正式消费，而不是把该项写成通过。

**后续标定与目标文件：** 每折 CPU 标定回执 `format='f05-fold-calibration-v1'` 包含 fold/round、base/folds SHA、selected checkpoint/tensor/variant/update、完整 train 角色 SHA、train score 文件 SHA、训练正负数、逐样本权重=1 和总和、固定 sklearn 参数、`coef`、`intercept`、`classes=[0,1]`、迭代次数、`converged=true` 与 sklearn 版本；回执文件 SHA 再进入 `f05-fit-oof-targets-v1` 的 `fold_sources[].calibration_sha256`。每折 heldout 应用**相应**标定函数，父汇总还绑定三个 heldout score SHA 和标定回执 SHA；完整 fit 每行只出现一次。没有真实 selected / 分数 / 已验证 producer 时只允许运行明确标记的纯合成 fixture。
