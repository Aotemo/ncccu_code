# F05 三折成员发现：helper 合成准备

本节点仅新增 `tools/diagnostics/f05_fold_candidates.py` 与直接合成测试，复用现有 F03、`paired_evidence._selected`、`f05_final_selection.validate_candidate_population` 与 `verify_bound_fold`；未运行真实发现、选员、融合或模型前向。08ac 审阅合入后以最终 main 全树做集成，正式执行前先封存完整候选及规则。

## 接口与血缘

- `discover(f03_ref, work_root=..., allow_fixture=False)` 输入真实文件 `{path, sha256}`，复核原 F03 字节、原始 manifest 与 dev 身份；原库存及其所有原有成员保持原样。输出是新扩充库存，保留原 `status` 以兼容现有选员，同时显式写入 `artifact_kind=f05-augmented-candidate-discovery-v1`、`augmented=true` 和 `source_f03_inventory` 的实际路径/SHA，不能把扩充件称作原 F03 文件。
- R265–R267 各有一行，包括缺失或无法核验时的 unknown。现有 F03 中的三折 unknown 原样保留，不自动升级或重复。折成员先独立核同轮/site 两份 SCIENCE、SITE/RUNTIME、原始及派生 manifest、封存折角色和原 dev 行与索引；通过后以派生 manifest SHA 绑定 `paired_evidence._selected` 的真实 selected、预测及精度证据。产生既有选员兼容的 `fold_binding={fold,site,runtime_binding,training_manifest}`。
- 同轮 checkpoint 完整枚举后，唯一匹配为 `prediction_verified_weight_pending`、零匹配为 `selected_checkpoint_missing`、多匹配为 `selected_checkpoint_ambiguous`。侧车缺失、枚举不完整、折证据不符等均为 unknown，不能当作已知缺失。
- 仅在三个已登记折的派生 manifest SHA 均独立核明时，对其他 completed 轮可凭同轮可核 freeze 声明的合法 SHA 非原始且非三个登记 SHA，明确记为 `freeze_declared_training_manifest_mismatch`；不打开、不声称验证该无关 manifest。缺失/损坏 freeze 或来源、含注册 SHA 的错绑，均为 unknown。
- `verify_augmented(augmented_ref, work_root=..., allow_fixture=False)` 同时核新库存及原 F03 的真实字节/SHA、显式标识、原字段/成员及三折恰一次，并重新只读扫描比较完整扩充结果；候选或 unknown 在封存后变化则拒绝。该复核只读取既有内部 dev 预测，不发前向。**现有选员入口尚未自动调用该函数；08ac 须在消费扩充库存前接线，不能直接把未验证的 dict 送入选员。**
- `allow_fixture=True` 仅用于带明确 fixture 标记的合成测试；生产默认 false，原 manifest SHA 与真实 5140 dev/167 positive 固定。目录枚举整体失败会中止该次发现，不产生可冻结库存。

## 验证与剩余接入

B Pod 独立副本，12–23 互斥 mask、隐藏设备、nice 19：6/6 合成通过，guard 回执 SHA `230f69aa27dad3c5fce0fe1bfa07c329ff2a710ec8a2848c78614af049a0230f`。覆盖三折科学血缘和预测复核顺序、原 F03 unknown 保留、缺/多匹配及缺侧车、三折齐全前拒绝排除无关 manifest、原库存 SHA/新增 unknown 被更改或删除的拒绝，以及 `require_fold_candidate_census` / `build_candidate_inventory` / `screen_candidates` 兼容。源码 SHA：helper `0e8c0ad4f41937a05146cca20096a6a8a18730d62a79c106a641823f9e823f10`，测试 `5d3ea3c4fcde359e4c3aee126e3a12f4cdcfea3143d88a30a4fe3834811ce40e`。

主线还需整合已推 helper 的 `f05_final_selection.py`（如最终 main 尚未包含）及本 helper，并在实际选员消费入口强制 `verify_augmented`；同时保护现有 F03 原件与 owner 负责的八 rank/activation 代码。真实文件结构或选点身份不符时保留 unknown，不修写旧回执、不以合成结论代替实际可部署证明。
