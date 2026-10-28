本材料只准备 R94 之后不依赖结果的接线；没有创建新正式包、clock、token 或启动。基线为已发布 R94 `212b41f9d3aa922519e62182701bfbea527c852d` / tree `662c2c2d6c05cc185994d5693eb5eec0b7078689`。用户随后明确下一 fresh 计划最多 24000 成功更新、连续 4 个完整验证轮无提升早停：800/轮，最多 30 轮，patience=4；warmup/EMA warmcopy 均保持绝对 1600，cosine horizon=24000。这些预算已写入模板，不再默认旧 6400/no-early-stop，也不再称 LR-only。R94 冻结运行仍按原合同结束。不用官方结果选择参数，不据 R94 中途指标定稿。

已准备的三件模板：

- [NEXT-SCIENCE.json.in](../../../configs/templates/NEXT-SCIENCE.json.in) 以 R94 为对照，预填用户新定的 24000/30/4 预算和已答复的内部 dev AUROC 监控，字段与 A 的 `R95-SCIENCE.json` 一致；仅 profile、比较说明保留显式 JSON slot，不把通用模板当已发布科学配置。其余科学数值保持 R94，`backbone_lr=1e-4` 不是 B 新选超参。slot 未填时不是合法 JSON，不能交给 prepare。R95 实际执行直接用 A 的版本化 `configs/R95-SCIENCE.json`，无需从模板再造另一份。
- [NEXT-IDENTITY.patch.in](../../../configs/templates/NEXT-IDENTITY.patch.in) 只列出现有 `experiment91.py` 的 `BINDING_VERSION`、`CANDIDATE.id`、`experiment_order` 三个同步身份位置；占位补丁未应用。最终 profile 与科学 JSON 一致，round_id 由入口显式输入。当前没有接口代码 delta，也没有预选下一轮编号或科学值。
- [NEXT-RUN.env.example](../../../configs/templates/NEXT-RUN.env.example) 已填 A 确认的 R95 profile/id/order 和科学配置路径；当前环境、份额、round_id、新 Git、TTL 留真实待绑定 slot。它是填写清单，不能原样 source 或执行。当前 B 准备目录不是正式 run 身份。

用户已直接选择“内部 dev AUROC（推荐）”，监控待答状态关闭：`metric=internal_dev_auroc, mode=max, variants=[online,ema], min_delta=0, min_cycles=1`，每个完整周期取两者最大 AUROC、首轮建立 baseline、严格大于 best 才算提升、同值算无提升；每轮只计一次，连续四轮无提升停止。新运行 CLOCK_POLICY 使用 A 的 train=10800 秒、total=16200 秒；由新源码复制/绑定，不新增 env 覆盖接口。早停 S<24000 是正常 completed，覆盖每个实际完成轮的 online/EMA 两份内部汇总后唯一选点；原 R94 的 6400/16条终态条件不适用于新计划。

已有源包与证据：

- A 的权威冻结源：`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/r94-frozen-212b41f9d3aa/source`，301 个 Git 文件。既有 `DEPLOYMENT.json` SHA256 `9e3489881a3a3eca665b187586fcf72d0d0999fa6f651710bdf55055ff7dd068`；Git blob、文件集合和 mode 已由 A 验证。
- 已将本机 `work/exp-v3-8card-review-20261028/r94-transfer/R94-DEPLOY.tar` 普通字节传输到独立 `exp-v3/technical-prep/next-round-ready-20261029-b/R94-DEPLOY.tar`，未重建 212b。但 B 静态读取该暂存 tar 时实际报 `tarfile.ReadError: unexpected end of data`、guard exit1，尚未证明这份暂存包完整，不能作为可用包交付。原因未进一步诊断，不把传输命令 exit0 当内容通过。见同目录 [PACKAGE-RECEIPT.json](PACKAGE-RECEIPT.json) 的派生失败摘要及其真实 Pod guard 原件 pin。A 的既有 301 文件冻结源/DEPLOYMENT 证据保留有效，本项归档不作为 R95 同步或启动前置；按最新收口指令停止补查。未来包内非 source 的旧调用/remote 元数据只作参考；Git 自带 records 不继承为新 CPU/SITE 证据。
- 新科学/身份定稿后由唯一 A 生成新 Git archive：沿既有 Git 文件清单、`COPYFILE_DISABLE=1`，排除 `._*`、`.DS_Store`、`__MACOSX`；只传新源码及明确的 remote 元数据，不夹入旧 formal-cpu/package、clock/token、运行 SITE 或 CPU evidence。212b 包不能冒充未来新 Git。

权威成功调用只读参考：本机 `work/exp-v3-8card-review-20261028/r94-transfer/{PREPARE-R94.py,ACTIVATE-NOW.py,ACTUAL-CONTINUOUS-CALL.txt}`；Pod 同一 R94 冻结根下有前两个原件。下面只是把其中既有入口参数换成 NEXT 变量，不执行旧脚本，不增加独立零 forward GPU 预演。

依赖顺序：A 完成用户指定 R95 的新源码合同与配置接线 → 完整内部结果归档及新 round 身份定稿 → A 唯一提交/同步并实际读回 Git commit/tree → 新隔离冻结源 → 当前 UID/Ready、解释器、validation allocation/guard/policy 与训练互斥池 → package-only → formal site → 现行 CPU 包 → 实际 Git 部署回执 → 新 preflight → 一次 activation → 一次真实训练。R94 安全终态及 GPU 独占交接由 A 核实；B 不提前 activation 或 launch。真实训练沿现有 8rank fresh/参数组/空 Adam/资源校验。

以下 `CPU` 是已有本机传输入口数组，所有 Python、SHA/JSON 读取仍在已核 Pod 的 CPU 通道；由 A 填全并 export env 清单后使用。`SHA256_...` 均从本次 Pod 输出实得，不在本机算，不复用 R94 digest。

```bash
CPU=(bash "$NEXT_LOCAL_REPO/tools/environment/pod_cpu.sh" run \
  "$NEXT_SSH_TARGET" "$NCCCU_POD_NAMESPACE" "$NCCCU_POD_NAME" \
  "$NCCCU_POD_UID" "$NCCCU_PYTHON")
"${CPU[@]}" "$NEXT_SOURCE/tools/activation/prepare.py" \
  --source-root "$NEXT_SOURCE" --science-config "$NEXT_SCIENCE_CONFIG" \
  --asset-receipt "$NEXT_SMALL_RECEIPT" --asset-receipt-sha256 "$NEXT_SMALL_SHA256" \
  --output "$NEXT_FREEZE_ROOT/technical-package" --work-root "$NEXT_WORK_ROOT" \
  --round-id "$NEXT_ROUND_ID" --git-base "$NEXT_GIT" --pcis "$NEXT_PCIS"
"${CPU[@]}" "$NEXT_SOURCE/tools/activation/prepare_gpu_site.py" \
  --technical-site "$NEXT_FREEZE_ROOT/technical-package/package/SITE_BINDING.json" \
  --technical-site-sha256 "$SHA256_TECHNICAL_SITE" \
  --relocation-receipt "$NEXT_RELOCATION" --relocation-receipt-sha256 "$NEXT_RELOCATION_SHA256" \
  --validation-cpu-allocation "$NCCCU_CPU_ALLOCATION" \
  --validation-cpu-allocation-sha256 "$NCCCU_CPU_ALLOCATION_SHA256" \
  --native-guard-receipt "$NEXT_NATIVE_RECEIPT" --native-guard-receipt-sha256 "$NEXT_NATIVE_RECEIPT_SHA256" \
  --resource-policy "$NEXT_RESOURCE_POLICY" --resource-policy-sha256 "$NEXT_RESOURCE_POLICY_SHA256" \
  --git-commit "$NEXT_GIT" --remote-commit "$NEXT_GIT" \
  --output "$NEXT_FREEZE_ROOT/gpu-inputs" --valid-seconds "$NEXT_GPU_VALID_SECONDS"
"${CPU[@]}" "$NEXT_SOURCE/tools/validation/check_candidate.py" \
  --source-root "$NEXT_SOURCE" --science-config "$NEXT_SCIENCE_CONFIG" \
  --asset-receipt "$NEXT_SMALL_RECEIPT" --asset-receipt-sha256 "$NEXT_SMALL_SHA256" \
  --site-binding "$NEXT_FREEZE_ROOT/gpu-inputs/SITE_BINDING.json" \
  --site-binding-sha256 "$SHA256_GPU_SITE" --output "$NEXT_FORMAL_PREP"
"${CPU[@]}" "$NEXT_SOURCE/tools/activation/freeze_deployment.py" \
  --source-root "$NEXT_SOURCE" --remote-tree "$NEXT_FREEZE_ROOT/REMOTE-TREE.json" \
  --remote-tree-sha256 "$SHA256_REMOTE_TREE" --tree-sha "$NEXT_TREE" \
  --git-commit "$NEXT_GIT" --remote-commit "$NEXT_GIT" \
  --remote-commit-json "$NEXT_FREEZE_ROOT/REMOTE-COMMIT.json" \
  --remote-commit-json-sha256 "$SHA256_REMOTE_COMMIT" \
  --prep-root "$NEXT_FORMAL_PREP" --package-manifest-sha256 "$SHA256_PACKAGE_MANIFEST" \
  --output "$NEXT_FREEZE_ROOT/DEPLOYMENT.json"
```

在控制面用新 Pod 实物输出 `POD-LIVE.json`，再按实际 owner 元数据情况沿已有 preflight 参数补 `--owner-live/--owner-live-sha256`；不在登录节点解析 JSON。重新核 UID/Ready/Running 和当前时间后，仍通过 CPU 通道执行：

```bash
"${CPU[@]}" "$NEXT_FORMAL_PREP/site_preflight.py" \
  --prep-root "$NEXT_FORMAL_PREP" --site-binding "$NEXT_FORMAL_PREP/package/SITE_BINDING.json" \
  --site-binding-sha256 "$SHA256_FORMAL_PREP_SITE" \
  --pod-live "$NEXT_POD_LIVE" --pod-live-sha256 "$SHA256_POD_LIVE" \
  --observed-at "$NEXT_OBSERVED_AT" --output "$NEXT_FREEZE_ROOT/SITE-PREFLIGHT.json"
"${CPU[@]}" "$NEXT_FORMAL_PREP/activate_audit.py" --activate-formal \
  --prep-root "$NEXT_FORMAL_PREP" --authorization-reference "$NEXT_AUTHORIZATION_REFERENCE" \
  --site-binding "$NEXT_FORMAL_PREP/package/SITE_BINDING.json" \
  --site-binding-sha256 "$SHA256_FORMAL_PREP_SITE" \
  --package-manifest-sha256 "$SHA256_PACKAGE_MANIFEST" --cpu-prep-receipt-sha256 "$SHA256_CPU_PREP" \
  --deployment-receipt "$NEXT_FREEZE_ROOT/DEPLOYMENT.json" --deployment-receipt-sha256 "$SHA256_DEPLOYMENT" \
  --site-preflight "$NEXT_FREEZE_ROOT/SITE-PREFLIGHT.json" --site-preflight-sha256 "$SHA256_SITE_PREFLIGHT" \
  --pod-uid "$NCCCU_POD_UID" --ready True --observed-at "$NEXT_OBSERVED_AT" \
  --audit-root "$NEXT_FREEZE_ROOT/activation-audit"
```

新 activation 成功退出后，A 重新获取 UID/Ready/Running/observed_at，在显式 GPU 通道启动新 formal 下 `launch_once.py`，参数沿现有入口：

```bash
"$NCCCU_PYTHON" -B -u "$NEXT_FORMAL_ROOT/launch_once.py" \
  --site-binding "$NEXT_FORMAL_ROOT/SITE_BINDING.json" --site-binding-sha256 "$SHA256_FORMAL_SITE" \
  --pod-uid "$NCCCU_POD_UID" --ready True --observed-at "$NEXT_OBSERVED_AT"
```

最后这条只在 Pod 内的 A GPU 通道运行，不通过 `CPU`：`CUDA_VISIBLE_DEVICES`、`HIP_VISIBLE_DEVICES`、`ROCR_VISIBLE_DEVICES` 显式为新 site 顺序；native 环境逐项读取新冻结 `RESOURCE-POLICY.json.native_affinity_environment`，不盲填。实际 R94 连续调用中的 exp-v3 编译缓存变量原样沿用其依赖目录，不复制旧 round 控制文件；`PYTHONDONTWRITEBYTECODE=1`、nice19 及当前身份 env 保留。先读取本次 `FORMAL-BINDING-AUDIT.json` 的成功、新 Git 身份再 exec，不能把 spawn 当首步成功。

已备齐的是模板、三个身份位置、既有入口顺序、A 已验证的基线源树引用及不可变依赖引用；B 暂存 tar 未通过，不能冒称包已备齐。本次 B 未改接口。真实接口差异是：212b 的 `experiment91.validate_job`/`prepare.build` 等仍核旧 S800/6400/1600，不能把新预算交给旧源码执行；A 独占适配 24000 上限、4 轮早停及 completed/freeze 条件。唯一有效 R95 身份为 profile `single-tiny22k-bc48a87f-fixed-9010-24k-earlystop-v1`、candidate `r95-convnext-tiny-imagenet22k-to-1k-24k-earlystop-9010`、order5；不混用曾拟的其它名称。A 的 R95 源码已经更换这三个值时，不再套用基于 R94 preimage 的占位补丁，B 实际 runtime delta=0。既有命令入口参数接线保留，不建新框架。A 已报告其增量 13/13 GREEN 与实际 prepare 生成通过；B 不重复这些检查。剩余是唯一 A 的新 Git 同步、新 round_id，以及新身份必然生成的 source/config/package digest、当前资源 TTL/site/preflight、一次 clock/token。它们不能提前填成完成。B 本窗口未运行模型、native 重验、旧测试或官方评估，命令已结束，无在跑 B CPU 任务；五件材料停止修改，交 A 唯一集成。
