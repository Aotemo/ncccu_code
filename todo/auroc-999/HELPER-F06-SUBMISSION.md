# HELPER F06 submission writer (2026-11-05)

新增 `tools/diagnostics/f06_oof_submission.py` 与 `tests/test_f06_oof_submission.py`（helper/f06-submit，
未合并 main）。复用 `fusion_submission.py` 的 `load_scores/moments/standardise/fuse/fit_platt/
apply_platt/_metrics/check_submission/sha`，不另建平台。

## 输入

- `FOLDS.json`：每行 `{"name","label","fold"}`；接受 JSON 数组、`{"rows":[...]}` 或 JSONL（按行）。
  非 fixture 模式要求恰好 5849 行、3 折、每折含两类、名字唯一。
- manifest（`f06-oof-members-v1`），三折各一条，所有文件按声明 SHA 校验：

```json
{"format": "f06-oof-members-v1",
 "folds": [
   {"fold": 0, "score_column": "logit",
    "oof":  {"path": ".../fold0-heldout.csv", "sha256": "..."},
    "test": {"path": ".../fold0-test.csv",     "sha256": "..."}},
   {"fold": 1, "...": "..."},
   {"fold": 2, "...": "..."}]}
```

- 参考顺序：`/home/hao/Desktop/challenge/submition/R120-98.45/submit.csv`
  （表头必须为 `Image_name,Probability,Pred_Label`，只读名字列，不读分数）。

## 逻辑

1. 每折留出 logit 用该折自身 mean/总体 std 做 z（常数写入 METADATA），按 FOLDS 行序拼成 5849 行；
   逐折校验覆盖、标签与 FOLDS 一致、无重复/非有限值。
2. OOF AUROC/AP（tie-aware，同 `fusion_submission._metrics`）与每折指标。
3. `decision = F06` 当且仅当 OOF AUROC **严格大于** `0.9894371866550443`（F04 官方），否则 F04。
4. F06 时：三折 test logit 用同一 z 常数标准化 → 等权平均 → 在 OOF z 与标签上拟合 Platt
   （`C=1e6`，`random_state=20261105`）→ `Probability`，`Pred_Label = p >= 0.5`。
5. 输出目录必须不存在：写 `submit.csv`（CRLF，顺序同参考文件）与 `METADATA.json`
   （输入 SHA、z 常数、Platt、OOF/每折指标、decision、`official_read=true` 且
   `official_read_scope=oof_decision_only`、`TTA=false`、`test_label_values_read=false`）。
   decision=F04 时只写 METADATA，不产生 submit.csv。

## 用法

```bash
python3.12 -B tools/diagnostics/f06_oof_submission.py \
  --folds /path/FOLDS.json --manifest /path/F06-OOF-MEMBERS.json \
  --reference /home/hao/Desktop/challenge/submition/R120-98.45/submit.csv \
  --output /public/home/acfwvv9ejn/SothisAI/exp-v3/f06-submission-<id> \
  --threads 6
```

测试 logit 尚未到时先判 OOF：`--oof-only`（不需要 `--reference`）。合成数据测试：

```bash
python3.12 -B -m unittest -v tests.test_f06_oof_submission
```

## 延迟（T4，2026-11-05 追加）

`tools/diagnostics/latency_bench.py` 复用提交链：模型由 `retrain.inference._load_model` 构造，
JPEG 由 `retrain.data.ImageDataset` 按 `run_prediction` 口径解码/裁边/缩放/letterbox/归一化
（显式关闭 prepared cache，与原图解码一致）。每批只预处理一次，成员共享；各成员前向用
`torch.cuda.synchronize` 括起计时；等权平均 logits 后 sigmoid。batch=10、预热后重复 ≥5 次取中位数，
分列 CPU 预处理 / 每成员 GPU / H2D / 融合 / 总每张 ms。FLOPs 用
`torch.utils.flop_counter.FlopCounterMode`（不可用记 `unknown`）。

估算：`t4_gpu_ms = flops / (effective_TFLOPS·1e12)·1000`（默认 fp16 12、fp32 4.5，写入 assumptions），
另一路 `dcu_ms = 实测 GPU ms × --dcu-t4-ratio`（默认 1.0，假设值）。总估算 = GPU 估算 + 实测 CPU 预处理
（`cpu_note` 注明 T4 主机 CPU 可能不同）。判定 `估算×1.5 < 100 ms`，两路已知估算都需通过；并给出
1/2/3 成员子集的最省估算。

成员清单：

```json
{"format": "f06-latency-members-v1",
 "members": [{"id": "m0", "variant": "ema",
              "model": {"path": "<export dir 或 checkpoint>", "sha256": "<该权重文件 SHA>"}}]}
```

```bash
python3.12 -B tools/diagnostics/latency_bench.py --members LATENCY-MEMBERS.json \
  --images <仅含 jpg 的目录> --device cuda:0 --precision fp16 --batch-size 10 \
  --repetitions 5 --t4-tflops 12 --dcu-t4-ratio 1.0 --threads 6 --output <fresh>/LATENCY.json
```

`--images` 只指向解包后的图片目录；`cost_data_test.tar` 同目录的 SSH 密钥/账号文件不读取、不复制。

提交判定接入：`f06_oof_submission.py --latency-report LATENCY.json`。若 OOF 选 F06 但报告 `pass=false`，
`decision=needs_reduction`、不写 submit.csv，METADATA 记录 3/2/1 子集估算；OOF 本就不足 F04 时仍为 F04。

## 阻塞（如实记录）

本机静态检查仅 `py_compile`/AST 通过；Pod CPU 内的 RED→GREEN **未执行**：
`exp-v3/technical-prep/` 下仍没有发给 `helper-f06-submit`、覆盖 mask 18–23 且未过期的 allocation
（ds2 最新仍为 1600Z 的 `6e2460bb` 0–11 与 `5ae33039` 12–23，均非本执行者）。按合同不自签、不复用他人
allocation；待主执行者签发后在其掩码内以 6 线程、nice 19、隐藏 GPU 运行。GPU 实测（cuda）由主执行者执行。
