# F06 official fine-tuning result and the four-member F04 submission (2026-11-05)

## F06 (official 5849 three-fold fine-tuning)

R120 selected EMA (`f21bffae…`, file `2ac4a025…`) warm start, fit 46,256 plus the two
non-held-out official folds, held-out official fold for selection; G64/P8, 400-update
cycles, 6,000 cap, patience 4, LR 1/10.

| fold | updates | selected variant | held-out AUROC | held-out AP |
|---|---|---|---|---|
| 0 | 2000 (early stop) | ema/online (tie at 400) | 0.985775 | 0.848684 |
| 1 | 3200 (early stop) | online at 1600 | 0.991260 | 0.836478 |
| 2 | 3200 (early stop) | ema at 800 | 0.985566 | 0.845813 |

Honest OOF (each row scored only by the fold model that never saw it, per-fold z):
AUROC **0.9876429**, AP **0.8431928** (5,849 rows, 191 positive) versus F04's official
0.9894372. F06 therefore does **not** replace F04; no F06 submission was produced.

## Latency (T4 estimate; end-to-end decode + members + fusion)

- 5 MP JPEG decode dominated a serial pipeline: ~107 ms/image. Bit-identical
  batch-parallel decode (4 threads, asserted equal to serial per batch) reduces it to
  ~32 ms/image and is now the production `run_prediction` default (`decode_workers=4`).
- fp16 end-to-end, x1.5 safety: one member 60.8 ms, three members 82.7 ms, four members
  **94.0 ms**, five members 105.4 ms (fail).
- fp32 end-to-end: only a single member passes (78.7 ms); four members 168.9 ms.

## Four-member submission

The official 5849 labels were authorized for validation, so the final member set was
chosen by the mean of internal-dev and official AUROC among subsets that pass the fp16
latency budget. Four members R120 + R134R1 + R250 + R249: official
**0.989743 / 0.853533**, dev **0.989934 / 0.8410**, mean AUROC 0.98984 (five-member
0.98985), fp16 62.7 ms raw / 94.0 ms with safety.

- z constants: the frozen F04 internal-dev logit moments per member.
- Platt: dev-fused score, same deterministic Newton fit as the five-member file
  (a=1.913952294801697, b=-6.102647497904602); threshold 0.5.
- Weights: four fp16 member files, 396,073,012 B total (<= 500,000,000).
- fp16 vs fp32 forward on 15 cost images: max |Δlogit| 0.215 (R250), max |Δz| 0.0145;
  with the Platt slope this bounds |Δp| < 0.007. The stored test logits are fp32 forward,
  so the file is conservative relative to the declared fp16 production forward.
- Output `/home/hao/Desktop/challenge/submition/F04-4member/`: `submit.csv` SHA256
  `ccf23fe0e9e1079d4be9c9974da8952051410e36277c28fbc91dc4a4f52a3132` (5,855 rows,
  R120 order, CRLF, 200 positives) and `METADATA.json`.

Production inference is pinned by `configs/F04-4MEMBER-INFERENCE.json`
(four members, fp16, `decode_workers=4`, z constants, Platt, latency assumptions).
