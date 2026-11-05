# F04 test submission (2026-11-05)

Delivered `/home/hao/Desktop/challenge/submition/F04-fusion/submit.csv`
(`Image_name,Probability,Pred_Label`, 5856 lines = header + 5855, `test_00000.jpg` order,
CRLF as the R120 reference). SHA256 `3a9dc5b73d763ed048bf1127fbe367e3ad9d6446f3ccc7835081a7d68e24d2a6`.
Metadata beside it in `METADATA.json`.

## Members

Five F04 frozen members, exported fp16 weights, `ema` variant, fp32 forward, single view, no TTA:
R120, R134R1, R250, R249, R135 (`records/analysis/20261104-f04/F04-FINAL-FREEZE.json`).
Each ran over all 5855 unlabeled test rows on its own DCU (GPU 0–4) through the R120
`inference.run_prediction` path; raw logits were captured by wrapping `inference._forward`.

## Fusion

Per member, z-normalize test logits with that member's internal-dev logit mean and population
std (`selected.prediction` dev-cycle CSV), then equal-weight average. Probability is a univariate
logistic (Platt) of the fused dev score: a = 1.9347664660965742, b = -6.13690213106862, fit on the
5140 dev rows (167 positive). `Pred_Label = Probability >= 0.5`; predicted positive rate 201/5855
= 3.43%. No official labels and no test-derived tuning.

## Verification

- 5855 rows, `Image_name` order byte-identical to `R120-98.45/submit.csv`, 0 out-of-range
  probabilities, label consistent with the 0.5 threshold.
- Spearman(F04 probability, R120 probability) on test = 0.9462923643342195.

## Runtime incident

The first launch used default torch threading (128) in five concurrent processes; after ten
minutes there was no CSV output and heavy CPU contention. Those five processes were killed
(logs kept as `run2.log`), and the same driver was relaunched with `torch.set_num_threads(8)`,
`OMP/MKL/OPENBLAS_NUM_THREADS=8` per process (`run3.log`). All five members then completed in
about 13 minutes. No model, preprocessing, precision or view setting changed between runs.
