# R112 — fit-only fixed-reference BN paired treatment

Actual Git `89712c91413f631ca6ac6b40db8ccce0248c5f6a` / tree `db9c18b83d8e7f383c68986f326503bee82f2313`. Direct matched control is [R111](../R111/README.md). Both use independent public ImageNet1K V2 ResNet50 fresh parameters, training seed20261029 and existing head seed20261027; fixed9:1, ImageNet-normalized512, G64/P8, single view, BCE, learning rates and800/24000/1600/patience4 remain unchanged. No R111 trained checkpoint, N06 or FLAIR parameters were used.

The sole intervention was the pre-registered `fit-bn-fixed-reference-v1` calibration: one eight-rank pass over46256 fit images,1448 local physical batches in total, collecting FP64 pooled input moments while every BN still used its original public eval statistics. Only after the complete pass were106 FP32 running mean/variance buffers replaced together. All named parameters and53 counters remained unchanged; all159 BN buffers were then fixed during training. This is not iterative AdaBN or a self-consistent population estimate for the network after replacement. The calibration receipt, rank exits and actual initial controls are retained in [the preparation record](../preparation/r112-fit-bn/README.md).

The matched parameter digest is `2207cf41c2e0cb9c41bc50dd56a56b2b82ca5fb7e3feca76887f1af00711fbf0`. R112's actual derived whole initial digest is `806e4c872e2369c69aba23bbed0d1c51438e3d02806a1e5a106c8bcb78c43030`, with BN digest `6c088d16941699e9ee89d6f39c455edd33f941432428ee8434fc76ec284318fc`. The derived106-buffer asset remains in exp-v3; no tensor asset is stored here.

Training completed normally at10400 successful=attempted updates,0 skips and13 complete cycles. Cycles10–13 did not improve the internal AUROC reference, so patience4 stopped the run. The unique internal selection is c9 EMA@7200: AUROC0.6869135246498758, AP0.06838528019747084, error_twice520031. All26 paired internal entries are preserved in `SINGLE-MODEL-FREEZE.json`; `TRAIN_RESULT.json` records component exit0, hard deadline met and no signals. Existing timing counters are training2113.123s, data wait707.928s, paired evaluation235.174s and checkpoint86.270s; train component wall2707.026s. These are observations, not a new benchmark. The initial very large finite loss fell to approximately0.314–0.402 in the final five logged updates, without recovering the main metric.

The internal result is substantially below R111's selected c5 EMA@4000 AUROC0.9754398301727533/AP0.757855599765702. This finite matched experiment does not support this one-pass, fixed-public-reference, all-layer BN replacement rule. It does not establish that every BN adaptation method fails, identify FLAIR's causal contribution, or justify another calibration/seed/algorithm scan. The authorized two fresh runs and one fit calibration are complete; no third run follows automatically.

Separately, the original single official5849 evaluation completed once: AUROC0.6971095923114934, AP0.07311415022835291, component757.240s/exit0. These values did not select the recipe or checkpoint. `ROUND_RESULT.json` preserves the original aggregate final comparison and completed round state.

At2026-10-30 03:30:57 Beijing, the normal read-only handoff confirmed the round and done state completed, all eight ranks inactive, devices idle with no users, and the original instance lock released. Owner178176 was stateZ and is not claimed to have been reaped. The unchanged handoff evidence remains at `exp-v3/technical-prep/r112-frozen-89712c91413f/PROGRESS-20261030-0330.json`, original Pod SHA `b367c148021d3e9169109b58427434c4613674a8d50b732d6f1dba5fe2b311c4`.

Original Pod SHA pins for the unchanged aggregate records copied here:

- `SINGLE-MODEL-FREEZE.json`: `b66e2067ad45e79b8e9daa3870ab1e9e98c12744fd829a12c13de7e14ffd6cd3`.
- `TRAIN_RESULT.json`: `4147ba7928412bafd49e125087dff3cadb2a00c632287efe9de57d60c09242da`.
- `ROUND_RESULT.json`: `2c8491dceef27b33dccbf2a79ed3da29a3eda43c5da3100220048b9631174252`.

Only this concise result and the three aggregate originals are archived. Weights, images, manifests containing sample rows, per-image predictions and submission CSVs remain outside Git.
