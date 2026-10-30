# R113 — freeze early Small features

Actual Git `7dc31c6c7995406e44df7511d253201d22028945` / tree `01dc0e65626a59e6ad4aedd8d199f60d120129f5`. The direct matched control is [R107](../R107/README.md). The sole intervention was `job.freeze_features_before=4`: parameters in `network.features[:4]` had `requires_grad=False`; later features, classifier and age parameters remained trainable. Module training modes, stochastic depth .1 and ordinary forward execution were retained. Small22K public initialization, seed20261029, single512 view, scale .85–1, global pairwise logistic loss, G64/P8, learning rates, fixed9:1 split and 800/24000/1600/patience4 were unchanged. No trained checkpoint, N06 or FLAIR parameters were used.

The actual complete fresh tensor digest remained `50c6f55c53f5656a2daff9f9bc1b83f4a34c821bf9a377aedf8c3462e0f027b5`. Separately, the formal CPU and eight-rank initial proofs checked all345 parameter tensors /49,455,459 elements, with62 frozen tensors /1,235,040 elements and283 trainable optimizer members /48,220,419 elements. Adam was initially empty. Frozen names were taken from the online model, and EMA copied those parameters on every successful update. Equal tensor initialization alone was not treated as proof of the mask or optimizer membership.

Training completed normally at6400 successful=attempted updates,0 skips and8 complete cycles. Cycles5–8 did not improve the internal AUROC reference, so patience4 stopped the run. The original selector uniquely chose **c4 ONLINE@3200**: internal AUROC **0.9838264352051979**, AP **0.7807106637265128**, error_twice **26864**. All16 online/EMA internal entries are retained in `SINGLE-MODEL-FREEZE.json`; `TRAIN_RESULT.json` records exit0, hard deadline met and no signals. The best EMA point was c5@4000, AUROC0.9822117277610474/AP0.8041990356906674; it does not replace the original selection.

The selected result did not strictly exceed R107's c4 EMA@3200 AUROC0.9884405731067525/AP0.8274326888154083. This particular early-feature freeze intervention is unsupported by the pre-registered main criterion. The branch stops: no seed27 confirmation and no freeze-depth or learning-rate scan. This result does not establish that every representation-preservation method fails or identify a causal explanation for the difference. The limited conclusion is also recorded in [the frozen decision](../../todo/auroc-999/R113-DECISION.md).

Existing timing counters were training1867.338s, data wait4.805s, paired evaluation144.175s and checkpoint98.813s; train component wall2334.634s. These are observed counters, not a new benchmark or a causal speed comparison.

Separately, the original single official5849 evaluation completed once: AUROC0.9832808662709891, AP0.7836110512813446, component748.660s/exit0. These values did not select the recipe or checkpoint. `ROUND_RESULT.json` preserves the original aggregate comparison and completed round state.

At2026-10-30 10:01:07 Beijing, the normal read-only handoff confirmed ROUND/done completed, all eight ranks inactive, devices idle with no users, and the original instance lock released. Owner187146 was stateZ and is not claimed to have been reaped. The unchanged handoff evidence remains at `exp-v3/technical-prep/r113-frozen-7dc31c6c7995/PROGRESS-20261030-1000.json`, original Pod SHA `3cd524295d1b57fa6c8d4265608d4792f9b27a67fedd60a6b58cafa9f66e26bc`. No active training remains from this run.

Original Pod SHA pins for the unchanged aggregate records copied here:

- `SINGLE-MODEL-FREEZE.json`: `45b9adc3710082468fe38748df36cabfc0f271f70d906a0924e30b0a6bf15c40`.
- `TRAIN_RESULT.json`: `888d977cc759dffbeb43d14bdbb1671927efb41db55d81cc97385cfccf38f3f4`.
- `ROUND_RESULT.json`: `356251621a7ecfb2ce1a572add4d8c6481b39044644ee78a09b154769d758291`.

Only this concise result and the three aggregate originals are archived. Weights, images, sample-level predictions and submission CSVs remain outside Git. This archival step performs no new Pod computation or evaluation.
