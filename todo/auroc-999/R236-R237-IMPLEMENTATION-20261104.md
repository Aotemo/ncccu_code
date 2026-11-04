# R236/R237 strong color Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. The user prohibits subagents; checkbox steps track the sole writer.

**Goal:** Add the fixed SSL-only strong color recipe without changing original supervision.
**Architecture:** Extend the existing ImageDataset photometric branch and SSL recipe registry. Reuse the same model, loss, sampler, entry, saving and evaluation.
**Tech Stack:** Existing Python 3.12, Pillow, torchvision and PyTorch in the verified Pod.
**Spec:** [fixed scientific contract](R236-R237-FIT-STRONG-COLOR-20261104.md).

## Global Constraints

- Current R234 source565/SITE c5ad stays frozen; R235 priority after complete strict dual gate.
- Only fit images/CSV; no medical weights, extra data or dev coordinates.
- Same G64/P8, 1600 SSL, original R107 24k/800/p4 supervision and official isolation.
- CPU only in verified Pod, 24 cores0–23/nice19/hidden GPU, disjoint from96 training cores.
- No new training/launch/save/eval/cache/monitor chain or independent GPU rehearsal.

## Review Focus

- Local augmentation RNG must leave global Python/numpy/torch and original geometry draws intact.
- None/identity/gamma and saved supervised job must retain exact original behavior.
- Gray and brightness semantics must agree with independent constant-pixel oracles.
- Strong policy must reject labeled/eval/shared-geometry use, never change original supervision.
- Both seed contracts and all existing accepted SSL recipes must remain valid.

## Task 1: fixed pixels and isolated RNG

Files: `runtime/bundle/retrain/data.py`, `tests/test_ssl_strong_color.py`.
Interface: `_ssl_strong_color(image, augmentation_seed)` returns a nonmutating RGB PIL image; ImageDataset accepts `ssl-strong-color-v1` only for label-free two-view ImageNet training.

- [ ] Write `StrongColorTest.test_pixels_rng_and_SSL_only_consumer`: assert helper exists before calling; gray-only RGB(100,150,200) produces141 in all channels; jittered constant gray100 produces floor(100*sampled_brightness), since the other operations are identity on constant gray. Test deterministic repeated seeds, varied views, no input mutation, global RNG isolation, valid dataset consumer and labeled/eval rejection.
- [ ] Run that target through existing pod_cpu guard on unchanged source; require assertion failure for missing helper.
- [ ] Implement local Random derived with `_seed(seed,'ssl-strong-color-v1')`; .8 gate, shuffle four operations, sample three[.2,1.8] and hue[-.2,.2], use existing torchvision functional operations, .2 grayscale. Consume original eight draws, replace original light photometry only in the new branch.
- [ ] Run target to pass; CPU driver additionally compares old565 and new None/identity/gamma tensors and original geometry for fixed fixtures/seeds. No model forward.

## Task 2: fixed science and original SSL loop

Files: `runtime/bundle/retrain/ssl_stage.py`, `runtime/bundle/retrain/experiment91.py`, two SCIENCE configs and the same test file.
Interfaces: `STRONG_COLOR_RECIPE=dict(NTXENT_RECIPE,version='fit-ntxent-strongcolor1600-v1',training_photometric='ssl-strong-color-v1')`; `FIT_NTXENT_STRONG_COLOR_SMALL_BINDING` and seed27 counterpart/group.

- [ ] Write `test_recipe_contract_and_original_streams`: assert recipe exists; compare full R107 job, R234 NT-Xent math/schedule and all R115 learning/sampling streams; require R236/R237 orders151/152 and fixed seeds, supervised training_photometric=None; reject altered recipe; validate existing accepted SSL recipes.
- [ ] Run target on unchanged source; require assertion failure for missing recipe.
- [ ] Register two bindings/recipe; select same existing NT-Xent path; pass recipe policy only into the existing SSL ImageDataset. Record its actual consumed policy in new-policy metrics/result only. Generate two configs from original R234/R235 using existing registered contracts in Pod.
- [ ] Run two new targets and four affected-file compilation; perform actual fit8/two-view decode with strict fit membership and 8×2×3×512×512 finite tensors. Compare unchanged model/loss/gather/optimizer/derived scopes to565 and reuse original P8/native proofs; no repeated model forward.
- [ ] Commit/push/API-read; along existing prepare/check_candidate/freeze_deployment build exact technical/formal packages and fill original entry text during R234 training. Activation remains conditional on complete gates and unique actualsafe.
