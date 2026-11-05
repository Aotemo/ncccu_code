# F06: official-5849 fine-tuning, 3 folds (registered 2026-11-05)

Authorization: user permits the official 5849 (191 positive) to be used for fine-tuning and
in-fold validation from 2026-11-05. F04 official result and one-shot claim remain unchanged.
Final submission still forbids TTA, keeps total fp16 member bytes <= 500,000,000, and forbids
medical-primary pretraining. Test labels are never read; official labels are used only in-fold.

## Folds

Official 5849 rows, stratified by label with seed 20261106. If the official CSV carries patient or
eye identifiers, grouping follows those identifiers; otherwise each image is its own group and the
split is label-stratified. Fold assignment is registered as a small JSON (row order, fold id,
counts, SHA) before any training; the same file is used by all three folds.

## Per-fold training

- init: R120 selected EMA (F04 strongest single member), exported fp32 tensor
  `f21bffae88acbf7c216c4141c0ef0b9d990a624e3e92a1baf750a501c415fc85`; weights-only warm start.
- train: fit 46,256 rows plus the two official folds not held out. Positive pool includes official
  positives; official rows may be oversampled x2 (registered in SCIENCE).
- validation/selection: the held-out official fold AUROC (internal dev also recorded as side
  evidence, not used for selection in this round).
- G64/P8 pairwise sigmoid, fp32, 512/512, ImageNet normalization, age as in R120.
- LR = R120/10 (backbone 1e-5, head 1e-4), short warmup 200 updates, eval every 400 updates,
  cap 6,000 successful updates, patience 4, min_cycles 1.
- Official images use the same preprocessing/cache as the F04 official evaluation.

## Resources

Pod A fold 1, Pod B fold 2, whichever finishes first runs fold 3. R272-A preparation is stopped,
never activated, and recorded as unused.

## Outputs and decision

Per-fold out-of-fold predictions over the official 5849, z-normalized per fold and merged, give the
honest OOF AUROC/AP. Compare with F04 official 0.98944. If OOF <= 0.98944, the final submission
stays F04 (existing file). Otherwise the final submission is the three-fold z-fusion (fp16, about
3x99 MB) over test, Platt-fitted on the official OOF fused score, written to
`submition/F06-official-ft/`. F05 remains secondary and must not delay F06.
