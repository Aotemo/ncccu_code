# R268-R1 / R269 results and three-soup dev (2026-11-05)

## R269/B (distill)

Completed normally with early stopping: 12000 successful updates, selected EMA cycle 11
AUROC `0.9890125239165747`, AP `0.8373606072483293`. `control/done.json` status `completed`;
no stop signal was sent. Checkpoints and receipts preserved.

## R268-R1/A (clean, fresh identity)

Completed with early stopping: 10400 successful updates, cycle 13. Best online AUROC
`0.985716883145031` (cycle 12); best EMA AUROC `0.9880239520958084` (cycle 9).

## Three uniform soups, full internal dev (5140, 167 positive)

GPU 8-rank `f05_soup_dev` path, fp32 arithmetic, single view, no TTA.
SUMMARY `4f818ef7a63e25b7b55f32481955a8d1a1002e68e5bb3e2a15e1d0dab7a8f6a2`.

| soup | fp32 AUROC | fp32 AP | fp16 AUROC | fp16 AP |
| --- | ---: | ---: | ---: | ---: |
| R120 | 0.9890040952 | 0.8432364671 | 0.9890004828 | 0.8432026839 |
| R134R1 | 0.9890028911 | 0.8413883624 | 0.9890101157 | 0.8414819058 |
| R201 | 0.9886621288 | 0.8365265056 | 0.9886573122 | 0.8365458794 |

Combined fp16 soup bytes 297,054,951. All three are below the F04 fused internal dev
(0.9902539582 / 0.8435746088); this is candidate evidence only, not a selection.
