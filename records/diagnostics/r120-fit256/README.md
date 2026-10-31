# R120 fixed-fit256 CPU diagnosis

The one authorized pass ended **unknown** after 23.740717 seconds, with **0 of 256** images evaluated. The selected R120 EMA checkpoint and its frozen tensor/blob identities passed the driver's checks. The first readonly prepared-cache input raised `ValueError` at `readonly_prepared_forward`; a private Pod-only inspection classified it as a cache entry or source-fingerprint mismatch. The available evidence does not distinguish those cases. The cache was not repaired, no source image fallback was used, and the diagnosis was not retried. No fit AUROC, error count, or comparison with dev can be inferred.

The fit-only roster was fixed from sealed metadata before forward evaluation: 128 names from each label, ordered by the frozen SHA256/name rule. Its private roster digest is `1da2fcd90fd9bba4b6a01a161f68841d8602678cac8d167611f850a5563c15ea`. Names, rows, per-image logits, and the detailed cache error remain in the private Pod preparation directory and are not in this repository.

- Token-free [aggregate](AGGREGATE.json): original Pod SHA256 `2a0c574f28af29b115d3a8f3122b283275c25669ea4e16c4bb2d615134e39244`.
- CPU guard receipt: original Pod SHA256 `73016405c8d0be5ed77f630bf39bb2561ada3f29ba3f5f837d45226596e26181`; raw copy in `work/r120-fit256-20261031-a/receipts/`, not committed.
- [Narrow driver](../../../tools/diagnostics/r120_fit256.py): Pod SHA256 `f5639a19d057bb5df582c3fff675049270837c1d022133b1ef1c11ea994288b9`.
- R120 source commit `7922dd788f982bcb72d30a1a8e6d6e7761e42e88`; freeze SHA256 `5f808304a0e4ae84501525a1634f0ca55ec80e16bcba67342a66e7e211f5bcf2`; selected checkpoint blob SHA256 `25229efdea3a91baeb0b5e5142a470c2a68a15bab9470f92e9de76b1b616cac6`; selected EMA tensor SHA256 `f21bffae88acbf7c216c4141c0ef0b9d990a624e3e92a1baf750a501c415fc85`.

The guard succeeded in enforcing the allocated CPU execution and returning a receipt; that is distinct from the diagnostic's `unknown` result. This attempt supplies no evidence about model fit ranking, label quality, generalization, or the 0.999 objective.
