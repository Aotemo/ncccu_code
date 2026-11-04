# R238/R239 implementation plan

Use the existing inline execution workflow; the user has delegated choices and prohibits subagents. Spec: [fit CSV label relation](R238-R239-FIT-LABEL-CONTRASTIVE-20261104.md), preregistered b489f61 before any R236R2 dev read.

Task 1 changes only `vicreg.py` and the new test file. Add a stable outside-log supervised contrastive loss over the same global two views, returning aggregate positive/negative counts. Preserve the original gather/VICReg/NT-Xent functions. Test independent hand computation and gradients, equivalence to NT-Xent for unique labels, view/row permutations, all-one-class and diagonal-mask stability. Watch the missing-function target fail in the verified Pod, then implement and pass.

Task 2 changes only `ssl_stage.py`, `experiment91.py`, the same test file and two SCI configs. Add a registered recipe inheriting all R236 views/budget/streams, a strictly fit-only label table, and a new loss branch consuming the existing loader indices and original rank-order gather. Keep the image dataset and original supervised job intact. Test forbidden dev/age access with traps, label/index validation, fixed contracts and original recipes, actual fit8 view/label alignment and a two-rank CPU Gloo ordering/gradient comparison. New completion evidence must explicitly report fit labels used and age unused; old recipe results remain unchanged.

- [x] Write two necessary missing-behavior targets and observe correct RED.
- [x] Implement the loss, fit label lookup, registry and science identities, with no extra training/launch/eval chain.
- [x] Run the narrow targets, compilation, real fit consumer and two-rank CPU collective; compare exact unchanged scopes and reuse original P8/native model evidence. Preserve the initial runnable-thread audit failure and the successful bounded-import collective-only recheck.
- [x] Publish source/config/check receipts and API-read before exact deployment. Source3bd/tree36eb, 2845 exact files; original priority R237 and backup R238 have independent unactivated 18/18 formal packages. R239 technical only. Both concrete entry commands and pre-activation capacity checks are prepared during current supervision.
- [ ] At complete unique actualsafe, publish full metrics and follow the preregistered gates; proactively inspect capacity before any activation. Never retry old failed activation identities.

Only CPU3's 24 cores0–23/nice19/hidden GPU may run preparation, disjoint from active R236R2's 96 cores24–119. Current frozen c800/SITE65bf/clock/token/grant and original CSV labels remain unchanged. No external data, medical weights/teacher, dev coordinates, source hot patch or independent GPU rehearsal.
