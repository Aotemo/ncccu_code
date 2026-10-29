# Public ImageNet R50 BN matched preparation

R111 retains the public BN buffers; R112 changes only fit-derived running means/variances. Both use the same current 9:1/age identity, public ImageNet1K V2 source and fresh parameter initialization. No N06/FLAIR weight, buffer, teacher or checkpoint is consumed.

Current Pod CPU10: calibration mathematics 3/3, paired binding 2/2, affected configuration binding 3/3 passed. The public full-1000 strict load has 320 state entries; the binary+age factory produced 321 state entries, 162 parameter tensors and 159 immutable BN buffers (53 layers). Empty Adam and actual source/structure/parameter groups were observed without GPU initialization or image reads. See the unchanged Pod originals in this directory.

Actual new public control: whole `b88683d2cd788e74127e86225bed1cbbbec2fbe3baa56e3b172a33b8ef92ea96`, parameters `2207cf41c2e0cb9c41bc50dd56a56b2b82ca5fb7e3feca76887f1af00711fbf0`, BN `21b9ef124340ced1d13ffe0ea1ef8214f800c05ea0f7bdb9572f06d26475b210`. R112 shares only the parameter control until its single later calibration produces real derived buffer/whole identities; null fields cannot pass formal preparation.

These are preparation results, not formal activation, GPU calibration or training. Each actual Git version still uses the existing native formal package and real eight-rank initial gate. The calibration entry is frozen with R111 and runs only after R111's full safe handoff, once, with its own future binding. Four explicit synthetic EMA boundary cursors in the R50 CPU package validate frozen buffers and update arithmetic; they are not 24000 training updates or usable training checkpoints.
