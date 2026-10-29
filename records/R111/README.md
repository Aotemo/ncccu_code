# R111 — public fixed-BN matched control

Actual Git `eee2012a766b932effe786ab83a03d4e8b3094c1` / tree `1a97c99b0ab3044ca55de6708dbd9da12c0ff6e7`. Independent public ImageNet1K V2 ResNet50 fresh; no N06/FLAIR or trained checkpoint initialization. Training seed20261029 and existing head seed20261027; common R111/R112 recipe and fixed9:1 remain as registered in the N06 replan.

Training completed normally at7200 successful=attempted updates,0 skips,9 complete cycles; c6–9 did not refresh the internal early-stop reference and patience4 stopped. The unique internal selection is c5 EMA@4000: AUROC0.9754398301727533, AP0.757855599765702. All18 internal entries are preserved in SINGLE-MODEL-FREEZE.json (Pod SHA0375464fb8e28c9bdbc5e9b3299adf463f9ff2ae8707e2b72369f574e8414615); TRAIN_RESULT.json original Pod SHA1964275715e3c8cac4526dbc9e8253ccac2d250dd0dc290d3183b6f450ed6fed. Train component1905.527s/exit0; observed training time1445.426s, data wait476.091s, paired evaluation161.294s and checkpoint59.817s are existing aggregate counters, not a new benchmark.

The original single official5849 evaluation completed once: AUROC0.9808231499114445, AP0.7731188181041838. These are separately reported final comparison results and did not choose the calibration recipe or R112 settings. Official component747.962s/exit0.

At2026-10-30 02:08:55 Beijing, the entire round was completed, ranks inactive, devices idle/no users and original instance lock released. Owner stateZ is recorded without claiming it was reaped. SAFE-HANDOFF.json is the unchanged read-only original (Pod SHA0eb1c7961347acba94ca86afb20472444b436e317cb432f9ccaf3ccfeddd0982). Only after this handoff was the one pre-registered fit calibration executed. R111 trained weights are not used by R112.
