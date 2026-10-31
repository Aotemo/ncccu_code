# R120：固定 256 张 fit 图像的有限拟合诊断

状态：2026-10-31 原 FIT256 任务已执行一次并以工程 `unknown` 收口，完成 0/256，无 AUROC/error 指标；不重试。独立缓存绑定元数据核对也已完成，仅定位首个失败 entry 缺少既定 namespace 的 receipt/npy，原 FIT256 结论不变。以下保留两项原合同与实际结果，不代表仍待执行；两项均不占 fresh、R123、order 或 SCIENCE，本页不构成额外审批门。

目的：补足保留模型在见过的 fit 图像、确定性验证视图上的排序证据。随机训练 batch 的低 loss 不能代替该证据；此前标签传播一致也不等于独立临床标注确认。结果仅用于重排下一问题，不选新 checkpoint、训练样本、权重或裁框。

1. **唯一模型与加载身份。** 只用 R120 所选 EMA c6@4800，tensor digest `f21bffae88acbf7c216c4141c0ef0b9d990a624e3e92a1baf750a501c415fc85`。checkpoint 路径及 blob pin 由原 freeze 消费；已知 blob SHA `25229efdea3a91baeb0b5e5142a470c2a68a15bab9470f92e9de76b1b616cac6` 须与实际原件一致，不能仅凭本文登记视作核验通过。原 factory/job、age 和输入身份不变，`initialize=False`，严格加载 EMA；缺失或身份不符即 unknown，不换 R107、末点、另一 variant 或 seed。可复用冻结 `inference._load_model` 的显式 checkpoint/CPU/EMA 路径，不调用会读取 dev 预测 hash 的单模型 full validator，不预约或运行 official，不修改旧 validator。

2. **名单预定与输入边界。** 只在当前 sealed manifest 的 fit indices 中，按每个 label 的 `(SHA256(row['name'] 原始 UTF-8), row['name'])` 排序各取前 128，共 256。原名字不 lower、不 trim；先固定名单，再生成预测。仅 metadata 选择及断言无 holdout；任一类不足、身份不符或缓存坏缺失即 unknown，不补样、修缓存或回退原图。只读取现有 readonly prepared512，使用原 identity eval 像素和原 age；不读 dev/official/test 图像或预测向量，不做全 fit 前向、原图扫描、新 group 或近重复扫描。`row['group']` 只是已有 content-group，不是 patient ID。

3. **一次有界执行。** A 在实时有效的本人 20% 准备份额、独立 technical-prep、nice 19、隐藏 CUDA/HIP/ROCR 的 guard 下执行。CPU FP32、`eval()`、`inference_mode()`，固定 batch 4、loader 0，线程不超过当次本人份额，每图一次；无增强、双视图、梯度、优化器、参数更新或 GPU。600 秒硬预算包含 checkpoint 读取、身份加载、256 次图像前向及聚合。超时、内存或依赖问题、非有限输出、未完整完成 256 张均为 unknown，只报已完成计数和工程原因，不据 partial 判科学；不重试、扩时、补样或扫描 batch/workers，不新增资源、grant、训练 clock。

4. **仅聚合输出。** 报实际 256/128+128、fit-only 身份断言、名单中的 unique 现有 content-group 数、raw-logit exact AUROC、error_twice、tie 对数及其占全部正负对的比例、涉及至少一对严格错序 `z_pos < z_neg` 的正/负样本数，以及 finite 与耗时。tie 与 strict wrong 分开；error_twice 按严格错序两倍、tie 一倍计。共享样本的正负对不是独立观测。名单、logits、逐行数据只留 exp-v3 私有准备目录，不导回、不进 Git；只回聚合与小回执。不输出阈值 accuracy、AP 跨样本比较、bootstrap 或逐图审阅。

5. **预定解释。** 完整且 error_twice=0，只证明这 256 张已见 fit 图在该 identity 视图下可分，不证明全 fit、0.999、细节充分、标签正确或过拟合根因。完整且 error_twice>0，说明该子集尚有排序误差，但优化、表征与监督仍混杂，不能诊断错标、删改标签、给样本设权或回灌 dev。平衡且小规模的 fit 子集不是总体泛化估计，与既有 dev 数值的差异不能直接充当统计意义上的泛化差距。unknown 不以部分结果补结论；一次任务结束后由协调方据完整聚合或确切工程缺口决定下一问题，不自动追加诊断或新轮。

6. **分工与不变合同。** A 独占必要窄 driver（可位于 `tools/diagnostics/r120_fit256.py`）、实际 Pod 执行及必要 Git 集成；B 无新任务，规划只维护本页及 `todo/README.md`。不重 formal/source/Gloo/旧套件，不建测试平台或 GPU 预演。R120 原冻结包、selected 及真实依赖继续保护。R107/R108 仍是科学比较锚点，R120 `.9894773092062407` 只为单次内部 best；本诊断不升级基线。固定 9:1、内部完整 dev 选点、`.996` 完整 freeze 后官方 5849 一次的独立政策、总体 raw AUROC `.999` 目标及原资源合同均不变，不授权平台 test，也不保证目标可达。

**原任务最终收口（协调方转交 A 实际回执）。** 固定 128+128 名单已在 private 目录冻结；R120 EMA c6@4800 的原 blob/tensor 严格加载 pin 通过。首个 readonly prepared-cache 输入发生 `ValueError`，完成 0/256，wall `23.740717s`，未超过 600 秒；主 guard exit0 不代表诊断成功。现有脱敏类别为 `entry_or_source_fingerprint`，原聚合不能进一步区分。无 AUROC/error 或可用的 fit 排序结论；既不能判 fit 可分，也不能判不可分，更不能推出标签或缓存全局结论。原任务关闭，不重试、换名单/模型、修 cache 或回退原图，原含名异常、名单和 logits 不导出。

溯源仅引用已交聚合，不由规划读取私有内容或重新校验：`work/r120-fit256-20261031-a/receipts/AGGREGATE.json`，原 SHA `2a0c574f28af29b115d3a8f3122b283275c25669ea4e16c4bb2d615134e39244`；CPU 回执 SHA `73016405c8d0be5ed77f630bf39bb2561ada3f29ba3f5f837d45226596e26181`。A 已实际同步 Git `0ca69fab19f236977f1d54785644afde2133753e` / tree `e0863e33a1a3169482c92e0318ec2662dd789fba`，该次仅 driver、aggregate、README 三件，nonforce/readback，旧原件不改。

**独立元数据核对的原范围（现已完成）。** A 唯一执行，B 无任务；当前有效 20% 本人 guard、nice 19、隐藏 GPU，一次 300 秒。先只用既有 private `WORKER-ERROR` / `WORKER.log` 异常链做脱敏分类：`entrymissing`、`receiptpolicy`、`sourcefingerprint`、`checksumformat`、`pathbinding` 或 `indistinguishable`；不导出样本名、样本路径或错误原文。最多对照两个现成冻结 metadata context：R120 SITE_BINDING/ASSETS/已有 cache manifest 与原 driver 绑定，以及 R122 成功训练 SITE_BINDING/ASSETS/已有 preflight；限 data/cache lexical-resolved 关系、prep version/Pillow/size/dataset_root、原 namespace 与 diagnostic cwd 对相对路径的影响，不重扫 source tree/目录或尝试别的 namespace。

仅当既有异常链能确定原 roster 首个实际失败 entry，才允许读取该一件 receipt JSON（不超过 16KiB）及其 npy/source 的 stat 元数据：存在性、symlink、size、receipt 期待字段及源 fingerprint 字段名差异。不能确定则不猜；不查第二 entry 或全 256，不读图像/npy 字节、不 hash 图像、不 `np.load`、不调用 `PreparedImageCache.read`，不加载模型或前向，不 repair、迁移 cache 或改 fingerprint。metadata 不足仍 unknown 停止，不重试或加探针。

只回 metadata 种类、entry 0 或 1、两个 context 字段一致性、存在性、可定位类别或仍不可分、耗时/局限与回执 SHA。无论结果如何，本次元数据核对都不授权修后重跑 256、前向补考或新训练；由协调方统一判断接续。总体 `.999` 未达、R120 单次 dev best、R107/R108 比较锚点和 `.996` 官方门均不变。

**独立元数据核对最终结果（协调方转交 A 实际回执）。** 2026-10-31 约 10:28 北京时间，一次任务完成：CPU16 当次本人 mask `0..11`、nice19/hiddenGPU、guard exit0；metadata 耗时 `0.2383430847s`，总 guard 约 18 秒。只定位原首批真正失败的 1 个 entry：既有异常链为 `FileNotFoundError`，既定 policy namespace 下 receipt 与 npy 的 stat 均不存在，source 仅 stat 确认为 regular，脱敏类别为 `entry_missing`。receipt 不存在，因此未打开 receipt；未读图像/npy 字节或模型，未 forward、扫描或 repair。

R120/R122 的 lexical/resolved data_root/cache_root/metadata_root、ASSETS binding 与 prepversion/Pillow/size/dataset_root 一致；namespace 同为 `429cbdbda7d0827c564d279fcfcef12d5bf3784f127c7226841d9b5948fee960`，路径均为绝对路径，private cwd 不影响本次绑定。此结果只说明该 1 条缓存项缺少，不知其余 coverage、缺失时间或原因，不等于训练数据缺失、原训练未消费该样本、模型阴性或缓存整体损坏。原 FIT256 聚合与原件不改，仍为 unknown/0 of 256，无 AUROC；不修后重跑、替换样本或扩查。

本次聚合原件 `work/r120-cache-binding-review-20261031-a/receipts/RESULT.json`，Pod SHA `9a43a493bed020bde9f28f2dfe70bc01f4b8ec0af83f2d38e6528a0bfb5e0ec5`；同目录 `CPU-RECEIPT.json`，Pod SHA `ad0c2db079c4334c2815f6fe17bb0c90e6588171bbe42aa68cee836b016aeab8`。规划仅引用协调方已读的脱敏结果，未查私有内容、Pod 或重算 hash。两项均收口，后续问题由协调方另行采用，本文不授权新增执行。
