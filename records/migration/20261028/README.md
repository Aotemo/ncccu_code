# exp-v3 依赖迁移

小批、raw/cache bulk、两组新索引和最终汇总均已真实通过。合计复制120380件、75373650642字节；train51396与valid5849条新缓存索引完成。B全部计算任务已正常退出，源码冻结，由A统一提交同步和冻结部署；本结果不替代GPU准入或正式训练。只迁移已有真实库存中的训练依赖，保留旧源、旧 FREEZE 原字节与 canonical/source_identity。三保留模型不作训练依赖，不迁删；官方 manifest 仅 opaque 字节复制/哈希，不解析标签语义。

最终 [RELOCATION.json](RELOCATION.json) 的 Pod SHA 为 `be516bfec741dd9c7f5cc4b41694be562c9e1fceb195fa1e41e535b715f9273e`。服务器原件为本页下述 prep 根的 `RELOCATION.json`；四个成功输入、逐件manifest pin、完整目标路径、当前UID与预算均在原件中。最后 [CPU-FINAL-RECEIPT.json](CPU-FINAL-RECEIPT.json) SHA 为 `f603ad360051f111335d52e4839c831b0d838d7c71d46d507a38745ca5922b24`，由 guard 确认 exit0 与完整任务树退出。传回仅普通字节传输，本机/登录节点未重新解析或哈希校验。

执行B独立 prep：`/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/migration-cpu-20261028-51fe3daf-b`。当前实核 UID `51fe3daf-ba0d-46e8-919c-99caec599c99`，B分配12核 mask12..23，A12核 mask0..11；allocation-2 SHA `25fe957398e81748c170d929b7b662b24270d1012e7f2940ec28ed315d9385ad`。这是阶段回执身份，不是脚本默认值。nice19、隐藏GPU；线程份额由A通道显式注入。

## 已完成小批

[SMALL-COPY.json](SMALL-COPY.json) 的 Pod SHA 为 `2025c819bcd56390e0c6d8ae9eba4fbefcf27496b6d378ae84aecf0283fd8d66`。实际起拷 UTC `2026-10-28T11:07:36.271601Z`，阶段 wall6.121秒；32件、268663792字节完成。逐件源读取SHA与独立目标读取SHA相同；源未由工具修改，目标无错内容覆盖。manifest 为 prep 下 `SMALL-COPY.files.jsonl`，Pod SHA `7a59927f829dc91b832b5d0c8ae725b97427c32737d478f43e9fceb4e5482955`。

| 对象 | 文件数 | 字节 |
| --- | ---: | ---: |
| stage1 原18冻件+FREEZE | 19 | 23776876 |
| 公开 original | 7 | 114437877 |
| 公开 converted | 3 | 114424744 |
| 恢复的精确 README | 1 | 15750 |
| sealed official manifest（仅字节） | 1 | 2182932 |
| train_identity | 1 | 13825613 |

目标精确路径见原件 `paths`；公开权重位于 `exp-v3/assets/public/tiny22k-bc48a87f/converted/convnext-tiny-tv-bc48a87f.pt`，stage1 位于 `exp-v3/assets/fixed-9010-stage1`。小批已交A做生产工厂集成，不等于全迁移或训练准入完成。

## bulk 与新缓存

真实库存 pin 为 `INVENTORY.json` SHA `3c60a944cdd43bbbd241cd87bbca1bd9fdb0650630f3f709ea4c38365fa422a2`，逐件 FILES.jsonl SHA `3d00a432c3b2bfc0f89c1e86e456c8bc200e86412d792552ab8cb1274434078b`；依赖补充 SHA `88bfb9d67a92894131dd2ed1408225951d7fd5fe0720aff5468a06e3cf318dc9`。不重复库存。

首次 bulk 在创建启动元数据后、pool启动前因通用 JSON writer 所需的 passed 字段缺失而退出，实际0B；保留原启动文本与失败CPU回执，未把该启动文本当成功。仅修该字段后使用新 `BULK-COPY-2` 阶段，不覆盖旧结果。真实新起拷 UTC `2026-10-28T11:14:34.853677Z`，末次完整进度 UTC `11:32:44.907353Z`；120348件、75104986850字节实际完成，阶段 wall约1154.5秒（包括前置库存解析）。raw为63103件40254109282字节，NPY为57245件34850877568字节；只复制两 split 必要 NPY，不复制旧验证缓存及旧控制日志。

[BULK-COPY-2.json](BULK-COPY-2.json) Pod SHA `f8e3da0eb36c6aa268aa30b37be3af352243c9c5efe5a1ab87ee0e6bc4bc774b`；逐件大 manifest 留在 prep 下 `BULK-COPY-2.files.jsonl`，Pod SHA `ac23aac59391f4d459b4c7b753bc7695a36f8c54735d4ee414bb21ad0322b05c`。源流 SHA 和独立目标 SHA 逐件相同，结束空余282050166784字节。BULK guard exit0，原件路径与 SHA 在最终交接中保留。

新缓存 policy 只把 dataset_root 换为 `exp-v3/data/glaucoma_first`；version `rgb-conservative-border-prepared-v2`、prepared_size512、Pillow12.3.0不变。旧 namespace `19b2e1abff37369711cf59110dd52a0fb4894dd35037ee4b3eb758f8bd201a48`，新 namespace `429cbdbda7d0827c564d279fcfcef12d5bf3784f127c7226841d9b5948fee960`；key算法与npy bytes不变。仅迁train/sealed、valid/sealed必要npy，不继承旧.verification、builder.lock、clock或progress。新 JSON receipt 绑定新raw的真实 source_file；当前内容 SHA 仍须逐件通过，不能用相同stat冒充。

cache子agent `0a198f7e-6cca-4280-9d17-2dffde4adde7` 只文本交付 `cache_rebind.py` 和 `test_cache_rebind.py`，未运行解析/哈希/测试/Pod；由父B在自身份额内部署验收。B拥有其余迁移源码、`test_asset_relocation.py` 与本目录；A是唯一Git写入者与GPU owner。

三项复制安全窄检查 [COPY-TESTS-GREEN.json](COPY-TESTS-GREEN.json) 实际通过，SHA `adb316d8eadb1a2eaa089414c9980d077070bafaf4f655f31387a10406d7096e`。随后六项缓存窄检查 [CACHE-TESTS.json](CACHE-TESTS.json) 实际通过，28.430秒，SHA `4570ab7742eea106656dacf3b192055e93dae32864cd65f77941931ad5e25319`，未增加项目全套测试。复制检查对应初版copy源码 SHA，后续仅启动/有限进度回执字段修正，bulk实际全路径另有真实终态。

缓存重绑不再读旧 raw/旧 NPY bytes；旧内容证明来自明确 pin 的 bulk 源/目标 SHA manifest。新索引使用点逐件读当前新 raw 和新 NPY，匹配旧 sealed source_sha/array_sha，随后写当前真实 source_file；这些读取不称为重复复制，也不以 stat 相同代替内容证明。两组均新建全部 JSON receipt，reused为0；原key、array_sha和npy bytes不变。

| 已通过回执 | Pod SHA256 | 实际计数/字节 |
| --- | --- | --- |
| [CACHE-TRAIN.json](CACHE-TRAIN.json) | `3123f225d9d64bdea23b6d1f5d8bbe517ea21bb0139f2f0149c915b22690df2b` | 51396条；raw32803270694B、array31286003712B |
| [CACHE-VALID.json](CACHE-VALID.json) | `36fbc40cbeb8253a370c08a0ecbc4786585f8d751b463bcbd155f7b277208c9b` | 5849条；raw3693437585B、array3564873856B |

大 entries JSONL 均留 prep：`CACHE-TRAIN.entries.jsonl` SHA `f1e2d1ac47dfa5b2ac1c4d09d36ecd5e2269ccea64dc319a29209eb16c0872a7`，`CACHE-VALID.entries.jsonl` SHA `4fdcccfa64bc53d843bd887c453d5e13590823c35830c1c6fc88392b6b8ecf25`。训练与最后官方各自使用顶层cache下的 `train/sealed`、`valid/sealed`；运行时再追加新namespace。未继承旧verification/builder/clock状态。

finalize只核四项成功回执和大manifest SHA、路径/计数/预算一致性，并消费bulk已算的 `data/glaucoma_first/train.csv` digest：`ec81a23ba7758eccdd09dba027ce1a50998d81c18ed30ab6d2d608b3397f67d7`，匹配原sealed身份；不解析CSV标签，不再读权重或图像bytes。最终空余280355667968字节。

| 冻结的迁移源码 | 最终 Pod SHA256 |
| --- | --- |
| `tools/migration/relocate_assets.py` | `a452d7f2d08395cd3c2308a1a33989f7779b4e3605e1bc3d7b61904e8dd35099` |
| `tools/migration/cache_rebind.py` | `d27e575be752464c9f230717417e861742329cf80a16f6c487d26e162ddb9d10` |
| `tools/migration/finalize_relocation.py` | `9ef343cc23fe5545b4c1af6829d05c4c1514574e5e659ad9edc554fe6d9893e2` |
| `tests/test_asset_relocation.py` | `61ba11780161064e91c92ae444aa638b0edfe1f36bf23afb22b928b67b4e7482` |
| `tests/test_cache_rebind.py` | `d9135c742b29a15ac59f4e74bbc60469d53c5c39964b92ce2d302b7a365d3516` |

原始失败保留：[BULK-COPY.started.json](BULK-COPY.started.json)仅为失败前写出的启动观察，不是完成回执；[CPU-BULK-START-FAILED.json](CPU-BULK-START-FAILED.json) SHA `25e75ab6c0e0168de497ad79e938b2a55f6f7287f1e4c308decbd280f97515b0`，exit1/0B。成功重试的 [CPU-BULK.json](CPU-BULK.json) SHA `160ba7c9d8bbdbb78924c24ea6c364b2f880d4ede25b9010446e19cf64a0db82`，exit0、无affinity违规；[CPU-CACHE-TESTS.json](CPU-CACHE-TESTS.json) SHA `ed2a2964f5b42c5600fb5dba3701143dd7e6fb271eaa2bdfb19aeb69eac6add0`，exit0。原复制功能RED回执也保留，未覆盖为成功。
