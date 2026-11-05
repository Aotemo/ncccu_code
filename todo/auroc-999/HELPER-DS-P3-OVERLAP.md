# P3 名单与 F05 v5 可疑标签的重合统计（2026-11-05）

状态：**已在 Pod A 准备池完成，只作描述，不用于调参。** 未读取官方标签、未加载模型、未前向。

## 目的与边界

回答“F05 三折 OOF 目标表标出的 204 条可疑标签，是否落在 P3 已固定的难例名单里”。结果仅描述两套清单的关系，不据此改标签、权重、成员或阈值；逐样本关联只在 `exp-v3`。

禁止项沿用 `AGENTS.md`：N06/FLAIR 不参与；官方 5849 不读；明细不进 Git。

## 输入（Pod 内实算 SHA）

| 输入 | 路径 | SHA-256 |
| --- | --- | --- |
| v5 可疑标签 | `…/takeover-20261104-08ac/F05-OOF-TARGETS-v5/SUSPICIOUS-private.json` | `d932f99cdd73621103ec030c1b87e7809156968a14f9c9871aaa000874e3a383` |
| P3 排名明细（含固定20、167 正例） | `…/P3-HARD-CASE/RANKS-v4/PRIVATE-RANKS.json` | `09c8a9b569ca6ece7dd7e0b86c5050c338c12beaf5b6ea1445f4a3533cc3c9d3` |
| P3 共有错序负例（603/241） | `…/P3-HARD-CASE/RANKS-v4/PRIVATE-SHARED-NEGATIVES.json` | `e040f882865484cacb38350661ce8d4848bd7d9ba1b02e72adb827df4c894667` |
| 固定 9:1 manifest | `…/assets/fixed-9010-stage1/manifest.json` | `7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377` |

`PRIVATE-SHARED-NEGATIVES.json` 的 SHA 与 P3 合同里登记的“603/241 封存私有清单 `e040f882`”一致，说明这是当时封存的原件，未重算或改写。

## 索引空间（本项的关键）

两套清单用的不是同一编号：

- v5 `SUSPICIOUS-private.json` 的 `indices` 是 **manifest 全局索引**（0–51395）。
- P3 的 `index` / `dev_index` 是 **内部 dev 人群内的位置**（0–5139），对应 `manifest.rows[dev_indices[position]]`。以 `PRIVATE-RANKS.json` 的 175 行按名字逐一核对，175/175 与 `dev_indices` 顺序对齐；直接按 manifest 索引读则为 0/175 命中。

因此先把 P3 的每个位置经 `dev_indices` 映射成 manifest 索引，再做交集。**先前不做映射直接取集合交集会得到 1 个假命中**（某个 dev 位置数值恰好等于某条 fit 索引），已作废，不作为结论。

## 结果（manifest 索引空间）

| 量 | 计数 |
| --- | --- |
| v5 可疑标签 | 204（全部在 fit 内，dev 内 0，manifest 外 0） |
| P3 固定 20 难正例 | 20（全部为 dev 正例） |
| P3 dev 正例 | 167 |
| P3 共有错序负例 | 603 |
| 其中支持 ≥3 个固定正例（241 子集） | 241（241 ⊆ 603） |
| **可疑标签 ∩ 固定20** | **0** |
| **可疑标签 ∩ 全部 dev 正例** | **0** |
| **可疑标签 ∩ 603** | **0** |
| **可疑标签 ∩ 241** | **0** |

## 观察（仅描述）

1. 四组交集全为 0。这与 F05 目标表只在 fit 行内标注可疑、P3 名单全部来自内部 dev 的构造一致，可作为两边彼此未串数据的旁证；它不能证明任何一方标注正确。
2. 因此“P3 难例”与“OOF 目标表可疑”目前是两个不相交的问题集：前者是 dev 上七成员共同错序的负例，后者是 fit 上标签一致性可疑的行。两者都不应被当作对方的外部验证。
3. 该结论不改变 P3 的归档状态（已按 c607 决定关闭、剩余 26334 张 fit 负例未扫描为 unknown），也不改变 F05 清洗/蒸馏目标的封存身份。

## 回执

- 输出 `…/helper-20261105-5ae33039/P3-OVERLAP-20261105-v2.json`，SHA `abadd06de1c8be765244ea511e95103aa863d22c6539457921313fbde9afcc98`。
- CPU guard 回执 `3d093b82456f3022075e79cb0b1a1964867d6eb9c4be6bba3d14b62e84fbfb77`，Pod A UID `04203d31-96b5-415d-b126-a30f3ac6107f`，12–23、nice 19、隐藏设备，exit 0。
- 首版脚本 `P3-OVERLAP-20261105.json`（未映射，SHA `61b9a20d3a140789f4cdf10769dbdc8951ea44089fa37c2b756c1047e0f8275e`）保留作索引空间错误的证据，不作结论。
