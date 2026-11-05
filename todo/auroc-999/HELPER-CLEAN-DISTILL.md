# Helper 清洗与蒸馏接口准备

根据 [三折合同](F05-OOF-CLEAN-DISTILL-768-20261105.md)，目标资产必须只覆盖原始 fit、按封存三折验证每个 OOF 教师未训练本行、绑定各折 selected 与仅训练子集的标定证明；任何缺项、重复、非有限值、错误标签或身份不符均失败。当前三折训练和 OOF 尚未完成，不推断或构造实际目标。

本助手范围限定在 `runtime/bundle/retrain/classification_loss.py`、`distributed_engine.py`、`data.py`、`config.py` 和相应测试。现有 `ImageDataset` 返回的是原 manifest 行索引，可按索引查询目标而不改变样本顺序与 batch 结构；辅助值必须与 logit/label 按 WORLD 同序聚集。清洗仅对被事前 confident-joint 标记的样本取 0.25，其余为 1；原 sigmoid 配对损失的权重为正负样本权重乘积，按总配对权重归一化。蒸馏不改变分类原标签，原损失加 `0.25 * BCEWithLogits(raw_logit, calibrated_oof_p1)`。

待 08ac 在其 `experiment91.py`、`pipeline.py` 和 activation 接线并绑定最终 OOF/标定来源；该助手不得修改这些路径。数学接口与其他草稿即使完成也不代表 OOF 来源已被验证或可以部署，须等待实测 CPU、Git 合并、配置/资产绑定与正式接续检查。

## 预定目标文件 schema

目标文件只存在 exp-v3，作为独立 JSON 文件通过 job 引用其绝对路径及 SHA256：`format="f05-fit-oof-targets-v1"`、`mode="clean"|"distill"`、`base_manifest_sha256="7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377"`、`folds_sha256="eb880273832b81be61830a069d5ae9ba8dc2e52be4af2f911f7a6b1ec9e7f6d8"`。`fold_sources` 是按折号 0、1、2 排序的三项，每项含 `fold`、`selected_model_sha256`、`calibration_sha256`，且各 SHA 均由正式接线解析为实际冻结回执并核对其折号、模型 selected、仅折训练子集标定及收敛状态；`rows` 按全局 fit manifest index 严格升序，每项含整数 `index`、原始整数 `label`、held-out 整数 `fold`，以及 clean 的 `weight` 或 distill 的 `soft_target`。所有行只能恰好覆盖原 fit 一次、不得包含 dev；`fold` 必须对应该折封存的 audit 集且不在该折 fit 集，标签与 manifest 完全一致。clean 值只能为 0.25 或 1，distill 值必须有限且位于 [0,1]。实际 OOF 分数、目标值、索引及路径不进 Git。

`mode=clean` 的权重须由完整 OOF 标定 p1 和原始标签，严格按合同 `t_j=max(0.95, mean(p_j | observed=j))` 的 confident-joint 重算，并与文件相符；`mode=distill` 的软标签须与按该行 held-out 折标定映射转换的 OOF 分数相符。生成与正式资产绑定必须同时验证每折模型及标定回执与 OOF 的来源，不能仅凭 JSON 自报折号或 SHA 断言折外身份。具体 OOF 及标定回执格式由 08ac 按现有折训练/推理链接线，缺来源时保持不可训练。配置固定蒸馏系数 0.25、温度 1，不允许通过数据文件更改；两个模式均保持原 sigmoid pairwise、G64/P8。

## 合成测试状态

首次 B guarded 双 rank 检查中，测试进程输出 4/4 OK、child exit 0，但 guard 回执 `5601cf8d3bd3259f3253497a264f6cb928aebedcdbe1992163a9c7d8136c839e` 总 exit **2**：affinity 均在 12–23，峰值 runnable 线程 24 超过助手份额 12，故本次整体为失败，不作为通过证据。测试在导入 NumPy/Torch 前将各进程 BLAS/OMP 及 Torch intra/interop 降至 1 后须以新源码身份重测；不调整实际 allocation，原回执保留。
