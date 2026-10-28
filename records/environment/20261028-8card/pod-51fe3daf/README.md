# 新 Pod CPU 验收

Pod UID `51fe3daf-ba0d-46e8-919c-99caec599c99`。原件来自 `/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep`；本目录仅普通字节传输的小回执。`EXPORT-SHA256.txt` 在该 Pod、A 的 CPU mask 内生成，本机与登录节点未做解析或哈希校验。

- 实核有效 CPU120，A/B 合计20%=24，A12(mask0–11)、B12(mask12–23)，nice19，隐藏GPU。此数值属于当次已 pin 的 allocation/TTL，不是默认环境参数。
- 实际 import Torch、TorchVision、NumPy、Pillow、timm；GPU未初始化。版本和实现 SHA 见 `DEPENDENCY-IMPORTS.json`。
- `PRODUCTION-CPU.json`：生产 Tiny22K→1K 的18项检查通过，含真实 public1000 strict、fresh binary+零age、参数组/空Adam、初态EMA、checkpoint strict回读及LR/EMA成功更新边界。初态tensor SHA为 `eba53feeafdf20bc86855a40648bc453358928d148078c6ae78b65f059846cc7`。
- `BINDING-CHECKS.json`：29项通过，含八卡/CPU角色、显式源码/部署门、Pod/PCI、一次性activation/launch/官方reservation和dev选择器的受影响检查；执行期间源码未变。
- `GPU-CONTRACTS.json`：新增11项CPU合同通过，覆盖0update/0forward准入的身份与退出拒绝、未来GPU的20% CPU分配；没有执行GPU准入。
- `DEPLOYMENT-TREE-CHECKS.json`保留初版2项通过，`COMMIT-TREE-LINK-CHECKS.json`是评审补齐commit→tree关系后的3项真实通过；新增拒绝回归的修复前失败与对应CPU树均保留。最终入口还需要同步后实际GET的commit/tree原件与SHA，不能用fixture当部署成功。
- `ADAPTATION-RED.json`、`INITIALIZATION-RED.json` 保留修复前真实失败；初始化白名单修复后的3项检查见 `INITIALIZATION-POLICY-CHECKS.json`。未改变的预算5项与样本重组3项在首次21项中通过，未机械重跑。
- 资产库存与18冻结件+FREEZE、缓存旧namespace依赖见两个 inventory/dependencies 回执。库存不等于迁移完成；迁移证据另见 `records/migration/`。保留模型仅stat，未在本阶段重验或加载。

CPU 包明确为 `cpu_technical_prep`、未提交源码的固定快照（基线 `cb939c8`），其实际 bundle SHA 清单单独记录。它不是远端Git同步、正式绑定或GPU准入证明。未读取样本图像或官方语义，未forward、训练、创建正式clock/token、取训练锁或运行GPU。完整官方5849没有新增前向。
