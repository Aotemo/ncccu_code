# 保留模型与参数

只保留以下三个唯一 EMA 模型。权重留在服务器，Git 保存既有参数原件、来源和模型索引；精确权重路径/SHA 以 [KEEP-MODELS.json](../records/storage/KEEP-MODELS.json) 为准。

| 模型 | 完整官方 AUROC | 参数与身份 | 使用边界 |
| --- | --- | --- | --- |
| N06 FLAIR ResNet50，EMA@2133 | 0.9952862924941565 | [N06/](N06/README.md) | 研究专用；禁止后续比赛初始化、teacher、BN 或训练参数来源 |
| R17 ConvNeXt Tiny focal2，EMA@2844 | 0.9893446521535555 | [R17/](R17/README.md) | 保留单模型 |
| R8-C25 ConvNeXt Tiny，无年龄，EMA@2844 | 0.9890383629536272 | [R8-C25/](R8-C25/README.md) | 保留单模型 |

每个模型目录保存服务器 `retained-models/<id>/` 已有的五份小型 JSON 原件，文件内容未重写：

- `MODEL.json`：精确 checkpoint 路径、SHA、EMA variant、更新数和原始参数记录路径。
- `*-job.json`：模型、学习率、精度、图像大小、年龄开关/统计和归一化等参数。
- `*-config.json`：原完整配置及数据/运行参数上下文。
- `*-contract.json`：原 source/manifest、公开来源、EMA、schedule 和 runtime 版本身份。
- `*-source-files.json`：当时源码的逐文件 SHA。

这些原件通过 SSH 只读传输保存，没有复制权重。三个权重曾由存储工作在原 Pod 核过 SHA，详见[存储状态](../records/storage/CLEANUP-STATUS.md)。**本次 checkpoint 内参数重新导出、加载核验和传输后 SHA 复核尚未执行：原 Pod 在暂存传输期间被移除（exit137 / NotFound）。** 原参数不能标成新增验证通过，也没有生成 `parameters.json` 或成功导出回执。

历史划分和 precision 不同，表格仅用于既定模型保留，不代表同条件科学比较或当前 9:1 任务达标。原件中的历史路径是来源身份，不是新环境默认配置；迁移必须记录实际路径并重核相同 SHA。

加载时使用权威清单中的精确 checkpoint 文件、原侧车与显式 `variant="ema"`，不依赖已清理的 `best`/`last` index。`runtime/bundle/retrain/persistence.py` 和 `inference.py` 记录原加载协议；历史 source digest 不等同于当前运行包，兼容性需在获准的计算环境中核实，不能直接套用当前配置。

后续经授权且具备合适环境时，可使用 [export_parameters.py](../tools/models/export_parameters.py) 对同一精确对象生成新增参数与核验回执；该新工具尚未在 Pod 执行，见[本次整理状态](../records/repository-layout/STATUS.md)。它不构建模型、不前向、不复评官方、不复制权重。
