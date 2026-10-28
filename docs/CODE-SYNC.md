# 源码同步与部署

统一远端为 `https://github.com/Aotemo/ncccu_code`，本地目录为工作区内的 `ncccu_code/`。原工作区包含数 GB 混合历史，保留原位；本仓库从远端原有 README 提交接续，导入当前实验的必要源码。

每次后续实验：

1. 按 `todo/` 文件分工在 `runtime/`、`tools/`、`configs/` 修改唯一源码，在 Git 中检查差异。只改科学合同允许的部分；本机仅文本和Git操作。
2. 按 [EXECUTION.md](EXECUTION.md) 动态发现并核对真实计算 Pod/解释器，在独立 `exp-v3/technical-prep` 对受影响内容执行必要 CPU 窄检查；不重复已经通过且未改变的模型/数据套件。配置解析、SHA验证、模型载入与测试均在 Pod。
3. 唯一 Git 写入者汇集具体文件列表、完整差异和真实 CPU 回执，包括删除的旧生成件以及全部untracked源码/测试/小回执，再通过 Git 或已连接的 GitHub 工具提交到远端，确认远端分支指向实际 commit。不因工具存在或本地已有commit就写“已同步”，也不能绕过对未知变更集发布的审批拒绝。规划与其他执行者不并发 stage/commit/push/ref 写入。
4. 当前唯一执行 owner 从已同步版本生成 `exp-v3` 冻结部署，在 Pod 核对 commit 对应文件的实际 SHA、配置及资产身份。绝对路径与 token 由当次正式绑定产生，不能从本地路径推断；发现新环境不授权改绑活跃运行。**成功同步和冻结身份核验之前，不进行任何新代码的 GPU 准入、试跑或正式训练。**
5. 用同一独占 GPU owner 串行运行，保持固定数据、官方隔离和真实退出回执。运行记录保存 commit、逐文件 SHA、配置和唯一 checkpoint 身份。
6. 结果只同步汇总和小型必要回执。所有非源码产物在服务器 `exp-v3` 按保留规则管理；旧来源先按真实清单核依赖、容量和SHA后迁入，不盲删或整体复制历史目录。

终端认证不可用而使用GitHub连接器时，仍由A唯一写入。具体diff和CPU证据齐全后，基于实核远端父commit构造tree；`create_tree`可批量提供inline content及`sha:null`删除，返回tree必须等于已审阅暂存内容的本地`git write-tree`。按实际父commit创建唯一实现commit，更新ref前复读租约，非强制更新后再次读取远端ref/commit/tree；远端变化则停止该次更新，不覆盖其他提交。工具能力已查明不等于实际写入成功，也不绕过未知变更集写main已遭拒绝的限制。

远端commit对象与代码tree分开核验。GitHub元数据接口可能将作者日期归一到UTC，不能仅凭返回字段无损重建原Git对象；只在实际对象SHA完全匹配后导入或移动本地ref。若唯一远端commit、代码tree与Pod部署SHA均已核实，但本地HEAD尚不能无损取回，则记录远端commit/本地HEAD的实际差异与取回阻碍，后置本地元数据归位。不声称fetch成功，不盲目reset，也不另造第二个实现commit来追逐元数据一致。部署必须由该远端commit的已核tree内容构建，并在Pod验证实际文件SHA；本地HEAD暂未归位本身不增加GPU门禁。

原训练入口会生成 `outputs/<run>/code/` 与 `source-files.json`，将模型绑定到当时真实执行的源码。活跃运行期间保留快照原 bytes；未保留模型的实验达到安全终态且结果核实后，源码快照和过程副本按存储规则清理，版本从 Git 追溯。后续维护以本仓库为准，不另维护每轮完整开发包；冻结部署从已同步 commit 导出。

当前 R91/R92 迁入仓库不会改变运行源码、配置、clock 或 token。仓库 commit 是迁入版本的来源记录，不追溯声称历史训练当时已经绑定此 commit；历史真实身份仍以服务器 source digest/部署清单为准。

当前待办统一维护在 [todo/](../todo/README.md)，目录职责见 [LAYOUT.md](LAYOUT.md)。参数归档工具的检查/提取可在独立 CPU `technical-prep` 暂存区完成；这不授权启动训练。正式部署仍须使用已同步 commit 和实际 SHA 清单。
