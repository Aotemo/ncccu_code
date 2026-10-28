# 清理真实状态

永久保留对象仅为 N06、R17、R8-C25，精确路径与 SHA 见 [KEEP-MODELS.json](KEEP-MODELS.json)。三个对象在 Pod 制定及续执行清单前均完成 SHA 核对；它们及同名 sidecar 不在删除清单中。最终清理后的重新核对尚未执行。

已先提取 188 条精简历史实验/技术终态记录，保存于 [实验结论](EXPERIMENT-CONCLUSIONS.md)。本地按明确路径列表实际删除 571 份旧说明文件，未发现本地 checkpoint 文件。

服务器原计划共 7,381 项：446 个历史 checkpoint、203 份说明文件、6,731 个过程文件、一个不再使用的 prepared768 缓存目录。清单 SHA 为 `8aa70d256c6a2b4a63760c1965a7e1d9b76903b72784a7aabb0b22b40fdb00c2`，详见 [原计划摘要](PRUNE3-PLAN-SUMMARY.json)。计划字节数不是已释放空间。

续执行期间，在原计算 Pod 只读观察到删除日志已写入 7,380 条，对应全部单文件项。最后一个缓存目录正在删除时，进程实际退出 137；随后原 Pod 查询返回 NotFound。没有最终成功回执，缓存删除程度与已释放空间未获终态复核，不能将本轮清理标为完成。[中断事实](PRUNE3-INTERRUPTED.json)保留真实退出状态。

R92 最终 audit 22/22、exit0 已核实，[实验结果](../R92/README.md)已保存；它的 checkpoint、CPU fixture 和过程副本已解除临时保护，但本次第二阶段删除尚未执行。没有 R91/R92 教学保留例外。stage1 的 split/age/schedule、原始数据、共用 prepared512、明确待用的公开 22K→1K 资产及来源链继续作为运行依赖保留。

后续实验处于用户暂停状态。存储收尾须先核验可用计算环境，读取原计划/日志并盘点缓存剩余内容，再生成精确剩余清单。目录已部分变化，不能直接以旧目录 identity 重新执行最后一个动作；也不能回退本机或登录节点运行校验。原始计划与完整删除日志保存在服务器 technical-prep 及工作区 `work/repository-init-20261028/remote-prune-state/`，不把大日志加入 Git。
