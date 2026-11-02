# R157/R158必要准备

[CPU-CHECKS](CPU-CHECKS.json)，Pod SHA `94f94f0db514c78486224bc0609a7567242e41c4b906b1ef02f8f8abb1933b3c`：新增配置目标通过、5文件compile、0failure/error；原6项损失/WORLD目标按相同loss和3调用者SHA复用。[真实最终guard](CPU-FINAL.json) SHA `ef9733a44755a21e1285a698d8c358d5d8d251e19a33330d1c49a5595e834d00` exit0。旧来源不支持新选项的RED exit2及原源码快照保留。

[公开来源汇总](PUBLIC-SOURCE-SUMMARY.json)：timm/convnext_small.fb_in22k_ft_in1k_384，revision `14a837353b6ffeced1c635dc5c8c799356e316f4`；模型卡/registry声明Apache2，原201017809B的LFS SHA `24f8525364a67ec20ae1ab21f45a525638950a8dc057534506b7aaee63b7e2a6`已匹配。转换SHA `218df88c96a379af7db5732c2adc684047ec487159ff0ebd7ee43acf59aaf3d8`，provenance `ec9c4e396a2a7a132841f703a17a6af7ee4f64665c09ba30b81c005a0e7a0800`，PUBLIC-HANDOFF `52ef8fa708987329882ffdef98d2521cd79c204d169a049629f6982e8ccd4704`。全1000类344张量严格映射/加载、FP32有限、41LN/36GELU-SD/40Conv及非张量语义通过；严格链forward_calls=0，这不是新增的源/目标前向输出一致性测量。

[真实fresh与冷加载汇总](PUBLIC-FRESH-SUMMARY.json)：R157 whole `91507a65d24dd90fce1c4a0d7e7950f55401015f0ec23161e6a9bcd8b86d0d67`；R158 whole `440ad1fff41c1166d0598a91b6e85b0cb04b4bc949bef195b40113476ccf7a96`。均345全训/5组空AdamW/0buffer；R158另经实际512合成输入前向、原CheckpointStore保存和inference严格冷加载，输出exact、loss身份相同，fixture已删除。原完整回执路径/SHA保留在汇总中，未加载训练checkpoint或读取官方。

新来源的384/squash只描述公开初始化；比赛仍原512预处理。R158只换同架构公开来源、回原logistic，不叠R156 loss；潜在收益不能纯归因于384像素。用户另开会话静态只读复核未见阻断问题，测试与模型计算由根在aa0f/0–22/23核完成，与peer23及训练24–119互斥，无GPU。现models/config/engine/EMA/inference均未改；唯一新增正式验收适配是把R157纳入原hard-loss检查tuple。

预算沿用原Small512满24k×2历史情景24154.892秒，小于train soft27750秒，非当前GPU实测或保证、不假设早停。当前待本次实际Git同步/回读与技术包；尚未新normalformal/grant/clock/activation/GPU。R157仅R156完整双门及safe后，其他情况独立R158，详见[固定合同](../../../todo/auroc-999/R157-R158-PREPARATION.md)。
