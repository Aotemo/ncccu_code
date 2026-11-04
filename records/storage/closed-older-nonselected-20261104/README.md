# 已闭轮非selected checkpoint清理

统筹授权第二批仅在并行容量不足时执行；先做[精确计划](PLAN-SUMMARY.json)，再按实物清单删除R135/R136/R152/R154/R156/R158/R160/R162共58个非selected checkpoint。计划SHA `536b79456211cc9ad2ce1532f3385611a74c8e8382ab3c2d8ca669aa18e56476`，清单SHA `a3507c4d097a0bb7b0ddd032ba950dd00656633735a7ca5a5ccd5a4f78634ebc`。

[删除回执](DELETE-RECEIPT.json) SHA `1898ec1a179fd0ac3c1e1d8a440a68792260b47d27d49665132786d4b54b6a7c`：58项、0跳过/0错误，unlinked allocated `42,738,476,544` bytes；共享盘free点测从`39,965,425,664`到`82,703,286,272` bytes，差值受同时活动影响，不作为后续容量保证。受保护selected及最小依赖、三KEEP、数据/共享缓存/待用公开源、活跃轮和R120提交依赖未动；保护stat和小文件SHA前后一致。启动新轮前仍重核实际容量。
