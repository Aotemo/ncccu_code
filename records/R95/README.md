# R95：24000步上限与连续4轮早停

实际Git为 `992c3de8a0793083d1bbd9d759d7e8655e015d44`，tree为 `1b960135ba61e7f6bf090be380a4b60beefcf0bf`。固定9:1、Tiny22K→1K公开fresh、backbone LR1e-4、head/age1e-3；上限24000成功更新，每800步配对验证online/EMA，取较高内部dev AUROC；首次建立基线，严格提升，连续4轮无提升早停。warmup与EMA warmcopy均1600，cosine horizon24000。

实际完成5600成功更新、7完整周期，attempted5600、skip0。第3轮后连续4轮未刷新内部AUROC，正常early_stopping；TRAIN、freeze及原一次官方组件均exit0，ROUND为complete。

| 来源 | checkpoint | AUROC | AP |
| --- | --- | --- | --- |
| 内部dev，5140条 | 第3轮EMA，2400步 | 0.9866645153288838 | 0.8193841805300731 |
| 官方，5849条，仅最终一次 | 同一冻结EMA | 0.9883091910818949 | 0.8341015210694182 |

内部唯一选点由实际14条配对结果产生，error_twice22150，tensor `82884e00eebc0ddd59b8c1165d4e5f8af3f6cc6119f1b3d154d0b49efaf6ce85`。`best_metrics`是online分支字段，最终selector正常选择EMA。官方不参与早停、选点或后续调参；raw AUROC未达0.999，总体任务未完成。

02:02:27北京时间的既有产物只读观测确认：8rank的原PID均不存在、freeze/official组件inactive、设备无使用者且8卡idle、原实例锁已释放。owner19269保留Z状态，不能称已reaped；未发送信号或重新取锁。本次观测无新前向、评估或launch。

非源码权威根：`/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r95-tiny22k-24k-p4-9010-20261029`。终态观测原件：`exp-v3/technical-prep/r95-frozen-992c3de8a079/MILESTONE-2.json`，Pod SHA `b7d44285745819a350b0e03cd1f126023dbc665997e9fa143a6e1083136777e5`。

本目录保存普通字节导回的小型原件，未在本机解析或重哈希：

- `SINGLE-MODEL-FREEZE.json`：完整14条内部结果及selected，Pod SHA `13aa989198ca9b4e64ed5e021f6da9b8733eda748878bd5750d19e2ebccc217b`。
- `TRAIN_RESULT.json`：Pod SHA `dd253b3aa0fdd33bb6a539f01ba596d7f5520dbea1547d543787fdae69f306c1`。
- `ROUND_RESULT.json`：Pod SHA `c98fcb7364fd59901816435480485ce817d9543d55452e7010a9337490b08e1e`。
- `TERMINAL-CPU-RECEIPT.json`：只读观测guard exit0，Pod SHA `e13a89fcc80614afcfe0307519e74dab1eb780396afbd6a15feb1d7d23fa59d8`。

权重、缓存、checkpoint与样本级预测仍保留exp-v3，不进入Git。后续科学比较只使用内部14条及运行合同，见todo中的R96决定。
