# R96：backbone LR5e-5

冻结Git为`35ad0cb11854d57511e07685d514316c09725abd`，tree为`a9a1decbe4fb9c6b1fa4c24eb97ecca100e56f68`。直接对照R95，仅将backbone LR由1e-4降为5e-5；head/age1e-3、公开fresh/seed、24k上限/800每周期/内部maxonlineEMA AUROC连续4轮无提升早停、绝对warmup与EMA warmcopy1600等均保留。

实际6400成功更新、8完整周期、attempted6400/skip0；第4轮EMA后c5–c8无新高，bad_cycles4，合法early_stopping。TRAIN、freeze及原一次官方组件均exit0，ROUND complete。

| 来源 | 唯一checkpoint | AUROC | AP |
| --- | --- | --- | --- |
| 内部dev，5140条 | c4 EMA@3200 | 0.983341180097075 | 0.825803312915609 |
| 官方5849，仅最终一次 | 同一冻结EMA | 0.9889791408726744 | 0.8389791978694786 |

完整16条内部结果选择error_twice27670，tensor `dcce91a2737f50d59e2d8b028d8fcf40dbb76e773d5bee420d357c487ea2d997`。相对同24k/p4政策下的R95，所选内部AUROC较低、AP较高；不能称所有指标均退步或据单seed诊断机制。官方不参与后续方案、早停或选点，raw AUROC未达0.999。

03:04观测时官方仍活跃；03:09:30的后续既有产物只读观测确认ROUND/done完成，8rank及train/freeze/official组件inactive，GPU无使用者且idle、原实例锁已释放。owner27571保留Z，不称reaped。未重新forward/评估、发送信号、取锁或重置clock。终态原件为`exp-v3/technical-prep/r96-frozen-35ad0cb11854/TERMINAL-OBSERVATION-2.json`，Pod SHA `4ebf9a8f47cfa5e5773613b6823f0ca6417d252915207733519c7fa2687b9367`。

权威运行根：`/public/home/acfwvv9ejn/SothisAI/exp-v3/rounds/r96-tiny22k-lr5e5-24k-p4-9010-20261029`。普通字节导回本目录原件：

- `SINGLE-MODEL-FREEZE.json`完整16条和selected，Pod SHA `4ecfa8cd95dfa438a6ae6cdfb73fd9ae1998179aff5b768892b5c85eefb84e8b`。
- `TRAIN_RESULT.json`，Pod SHA `967c225f596fb3109cab091c04b139b8c85df7e9c9533debf66e47374adb45da`。
- `ROUND_RESULT.json`含原一次官方终态；其Pod SHA见上述终态观测的source_snapshots。

本次最终只读与R97配置窄验共享一个验证池任务，guard原Pod SHA `41f06fd261ce481adbb94814bb2a2382701950b427b21ba2ebd82a834045a507`，原件归于`records/preparation/r97-head5e4/CPU-RECEIPT.json`。未在本机解析或重哈希；权重、checkpoint与样本级预测留exp-v3。后续科学决定使用完整内部证据，见R97决策，不因官方结果更改既定方案。
