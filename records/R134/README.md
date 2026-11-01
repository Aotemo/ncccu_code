# R134：MixStyle 初态前技术失败

运行Git `15f78f725c6b5aad23da308bb7076801a831e646`，parent `a6e65dedee3a460164ff70b33b5f1e4748c25c9f`，tree `58d8d135893407ef0acb19e122f3b976a81f0d45`；order45、SCI SHA `94116a481cc3f35aadd7a82410b1e850d17a6c9315d970f49055af88755a5711`。直接R107/public fresh345全训，只加固定早两stage rank-local8 MixStyle，无Tri3/outputreg/LoRA/layerwise或旧checkpoint。原R133完整科学阴性另存，不把本轮技术失败与其拼接。

[精简失败记录](R134-FAILURE.json)与[唯一completion](COMPLETION-FAILED-R134.json)记录：09:22:53.495411 status failed，09:23:58.1901767 done failed、selected null，TRAIN/ROUND/freeze/official缺件；成功/尝试/eval游标字段未报告，不补写0。没有initial-ranks、ACTUAL_FRESH_INITIAL或记录成功update；两门为null，MixStyle尚无科学结论，官方未进入，不套threshold_not_met。

失败摘要SHA `f7ec5c14620a71d9cd0e8c274ad6d7bf50600f3f68542a411a1595daa5a84a1a`，completion SHA `dad39c1bc71de8bfca90ab2473b62872f9e0eb08386dca8cdb350b1ae5549bf0`；均在Pod内从已存在原件校核，原raw诊断/启动pending/全部阶段日志仍留独立prep，不把归档等同新的资源观察或恢复验证。

各rank的实际首异常为`ValueError: exact MixStyle policy and explicit attempt context required`。调用链`distributed_engine.py:324 prime_reducer → :122`既有dummy forward漏context，经`age.py:107 → models.py:466 → mixstyle.py:129/:53`拒绝。CPU数学/保存目标、单项identity及normalformal未覆盖该真实reducer边界；这些已过证据各自保持，不冒八rank训练成功。完整原bounded诊断留`exp-v3/technical-prep/r134-20261101-7c17/FAILED-DIAGNOSTIC.json`，SHA `7c28afdb014cc1c6c2b254471c4d49c2d27dfdb21c413c04e068a7858cc8678c`；Git只存精简trace，不复制NCCL环境噪声。

本轮唯一身份实际RED 1FAIL/0ERROR→GREEN 1/1；normalformal唯一一次18/18（98.012739658秒），745源Git blob/mode exact，公开whole50c、345全训/实际5组/Adam0及正式两site descriptor/onlineEMA/默认factory strict重建通过。正式CPU SHA `1188136e3bba09c6c4c9ac37e55b5c4b82a81918676ec7e3bc0b90826885d599`，deployment `e7d86f27e6eaaf83b6f851d84e0b9b023b11f434978013c872066dd88a9c3dc2`、SITE `c1aa829f7112fdcd619cee9b42b6e1811fa4656187e73686169717f1428ee869`。13项absence/原调用0后一次activation0和live launch0；这仅证明入口成功。第一小件09:22:17.676596 owner366816/start352346560及8rank366828–35 argv/start active、真实96核/control112–119+rank24–111，第二小件已退出；初态门未完成，未观察首步Adam或成功log25/采样trace。

actualsafe true/terminal_idle：owner同startZ非reaped、八rankabsent、锁文件仍在/holders空、设备users空、scanDenied0、八expectedPCI全seen/busy0。[原CPU guard](TERMINAL-CPU-RECEIPT.json) SHA `bc623050dcb9988b7ebdb76758a413798f2132ccf3546b667d083a5f7c89fb4a`/exit0。安全来自唯一已有completion，归档不重复观察，不hash/load旧blob或新增GPU forward。

原one-shot已消费；运行包、全部clock/token/grant/失败记录和真实成本保留，不重入/热改/删除marker。后续最小技术修复及独立新身份由根另冻结，不能将重启伪装本轮合法恢复或把技术失败记科学阴性。所有R1301503、R1/R131/R132/R133/R120/KEEP及共享依赖保护。raw AUROC `.999`仍未证。
