# 固定 ViT-B CLS 与条件 Small-CLS 后续准备

R190的37eeae6技术包已实际通过；R191同源首次技术打包nxk8yk6x/exit1暴露prepare.py仅给Small格式注入public162字段、Base漏注入，原失败产物保留。最小修复只将Base格式纳入同一162分支，验证器未放宽。[真实原prepare→load_config集成](PREPARE-PUBLIC162-CHECK.json)6.053秒通过，1evksm6d/`8d1b34feaf1f001dd15996c02a54cce64fb91cf93899f75a20c573b000f9ec4e` exit0，0模型/forward/GPU；其git-base37仅为未提交修复CPU fixture的基线标识，不表示新Git部署。[当前R191精确证据](PREPARE-FIX-CHECKS.json)为原7目标/13编译按SHA复用，加1真实prepare集成/1源码编译，共8目标/14唯一源码编译；原CPU-CHECKS.json完整保留。新prepare SHA `6687931e964e9652f2587ab0e9d005594023adb5c2624eb94f285634f167489e`，最终sxb7gt7n/`254883005828a1f1aab509ac11fc8717149e930a53fee54d04406fde44f2baba` exit0。R191将使用新未启动准备阶段，尚无formal/clock/GPU，原R189冻结与R190已就绪包不改。下文为首次模型准备记录。

R190/order103仅在完整R189严格过R107/R120双门且safe后，以原Small-CLS/3bf4720a/seed27独立fresh复核；否则仅技术准备。R191/order104为独立Base-CLS/c6a5fb7d/seed29，保留R189原512/ImageNet/logistic/G64P8/AdamW/LR/EMA/age/9:1/24k-800-p4，只改变同家族骨干容量和匹配公开初始化。现役R189仍原0f3冻结，后续尚未formal/clock/GPU。

[实际7个必要目标与13个唯一源码编译](CPU-CHECKS.json)通过。[模型3目标](MODEL-CHECKS.json)含Base原生CLS/逐tensor与RNG/age一次的独立oracle、真实CPU P8全部165有限梯度/5空Adam组/真实noBN EMA/CheckpointStore→inference冷图，以及Small None/CLS兼容；guard6sbt5vtu/781736f1…e92，78.845s。[另3目标](REMAINING-CHECKS.json)复用同一真实非持久RoPE copy/blend/恢复/冻结冷图fixture核Small与Base，并核两候选/保存job/来源绑定，zomn7j_h/44041cce…9dce，21.088s。该冻结fixture只将来源/数据准入与选点设为fixture，模型、EMA、blob、freeze与冷图真实执行，不冒充八rank准入。存储函数目标6nbxv49e/caf3a89c…8ec5亦0。

传输顺序错误77axtw6o/exit1保留：旧源码包覆盖了提前传入的测试文件，新模型断言未执行。确认传输退出后重传，真实RED w82b_1oh/dcb06f00…06ee复现实际工厂384维与独立Base oracle预期768维不符，随后GREEN通过。旧不变loss/WORLD/优化器实现按精确SHA复用，不重跑全套或GPU预演。

[R190实际public fresh](R190-PUBLIC-FRESH-INITIAL.json) whole `8c62435ac0c27cbbe3fa2dfd113386fb768582d3b877ee0e0eab13a374ba3427`/SCI `7fcf341895b74851988bd3a85704ebc87a0a11ee0bee2272353dd737377a089f`；[R191实际public fresh](R191-PUBLIC-FRESH-INITIAL.json) whole `c3495f5af134d01496fee241554a8e6113972f554bccb899eb6184cb92c09325`/SCI `f41f43a2f80fd93c1ffed36dc2a86bf1f22dc47a02610402850fbe25f310ef73`。两者165全训/5空Adam、原生pool token、真实首noBN warm-copy、0实图/0GPU，r2hy4d78/f44caa65…4f8a exit0。初始化占位未进Git。Base完整公开342624338B/162张量strict及许可在[来源证据](../vit-base-source-20261102/README.md)，非自有teacher或医学初始化。

复用现ViT工厂和消费者，仅显式区分Small384/Base768、模型/公开格式/保存job和描述身份，并将现非持久RoPE白名单扩到Base；原native512/12层/5prefix全部保留。Base为12头，Dropout.2/Linear768→1/age各一次。源256与比赛512区分；所列4个timm文件pin未覆盖整个transitive包，CPU证据不证明临床机制、AUROC、DCU峰值或吞吐。

R191精确新round使用16GiB，默认12GiB/全部旧例外/free16GiB不变。按实际85,641,987参数、训练model+EMA+Adam两moment共4份FP32，8个既有blob+初态+一个原子新blob的张量情景为13,017,582,024B，超过旧12GiB。该数不含metadata或全部故障并发峰值，16GiB实际GPU成功尚未证明。P8 CPU saved storage6,772,498,528B也不是DCU峰值上界。

完整24k时间场景以R1实际dispatch-inclusive 7200更新1811.056秒、按宽度平方4投影为24147.408秒，低于27750训练soft/31500总clock；没有假设提前早停。该运算量比例不是实际GPU kernel耗时上界，仍由现时窗/资源准入及真实运行预算约束。最终[编译/精确证据核对回执](CPU-FINAL-GUARD.json)u5lfpyk_/`6ef579b9c93e08c800f2fbe84e6c5bb18a8f19c2e69b24bda6e930a1130da56f` exit0，未重跑已过测试、加载模型/大权重或forward。原件在exp-v3/technical-prep/vit-base-20261102-32e3/；5de准备0–11与训练24–119互斥。raw .999未达。
