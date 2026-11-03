# R208/R209：原生 AMSGrad CPU 证据

[最终CPU汇总](CPU-CHECKS.json) SHA `862ad766e5f0aa536a70bbead4c7a6049934dbbd93acacbeb2cebb981483184d`：6个必要目标、6份源码编译、两actual public fresh通过；[最终guards8jhvna9](CPU-FINAL-GUARD.json) / `fb7f5ebfa8fbe0b83e156cde232866c558f43accc332699205cb477d25d4e9c5` exit0。ec5/root0–11/nice19/隐藏GPU，与当前R203训练24–119互斥。尚待Git冻结/实际回读与两technical包，无新normal、clock或GPU。

[新优化器两目标](MATH-GREEN.json) guardxvs5o8k0 exit0，1.573秒：真实原生AdamW五组全amsgrad=True、原LR/WD/age/默认beta(.9,.999)/eps1e-8/foreach与fused=None，初始state空、无RNG采样。三步独立FP64 oracle核参数/m/v/raw vmax/step；首大梯度后小/零梯度令vmax严格大于当前v，能区分普通AdamW，也明确先raw max再bias correction。Toy原生torch.save/load后第4步参数与全部4种state逐项exact；不冒充生产CheckpointStore或八rank断点恢复实测。None/省略的原生默认三步、RNG、配置load和保存身份保持，非法policy及beta095/Lion叠加拒绝。

[候选两目标](BINDING-GREEN.json) guardmo0qncoi exit0，9.736秒：新R208/R209及旧R206/R207身份/source/9:1/结构/日程/早停/保存job核验。R208仅R207逐pair平方hinge的seed27条件复核；R209直接R107原Small/publicfe6/seed29/512/logistic/G64P8，仅新增`adamw_amsgrad_policy='adamw-amsgrad-v1'`，同名字段进入candidate与job，validate_job逐项比较。R209显式官方>=.996门沿现R120，历史R107不改。[旧beta095/默认两目标](LEGACY-GREEN.json) guardyl7q0bq8 exit0，1.524秒，原行为和更新数值保持。

[两actual public fresh](PUBLIC-FRESH-SUMMARY.json) guardjkpjy1jv exit0：R208 whole91507a65 / SCI `e4c99791df4441e0cbe09fca65628b20a00c30317c18d5a6c1ff81541faf9288`；R209 whole50c6 / SCI `86272648c18e596c702eb46c131fc55c893897f89df7cf8f1b4cfb538cebf7d9`。均345全训/49,455,459参数/无buffer/5空组，0图像forward或训练checkpoint/官方读取。R209五组实际amsgrad=True，R208全False，原beta/eps/LR/WD及age保持。

实际FP32参数字节197,821,836，AMSGrad每次完整成功状态增加同量max_exp_avg_sq。R209主要浮点情景9C+两份参数initial为9,297,626,292字节，再加一个原子新C为10,286,735,472，小于原12GiB；排除步数/RNG/metadata/序列化、workspace及其他并发写，不是保存/设备峰值或容量保证。默认12GiB/旧例外/free16GiB不改，正式资源与真实保存继续由原guard消费。

永久实现只动config/engine/experiment91/check_candidate及必要测试/两SCI。去掉新增4行AMSGrad构造分支后原_optimizer逐字符不变；model/loss/gather/DDP/EMA/age/precision/fresh/保存推理等按不变SHA复用。既有prepare私有helper只加两身份与本6目标6编译证据消费，不新建运行链。[设计](../../../todo/auroc-999/R208-R209-PREPARATION.md)依据已读PyTorch2.10原生公式；raw v历史max不等于校正分母或有效LR必单调，不证明临床机制、收敛或AUROC收益。

prepare helper SHA `bc14180349f1b808a9f042268596f7a44683d079414e3179fb6bbb3e5fde718d`，逆向移除四处身份/证据增补后与原89857348逐字符相同，原helper/storage消费段保持；[guardhzbnpv8x](CPU-TRANSPORT-HELPER.json) / `6037dd31c73e04ad61c2a46e4f8ffd688ac947baeeadeba097ef95857d51d201` exit0及compile通过。后续仍沿同一normal/冻结/启动入口，尚未创建正式clock或启动GPU。

失败原件：[MATH-RED](MATH-RED.json) guard0tg0d2c6/0f81e033因原amsgrad=False与新None身份未省略，两项预期FAIL；[BINDING-RED](BINDING-RED.json) guardninimcw2/be10c71a因新profile未登记。[6源码compile](CPU-COMPILE.json) guardcq54p9m4/79928dc5及[fresh guard](CPU-FRESH.json) / 51e125d6均exit0。原件在`exp-v3/technical-prep/amsgrad-20261103-32e3`；本准备未改R203/573或既定7ead/78冻结。
