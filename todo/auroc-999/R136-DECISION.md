# R136：raw Conv2d/Linear weight gradient centralization

用户在固定GC及7既有文件方案后明确“继续执行”；根冻结派单授权reference `46de68b4-2831-4210-a11a-02f90e0d4af8`。唯一A负责源码/Git/Pod/GPU，规划和根只读review。R136/order48已查无占用；三个必要正向目标已有分阶段真实通过及7源compile，正在实际Git同步/精确部署与唯一normalformal准备，GPU尚未启动。旧五件/未知analysis保持隔离；不改或重入R135及旧clock/grant。

实际RED：两个算法2FAIL/0ERROR（1.444秒）、身份1FAIL/0ERROR（2.490秒）。GREEN1第1算法目标PASS，第2目标FAIL；定向DIAG1证实toy norm.bias为pairlog公共平移理论零梯度，约1e-9的FP32归约差被原Adam epsilon放大成1.08e-4参数差。仅给测试factory增加固定input gain使该方向可辨识，未改生产或放宽容差；第1目标不调用forward，保持原通过证据。随后仅第2目标GREEN2 1/1（9.372秒、CPU `61bbe7e8…`），身份GREEN1 1/1（3.413秒、CPU `03a710a5…`），不写成同一次最终3/3全跑。一次SSH在kex阶段失败、无Pod执行，原件保留后原连接普通只读重试成功。所有失败、诊断及最终7源pins见prep的`CPU-CHECKS.json` SHA `f7c10ce7163343b64a76584b21ca816637be416aecf26ca278bf13f19ba848ff`；最终SCI SHA `098768cc3eeb0e02f9fa742daa72b2fc076d461787538ad1b56caefaeb49817f`。没有独立public构造或GPU预演；正式50c/113目标/初态/重建及实际GPU事实仍待本轮既定正式链。

配置`configs/R136-SCIENCE.json`；profile `single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-rawgc-convlinear-v1`，candidate `r136-convnext-small-imagenet22k-to-1k-24k-p4-seed20261029-scale085-pairlog-rawgc-convlinear-9010`，round `r136-small22k-seed20261029-scale085-pairlog-rawgc-convlinear-24k-p4-9010-20261101`，prep `technical-prep/r136-20261101-7c17`。

直接回RGB3 R107 seed20261029/public fe6b fresh/345全训/预期whole50c/空Adam，不继承R135的XY、75482初态或任何checkpoint。唯一不可调policy为`raw-convlinear-weight-gc-v1`：只对实际nn.Conv2d和nn.Linear的直接weight（含binary head）FP32 raw gradient按所有非output轴减均值，在原DDP平均、mirror.merge、unscale之后及同一次globalclip1之前执行。bias/LN/age/LayerScale不纳入；实际Small目标预期40Conv+73Linear=113，由正常formal实证。GC改变clip系数和Adam矩，不宣称其他参数缩放不变或权重均值守恒。不新增WORLD系数、collective、forward/backward或grad clone/hash，不做post方向GC/权重投影菜单。

其余科学整件保持R107：固定9:1/原split seed20261024、512单视图scale085、原旋转光度/G64P8/FP32-noTF32/pairlog、原AdamW/LR-WD-clip-SD-dropout-age/1600warmup+EMA、max24000成功/eval800/p4/唯一selector。其他关闭optional缺省/None。只改批准的config/engine/distributed_engine/experiment91/check_candidate/test_gradient_clip_policy/test_candidate_binding七既有文件，无新源码或调用层，models/inference/EMA/persistence/launch/monitor保持。

两个算法目标和一个身份目标先真实RED再最小GREEN：独立逐行值/排除项/clip顺序/None RNG和序列化；实际两rank Gloo/DDP+原gather loss对独立global raw-gradient reference的Adam一二阶矩/step及保存加载后下一步EMA；旧R13547/新R13648精确身份及R107科学整件差异。只跑受影响目标/必要compile，不重旧套件或GPU预演。模型未变，无独立public控制构造；新实际Git push/readback后精确部署，唯一正常formal承接public strict/50c/113目标/345/实际组/Adam0及online-EMA/defaultfactory strict。真实8rank初态与log25自然成功沿原入口，不补step1/event/collective。

新CPU allocation SHA `9c1b00249e32a6fbc31b6c6bcb00f9978b11d52491692fde46d8eb17743695cf`，实测有效120，prep24/A12 mask0–11/root12–23保留不用，有效至1793540480.1967466；74bd已过期保持原文件，不修改续期。GPU新阶段仍独立当前UIDReady/资源与80%训练整树准入、9h grant/6h训练/8h整轮，Small12GiB/free20/floor16/300s保存保护不扩；既有8卡显式可见/uncontracted/native绑定/正确-i。上级Job不可读记unknown，不扩RBAC/资源/租期，不新agent/timer。R130全部/R120/KEEP/R134R1/R135 selected及public/prepared512和全部失败成本保护，fit诊断不修不重跑。

完整工程有效selected同时strict>R107 `.9884405731067525`且>R120 singlebest `.9894773092062407`才固定seed20261027对R108 `.9872328538178018`；任一不过关闭此raw Conv+Linear GC整体，不变位置/范围/均值组/系数/LR/seed救场。工程截断记相应unknown而非科学阴性；完整freeze≥.996才官方5849一次，raw.999未证不称总体完成。
