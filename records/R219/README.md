# R219：Small 自然图像12K→1K公开初始化

14:47[真实八rank与首成功](R219-START.json)已证：50 success=attempt、0跳过，345全训、五空优化器组、初态568263ab与正式CPU一致。自然首25 loss .5033292770385742有限，WORLD8/G64P8/448pairs，每rank8主图/8视图。14:53实观850次成功更新、eval1，暂内部AUROC .976749898553988，尚无完整selected或收益结论，raw .999未达。

直接R107原Small/seed29/512/ImageNet输入、原logistic/default AdamW-LR-WD、增强、age、globalclip、EMA1600/.999、24k/800/p4；仅换为合法自然公开Small12k→1k初始化。来源revision e9f96b3e、original0a127fb2、converted553fd294；来源384/squash元数据不改变比赛512预处理，不叠加R217长EMA。完整内部唯一所选>=.996才官方5849一次。

冻结source `87cc9198faa392c5cb9ed8346e3d14dd19f72516` / tree `db77f0baff725583f09a50475d438ef1834b952a`，SCIca352c2e，actual SITE5c4672b8。[7目标5compile与两fresh](../preparation/small12k-20261103/README.md)按SHA[复用](R219-CPU-REUSE.json)；[正式CPU18/18](R219-CPU-PREPARE.json)与[精确部署](R219-PREPARATION.json)均在R217训练期间完成，完整2369文件bytes/mode已核。默认12GiB/free16GiB保持。

[窗口检查](R219-CPU-WINDOW.json)、[一次activation](R219-CPU-ACTIVATION.json)及显式8设备direct launch均0；start1/start2未知阶段原件保留，start3及[首成功guard](R219-CPU-FIRST-SUCCESS.json)为实际证据，不追加GPU预演或重复launch。generic非Lion字段不证明逐rank完整AdamW kwargs或moment数量。未来准入12h为已验证配置，训练clock31500秒及旧grant均未改。

R217完整EMA .9875435134155578、双门false，R218不eligible，唯一actualsafe后接续。[实际间隔](R219-HANDOFF-GAP.json)：整轮结束至dispatch87.311秒、设备143.537秒、首25记录202.611秒；末训练事件至三项150.327/206.553/265.627秒，均<300。连续DCU利用率未测。

owner757838/start374447810，rank757857–757864；实际birth与mask见初态原件。训练24–119/96核与准备7c7af22c/root0–11、保留peer12–23互斥，当前私有r219-pod-context.env/ready-headroom/start3；32e3唯一Git/GPU，不恢复派生或发消息。R217/R219同封存256图fit诊断为独立CPU描述，不改变本轮冻结或选点。
