# R99绑定Dropout配置窄验

主对照R95，仅job.dropout从0.2提高到0.4；backbone1e-4/head1e-3/age1e-3、24k/800/p4/绝对1600不变。原生模型factory已消费job.dropout，本次仅把experiment91三处旧固定p期望改读同一pinnedSCIENCE，保留真实nn.Dropout类型及实际p的非tensor配置校验。既有fixture使用该绑定配置的p，无新增套件/拒绝case/模型默认修改。

实时Pod UID/Ready核对后，validation4预算的A mask0..11、nice19、隐藏CUDA/HIP/ROCR中执行现有9项candidate tests，9/9通过、exit0；测试0.638秒、配置检查wall0.887秒，前后源码SHA一致。脚本在Pod确认相对R95 job仅dropout.2→.4；没有新模型load、forward或GPU预演。

独立prep `/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep/r99-drop04-20261029-a`。CHECK原件Pod SHA `de9c0645edc2e86b581c60d85df8ee1a91fa7dc9fe00d746a25c0e1f0beab474`，CPU原件SHA `5960a449757bd29f5ce9c6ea2c646f020a0493bb28a9f8ed603ff869e313e9b0`。配置SHA `f9a15261bda9210c42054623038a300c64157a2ff02a041f0da32984aae07e90`，experiment91源码 `8ee0256397c6e3976835dc639866c6958a8d88c3b76942dbf24957935591d292`，fixture `b9cff117319c168eeae61b5e9e6c6969f604635ab9d1fda7980908b67956ea21`。均来自Pod原件，本机只普通字节传输，不解析/重哈希。

新正式包仍经现有原生CPU身份链产生真实模型p、fresh初态/参数组/空Adam证据，不能以本9项冒称GPU已启动。R98原一次官方及整轮尚未安全退场前，不启动R99 GPU。
