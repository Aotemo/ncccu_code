# R96配置接线窄验证

在实时UID/Ready核验的既有计算Pod、验证allocation3的A12份额内，现有candidate binding的9项检查全部通过，wall3.340秒；源码在检查期间不变，无模型载入或GPU。没有新增case、重复早停套件或独立GPU预演。

`CONFIGURATION-CHECKS.json`原Pod SHA为`4917a2b0e0dfc6dada12e861170cb802ecc70951dcaea0cade24530204e72444`；guard `CPU-RECEIPT.json`原Pod SHA为`263834a1b2767a979b2a2c7d15f2680bac85978513e3bf3fd9a2a89cc2fb6809`，exit0。原始输出为`CHECKS-OUTPUT.txt`，三件均普通字节导回，未本机解析/重哈希。

实际配置对照确认R95→R96仅job.backbone_lr从0.0001变为0.00005；配置profile/比较说明及候选identity另作版本登记，其余科学字段相同。此证据是配置接线检查，不宣称R96已训练或实际GPU通过。后续由新Git冻结包与真实训练初态接管。
