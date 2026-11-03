# fit-only 遮挡空间重建准备

R224固定检验公开Small在fit-only空间RGB重建后，原R107监督合同能否改善完整内部排序；R225仅完整selected严格胜R107及R120后的固定seed27。适配仍1600成功更新，不延长R115；从16×16个32像素块中遮154块，单视图、归一化零遮挡、masked像素L1，以局部空间重建补充原VICReg的全局两视图信号。监督512/增强/age/G64P8/448 logistic/AdamW/LR/EMA/24k/800/patience4保持，临时decoder丢弃，监督空优化器重建。

7个必要CPU目标和6源编译通过：4项mask/无泄漏/独立梯度与DDP均值/固定配方检查，2项旧SSL采样及derived交接回归，1项两候选绑定。前三新目标的实现SHA及测试方法AST一致后复用，不重复运行。原始3FAIL及候选1FAIL回执保留。[CPU汇总](CPU-CHECKS.json) SHA `e7d86b007f4f858fed03fc5c6823556cedce3b79f237968243b2245a2ae671ab`；最终实核c53b14e6 exit0，未受影响链路没有重跑全套。

[真实P8路径](ACTUAL-P8-CPU.json)使用同公开Small源和8张真实fit图，57.817秒：实际初态50c6f55c与原R107一致、公开344张量strict exact/FP32 finite，8×3×512×512空间重建、344梯度finite、encoder改变/head与age不变、冷derived严格重建/监督5组空AdamW均通过。CPU fixture已删除且从不进入训练；不是GPU峰值、八rank实测或1600次正式适配。R225当前仅引用同job/public的R108初态91507a65控制，未声称新fresh构造或启动。

原技术package（site979865ac）仅CPU准备，无clock/GPU。首裸fresh脚本因缺技术site上下文被生产校验拒绝，未前向/未改生产门；后沿现有prepare入口真实绑定再验证。结果归档的两次接入失败保留，均发生在必要测试通过后，未重跑已过测试或变更模型；后续源身份核验独立exit0。正式Git冻结/正式CPU/部署/启动由后续回执另证，不能把本页登记当成已运行。

依据为[RETFound眼底masked重建研究](https://www.nature.com/articles/s41586-023-06555-x)和[SimMIM空间重建研究](https://arxiv.org/abs/2111.09886)。本项目的密集CNN、样本量和短预算不同，不复现其规模或承诺收益；方法无法找回512监督输入以外的细节。先前[36例辅助盲态审阅](../../analysis/20261103-structure-review/README.md)没有确诊唯一瓶颈、临床决定性或标签一致性。只用现有fit图，未读取官方、dev训练图/标签、额外任务数据、医学预训练或旧teacher；所有计算位于24核0–23准备池，与训练24–119互斥。
