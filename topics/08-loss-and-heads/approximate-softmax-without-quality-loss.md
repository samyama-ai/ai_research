---
id: 08-loss-and-heads/approximate-softmax-without-quality-loss
title: "Approximate Softmax Without Quality Loss"
topic: 08-loss-and-heads
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Approximate Softmax Without Quality Loss

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/approximate-softmax-without-quality-loss` · **Status:** solved-but-impractical

## 1. Problem Statement

A language model ends in a linear head $W \in \mathbb{R}^{V \times d}$ followed by a softmax over $V$ tokens. Computing the exact normalizer costs $\Theta(Vd)$ per token. The question: can that be reduced to $o(Vd)$ — sublinear in vocabulary — with **no** measurable loss in downstream quality?

Three variants, routinely conflated:

- **Method variant.** Build an estimator $\hat{p}$ of the softmax $p$ using $o(V)$ inner products per token, such that a model trained (or decoded) with $\hat{p}$ matches a full-softmax control on held-out perplexity and task accuracy.
- **Theory variant.** Bound the excess risk of an $\epsilon$-approximate normalizer, and decide whether sublinear-time exact-argmax retrieval over an *adversarially chosen* $W$ is possible at all.
- **Measurement variant.** Fix a metric that detects tail damage. Perplexity averages $-\log p$ over a corpus dominated by high-frequency tokens; approximation error concentrates on rare tokens, which the metric barely weighs.

**Solved-but-impractical** is the honest status: many methods hit $o(Vd)$ with small perplexity deltas, but on modern hardware exact fused kernels erased the memory motivation, and Amdahl's law caps the compute win at ~20% of training FLOPs. The approximations win only in a regime ($V \gtrsim 10^6$, small $d$) nobody trains in.

## 2. Formal Setting

Hidden state $h \in \mathbb{R}^d$, logits $z_i = w_i^\top h + b_i$, and

$$p_i = \frac{e^{z_i}}{Z}, \qquad Z = \sum_{j=1}^{V} e^{z_j}.$$

**Quantities as measured.**

- **Cost.** $C = $ FLOPs per token in the head, $2Vd$ forward, $6Vd$ forward+backward. Measured as wall-clock ms/step on a named accelerator, not as an asymptotic.
- **Head share.** $\rho = 6Vd / 6N$ where $N$ is total parameters — the fraction of training FLOPs an approximation can touch. Any end-to-end speedup is bounded by $\rho$.
- **Normalizer error.** $\delta = |\log \hat{Z} - \log Z|$, measured by computing $Z$ exactly on a held-out sample of $10^4$–$10^5$ positions.
- **Quality.** $\Delta\text{PPL}$ against a full-softmax control trained on identical data, seed, and step count. Tail-sensitive companion: $\Delta\text{PPL}$ restricted to tokens with corpus frequency below $10^{-5}$, and $\mathrm{KL}(p \,\|\, \hat{p})$ at matched positions.
- **Gradient bias.** $b = \mathbb{E}[\hat{g}] - g$ where $g = \nabla_\theta(-\log p_y)$; measurable by Monte Carlo against an exact head at small $V$.

**Assumptions, and which fail.**

1. *Logits are approximately low-rank / clusterable.* Holds partially; a rank-$d$ head with $d \ll V$ is exactly the softmax bottleneck (§4), which is a modeling defect, not a free structure.
2. *Sampled negatives give unbiased gradients.* **Violated.** Blanc & Rendle (ICML 2018) show sampled softmax is unbiased only when the proposal is $\propto e^{z_j}$ — the quantity being avoided.
3. *Norms are bounded so MIPS reduces to nearest-neighbor search.* Violated in practice: token-embedding norms in trained heads span more than an order of magnitude and correlate with frequency.
4. *Perplexity detects the damage.* **Violated by construction** — see §6.

## 3. State of the Art

**Established (with control arms and independent replication).**

- **Adaptive softmax** (Grave, Joulin, Cissé, Grangier, Jégou, ICML 2017): frequency-clustered hierarchical head tuned to GPU matmul shapes. 2–10× head speedup; LSTM reaching 43.9 perplexity on One Billion Word using a single GPU. Widely reproduced in fairseq.
- **Comparative ablation** (Chen, Grangier, Auli, ACL 2016): full softmax, hierarchical softmax, differentiated softmax, target sampling, NCE, self-normalization, all at equal training time on One Billion Word. Conclusion that has held: at *equal compute* the cheap heads win; at *equal steps* full softmax wins. No approximation dominates on both axes.
- **NCE consistency** (Gutmann & Hyvärinen, JMLR 2012; Ma & Collins, EMNLP 2018): consistent and asymptotically normal for conditional models under stated conditions; finite-sample variance grows with the ratio of noise to data samples.
- **Exact-but-cheap kernels** (Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models*, 2024): computes cross entropy without materializing the logit matrix. Reported reduction of loss-computation memory from 24 GB to 1 MB for Gemma 2 (2B) at batch 8192. This is exact — it removes the main reason approximations were adopted.

**Claimed but unablated / benchmark-only.**

- LSH- and graph-based MIPS decoding (Mussmann & Ermon ICML 2016; Zhang et al., *Navigating with Graph Representations*, NeurIPS 2018; Chen et al., *Learning to Screen*, ICLR 2019): report 10–100× decode speedups with "negligible" accuracy loss, but almost always on top-1/top-5 recall, not on calibrated sampling or long-form generation quality. No published evaluation of nucleus sampling under a screened head at $V > 100\text{k}$.
- SVD-softmax (Shim et al., NeurIPS 2017): two-pass low-rank preview then exact refinement of a candidate set. Benchmark numbers only; no training-time ablation at modern scale.

## 4. What Is Known

- **Head share is small and shrinking.** Llama 3 8B: $V=128{,}256$, $d=4096$ → 525M head params, 6.4% of the model. Gemma 2 2B: $V=256{,}128$, $d=2304$ → 590M, about 23% of training FLOPs. For a 70B model with $V=128$k, $\rho < 1\%$.
- **Softmax bottleneck.** Yang et al. (ICLR 2018): a rank-$d$ log-probability matrix cannot express an empirical log-probability matrix of rank $> d$. Mixture-of-Softmaxes reached 54.44 PPL on Penn Treebank and 61.45 on WikiText-2, against ~57.3 / ~65.8 for the AWD-LSTM baseline — at roughly 2–3× head cost. Approximation and expressivity pull in opposite directions.
- **Unargmaxable classes.** Demeter, Kimmel, Downey (ACL 2020) and Grivas, Bogoychev, Lopez (2022): tokens whose embeddings lie inside the convex hull of the others can never be argmax for any $h$. Present in real trained models with $d \approx 512$; rare at $d \ge 2048$.
- **Sublinear MIPS is conditionally hard.** Under SETH, approximate near-neighbor search in high dimension admits no truly subquadratic batch algorithm (Rubinstein, STOC 2018). Data-dependent indices work because trained $W$ is benign, not because worst-case sublinear retrieval exists.
- **Self-normalization works at inference, not training.** Devlin et al. (ACL 2014) penalize $(\log Z)^2$ and skip normalization at decode; Andreas & Klein (2015) show the penalty holds only near the training distribution.

## 5. What Is Not Known

- **Theoretically open.** No excess-risk bound linking normalizer error $\delta$ to downstream generation quality. We do not have a theorem of the form: $\delta \le \epsilon$ uniformly $\Rightarrow$ total-variation distance of $T$-step generated sequences $\le f(\epsilon, T)$. Also open: whether *any* estimator achieving $o(V)$ inner products can have gradient bias that vanishes without a proposal depending on $Z$.
- **Empirically open.** Nobody has trained a $\ge$7B model to $\ge$1T tokens with an approximate head and reported a matched-compute, matched-data comparison against full softmax on tail perplexity and long-form generation. Every published ablation is at LSTM/1B-word scale or below.
- **Methodologically blocked.** There is no accepted metric for "tail damage". Rare-token perplexity, calibration error on the bottom 90% of the vocabulary, and generation-diversity statistics all measure different things, and no study reports which one predicts human-visible degradation.

## 6. Why It Is Hard

The obstruction is **an evaluation that does not measure the thing it names**, compounded by **Amdahl's law**.

Perplexity is $\exp$ of a frequency-weighted mean of $-\log p_y$. In a natural corpus the top 1000 tokens carry ~85–90% of the mass. An approximation that reconstructs the head of the distribution and mangles the tail moves perplexity by well under 1% while changing which rare tokens are reachable at all. So the standard metric certifies a method whose failure mode it cannot see — and the failure surfaces at decode, over hundreds of sampling steps, where the errors compound.

Second, the payoff ceiling. With $\rho \approx 0.23$ for a 2B model and $\rho < 0.07$ for an 8B, eliminating 97% of head FLOPs buys at most ~22% and ~6% end-to-end. Exact chunked kernels already deliver the memory saving with zero bias. The remaining prize is too small to justify an unbounded risk on a $10^{25}$-FLOP run — which is why the problem is *solved-but-impractical* rather than open.

## 7. Current Research (as of 2026)

- **Exact-kernel displacement.** Cut Cross-Entropy (Apple) and the Liger fused linear-cross-entropy kernels (LinkedIn) are the default answer in open training stacks. Direction: chunked/streaming loss with recomputed logits.
- **Vocabulary scaling laws.** Tao et al. (2024) argue optimal vocabulary grows sublinearly with parameters — if $V$ grows slower than $N$, $\rho$ shrinks and the problem self-dissolves. *(frontier — verify the exponent.)*
- **Retrieval-shaped heads.** Ongoing work at Meta/Google on multi-token-prediction and byte-level heads sidesteps $V$ entirely rather than approximating it. *(frontier — verify.)*
- **Decode-time candidate generation.** Speculative decoding replaced most of the motivation for MIPS-screened heads: it accelerates the whole forward pass, not just the head, and is exactness-preserving by construction.

## 8. Concrete Next Experiment

**Question.** Does an approximate head cost anything a matched-compute control can detect?

- **Scale.** Two 1.4B-parameter decoder models, $V = 256{,}128$, $d = 2048$, trained on 100B tokens of the same shuffled corpus with identical seed, schedule, and step count.
- **Arms.** (A) Control: exact fused cross-entropy. (B) Sampled softmax, 8192 shared negatives per batch drawn from the unigram$^{0.75}$ proposal, with the standard log-proposal correction.
- **Budget.** ~$1.7\times10^{21}$ FLOPs total; roughly 3k H100-hours per arm.
- **Deciding number.** Perplexity restricted to tokens with corpus frequency below $10^{-5}$, on a 50M-token held-out set. **Decision rule: if $\Delta\text{PPL}_{\text{tail}} \le 1\%$ with a bootstrap 95% CI excluding 2%, the approximation is quality-neutral at this scale; if it exceeds 3%, sampled softmax is disqualified for production heads.** Report overall perplexity alongside — the expected result is $\Delta\text{PPL}_{\text{overall}} < 0.3\%$ regardless, which is precisely the point about the metric.
- **Secondary readout.** Measured wall-clock speedup. If it lands below 15%, the experiment also settles the practical question in the negative.

## 9. Key References

- **[Foundational]** Morin, F., Bengio, Y. *Hierarchical Probabilistic Neural Network Language Model.* AISTATS, 2005.
- **[Foundational]** Gutmann, M., Hyvärinen, A. *Noise-Contrastive Estimation of Unnormalized Statistical Models, with Applications to Natural Image Statistics.* JMLR 13, 2012.
- **[Foundational]** Bengio, Y., Sénécal, J.-S. *Adaptive Importance Sampling to Accelerate Training of a Neural Probabilistic Language Model.* IEEE TNN, 2008.
- **[SOTA]** Grave, E., Joulin, A., Cissé, M., Grangier, D., Jégou, H. *Efficient Softmax Approximation for GPUs.* ICML, 2017. — arXiv:1609.04309
- **[SOTA]** Wijmans, E., Huval, B., Hertzberg, A., Koltun, V., Krähenbühl, P. *Cut Your Losses in Large-Vocabulary Language Models.* 2024. — arXiv:2411.09009
- **[Ablation]** Chen, W., Grangier, D., Auli, M. *Strategies for Training Large Vocabulary Neural Language Models.* ACL, 2016. — arXiv:1512.04906
- **[Theory]** Blanc, G., Rendle, S. *Adaptive Sampled Softmax with Kernel Based Sampling.* ICML, 2018.
- **[Theory]** Ma, Z., Collins, M. *Noise Contrastive Estimation and Negative Sampling for Conditional Models: Consistency and Statistical Efficiency.* EMNLP, 2018.
- **[Theory]** Yang, Z., Dai, Z., Salakhutdinov, R., Cohen, W. *Breaking the Softmax Bottleneck: A High-Rank RNN Language Model.* ICLR, 2018. — arXiv:1711.03953
- **[Theory]** Rubinstein, A. *Hardness of Approximate Nearest Neighbor Search.* STOC, 2018.
- **[Related]** Demeter, D., Kimmel, G., Downey, D. *Stolen Probability: A Structural Weakness of Neural Language Models.* ACL, 2020.
- **[Related]** Shim, K., Lee, M., Choi, I., Boo, Y., Sung, W. *SVD-Softmax: Fast Softmax Approximation on Large Vocabulary Neural Networks.* NeurIPS, 2017.
- **[Related]** Devlin, J., Zbib, R., Huang, Z., Lamar, T., Schwartz, R., Makhoul, J. *Fast and Robust Neural Network Joint Models for Statistical Machine Translation.* ACL, 2014.

## 10. Worked Example

Take a Gemma-2-2B-shaped head: $V = 256{,}128$, $d = 2304$, $N = 2.6\times10^9$.

**Cost.** Head forward+backward: $6Vd = 6 \times 256{,}128 \times 2304 \approx 3.54$ GFLOP/token. Whole model: $6N \approx 15.6$ GFLOP/token. So $\rho = 22.7\%$.

Replace the head with sampled softmax, 8192 shared negatives: $6 \times 8192 \times 2304 \approx 0.11$ GFLOP/token, a $31\times$ head reduction. End-to-end:

$$\text{speedup} = \frac{1}{0.773 + 0.227/31} = 1.28\times.$$

A 28% ceiling, before any kernel-efficiency loss from the irregular gather.

**Quality.** Now the same head under the metric. Suppose at a given position the true $Z = 100$, the top token contributes $e^{z_1} = 60$, the next 999 tokens contribute 35, and the remaining 255,128 tokens contribute 5. Sampling 8192 negatives uniformly covers 3.2% of the vocabulary; the tail estimate $\hat{Z}_{\text{tail}}$ has relative standard deviation on the order of $\sqrt{(1-0.032)/(0.032 \cdot m_{\text{eff}})}$, and with heavy-tailed $e^{z_j}$ the effective sample count $m_{\text{eff}}$ is far below 8192 — a single unsampled token with $z_j$ two nats above the tail mean shifts $\hat{Z}$ by several percent.

Push that through both metrics. A 5% error in $\hat{Z}$ is $\log 1.05 = 0.049$ nats. On the target token, if it is a frequent one, $-\log \hat{p}_y$ moves from 0.51 to 0.56 nats — but frequent tokens are also the ones the sampler reliably includes, so in practice their error is near zero. The error lands on rare targets. If rare tokens are 8% of positions and each absorbs 0.05 nats of bias, overall cross entropy rises by $0.08 \times 0.05 = 0.004$ nats: **perplexity moves 0.4%** — inside seed noise for a 100B-token run. Tail-restricted perplexity moves the full 5%.

**The obstruction, made visible.** The method buys at most 28% wall-clock, injects a bias that the headline metric reports as 0.4% (indistinguishable from noise), and concentrates that bias exactly where no standard evaluation looks. Meanwhile the exact fused kernel delivers the memory saving — 24 GB to ~1 MB of logit storage — at zero bias and roughly the same FLOPs. That asymmetry, not any missing algorithm, is why the problem sits at *solved-but-impractical*.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*