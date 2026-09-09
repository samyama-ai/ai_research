---
id: 11-inference-and-serving/multi-token-prediction-correctness
title: "Verification-Free Multi-Token Prediction Correctness"
topic: 11-inference-and-serving
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Verification-Free Multi-Token Prediction Correctness

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/multi-token-prediction-correctness` · **Status:** open

## 1. Problem Statement

Multi-token prediction (MTP) attaches $k$ heads to a transformer trunk so that one forward pass emits $k$ tokens. Two deployment modes exist:

- **Verified.** The $k$ tokens are a draft; a target-model pass accepts or rejects them by the speculative-sampling rule. Output distribution is provably unchanged; the only cost is throughput.
- **Verification-free.** The $k$ tokens are committed directly. Throughput gain is the full $k\times$ minus head cost, but nothing guarantees the emitted string is a sample from the model.

The problem is the second mode. **Input:** a context $c$, an MTP model, a block size $k$, a temperature $T$. **Output:** a decision predicate — for which $(k, T, \text{task}, \text{model scale})$ does verification-free block decoding leave end-task performance statistically indistinguishable from autoregressive (AR) decoding? **Solved** means: a computable, pre-deployment criterion that predicts, per context or per model, the block size at which committing without verification is safe, with an error bound.

Three variants that are usually conflated:

- **Measurement.** What is the right correctness metric? Greedy benchmark accuracy, KL to the AR distribution, and human preference disagree, and the field mostly reports the first.
- **Method.** Can heads be trained (joint factorization, latent conditioning, diffusion-style refinement) so the block distribution is close enough to the AR joint that verification is redundant?
- **Theory.** Is there a bound on end-task degradation in terms of a quantity estimable from the AR model alone, before any MTP head is trained?

## 2. Formal Setting

Let $p_\theta$ be the AR model over vocabulary $\mathcal{V}$. For context $c$ the true block distribution is
$$p(x_{1:k}\mid c)=\prod_{i=1}^{k} p_\theta(x_i \mid c, x_{<i}).$$
An MTP model with heads $q^{(1)},\dots,q^{(k)}$ conditions each head only on $c$ (and shared trunk state), giving
$$q(x_{1:k}\mid c)=\prod_{i=1}^{k} q^{(i)}(x_i\mid c).$$

**Quantities, as measured.**

- *Block divergence* $\Delta_k(c) = D_{\mathrm{KL}}\!\left(p(\cdot\mid c)\,\|\,q(\cdot\mid c)\right)$. Measured by importance sampling: draw $N$ blocks from $p$ by ordinary AR sampling, score each under both models, average $\log p - \log q$. $N=256$ per context gives a standard error near $0.03$ nats at $k=4$ in practice.
- *Total correlation* $C_k(c)=\sum_{i} H(X_i\mid c) - H(X_{1:k}\mid c)$, the KL from the true joint to the product of its own marginals. This is the **irreducible** part of $\Delta_k$: no head that conditions only on $c$ can go below it. Critically, $C_k$ is estimable from the AR model alone — marginals $H(X_i\mid c)$ come from Monte Carlo rollouts, $H(X_{1:k}\mid c)$ from the chain rule on the same rollouts. No MTP training required.
- *Acceptance rate* $\alpha_k$: the fraction of drafted tokens a verifier would have accepted. Reported by every speculative system; it is a proxy, not a correctness metric.
- *End-task delta* $\delta = \text{Acc}_{\mathrm{MTP}} - \text{Acc}_{\mathrm{AR}}$, paired over the same prompts and seeds, with a bootstrap CI.

**Assumptions, and which fail.**

1. *Heads are conditionally independent given trunk state.* Holds by construction in Medusa-style heads; **violated by design** in DeepSeek-V3-style sequential MTP modules, which chain the $i$-th prediction on the $(i-1)$-th embedding and so partially recover the joint.
2. *Degradation is monotone in $k$.* Assumed everywhere, unproven.
3. *Greedy accuracy tracks distributional fidelity.* **Known false.** Greedy decoding collapses the product of marginals to the product of argmaxes, which can coincide with the AR argmax path even when $\Delta_k$ is large. See §10.
4. *Benchmark prompts are representative of the $C_k$ distribution in deployment.* Unverified; code and math prompts are unusually low-entropy.

## 3. State of the Art

**Established (verified mode).** Speculative decoding is *exactly* distribution-preserving: Leviathan et al. (ICML 2024, arXiv:2211.17192) and Chen et al. (arXiv:2302.01318) both prove the modified-rejection-sampling step yields samples from the target distribution regardless of draft quality. This is a theorem, independently reproduced, and it is the reason verification is the default.

**Established (MTP as training signal).** Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (ICML 2024, arXiv:2404.19737): $n=4$ auxiliary heads improve a 13B model by roughly $+12\%$ relative on HumanEval and $+17\%$ on MBPP versus a next-token baseline at matched data. The gain is a *pretraining* effect and does not depend on how the heads are used at inference.

**Claimed but under-ablated (verification-free).** Systems that commit unverified tokens report throughput and benchmark scores but rarely report distributional distance. Blockwise parallel decoding (Stern, Shazeer, Uszkoreit, NeurIPS 2018) already offered "approximate" variants with a top-$k$ or distance-based acceptance relaxation — the paper reports the BLEU cost, but the modern literature has not repeated that ablation at LLM scale. DeepSeek-V3's MTP module (DeepSeek-AI, technical report, 2024) reports second-token acceptance in the 85–90% range; the deployed inference path still verifies.

**Benchmark-number-only results.** Medusa (Cai et al., ICML 2024, arXiv:2401.10774) reports 2.2× (Medusa-1) to 2.3–3.6× (Medusa-2) speedup; EAGLE-2 (Li et al., EMNLP 2024) reports up to ~4×. These are throughput numbers under verification. Where a "typical acceptance" relaxation is used, the reported quality evidence is MT-Bench score parity — a single noisy scalar on 80 prompts, not a distributional test.

## 4. What Is Known

- Speculative verification preserves the target distribution exactly, for any draft model, at any $k$. Proven; reproduced across implementations.
- Removing conditioning across positions is *costly in principle*: non-autoregressive machine translation, the earliest verification-free block decoder, lost roughly 2–5 BLEU against AR baselines on WMT14 En–De at ~65M-parameter scale (Gu et al., ICLR 2018), attributed to the "multimodality problem" — exactly $C_k > 0$. Iterative refinement (Mask-Predict, Ghazvininejad et al., EMNLP 2019) recovers most but not all of the gap, at the cost of multiple passes.
- MTP heads help pretraining at scale $\ge 3$B and appear to *hurt* small models: Gloeckle et al. report the effect reverses below roughly 1–3B parameters.
- Acceptance rates decay steeply with position: reported second-token acceptance near 85–90% (DeepSeek-V3, 671B MoE) falls off fast by position 3–4 in Medusa-style heads, which is why practical tree depths stay at 4–5.
- Jacobi/lookahead decoding (Fu et al., ICML 2024) obtains speedup with *no* extra parameters and remains greedy-exact, showing part of the MTP gain is not about learned heads at all.

## 5. What Is Not Known

- **Theoretically open.** No bound of the form $\delta \le f(C_k, T, \text{task})$. Nobody has proven that end-task degradation is controlled by any estimable divergence, nor that it is monotone in $k$. Whether error compounds or self-corrects across blocks — the model re-conditions on its own committed block — is unproven in both directions.
- **Empirically open.** The paired experiment — same model, same prompts, verified vs verification-free, at $k \in \{2,4,8\}$ and $T \in \{0, 0.7, 1.0\}$, with a bootstrap CI on $\delta$ — is runnable today on a 7B model for a few thousand GPU-hours. It has not been published at that resolution. The distribution of $C_k$ across real serving traffic has, to our knowledge, never been measured at all.
- **Methodologically blocked.** "Quality preserved" has no agreed operational definition for stochastic decoders. Greedy pass@1 parity is the field's default and is provably insensitive to the failure mode (§10). A per-token KL is not comparable across tokenizers or block sizes.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure what it names**, compounded by **absent ground truth**.

The failure introduced by dropping intra-block conditioning is *mode mixing*: the decoder emits a string that is high-probability under each positional marginal but near-zero under the joint. Greedy decoding — the setting in which almost all MTP speedup claims are validated — is the one regime where mode mixing is least likely to fire, because the product of argmaxes often equals the argmax of the product. So the standard evaluation is close to blind to the defect it should detect. Meanwhile there is no reference distribution to score against other than the AR model itself, and evaluating $p(x_{1:k}\mid c)$ densely costs exactly the AR forward passes the method exists to avoid. Measuring correctness costs more than the speedup is worth, which is why nobody does it.

## 7. Current Research (as of 2026)

- **Sequential MTP modules** (DeepSeek lineage) that chain heads to restore some intra-block dependence, trading parallelism for a smaller $C_k$ residual. Deployed under verification; the verification-free limit is unstudied.
- **Discrete-diffusion and any-order decoders** as a principled verification-free block decoder: iterative denoising re-couples positions. *(frontier — verify)* Reported throughput is competitive; distributional fidelity against a matched AR model is the open question.
- **Entropy- and confidence-gated block commit** — commit unverified only when a cheap uncertainty statistic is low, verify otherwise. This is the most likely near-term practical answer and is largely an engineering literature at present. *(frontier — verify)*
- **Serving-stack work** (vLLM, TensorRT-LLM, SGLang) exposing speculative depth as a tunable; correctness stays verified by policy, so the empirical question remains unforced.

## 8. Concrete Next Experiment

**Scale.** One 7–8B open-weights base model with 4 MTP heads trained on 50B tokens (roughly 2–3k A100-hours). Evaluate on 4 suites spanning the entropy range: HumanEval (164), GSM8K (1319), MT-Bench (80), and 2000 held-out open-ended web-text continuations.

**Arms.** (a) AR baseline, full autoregressive sampling. (b) Verified: MTP draft + speculative verification — must match (a) in distribution by construction; serves as an implementation check. (c) Verification-free at $k=2,4,8$. All arms at $T \in \{0, 0.7, 1.0\}$, 5 seeds, paired prompts.

**Instrumentation.** For every prompt, estimate $C_k(c)$ from the AR model with 256 rollouts, before running any MTP arm.

**The deciding number.** The Spearman correlation $\rho$ between per-prompt $C_k(c)$ and per-prompt degradation in arm (c) at $T=1.0$, $k=4$. If $\rho \ge 0.5$ with a 95% bootstrap CI excluding 0.2, then total correlation — computable from the AR model alone, with no MTP training — is a valid pre-deployment gate, and the practical problem becomes threshold selection. If $\rho < 0.2$, the natural information-theoretic quantity does not predict end-task harm and the problem is genuinely methodologically blocked, not merely unmeasured.

Secondary readout: the largest $k$ at which the 95% CI on $\delta$ excludes $-1.0$ accuracy point on GSM8K at $T=0.7$.

## 9. Key References

- **[Foundational]** Mitchell Stern, Noam Shazeer, Jakob Uszkoreit. *Blockwise Parallel Decoding for Deep Autoregressive Models.* NeurIPS, 2018. — arXiv:1811.03115
- **[Foundational]** Jiatao Gu, James Bradbury, Caiming Xiong, Victor O.K. Li, Richard Socher. *Non-Autoregressive Neural Machine Translation.* ICLR, 2018. — arXiv:1711.02281
- **[Foundational]** Marjan Ghazvininejad, Omer Levy, Yinhan Liu, Luke Zettlemoyer. *Mask-Predict: Parallel Decoding of Conditional Masked Language Models.* EMNLP, 2019. — arXiv:1904.09324
- **[Theory]** Yaniv Leviathan, Matan Kalman, Yossi Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML, 2023. — arXiv:2211.17192
- **[Theory]** Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023. — arXiv:2302.01318
- **[SOTA]** Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, Gabriel Synnaeve. *Better & Faster Large Language Models via Multi-token Prediction.* ICML, 2024. — arXiv:2404.19737
- **[SOTA]** Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML, 2024. — arXiv:2401.10774
- **[SOTA]** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* ICML, 2024. — arXiv:2401.15077
- **[SOTA]** Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang. *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding.* ICML, 2024. — arXiv:2402.02057
- **[Systems]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Survey]** Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, Zhifang Sui. *Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding.* Findings of ACL, 2024. — arXiv:2401.07851

## 10. Worked Example

Context: `"The article referred to the city simply as"`. Suppose the AR model places its mass on two continuations of length 2:

| block | AR joint $p$ |
|---|---|
| `New York` | 0.50 |
| `the Big` | 0.48 |
| all others | 0.02 |

Positional marginals: $p_1(\texttt{New})=0.50$, $p_1(\texttt{the})=0.48$; $p_2(\texttt{York})=0.50$, $p_2(\texttt{Big})=0.48$.

An ideal verification-free MTP head with $C_k$-limited capacity samples the product of marginals:

$$q(\texttt{New Big}) = 0.50 \times 0.48 = 0.24,\qquad q(\texttt{the York}) = 0.48 \times 0.50 = 0.24.$$

**48% of blocks are strings with AR probability near zero**, even with heads that are *perfect* — each marginal matched exactly. The irreducible cost is the total correlation:

$$C_2 = H(X_1)+H(X_2)-H(X_{1:2}) \approx 0.693 + 0.693 - 0.703 \approx 0.68 \text{ nats}.$$

Now decode greedily. Head 1 argmax is `New` ($0.50 > 0.48$); head 2 argmax is `York` ($0.50 > 0.48$). Output: `New York` — **identical to the AR greedy path**. Greedy pass@1 registers zero degradation.

Flip the tie: make `the Big` the 0.50 mode and `New York` the 0.48 mode. Greedy MTP now emits `the Big`, still correct. Only at $0.50/0.50$ does the greedy product break. So on this instance the standard evaluation reports perfect parity while nearly half of all temperature-1.0 samples are malformed, and the acceptance-rate proxy also looks healthy — each token individually would clear a top-1-agreement check.

That is the obstruction in one instance: the defect lives in the joint, the metric lives in the marginals, and the deciding statistic ($C_k$, here 0.68 nats) is computable from the AR model but is not reported by any published MTP system.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*