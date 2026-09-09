---
id: 30-synthetic-data/self-generated-data-information-ceiling
title: "Information-Theoretic Ceiling of Self-Generated Training Data"
topic: 30-synthetic-data
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Information-Theoretic Ceiling of Self-Generated Training Data

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/self-generated-data-information-ceiling` · **Status:** open

## 1. Problem Statement

A model $\hat{P}_\theta$ generates data, is retrained on it, and improves on held-out tasks. The data processing inequality says the samples carry no new information about the true data distribution $P^\star$. Both statements are true. The problem is to reconcile them into a predictive bound.

- **Measurement variant.** Given a self-training pipeline, measure how many bits about $P^\star$ enter per iteration, and from which channel (verifier, tool, human filter, retained real data). No standard estimator exists.
- **Method variant.** Given a fixed budget of $n$ external-oracle calls, design the pipeline maximizing downstream capability. This is the one practitioners run.
- **Theory variant.** Prove a bound of the form: after $T$ rounds of self-generation with an oracle emitting $B$ bits total, generalization error on $P^\star$ cannot fall below $f(B, T, \text{compute})$ — and show the bound is tight enough to be violated by a real system.

Solving it means producing a quantity, computable from a training run, that predicts the iteration at which self-improvement saturates. Nothing currently does this.

## 2. Formal Setting

$P^\star$ is the target distribution over sequences $x \in \mathcal{X}^\star$, treated as a random object with prior $\mu$ so mutual information with it is defined. $D_0 \sim (P^\star)^{\otimes n_0}$ is the real corpus. A learner $A$ maps data to parameters: $\theta_0 = A(D_0)$.

**Self-training round $t$.** Sample $\tilde{D}_t \sim \hat{P}_{\theta_{t-1}}^{\otimes m}$, apply a filter $\Phi$, set $\theta_t = A(\Phi(\tilde{D}_t) \cup R_t)$ where $R_t$ is any retained real data.

**Closed-loop ceiling.** If $R_t = \emptyset$ and $\Phi$ depends only on $\tilde{D}_t$, then $P^\star \to \theta_0 \to \tilde{D}_1 \to \theta_1 \to \cdots$ is a Markov chain and

$$I(\theta_T; P^\star) \;\le\; I(\theta_0; P^\star) \;\le\; H(D_0).$$

**Open-loop injection.** Let $V: \mathcal{X}^\star \to \mathcal{Y}$ be a verifier (unit test, proof checker, human label) with $|\mathcal{Y}| = k$, called $N$ times per round. Then

$$I(\theta_T; P^\star) \;\le\; I(\theta_0; P^\star) + T \cdot N \log_2 k \ \text{bits},$$

with $\log_2 k = 1$ for a binary checker. This is the only rigorous, computable ceiling in the literature, and Section 10 shows it is numerically vacuous.

**Measured quantities.**
- Capability: $L_T = \mathbb{E}_{x\sim P^\star}[-\log \hat{P}_{\theta_T}(x)]$ on a held-out real corpus, in nats/token.
- Diversity: $\hat{H}_t = -\frac{1}{|S|}\sum_{x \in S}\log \hat{P}_{\theta_t}(x)$ over model samples $S$ (self-entropy, an upper-biased plug-in estimate).
- Coverage: $\text{pass@}k$ at large $k$ ($k \ge 256$), the operational proxy for support of $\hat{P}_{\theta_t}$.
- Injected bits: $\hat{B}_t = N \cdot \hat{H}(V(\tilde{x}))$, the empirical entropy of verifier verdicts — an upper bound, since correlated verdicts carry fewer bits.

**Assumptions, and which fail.** (i) $A$ is a fixed learner — violated: hyperparameters and data mixes are tuned per round using real validation sets, leaking real bits off-books. (ii) The filter uses no external information — violated whenever a tool, retriever, or human is in the loop. (iii) Closed-loop purity — violated in practice: web crawls after 2023 contain model output, so $D_0$ itself is partly synthetic and the chain is not clean. (iv) Mutual information is the right currency — violated in spirit: a bound on bits is agnostic to compute, and self-training is plausibly a *computational* re-encoding of information already present.

## 3. State of the Art

**Theory (established).** Dohmatob, Feng, Kempe et al. (ICML 2024; NeurIPS 2024) give exact model-collapse formulas for linear/ridge regression and a modified scaling law: with a fixed synthetic corpus, test error acquires a term that does not vanish as $N \to \infty$, so scaling curves plateau rather than continue. Gerstgrasser et al. (COLM 2024) prove that if synthetic data is *accumulated* alongside real data rather than replacing it, linear-regression test error stays bounded by a constant multiple ($\pi^2/6$ factor) of the real-data error, so collapse is not inevitable. Bertrand et al. (ICLR 2024) give a local-stability condition: iterative retraining is stable if the real-data fraction exceeds a threshold and the initial model is close enough to $P^\star$. Seddik et al. (COLM 2024) bound total-variation drift as a function of the synthetic/real mixing ratio.

**Empirical (established).** Shumailov et al. (*Nature*, 2024) show tail loss and degradation over 9 generations of full-replacement retraining. Alemohammad et al. (ICLR 2024) show the same for image models and name the regime "MAD" (Model Autophagy Disorder).

**Claimed but unablated.** That verification "breaks" the ceiling. Feng et al. (2024) show verifier-pruned synthetic data restores scaling, but no work isolates how many verifier bits were needed versus how much came from the retained real data or the tuned learner. Positive self-improvement results — STaR (NeurIPS 2022), ReST (2023), ReST$^{EM}$ (TMLR 2024), Self-Rewarding LMs (2024) — report downstream benchmark gains only. None reports an information budget, and none runs a matched control with the oracle bits held fixed.

**Benchmark-number-only.** Every reported synthetic-data scaling gain in the phi-model line (Gunasekar et al., 2023) rests on benchmark scores with an unpublished generator prompt distribution; contamination-controlled replications are absent.

## 4. What Is Known

- **Full replacement degrades.** OPT-125M fine-tuned on WikiText-2 for 9 generations of sample-and-retrain loses tail mass and perplexity on real data rises monotonically (Shumailov et al., *Nature* 631:755–759, 2024). Scale: 125M params, ~2M tokens.
- **Accumulation does not degrade.** Transformers on TinyStories, VAEs on CelebA, diffusion on CIFAR: with data accumulation, test loss is flat across generations; with replacement it grows roughly linearly (Gerstgrasser et al., COLM 2024). Scale: 9M–125M params, up to 8 generations.
- **Truncation causes the tail loss.** Top-$p$/top-$k$ sampling, not self-training per se, cuts the Zipfian tail; the modified scaling law follows from the truncated exponent (Dohmatob et al., ICML 2024).
- **Self-training saturates fast in practice.** ReST$^{EM}$ on PaLM 2-L: MATH and APPS gains saturate after 1–3 iterations, then overfit (Singh et al., TMLR 2024). Scale: ~$10^2$B params.
- **RLVR narrows support.** At $k=256$, base models match or exceed their RLVR-tuned descendants on pass@$k$ across math benchmarks for Qwen2.5-7B/14B/32B and LLaMA-3.1-8B (Yue et al., 2025). The tuned model reallocates probability mass; it does not obviously add reachable solutions.
- **The real-data stock is finite.** ~$3 \times 10^{14}$ tokens of public human text, projected exhaustion between 2026 and 2032 (Villalobos et al., ICML 2025).

## 5. What Is Not Known

- **Theoretically open.** Whether any bound tighter than the trivial $T N \log_2 k$ oracle-bit ceiling holds for a compute-bounded learner. No separation theorem exists showing that a self-training loop with $B$ bits provably cannot reach loss $L$ that a direct learner with $B$ bits of real data reaches. Also open: whether "amplification" — the loop converting existing bits into a better-conditioned representation — admits a formal statement in which the information stays constant while measurable capability strictly increases.
- **Empirically open.** No published run has held the oracle-bit budget fixed while varying the number of self-training rounds, at $\ge 7$B parameters. The experiment is a few thousand GPU-hours; it has not been run.
- **Methodologically blocked.** $I(\theta; P^\star)$ is not estimable for a real model. Every proxy fails somewhere: self-entropy conflates diversity with error, pass@$k$ saturates at attainable $k$, and held-out loss is contaminated by pretraining exposure. Until a proxy is validated against a synthetic ground-truth setup where $P^\star$ is known, the central quantity is unmeasured.

## 6. Why It Is Hard

**Non-identifiability of the bit source.** In every real pipeline, at least four channels inject information about $P^\star$ simultaneously: retained real data, the verifier, the human-authored prompt/seed distribution, and hyperparameter selection against a real validation set. They are not separately observable from a training log. A run that improves over 3 rounds cannot be attributed to the verifier without an arm that removes the other three — and removing the seed distribution is not obviously possible, since the seeds are what define the task.

**Compounding this: the valid bound is vacuous.** The DPI ceiling is correct and unhelpful, because bits about $P^\star$ are not the binding constraint — reachable-with-compute bits are. That means the honest obstruction is not measurement noise but that the quantity with a theorem attached is the wrong quantity.

## 7. Current Research (as of 2026)

- **Curated self-consumption.** Ferbach et al. (NeurIPS 2024) prove curated self-consuming loops optimize the curator's implicit reward — reframing the loop as preference optimization rather than density estimation. Active at Mila/DeepMind.
- **Mixing-ratio theory.** Kempe's group (NYU) and Kazdan et al. (Stanford) on accumulate-vs-replace and real/synthetic ratios.
- **Entropy collapse in RLVR.** Cui et al. (2025) relate policy entropy decay to a covariance term and propose clipping to arrest it. Direct empirical handle on support narrowing.
- **Verification-as-oracle scaling.** Whether test-time compute plus a cheap verifier substitutes for data. *(frontier — verify)*
- **Bit-accounting for pipelines.** No group is publicly running the controlled bit-budget ablation. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Do additional self-training rounds add capability when the oracle-bit budget is held constant?

**Scale.** A 7B base model (Qwen2.5-7B or Llama-3.1-8B), math reasoning with a binary answer checker. Fixed total budget $B = 10^7$ verifier calls. Four arms, matched on both verifier calls *and* gradient-update FLOPs:

| Arm | Rounds | Calls/round |
|---|---|---|
| A (control) | 1 | $10^7$ |
| B | 4 | $2.5\times10^6$ |
| C | 16 | $6.25\times10^5$ |
| D (null) | 4 | 0 (self-consistency filter only) |

**Deciding number.** $\Delta = \text{pass@}256_{\text{C}} - \text{pass@}256_{\text{A}}$ on a held-out, contamination-audited set (e.g. a 2026 competition set post-dating all checkpoints). pass@256, not pass@1: it measures support, not mass reallocation.

- $\Delta \le 0$: rounds buy nothing beyond bits; the ceiling is real and oracle-call count is the whole story.
- $\Delta \ge 3$ points with non-overlapping bootstrap CIs: iteration performs computational amplification not accounted for by bit counting, and the DPI framing is the wrong model.
- Arm D above baseline: the closed loop is generating capability with zero external bits, which forces the seed distribution into the accounting.

Cost estimate: ~4,000 A100-hours.

## 9. Key References

- **[Foundational]** Shumailov, Shumaylov, Zhao, Papernot, Anderson, Gal. *AI models collapse when trained on recursively generated data.* Nature 631, 755–759, 2024. (Preprint: *The Curse of Recursion*, arXiv:2305.17493)
- **[Foundational]** Alemohammad, Casco-Rodriguez, Luzi, Humayun, Babaei, LeJeune, Siahkoohi, Baraniuk. *Self-Consuming Generative Models Go MAD.* ICLR 2024. — arXiv:2307.01850
- **[SOTA — theory]** Gerstgrasser, Schaeffer, Dey, Rafailov, et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM 2024. — arXiv:2404.01413
- **[SOTA — theory]** Dohmatob, Feng, Yang, Charton, Kempe. *A Tale of Tails: Model Collapse as a Change of Scaling Laws.* ICML 2024. — arXiv:2402.07043
- **[Theory]** Bertrand, Bose, Duplessis, Jiralerspong, Gidel. *On the Stability of Iterative Retraining of Generative Models on their own Data.* ICLR 2024. — arXiv:2310.00429
- **[Theory]** Seddik, Chen, Hayou, Youssef, Debbah. *How Bad is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse.* COLM 2024. — arXiv:2404.05090
- **[Theory]** Ferbach, Bertrand, Bose, Gidel. *Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences.* NeurIPS 2024. — arXiv:2407.09499
- **[SOTA — empirical]** Singh, Co-Reyes, Agarwal, et al. *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models.* TMLR 2024. — arXiv:2312.06585
- **[Empirical]** Zelikman, Wu, Mu, Goodman. *STaR: Bootstrapping Reasoning With Reasoning.* NeurIPS 2022. — arXiv:2203.14465
- **[Empirical]** Yue, Chen, Lu, et al. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* 2025. — arXiv:2504.13837
- **[Empirical]** Feng, Dohmatob, Yang, Charton, Kempe. *Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification.* 2024. — arXiv:2406.07515
- **[Survey/context]** Villalobos, Ho, Sevilla, Besiroglu, Heim, Hobbhahn. *Will We Run Out of Data? Limits of LLM Scaling Based on Human-Generated Data.* ICML 2025. — arXiv:2211.04325
- **[Foundational]** Cover, Thomas. *Elements of Information Theory,* 2nd ed. Wiley, 2006. (Data processing inequality, Thm. 2.8.1)

## 10. Worked Example

**Setup.** STaR-style loop on a 7B model, GSM8K-scale math: $10^5$ seed problems, 64 samples each, binary answer checker.

**Bits injected.** $10^5 \times 64 = 6.4\times10^6$ verifier calls, each $\le 1$ bit:

$$B \le 6.4 \times 10^6 \ \text{bits} = 0.8\ \text{MB}.$$

The true figure is smaller: verdicts within a problem's 64 samples are strongly correlated. Empirically the per-problem verdict distribution is near-degenerate for ~70% of problems (all-correct or all-wrong), so the realized entropy is closer to $\hat{B} \approx 2\times10^6$ bits $\approx 250$ KB.

**What the model absorbs.** The fine-tuned model has $7\times10^9$ parameters in bf16, $1.1\times10^{11}$ bits of capacity. LoRA at rank 16 across attention projections still updates ~$4\times10^7$ parameters, $6\times10^8$ bits. Reported gains: ~+10 points pass@1 on GSM8K.

**The obstruction, made visible.** The bound says: at most 250 KB of information about $P^\star$ entered. The observed change is a 10-point capability shift driven by parameter updates carrying $10^9$ bits of description length — three orders of magnitude more than the ceiling permits *as new information*. Both are consistent: the extra $10^9$ bits are a re-encoding of pretraining information, not new information. But that consistency is exactly the problem. The information budget is satisfied by any outcome from "no gain" to "solves every problem it could ever verify". A bound that 250 KB permits a 10-point jump also permits a 40-point jump, and permits zero.

**Consequence.** The DPI ceiling cannot answer the practitioner's question — *how many more rounds are worth running?* Answering it requires a resource-bounded quantity (bits reachable within compute $C$), and no such quantity has been defined for neural learners. That is the open problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*