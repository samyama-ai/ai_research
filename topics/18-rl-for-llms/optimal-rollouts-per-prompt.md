---
id: 18-rl-for-llms/optimal-rollouts-per-prompt
title: "Optimal Rollout Count per Prompt"
topic: 18-rl-for-llms
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Rollout Count per Prompt

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/optimal-rollouts-per-prompt` · **Status:** empirically-open

## 1. Problem Statement

Group-based policy-gradient methods for LLMs (GRPO, RLOO, and their variants) sample $G$ completions per prompt and use the within-group reward spread as the baseline. At a fixed generation budget, $G$ trades against the number of distinct prompts $N$: $B = N \cdot G$ rollouts per optimization step. Larger $G$ gives a lower-variance advantage estimate per prompt; larger $N$ gives more distinct gradient directions.

- **Input:** a base policy $\pi_\theta$, a prompt distribution $\mathcal{D}$, a reward $r$ (usually binary verifier output), and a fixed budget — either total rollouts $B$ or wall-clock on fixed hardware.
- **Output:** the group size $G^\star$ (possibly prompt-dependent, $G^\star(x)$) maximizing final downstream accuracy.
- **Solved** would mean: a rule predicting $G^\star$ from measurable quantities (model scale, prompt pass-rate distribution, sequence length, reward noise) that beats a fixed-$G$ baseline at matched wall-clock across at least two model scales and two task families.

Three variants, different difficulty:
- **Measurement:** does $G$ matter at matched budget, and by how much? Runnable now; not run as a clean sweep at frontier scale.
- **Method:** adaptive per-prompt $G$ (spend more rollouts where the group is informative). Partially attacked by dynamic sampling and difficulty filtering.
- **Theory:** derive $G^\star$ from a variance/bias decomposition of the group-baselined estimator. Open — existing analyses assume unbiasedness that the practical estimators do not satisfy.

## 2. Formal Setting

Policy $\pi_\theta(y \mid x)$, prompt $x \sim \mathcal{D}$, completion $y$, reward $r(x,y) \in \{0,1\}$ for verifiable tasks. Per-prompt success rate:

$$p(x) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}[r(x,y)]$$

**Measured as:** the empirical mean over the $G$ sampled rollouts at the current step, $\hat{p}(x) = \frac{1}{G}\sum_{i=1}^{G} r_i$. This is the only estimate available online; it is itself $O(G^{-1/2})$-noisy, which is the source of most of the difficulty below.

**RLOO / leave-one-out advantage** (unbiased baseline, Kool et al. 2019; Ahmadian et al. 2024):

$$A_i^{\mathrm{LOO}} = r_i - \frac{1}{G-1}\sum_{j \ne i} r_j$$

**GRPO advantage** as originally specified (Shao et al. 2024): $A_i = (r_i - \mu)/\sigma$ with $\mu, \sigma$ the group mean and standard deviation. The $\sigma$ division and the length normalization are *biased* — Liu et al. (2025, "Dr. GRPO") show the $1/\sigma$ term up-weights prompts with near-0 or near-1 pass rate and the token-length normalization biases toward long wrong answers. Bias is $O(1/G)$ and does not vanish at fixed $G$.

**Gradient estimator and its variance.** Let $g_\theta(x) = \mathbb{E}[A \nabla_\theta \log \pi_\theta(y|x)]$. For the budget-constrained estimator $\hat{g} = \frac{1}{N}\sum_{n=1}^{N}\frac{1}{G}\sum_{i=1}^{G} A_{ni}\nabla \log \pi(y_{ni}|x_n)$, the variance decomposes as

$$\mathrm{Var}(\hat g) = \underbrace{\frac{1}{N}\mathrm{Var}_x\!\left(g_\theta(x)\right)}_{\text{prompt diversity}} + \underbrace{\frac{1}{NG}\mathbb{E}_x\!\left[\mathrm{Var}_{y|x}(\cdot)\right]}_{\text{within-prompt noise}}$$

With $B = NG$ fixed, the second term is $\propto 1/B$ — **invariant to $G$** — while the first is $\propto G/B$, strictly increasing in $G$. Under this decomposition alone, $G = 2$ is optimal. The decomposition is a fact about the estimator; it is not the answer, because it holds only under assumptions that fail:

- **A1 (unbiased advantage).** Violated by GRPO's $\sigma$-normalization and length normalization.
- **A2 (rollout cost independent of $G$).** Violated: rollouts in a group share a prompt prefix, so KV-cache prefix reuse makes the marginal rollout cheaper within a group than across groups. At 8k-token prompts and 1k-token completions the prefill share is large; at 200-token prompts and 16k-token completions it is negligible.
- **A3 (advantage defined for every group).** Violated: if all $G$ rewards are equal, every $A_i = 0$ and the group contributes zero gradient. Probability of a **degenerate group**: $p^G + (1-p)^G$.
- **A4 (stationary $p(x)$).** Violated: pass rates move during training, so $G^\star$ is a function of the training step, not a constant.

**Effective (informative) budget** — the quantity that arguably should be optimized:

$$B_{\mathrm{eff}} = N \cdot G \cdot \mathbb{E}_{x}\!\left[1 - p(x)^G - (1-p(x))^G\right]$$

## 3. State of the Art

**Empirical/systems SOTA — established.**
- Dynamic sampling (DAPO, Yu et al. 2025): oversample and discard prompts whose group is all-correct or all-wrong, refilling the batch to a constant count of informative groups. Ablated in the paper; removing it degrades AIME24 accuracy and slows convergence. This is the strongest *engineered* answer, and it changes $N$ rather than $G$.
- Unbiased-baseline variants: RLOO (Ahmadian et al. 2024) at $k \in \{2,4\}$ matches or beats PPO on TL;DR and HH at 7B; Dr. GRPO (Liu et al. 2025) removes GRPO's two bias terms and matches accuracy with visibly shorter incorrect responses.

**Claimed but unablated.** Production group sizes are reported, not justified: DeepSeekMath/R1 (64), Open-Reasoner-Zero (64), DAPO (16), Kimi k1.5 (8–16), Magistral (Mistral, 2025). None publishes a budget-matched $G$ sweep. The numbers are configuration disclosures, not evidence about $G^\star$.

**Benchmark-number-only results.** Nearly every reported $G$ comparison is a single-seed AIME24/AIME25 delta of 1–4 points on a 30-problem set — inside the seed-to-seed noise band for pass@1 at that set size. Treat any such comparison as uninformative about $G$.

**Theory SOTA.** Greensmith, Bartlett & Baxter (JMLR 2004) give the general variance-reduction bounds for baselined policy gradients; Kool et al. (2019) prove the LOO baseline is unbiased and optimal within its class. Neither addresses the $B = NG$ allocation problem with degenerate groups and non-stationary $p(x)$. Best-arm-identification complexity (Kaufmann, Cappé & Garivier, JMLR 2016) gives the right shape for a per-prompt stopping rule but has not been instantiated for this setting.

## 4. What Is Known

- **Small $G$ works.** RLOO with $k=2$ and $k=4$ matches PPO on summarization and dialogue preference tasks at 6–7B (Ahmadian et al., ACL 2024). Established, independently reproduced in open frameworks.
- **Degeneracy is the dominant waste.** DAPO (Yu et al. 2025) reports that with $G=16$ on their Qwen2.5-32B math setup, the fraction of prompts with all-correct or all-wrong groups grows through training until a large share of each batch produces no gradient — the motivation for dynamic sampling. Measured at 32B on DAPO-Math-17k.
- **Repeated sampling has a wide dynamic range.** Brown et al. (2024) show coverage (pass@$k$) rising log-linearly over four orders of magnitude in $k$ on GSM8K/MATH/SWE-bench at 7B–70B. This bounds how much extra signal a larger $G$ can *possibly* expose per prompt.
- **RL narrows rather than extends the base-model support.** Yue et al. (2025) find RL-trained models' pass@$k$ at large $k$ does not exceed the base model's, measured at 7B–32B on math and code. If correct, the value of large $G$ is variance reduction, not discovery of new solutions.
- **Test-time compute allocation is prompt-difficulty dependent.** Snell et al. (2024) show compute-optimal test-time sampling varies by difficulty bin, with up to $4\times$ efficiency gains over uniform allocation on MATH at PaLM-2 scale. This is *inference*, not training, but it is the closest measured analogue of adaptive $G(x)$.

## 5. What Is Not Known

- **Empirically open.** No published budget-matched sweep of $G \in \{2,4,8,16,32,64\}$ at fixed $B$ (and separately at fixed wall-clock), with $\geq 3$ seeds and an evaluation set large enough to resolve 2-point differences. The experiment is entirely runnable; nobody has paid for it at frontier scale.
- **Empirically open.** Whether $G^\star$ scales with model size, sequence length, or training step. All four are confounded in every existing report.
- **Theoretically open.** The optimal $(N,G)$ split under the *biased* GRPO estimator with degenerate groups. No proof either way; the clean variance argument (Section 2) applies only to the unbiased LOO case with A2–A4 satisfied.
- **Methodologically blocked.** "Signal per rollout" has no agreed definition. The two natural proxies — informative-prompt count and per-prompt gradient SNR — rank $G$ in opposite orders (Section 10). Until one is validated against downstream accuracy, sweeps measure the proxy, not the objective.

## 6. Why It Is Hard

Three named obstructions.

1. **Compute cost of the decisive experiment.** A clean sweep needs $\geq 6$ arms $\times$ 3 seeds $\times$ a full RL run. At 8B with 4k-token rollouts and $5 \times 10^6$ rollouts per run, that is $\approx 2\times10^{10}$ generated tokens per arm; on 32 H100s at ~80k tok/s aggregate this is ~3 days per arm and ~50 GPU-days-of-cluster for the full grid. That is why it is published as a config line, not a sweep.
2. **Confounded measurement.** Changing $G$ at fixed $B$ changes the number of distinct prompts, the KV-cache prefix reuse rate, the effective batch composition after filtering, and the advantage scale simultaneously. A raw $G$ delta attributes all four to $G$.
3. **Evaluation that does not measure what it names.** AIME24/AIME25 have 30 problems each; a single-seed pass@1 difference of 3.3 points is one problem. Most claims about $G$ rest on differences at this resolution.

## 7. Current Research (as of 2026)

- **Dynamic sampling and difficulty filtering** — DAPO (ByteDance Seed), online difficulty filtering (Bae et al., 2025), curriculum variants in Skywork-OR1 and ProRL (NVIDIA). All keep $G$ fixed and adapt $N$ or the prompt pool.
- **Debiasing the group estimator** — Dr. GRPO (Sea AI Lab / NUS), minimalist REINFORCE-style variants (Xiong et al., 2025). These make the $G$ question cleaner by restoring A1.
- **Adaptive per-prompt allocation** — bandit-style budget allocation as a training-time analogue of Snell et al.'s test-time result. *(frontier — verify; no budget-matched published result at scale that I can confirm.)*
- **Asynchronous rollout engines** decoupling $G$ from the optimizer batch, which turns $G$ from a systems constraint into a free hyperparameter and makes the sweep affordable. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Scale.** Qwen3-1.7B-Base (small enough to afford, large enough to show RL gains), DAPO-Math-17k prompt pool, binary verifier reward, 4k-token generation cap, Dr. GRPO objective (unbiased — removes confound A1). Fixed budget $B = 8192$ rollouts per optimizer step, 400 steps ($3.3\times10^6$ rollouts per arm). Arms: $G \in \{2, 4, 8, 16, 32, 64\}$ with $N = B/G \in \{4096, \dots, 128\}$. Three seeds per arm: 18 runs, $\approx 1.3\times10^{10}$ tokens total, ~2 cluster-weeks on 16 H100s.

**Control arm.** $G = 8$, $N = 1024$, no dynamic sampling — the modal open-source configuration. A second control at matched *wall-clock* rather than matched rollouts isolates the prefix-reuse effect (A2): report tokens/sec per arm.

**Deciding number.** Mean pass@1 on a held-out 500-problem math set (MATH-500 plus AIME-style items), estimated as avg@32 to cut evaluation variance to $\pm0.6$ points, averaged over 3 seeds. **The question is settled if the best and worst $G$ arms differ by more than 2.0 absolute points with non-overlapping 95% seed intervals.** If the spread across $G \in \{2,\dots,64\}$ is under 2.0 points, $G$ is not a meaningful knob at this scale and the field should stop tuning it; if it exceeds 2.0 points, report the argmax and whether it tracks the median pass rate $\bar p$ as $G^\star \approx 1/\min(\bar p, 1-\bar p)$.

## 9. Key References

- **[Foundational]** Evan Greensmith, Peter L. Bartlett, Jonathan Baxter. *Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning.* JMLR 5, 2004.
- **[Foundational]** Wouter Kool, Herke van Hoof, Max Welling. *Buy 4 REINFORCE Samples, Get a Baseline for Free!* Deep RL Meets Structured Prediction Workshop, ICLR 2019.
- **[Foundational]** Zhihong Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. — arXiv:2402.03300 (introduces GRPO).
- **[SOTA]** Arash Ahmadian et al. *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs.* ACL 2024. — arXiv:2402.14740 (RLOO).
- **[SOTA]** Qiying Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025. — arXiv:2503.14476 (dynamic sampling).
- **[SOTA]** Zichen Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective.* 2025. — arXiv:2503.20783 (Dr. GRPO; GRPO bias terms).
- **[Empirical]** Bradley Brown et al. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787.
- **[Empirical]** Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314.
- **[Empirical]** Yang Yue et al. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* 2025. — arXiv:2504.13837.
- **[Empirical]** Wei Xiong et al. *A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce.* 2025.
- **[Empirical]** Sanghwan Bae et al. *Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning.* 2025.
- **[Theory]** Emilie Kaufmann, Olivier Cappé, Aurélien Garivier. *On the Complexity of Best-Arm Identification in Multi-Armed Bandit Models.* JMLR 17, 2016.
- **[Systems]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* Nature, 2025. — arXiv:2501.12948.
- **[Systems]** Kimi Team. *Kimi k1.5: Scaling Reinforcement Learning with LLMs.* 2025. — arXiv:2501.12599.

## 10. Worked Example

Fix $B = 1024$ rollouts per step. Take a prompt pool where every prompt has pass rate $p = 0.9$ — a realistic mid-training state after the easy prompts have been learned but before filtering.

**Proxy 1: informative prompts per step.** Non-degenerate with probability $1 - p^G - (1-p)^G$.

| $G$ | $N = B/G$ | $P(\text{informative})$ | Informative prompts | Informative rollouts |
|---|---|---|---|---|
| 4 | 256 | $1-0.9^4 = 0.344$ | 88.0 | 352 |
| 8 | 128 | $1-0.9^8 = 0.570$ | 72.9 | 583 |
| 16 | 64 | $1-0.9^{16} = 0.815$ | 52.1 | 834 |
| 64 | 16 | $1-0.9^{64} = 0.999$ | 16.0 | 1023 |

Proxy 1 ranks $G = 4$ best on *distinct informative prompts* (88.0 vs 16.0) and $G = 64$ best on *rollouts that land in an informative group* (1023 vs 352). Two readings of the same proxy, opposite winners.

**Proxy 2: per-prompt gradient SNR.** Conditional on being informative, the LOO advantage magnitude for the single successful-or-failed outlier is $\approx 1$, but the estimator's standard error scales as $\sqrt{p(1-p)/(G-1)}$: $0.173$ at $G=4$, $0.113$ at $G=8$, $0.077$ at $G=16$, $0.038$ at $G=64$. Proxy 2 ranks $G = 64$ best, monotonically.

**The obstruction, made visible.** At $p = 0.9$, $G = 4$ wins proxy 1 by $5.5\times$ and loses proxy 2 by $4.5\times$. The clean variance decomposition (Section 2) says $G = 2$; the degeneracy calculation says most of a $G=2$ batch produces literally zero gradient ($1 - 0.9^2 - 0.1^2 = 0.18$, so 82% of prompts contribute nothing). Neither proxy is the objective, and they do not agree even on the sign of $\partial(\text{quality})/\partial G$. Add prefix reuse — at 8 rollouts sharing one 2k-token prompt, prefill cost per rollout drops ~8× relative to $G=1$, so equal-$B$ arms are not equal-wall-clock arms — and the ranking shifts again. This is why the question survives: it is decidable only by running Section 8's sweep and reading downstream accuracy, and the sweep costs a full training run per arm.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*