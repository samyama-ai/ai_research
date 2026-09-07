---
id: 18-rl-for-llms/curriculum-selection-problem-difficulty
title: "Curriculum Selection by Problem Difficulty"
topic: 18-rl-for-llms
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Curriculum Selection by Problem Difficulty

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/curriculum-selection-problem-difficulty` · **Status:** open

## 1. Problem Statement

Given a fixed pool of RL training prompts with verifiable rewards (math, code, formal proofs), decide **which prompts to sample at each step** so that a fixed rollout budget buys the most downstream capability.

- **Input:** prompt pool $\mathcal{D}$ (typically $10^4$–$10^6$ items), a policy $\pi_\theta$, a verifier $R$, a total rollout budget $N$.
- **Output:** a sampling distribution $q_t(\cdot)$ over $\mathcal{D}$ for each step $t$.
- **Objective:** maximise held-out pass@1 (or pass@$k$) at budget $N$, not at fixed step count.

Three variants, routinely conflated:

- **Measurement:** is "difficulty" identifiable and stable? Define $p_\theta(x)$ = policy success rate on $x$. It is policy-relative and drifts during training. A static human or model-assigned difficulty label is a different quantity.
- **Method:** does any online selection rule beat uniform sampling at matched rollout cost, including the cost of estimating difficulty?
- **Theory:** is there a regret bound for curriculum selection under policy-gradient dynamics, or a proof that the greedy "maximise learning progress" rule is suboptimal?

Solving it means: a selection rule with a stated estimator for $p_\theta$, an ablation at matched total rollouts, and reproduction across seeds and two base-model families.

## 2. Formal Setting

Policy $\pi_\theta$, prompt $x$, sampled response $y \sim \pi_\theta(\cdot\mid x)$, binary verifier $r = R(x,y) \in \{0,1\}$.

**Difficulty, as measured.** $p_\theta(x) = \mathbb{E}_{y\sim\pi_\theta}[R(x,y)]$, estimated by $\hat p = \frac{1}{G}\sum_{i=1}^{G} r_i$ from $G$ rollouts at temperature $T$. The estimator has standard error $\sqrt{p(1-p)/G}$: at $G=16$, $p=0.5$, SE $=0.125$. Difficulty measured at $T=1.0$ is not difficulty at $T=0.6$; evaluation usually uses the latter.

**Gradient signal.** Under GRPO (Shao et al., 2024), group-normalised advantage
$$A_i = \frac{r_i - \bar r}{\mathrm{std}(r_{1:G})},\qquad \bar r = \tfrac1G\textstyle\sum_i r_i .$$
If all $r_i$ are equal the advantage is $0/0$ (implementations set it to zero) and the group contributes nothing. Probability a group is informative:
$$P_{\mathrm{inf}}(p) = 1 - p^{G} - (1-p)^{G},$$
and per-sample reward variance is $p(1-p)$, maximised at $p=1/2$.

**Budget.** Cost is rollouts, not steps: $N = \sum_t G\,|B_t| + N_{\text{probe}}$, where $N_{\text{probe}}$ is rollouts spent estimating difficulty. Papers that report "fewer steps" while hiding $N_{\text{probe}}$ are not reporting a saving.

**Objective.** $\max_{q_{1:T}} \; \mathbb{E}[\,\mathrm{Acc}_{\text{held-out}}(\theta_T)\,]$ s.t. $N \le N_{\max}$.

**Assumptions known to be violated in practice:**
1. *Reward is a clean binary signal* — verifiers accept wrong-but-lucky answers; format rewards leak.
2. *Difficulty is stationary within a phase* — $p_\theta$ moves fastest exactly on the prompts a curriculum concentrates on.
3. *Prompts are exchangeable draws from one distribution* — pools mix contest math, synthetic templates, and near-duplicates, so $p$ correlates with cluster identity rather than with any latent difficulty.
4. *Pass@1 gains transfer across benchmarks* — AIME-tuned curricula need not transfer to code.

## 3. State of the Art

**Empirical SOTA (established, with ablation).** DAPO (Yu et al., 2025, arXiv:2503.14476) adds *dynamic sampling*: resample any prompt whose group is all-correct or all-wrong until the batch is full of $0<\hat p<1$ groups. Reported AIME 2024 avg@32 of 50 on Qwen2.5-32B base, above DeepSeek-R1-Zero-Qwen-32B's 47 at roughly half the training steps. The cumulative ablation table attributes several points to dynamic sampling — but it is single-run, and the discarded rollouts are counted as free.

**Established mechanism, not a curriculum.** The $p\in\{0,1\}$ zero-gradient fact is exact and implementation-independent for group-normalised advantages. Filtering on it is engineering, not a difficulty theory.

**Claimed but unablated.** Kimi k1.5 (Kimi Team, 2025) and several open recipes describe "curriculum sampling, easy to hard" and prioritised replay by success rate; the reports give end-to-end benchmark numbers with no matched-budget control arm. Self-Evolving Curriculum-style bandits over difficulty bins (2025) report gains on single seeds. Absolute Zero (Zhao et al., 2025, arXiv:2505.03335) uses an explicit *learnability* reward $1-\bar r$-shaped around mid-accuracy in a self-play proposer; the curriculum term is not ablated against uniform proposal.

**Theory SOTA (adjacent, not LLM).** Narvekar et al., *Curriculum Learning for RL Domains: A Framework and Survey* (JMLR 2020). ProCuRL (Tzannetos et al., TMLR 2023) derives, in a simplified setting, a selection rule of the $\arg\max_x p(x)\,(1-p(x))$ family — the formal statement closest to "train at the edge of ability". Prioritized Level Replay (Jiang et al., ICML 2021) shows TD-error-based level selection beats uniform in Procgen. Neither has been ported to LLM RL with a proof that survives non-stationary $p_\theta$.

## 4. What Is Known

- **Zero-variance groups are dead compute.** At $G=16$: $P_{\mathrm{inf}}(0.5)=0.99997$, $P_{\mathrm{inf}}(0.05)=0.56$, $P_{\mathrm{inf}}(0.01)=0.15$. Late in training on a math pool, 30–60% of groups are all-correct in reported runs (DAPO, 32B scale).
- **Filtering saturated prompts helps at 32B.** DAPO, Qwen2.5-32B, AIME'24 avg@32 = 50 with the full recipe; the paper's own progression starts near 30 for naive GRPO. Single seed.
- **RL sharpens rather than extends at small budgets.** Yue et al. (2025, arXiv:2504.13837) find RL-trained models beat base at pass@1 but base models match or exceed them at large $k$ across several 7B–32B pairs — so curricula measured by pass@1 may be optimising sharpening, not new capability.
- **Long-horizon RL keeps improving without an explicit curriculum.** ProRL (Liu et al., NVIDIA, 2025, arXiv:2505.24864) trains a 1.5B model for >2k steps with a KL-reset schedule and reports gains beyond the base pass@$k$ envelope. Its lever is regularisation and step count, not difficulty selection.
- **SFT-side analogue works.** DART-Math (Tong et al., NeurIPS 2024) shows difficulty-aware rejection sampling — more samples for harder queries — beats uniform rejection sampling at matched sample count on 7B models. This is the clearest matched-budget evidence for difficulty-weighting anywhere in the LLM stack, but it is SFT, not RL.
- **Difficulty is estimable cheaply offline.** IRT-style latent difficulty fits (Polo et al., *tinyBenchmarks*, ICML 2024) recover benchmark item difficulty from ~100 items with a few-percent error — for a *fixed* model set, not for a policy mid-training.

## 5. What Is Not Known

- **Empirically open.** Does any difficulty curriculum beat uniform sampling *at matched total rollouts including probe cost*, with ≥3 seeds, at ≥7B? No published run controls for probe cost. Runnable today for ~$20–60k of H100 time.
- **Empirically open.** Is the optimal target band $p\approx 0.5$ (variance-maximising) or lower, e.g. $p\in[0.1,0.4]$ (where headroom exists)? Reported bands range from $(0,1)$ exclusive to $[0.3,0.7]$ with no head-to-head.
- **Theoretically open.** No regret bound for prompt selection under a non-stationary policy-gradient learner. The greedy $p(1-p)$ rule is myopic; whether it is within a constant factor of the optimal schedule is unproven even in a tabular bandit abstraction.
- **Methodologically blocked.** "Difficulty" has no policy-independent definition. Human labels, base-model $\hat p$, and current-policy $\hat p$ correlate weakly and reorder prompts differently; there is no agreed ground truth to validate an estimator against.
- **Methodologically blocked.** Curriculum gains and exploration-entropy gains are not separable with current instrumentation: mid-$p$ prompts also carry higher response entropy, so a curriculum arm changes two variables at once.

## 6. Why It Is Hard

The binding obstruction is **compute-cost-of-measurement compounded by non-stationarity**. $p_\theta(x)$ must be re-estimated because it moves, and each estimate costs $G$ rollouts — the same currency as training.

Concretely at 7B with batch 512 prompts, $G=16$: one step = 8,192 rollouts; 50 steps = 409,600. Re-scoring a 100k-prompt pool at $G=16$ costs 1.6M rollouts — **4× the training it informs**. Any curriculum must recover its own probe cost before it produces a net gain, and no paper reports the net.

Second obstruction: **non-identifiability**. A prompt with $\hat p = 3/16$ is indistinguishable at that sample size from one with true $p=0.05$ or $p=0.35$ (SE $\approx 0.10$). Selection on a noisy statistic preferentially picks estimation noise — the same winner's-curse failure as prioritised experience replay.

Third: **the evaluation does not measure the named thing**. Curriculum papers report AIME/MATH pass@1, which Yue et al. show can rise while the pass@$k$ envelope does not. A curriculum that reorders the sharpening schedule scores as a capability gain.

## 7. Current Research (as of 2026)

- **Filter-based curricula in production recipes** — dynamic sampling (ByteDance Seed/DAPO lineage), difficulty-bucketed staged training (Kimi, Qwen math teams). Established as engineering; ablations remain single-seed.
- **Bandit and self-play proposers** — treating difficulty bins as arms with learning-progress reward; Mila/Microsoft-adjacent groups. *(frontier — verify)*
- **Amortised difficulty prediction** — a value head or small probe predicting $p_\theta(x)$ so the pool can be re-ranked without rollouts. This directly attacks the probe-cost obstruction and is the highest-value open line. *(frontier — verify)*
- **Pass@$k$-aware objectives** — selecting prompts by expected change in the pass@$k$ envelope rather than pass@1, following Yue et al.'s critique. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Qwen2.5-Math-7B base, 8k-prompt verified math pool, GRPO, $G=16$, batch 256 prompts, budget fixed at $6\times10^6$ *total* rollouts per arm (probe rollouts counted), 3 seeds. ~8×H100 for ~5 days per arm.

**Arms.**
- **Control:** uniform sampling, no filtering.
- **A:** DAPO dynamic sampling (resample until $0<\hat p<1$), discarded rollouts charged to budget.
- **B:** band curriculum, sample from $\hat p \in [0.2,0.8]$ using a value head trained online to predict $p_\theta$; probe only 5% of the pool per 50 steps.
- **C (staleness control):** arm B's rule but with difficulty scored once at step 0 and never refreshed.

**Deciding number.** AIME'24 + AMC'23 avg@32 at $T=0.6$, and pass@256 on the same sets, at equal total rollouts. The question resolves *yes* if any curriculum arm beats the control by **>2.0 points avg@32 with non-overlapping ±1 SE across 3 seeds** *and* does not lose pass@256. If B ≈ C, difficulty non-stationarity is irrelevant and the whole re-estimation literature is over-engineered. If A ≈ control at matched rollouts, dynamic sampling is a step-count illusion.

## 9. Key References

- **[Foundational]** Yoshua Bengio, Jérôme Louradour, Ronan Collobert, Jason Weston. *Curriculum Learning.* ICML, 2009.
- **[Foundational]** Alex Graves, Marc G. Bellemare, Jacob Menick, Rémi Munos, Koray Kavukcuoglu. *Automated Curriculum Learning for Neural Networks.* ICML, 2017. — arXiv:1704.03003
- **[Survey]** Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko Sinapov, Matthew E. Taylor, Peter Stone. *Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey.* JMLR, 2020.
- **[SOTA]** Qiying Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025. — arXiv:2503.14476
- **[Foundational]** Zhihong Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. — arXiv:2402.03300
- **[SOTA]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* Nature, 2025. — arXiv:2501.12948
- **[SOTA]** Yang Yue et al. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* 2025. — arXiv:2504.13837
- **[SOTA]** Mingjie Liu et al. *ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models.* NVIDIA, 2025. — arXiv:2505.24864
- **[SOTA]** Yuxuan Tong et al. *DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving.* NeurIPS, 2024. — arXiv:2407.13690
- **[Theory]** Georgios Tzannetos, Bárbara Gomes Ribeiro, Parameswaran Kamalaruban, Adish Singla. *Proximal Curriculum for Reinforcement Learning Agents.* TMLR, 2023.
- **[Theory]** Minqi Jiang, Edward Grefenstette, Tim Rocktäschel. *Prioritized Level Replay.* ICML, 2021.
- **[Method]** Andrew Zhao et al. *Absolute Zero: Reinforced Self-play Reasoning with Zero Data.* 2025. — arXiv:2505.03335
- **[Method]** Felipe Maia Polo et al. *tinyBenchmarks: Evaluating LLMs with Fewer Examples.* ICML, 2024.

## 10. Worked Example

A 7B policy, 100k-prompt pool, $G=16$, batch 512 prompts. One step costs $512\times16 = 8{,}192$ rollouts.

**Step 1 — signal accounting.** Suppose the pool splits 40% at $p\approx0.95$, 35% at $p\approx0.5$, 25% at $p\approx0.02$. Expected informative groups per uniform batch:

| bucket | $p$ | $P_{\mathrm{inf}}$ ($G{=}16$) | share | informative groups /512 |
|---|---|---|---|---|
| saturated | 0.95 | 0.560 | 0.40 | 115 |
| mid | 0.50 | 1.000 | 0.35 | 179 |
| unsolved | 0.02 | 0.276 | 0.25 | 35 |
| **total** | | | | **329 (64%)** |

A perfect mid-band curriculum yields 512 informative groups — a **1.56×** increase in usable groups per step. Reward variance per sample rises from a weighted $0.40(0.0475)+0.35(0.25)+0.25(0.0196)=0.111$ to $0.25$, a **2.25×** signal gain.

**Step 2 — pay for the measurement.** To know which prompts sit mid-band you must estimate $\hat p$. Full pool re-score: $100{,}000\times16 = 1.6\text{M}$ rollouts $=195$ training steps. Amortised over a 50-step refresh interval, the probe costs $195/50 \approx 3.9$ steps of overhead *per step trained* — a 4.9× budget multiplier against a 2.25× signal gain. **Net loss.**

**Step 3 — cheapen it and hit the noise wall.** Drop to $G=4$ probes and 5% of the pool: cost falls to $20{,}000$ rollouts per refresh (2.4 steps, negligible). But $\hat p$ now takes values in $\{0,\tfrac14,\tfrac12,\tfrac34,1\}$ with SE $=0.25$ at $p=0.5$. Of prompts scoring $\hat p = 0.5$, under the pool prior above, a large fraction are truly $p=0.95$ or $p=0.02$ items that fluctuated — the selected "mid-band" set is substantially estimation noise, and the realised variance gain collapses well below 2.25×.

**The obstruction, visible.** The signal gain is real and bounded (≈2.25×); the measurement is either more expensive than the training it improves, or too noisy to identify the band it targets. Everything between those poles — cheap amortised predictors of $p_\theta$ — is the unexplored middle, and it is exactly what §8 arm B tests.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*