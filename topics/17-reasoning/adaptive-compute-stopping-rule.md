---
id: 17-reasoning/adaptive-compute-stopping-rule
title: "Optimal Stopping Rule for Adaptive Test-Time Compute"
topic: 17-reasoning
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Stopping Rule for Adaptive Test-Time Compute

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/adaptive-compute-stopping-rule` · **Status:** empirically-open

## 1. Problem Statement

A reasoning model can spend more compute per query: sample more chains, extend a chain of thought, run more verifier passes, expand more tree nodes. Accuracy rises with budget and then flattens. The per-query optimum varies by orders of magnitude — an arithmetic query is settled in 200 tokens, a competition geometry problem may still be improving at 32k.

**The problem.** Given a query $x$ and a partial computation trace, decide *when to stop* so that expected accuracy per unit compute is maximized, using only information available at inference time (no ground-truth label, no oracle verifier).

Three variants, different difficulty:

- **Measurement.** Define the quantity a stopping rule should track: the marginal probability that continuing changes the emitted answer from wrong to right. Estimating it requires a counterfactual that is never observed on a deployed query.
- **Method.** Build a policy $\pi$ mapping trace state to {continue, stop} that dominates fixed-budget baselines on the accuracy-vs-compute frontier. Partially achieved; see §3.
- **Theory.** Characterize the optimal stopping time under a specified model of how answer correctness evolves with budget, and bound the regret of any rule that must estimate that model online. Open.

**Solved** would mean: a rule that, at matched *total* compute across a test distribution, beats the best fixed budget by a margin that survives per-instance ablation, and whose gap to a label-oracle stopping rule is bounded and small.

## 2. Formal Setting

Let $x \sim \mathcal{D}$ be a query with unknown answer $y^\star(x)$. Compute is spent in increments $t = 1, 2, \dots$; increment $t$ costs $c_t$ and produces evidence $Z_t$ (a sampled chain, a token, an expanded node). Define the filtration $\mathcal{F}_t = \sigma(x, Z_1, \dots, Z_t)$.

- **Answer map** $a_t = A(x, Z_{1:t})$ — the answer the system would emit if stopped now (majority vote, argmax verifier score, current partial-decode continuation).
- **Reward** $R_t = \mathbb{1}[a_t = y^\star(x)]$. *Measured* by exact-match grading against a reference on a held-out set; on deployed queries it is unobservable, which is the whole difficulty.
- **Cost** $C_\tau = \sum_{t\le\tau} c_t$. *Measured* as generated tokens (reproducible), FLOPs (requires an arithmetic model), or wall-clock (hardware- and batch-dependent, so not comparable across papers). Prefer tokens; report the others.

A stopping rule is a stopping time $\tau$ adapted to $\mathcal{F}_t$. The Lagrangian objective at price $\lambda>0$ (compute per unit accuracy):

$$V^\star(x) \;=\; \sup_{\tau}\; \mathbb{E}\big[\,R_\tau - \lambda\, C_\tau \;\big|\; x \,\big].$$

By the standard optimal-stopping characterization (Snell envelope), the optimal rule stops at the first $t$ where the current value dominates the continuation value:

$$\tau^\star = \min\{t : R_t^{\text{exp}} \;\ge\; \mathbb{E}[\,V_{t+1} \mid \mathcal{F}_t\,] - \lambda c_{t+1}\}, \qquad R_t^{\text{exp}} = \Pr[a_t = y^\star \mid \mathcal{F}_t].$$

Everything hinges on two estimands:

$$p_t \;=\; \Pr[a_t = y^\star \mid \mathcal{F}_t] \quad\text{(calibrated correctness)}, \qquad \Delta_t \;=\; \mathbb{E}[p_{t+k} - p_t \mid \mathcal{F}_t] \quad\text{(marginal gain)}.$$

$p_t$ is estimated by a confidence signal $\hat p_t$: self-consistency vote share $\hat p_t = \max_v n_v/t$, mean token logprob, a process reward model score, or a learned probe. $\Delta_t$ is estimated by regression on offline traces where $y^\star$ is known.

**Assumptions, and which fail.**

1. *$R_t$ is measurable offline.* Holds for math/code with checkers; fails for open-ended generation — the whole framework is undefined there.
2. *Monotone value of compute*, $\Delta_t \ge 0$. **Violated:** long-chain reasoners exhibit non-monotone accuracy — extended sampling flips correct answers to incorrect (overthinking).
3. *Calibration*, $\hat p_t \approx p_t$. **Violated:** self-consistency vote share is systematically overconfident on out-of-distribution and adversarial items; RLHF-tuned models are miscalibrated relative to their base checkpoints.
4. *Exchangeable increments.* Holds for i.i.d. sampling, **violated** for sequential decoding where $Z_{t+1}$ is conditioned on $Z_{1:t}$ — the sequential case has no i.i.d. concentration bound to lean on.
5. *A single global $\lambda$.* Serving systems have queue-level constraints and deadlines, so the true problem is a constrained MDP over a batch, not $N$ independent stopping problems.

## 3. State of the Art

**Established (ablated, reproduced).**

- **Adaptive-Consistency** (Aggarwal, Yang, Mausam, Nakov; EMNLP 2023) — stop sampling when a Dirichlet/Beta posterior on vote shares says the majority is stable. Reports ~$3.3\times$ fewer samples than fixed 40-sample self-consistency at an average accuracy drop under $0.1$ points across 17 dataset/model pairs. Simple, cheap, reproduced widely.
- **Compute-optimal scaling** (Snell, Lee, Xu, Kumar; ICLR 2025) — allocating budget by *estimated question difficulty* (PRM-bin) beats uniform best-of-N by roughly $4\times$ in test-time compute at matched accuracy on MATH with PaLM-2-S\*. This is difficulty-conditioned *allocation*, not within-trace stopping; the two are conflated in much follow-on writing.
- **Learning How Hard to Think** (Damani, Shenfeld, Peng, Bobu, Andreas; ICLR 2025) — a lightweight predictor of marginal reward gain routes budget across queries; reports matched performance at up to ~50% of the compute, or higher accuracy at matched compute, on math and code benchmarks with 7B–70B models.

**Claimed but unablated / benchmark-number-only.**

- **Budget forcing** (s1, Muennighoff et al., 2025) — forcing "Wait" to extend or an end token to truncate. Reported as a test-time scaling curve on AIME24/MATH500 with s1-32B; the stopping decision is a fixed token cap, not adaptive, and there is no per-instance oracle comparison.
- **Certaindex / Dynasor** (Fu et al., 2024) — a serving-level signal combining vote entropy and PRM score to terminate reasoning programs early; reported throughput gains (multi-fold) come from a serving system, so accuracy-vs-token and throughput-vs-latency effects are entangled.
- **Length-controlled RL** (L1, Aggarwal & Welleck, 2025) — trains the model to hit a requested token budget. Controls length; does not decide the budget.
- **Mid-generation self-assessment** (Manvi, Singh, Ermon, 2024) — the model predicts whether restarting will help. Promising numbers, single-lab, no independent replication at frontier scale that we can verify.

**Theory SOTA** is classical and not specialized to this setting: Wald's SPRT (1945) is optimal for a binary composite test with i.i.d. observations; Chow–Robbins and the Gittins index give optimal policies for exponential-family bandit/stopping problems. None of these apply directly, because the "arm" here is a non-stationary, self-conditioned generator with unknown correctness.

## 4. What Is Known

- Self-consistency accuracy saturates: on GSM8K with PaLM-540B, 40 samples gave $+17.9$ points over greedy, with most of the gain by ~10 samples (Wang et al., ICLR 2023). The saturation point is heavily query-dependent — this variance is exactly what a stopping rule can exploit.
- Vote share is a usable but weakly calibrated correctness proxy. Empirically, items with unanimous votes at $t=5$ are correct far more often than split items, but the mapping from vote share to accuracy shifts across datasets, so a threshold tuned on GSM8K does not transfer to AIME.
- Process reward models beat outcome reward models for best-of-N selection: PRM-supervised selection solved 78.2% of a MATH500-style subset at $N=1860$ versus 72.4% for outcome-supervised (Lightman et al., ICLR 2024, GPT-4-scale verifier). Better selection changes the shape of the accuracy-vs-budget curve, and so changes the optimal stopping point.
- Non-monotonicity is real. Long-CoT models (DeepSeek-R1-class, 2025) show accuracy on easy items that *degrades* with forced extension; several groups report overthinking on GSM8K-easy splits. Assumption 2 of §2 fails empirically.
- Difficulty is predictable enough to route on: binning by predicted difficulty recovers most of the oracle-allocation gain in Snell et al. at PaLM-2-S\* scale — but the oracle-difficulty arm still dominates the predicted-difficulty arm, so the estimation gap is not closed.

## 5. What Is Not Known

- **Theoretically open.** No regret bound for stopping a self-conditioned sequential generator. There is no proof that any $\mathcal{F}_t$-measurable rule attains $o(T)$ regret against the label-oracle rule without an assumption (calibration, or monotone $\Delta_t$) that is known to fail. Nor is there an impossibility proof.
- **Empirically open.** Nobody has published, at frontier scale (≥70B or a top-tier reasoning API), a clean three-arm comparison — best fixed budget vs. adaptive rule vs. label-oracle stopping — on the *same* token budget, with the oracle arm reported. Without the oracle, "adaptive beats fixed by 3×" says nothing about how much headroom remains. This is runnable today for a few thousand GPU-hours.
- **Methodologically blocked.** The marginal-gain estimand $\Delta_t$ is not identified from deployment data: on a live query you observe one trajectory and no label. Offline estimation on labelled sets gives $\Delta_t$ under the *offline* distribution of traces, which the stopping policy itself then shifts — a distribution-shift loop nobody has instrumented.

## 6. Why It Is Hard

**The specific obstruction is non-identifiability of "already right" from "cheap to be right".** Every deployable stopping signal — vote concentration, logprob, PRM score, self-reported confidence — is high on items the model finds *easy*. Easy items are both (a) already correct and (b) the items where continuing is cheapest and least useful. So the signal is maximally informative exactly where the decision does not matter, and near-uninformative on the hard, high-variance items where the stopping decision carries all the value. Formally, $\hat p_t$ correlates with $p_t$ but is nearly independent of $\Delta_t$, and it is $\Delta_t$ the rule needs.

Two secondary obstructions:

- **Confounded evaluation.** Papers report a population curve, mean accuracy against mean tokens. By Jensen, a policy that merely *reallocates* budget can move that curve without any per-instance decision quality. A rule that spends 10× on hard items would score identically to one that spends 10× on a random 10% — unless the oracle arm is reported. It usually is not.
- **Cost of the ground truth.** Building the $\Delta_t$ target requires, per query, many independent trajectories at many budgets with labels: $O(K \times B)$ generations. At $K=64$, $B=8$ budget levels, 2k queries, 8k tokens each, that is ~8B generated tokens for one model — feasible, but not casually, and it must be redone per model.

## 7. Current Research (as of 2026)

- **Confidence-gated early exit in serving stacks** — vLLM/SGLang-adjacent work on terminating reasoning requests via entropy or vote-stability signals, driven by the cost of long-CoT inference. Certaindex-style signals are the reference point *(frontier — verify current implementations)*.
- **RL with an explicit length or budget term in the reward** — training the model to internalize stopping rather than bolting a controller on top (L1; length-penalized variants in open reasoning recipes). The open question is whether an internalized stop generalizes off the training difficulty distribution *(frontier — verify)*.
- **Inference-aware fine-tuning** — training the policy for the best-of-N or verifier-selection objective it will be deployed under (Chow et al., 2024), which changes the accuracy-vs-$N$ curve and so relocates $\tau^\star$.
- **Calibration of reasoning traces** — probes on hidden states for "will this answer survive more sampling", including mid-generation self-evaluation (Manvi et al.). Academic groups: MIT (Andreas), Stanford (Ermon), CMU (Welleck), Berkeley (Kumar/Levine-adjacent).

## 8. Concrete Next Experiment

**The oracle-gap experiment.** Scale it small enough to run, large enough to be believed.

- **Scale.** One open-weights reasoning model at 32B (e.g. a DeepSeek-R1-distill-32B or Qwen-class reasoner), 1,000 problems spanning three difficulty tiers (GSM8K-hard 400, MATH500 400, AIME-style 200). Sample $K = 64$ independent chains per problem, cap 8k tokens, temperature 0.7. Record per-chain tokens, final answer, correctness. Cost: ~$1000 \times 64 \times 3\text{k}$ mean tokens $\approx 2\times10^{8}$ tokens — a few hundred A100-hours. Log everything once; all arms are then computed offline by replaying prefixes of the sample stream.
- **Arms**, all at matched *total* token budget $B_{\text{tot}}$, swept over five values:
  1. **Control:** best fixed $k$ per budget (the strongest honest baseline; tune $k$ on the same data — give the control every advantage).
  2. Adaptive-Consistency (Beta-posterior stopping).
  3. Learned $\Delta_t$ predictor (trained on a disjoint 1,000-problem split).
  4. **Label-oracle stop:** stop at the first $t$ where $a_t$ is correct *and* stays correct for all $t' > t$; if never correct, spend $k=1$. This is the ceiling.
- **The deciding number.** $\rho = \dfrac{\text{Acc}_{\text{adaptive}} - \text{Acc}_{\text{fixed}}}{\text{Acc}_{\text{oracle}} - \text{Acc}_{\text{fixed}}}$ at matched $B_{\text{tot}}$, at the budget where the fixed-$k$ curve is at 90% of its asymptote.

Interpretation: $\rho < 0.15$ means current adaptive rules capture almost none of the available headroom, and the problem is wide open on the *method* axis. $\rho > 0.6$ means the remaining gap is oracle-sized and the field should move to sequential (within-chain) stopping, where the i.i.d. structure that makes this replay possible no longer holds. No published paper reports $\rho$.

## 9. Key References

- **[Foundational]** Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR, 2023. — arXiv:2203.11171
- **[Foundational]** Wald. *Sequential Tests of Statistical Hypotheses.* Annals of Mathematical Statistics, 1945.
- **[Foundational]** Chow, Robbins, Siegmund. *Great Expectations: The Theory of Optimal Stopping.* Houghton Mifflin, 1971.
- **[SOTA]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* ICLR, 2025. — arXiv:2408.03314
- **[SOTA]** Damani, Shenfeld, Peng, Bobu, Andreas. *Learning How Hard to Think: Input-Adaptive Allocation of LM Computation.* ICLR, 2025. — arXiv:2410.04707
- **[SOTA]** Aggarwal, Yang, Mausam, Nakov. *Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs.* EMNLP, 2023. — arXiv:2305.11860
- **[SOTA]** Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever, Cobbe. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050
- **[Method]** Muennighoff, Yang, Shi, Li, Fei-Fei, Hajishirzi, Zettlemoyer, Liang, Candès, Hashimoto. *s1: Simple Test-Time Scaling.* 2025. — arXiv:2501.19393
- **[Method]** Manvi, Singh, Ermon. *Adaptive Inference-Time Compute: LLMs Can Predict If They Can Do Better, Even Mid-Generation.* 2024. — arXiv:2410.02725
- **[Method]** Aggarwal, Welleck. *L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning.* 2025. — arXiv:2503.04697
- **[Systems]** Fu, Chen, Jia, Sharma et al. *Efficiently Serving LLM Reasoning Programs with Certaindex.* 2024. — arXiv:2412.20993
- **[Method]** Chow, Tennenholtz, Gur, Zhuang, Dai, Thiagarajan, Boutilier, Agarwal, Kumar, Faust. *Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models.* 2024. — arXiv:2412.15287
- **[Model]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* 2025. — arXiv:2501.12948

## 10. Worked Example

One MATH-level problem, self-consistency with a Beta-posterior stopping rule at $t=8$ samples so far, votes: answer $A$ = 5, answer $B$ = 3.

Posterior on the majority's true share with a Jeffreys prior: $\theta_A \sim \text{Beta}(5.5, 3.5)$, so $\mathbb{E}[\theta_A] = 0.61$ and $\Pr[\theta_A > 0.5] = 0.74$. Adaptive-Consistency's stability criterion (stop when the probability that the current majority survives to the budget cap exceeds $0.95$) says **continue** — correctly, on its own terms.

Now the obstruction. What the rule actually needs is $\Delta_t$: does continuing *change the emitted answer to the right one*? Replay the full 64-sample stream for this item and you find one of two regimes, and $\mathcal{F}_8$ cannot tell them apart:

| Regime | vote share at $t{=}64$ | $R_8$ | $R_{64}$ | $\Delta$ per 56 extra samples |
|---|---|---|---|---|
| Split-then-converge to $A$ | $A$: 0.62 | 1 | 1 | $0$ |
| Split-then-flip to $B$ | $B$: 0.55 | 0 | 1 | $+1$ |

Both regimes produce the identical evidence $(5, 3)$ at $t=8$. Under a plausible empirical mix — on MATH-hard items with a 5–3 split, roughly 70% converge to the current majority and 30% flip — the expected gain is $\mathbb{E}[\Delta_8] \approx 0.30 \times \Pr[\text{flip is to the correct answer}] \approx 0.30 \times 0.6 = 0.18$ accuracy points per item, at a cost of 56 extra chains $\times$ ~3k tokens $=$ 168k tokens. At $\lambda$ set so that 168k tokens is worth 0.10 accuracy points, continuing is correct here — but the same $(5,3)$ evidence on GSM8K, where the converge rate is ~92%, gives $\mathbb{E}[\Delta_8] \approx 0.05$ and continuing is wrong.

The rule's input is identical in both cases. What separates them is the dataset-level flip rate, which is a property of the *distribution*, not of $\mathcal{F}_t$. That is the non-identifiability in §6, made concrete: a threshold tuned on one distribution is misspecified on the next, and no amount of within-trace confidence recovers the missing term.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*