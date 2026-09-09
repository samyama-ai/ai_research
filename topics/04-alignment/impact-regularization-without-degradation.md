---
id: 04-alignment/impact-regularization-without-degradation
title: "Impact Regularization Without Task Degradation"
topic: 04-alignment
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Impact Regularization Without Task Degradation

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/impact-regularization-without-degradation` · **Status:** open

## 1. Problem Statement

An impact regularizer penalizes an agent for changing the world (or itself) more than the task requires. The failure mode it targets is the side effect: an unspecified-but-unwanted consequence of competent task pursuit. The failure mode it *creates* is task degradation: the same penalty that suppresses side effects also suppresses the state changes the task needs.

**Input.** A task objective $R$ (reward model, preference dataset, or verifier), a policy class $\Pi$, a baseline policy or baseline state, and an unpenalized reference policy $\pi^\*$ that maximizes $R$.

**Output.** A policy $\hat\pi$ and a scalar $\lambda$ such that $\hat\pi$ scores near $\pi^\*$ on $R$ while scoring far below $\pi^\*$ on a side-effect measure the regularizer was not fit to.

**Decision predicate.** For a held-out side-effect measure $S$ (not used in training) and task return $J$, does there exist $\lambda>0$ with

$$J(\hat\pi_\lambda) \ge (1-\delta)\,J(\pi^\*) \quad\text{and}\quad S(\hat\pi_\lambda) \le \gamma\, S(\pi^\*)$$

for $\delta \le 0.05$ and $\gamma \le 0.5$? Solving the problem means exhibiting a method whose $(\delta,\gamma)$ frontier dominates plain reward-scale tuning, on tasks not designed around the regularizer.

Three variants, of different difficulty:
- **Measurement:** define $S$ so that it is not a relabeling of "distance from the baseline". Currently the weakest link.
- **Method:** construct a penalty with a favorable $(\delta,\gamma)$ frontier. Several exist for gridworlds; none is established for language models.
- **Theory:** prove that any penalty depending only on state-distance must trade $\delta$ against $\gamma$ at some rate, or exhibit a penalty class that escapes it.

## 2. Formal Setting

MDP $M=(\mathcal{S},\mathcal{A},T,R,\mathcal{S}_0,\gamma_d)$; for language models, $\mathcal{S}$ is the prompt-plus-prefix, $\mathcal{A}$ the token vocabulary, and $T$ is deterministic concatenation.

**Task return.** $J(\pi)=\mathbb{E}_{s_0\sim\mathcal{S}_0,\ \tau\sim\pi}\left[\sum_t \gamma_d^t R(s_t,a_t)\right]$. *Measured as:* mean reward-model score, or win-rate against $\pi_0$ under a fixed judge, on $n\ge 1000$ held-out prompts.

**Deviation.** For LLMs the standard regularizer is sequence-level KL to the SFT reference,
$$D_{\mathrm{KL}}(\pi\Vert\pi_0)=\mathbb{E}_{s\sim d^\pi}\big[\textstyle\sum_a \pi(a\mid s)\log\tfrac{\pi(a\mid s)}{\pi_0(a\mid s)}\big],$$
*measured as* the mean per-token log-ratio on on-policy samples — an estimator of the *reverse* KL under the policy's own state distribution, not the symmetric quantity people usually mean by "distance".

**Attainable Utility Preservation** (Turner et al., 2020) penalizes loss of optionality over an auxiliary reward set $\mathcal{R}=\{R_1,\dots,R_k\}$:
$$\mathrm{Pen}(s,a)=\frac{1}{k}\sum_{i=1}^{k}\big|Q_{R_i}(s,a)-Q_{R_i}(s,\varnothing)\big|,$$
with $\varnothing$ the no-op. *Measured as:* $k$ learned $Q$-networks (in the SafeLife work, $k=1$ with $R_1$ a random linear function of a VAE latent), divided by a running scale term.

**Relative reachability** (Krakovna et al., 2018) penalizes reachability lost relative to a baseline $s'$:
$$d_{\mathrm{RR}}(s;s')=\frac{1}{|\mathcal{S}|}\sum_{x\in\mathcal{S}}\max\big(0,\ \mathcal{R}(x\mid s')-\mathcal{R}(x\mid s)\big),$$
$\mathcal{R}(x\mid s)$ being discounted reachability of $x$ from $s$. *Measured as:* exhaustive tabular rollout — tractable only in gridworlds.

**Baseline.** Starting state, inaction ($\pi$ replaced by no-ops from $s_0$), or stepwise inaction (no-ops from $s_{t-1}$). Choice is not innocuous: starting-state baselines penalize irreversible changes the environment would have made anyway; stepwise-inaction baselines admit offsetting, where the agent undoes its own useful effect to return to the baseline.

**Assumptions, and where they break.**
1. *A no-op action exists and is safe.* Violated for deployed assistants: refusing is an action with consequences.
2. *The auxiliary reward set spans what we care about.* Unverifiable; AUP's own results use a single random function, so the span claim is not made.
3. *Side effects are measurable as state-distance.* Violated whenever the harm is a distribution-level property (loss of output diversity, sycophancy) with no state-space signature.
4. *Held-out $S$ is independent of $D$.* In practice both are computed from the same corpus, so $\gamma$ is partly mechanical.

## 3. State of the Art

**Theory SOTA.** Power-seeking results (Turner et al., NeurIPS 2021) prove that for most reward functions under symmetry conditions on the MDP, optimal policies favor options-preserving states — the formal justification for optionality-based penalties. Established. There is no matching impossibility theorem for the $(\delta,\gamma)$ trade-off.

**Empirical SOTA, gridworlds.** AUP on SafeLife (Turner, Ratzlaff, Tadepalli, NeurIPS 2020) reports large side-effect reductions at near-parity task score using a single randomly generated auxiliary reward. *Established within the benchmark; independently reproduced only partially, and SafeLife's side-effect metric is itself a state-difference measure, so this is the confounded case.* Relative reachability and future-task preservation (Krakovna et al., NeurIPS 2020) solve the AI Safety Gridworlds side-effect suite including the offsetting cases; these are tabular results on $\le 10^2$ states.

**Empirical SOTA, language models.** KL-regularized RLHF is the only impact regularizer deployed at scale. Gao, Schulman, Hilton (ICML 2023) fit proxy-vs-gold reward as a function of $d=\sqrt{\mathrm{KL}}$, with best-of-$n$ following $R(d)=d(\alpha_{bon}-\beta_{bon}d)$ and RL a similar concave form — a measured optimum in $\lambda$, not an argument that the frontier is good. PPO-ptx (Ouyang et al., 2022) mixes pretraining gradients to repair regressions; reported to remove most of the public-NLP-benchmark drop, **unablated against a simple learning-rate or KL-coefficient sweep**. Model souping / WARM-style weight averaging is *claimed* to reduce alignment tax; the claim rests on benchmark aggregates, not on a matched-$J$ comparison.

## 4. What Is Known

- **The alignment tax is scale-dependent.** Bai et al. (2022), on HH-RLHF with models from ~13M to 52B parameters: RLHF degrades zero-shot NLP evaluations for small models and becomes neutral-to-positive above roughly 10B parameters. Measured on a fixed suite of zero-shot tasks; the crossover point is suite-dependent.
- **Regression is real at 175B.** InstructGPT (Ouyang et al., 2022) shows PPO performance drops on SQuADv2, DROP, HellaSwag and WMT translation relative to the SFT model; PPO-ptx recovers most of it.
- **Over-optimization is non-monotone in $\lambda$.** Gao et al. (2023), policy ~1.2B, reward models 3M–3B: gold reward rises then falls as KL grows, with the peak location scaling predictably in RM size and data.
- **KL-regularization does not preserve diversity.** Kirk et al. (ICLR 2024): RLHF improves out-of-distribution generalization over SFT but sharply reduces across-input output diversity at 7B scale — a side effect KL was supposed to bound and does not.
- **Locality of a penalty does not follow from locality of a mechanism.** Hase et al. (NeurIPS 2023) show causal-tracing localization does not predict where editing works; the analogous lesson for impact penalties is that "small parameter change" ≠ "small behavioral change".

## 5. What Is Not Known

- **Methodologically blocked.** Whether any deployed measure of "side effect" for language models is independent of the regularizer. Every current $S$ (KL, benchmark delta, edit-locality score) is a distance from the pre-training reference, which is what $D$ penalizes. Until $S$ is constructed from consequences rather than distances, $\gamma$ is not interpretable.
- **Theoretically open.** Whether a state-distance penalty must trade $\delta$ against $\gamma$. No lower bound of the form "for any $D$ measurable from state alone, there is a task where $\delta \ge f(\gamma)$" exists, nor a construction avoiding it.
- **Empirically open.** Whether AUP-style optionality penalties transfer to LLMs. The auxiliary-$Q$ machinery is runnable at 7B with LoRA heads; nobody has published the frontier.
- **Empirically open.** Whether the small-model alignment tax at 1B–10B is caused by KL regularization or by reward-model error; the two are confounded in every published run.

## 6. Why It Is Hard

**The primary obstruction is confounded measurement, not compute.** The side-effect score and the regularizer are computed from the same reference distribution. A method that reduces $S$ by moving the policy toward $\pi_0$ scores well while doing nothing about unanticipated consequences — the thing the regularizer names. SafeLife's side-effect metric literally counts cell-state differences from an inaction rollout; AUP is a function of the same rollout.

**Secondary: non-identifiability of the auxiliary set.** AUP's guarantee, such as it is, holds if $\mathcal{R}$ spans the human-relevant utilities. Nothing measures span. A random $\mathcal{R}$ that works on SafeLife tells you the environment's geometry is benign, not that the method is.

**Third: absent ground truth.** For LLMs there is no held-out list of "effects we did not want". Benchmark deltas are a proxy that measures capability retention, which the regularizer's own objective already targets.

## 7. Current Research (as of 2026)

- **Weight-space tax mitigation** — model souping, task arithmetic, and WARM-style reward-model averaging as substitutes for a KL penalty (Google DeepMind, UW/AI2 lineage). *(frontier — verify)* Matched-$J$ ablations remain scarce.
- **Adaptive and per-token KL schedules** in open post-training stacks; treated as an engineering knob, with the frontier unreported.
- **Unlearning ripple effects** (NYU, Allen Institute, academic unlearning workshops): measuring collateral capability loss from targeted removal. This is the same $(\delta,\gamma)$ question with a cleaner $S$, since the removal target is specified.
- **Agentic side effects** — irreversibility penalties for tool-using agents (file deletion, transactions). Early, mostly evaluation-suite work. *(frontier — verify)*
- **Formal optionality** — successor-representation and empowerment approximations as scalable stand-ins for reachability.

## 8. Concrete Next Experiment

**Question.** Does an optionality penalty beat a KL penalty at matched task return on a side-effect measure it was not fit to?

**Scale.** One 7B–8B open base model, standard SFT checkpoint, PPO or GRPO on a helpfulness reward model. ~$10^4$ prompts, 3 seeds. Budget ≈ 400–800 A100-hours total across arms. This is small enough to run in a week.

**Arms.**
- *Control:* KL penalty, $\lambda$ swept over 6 values spanning per-token KL $\in[0.005,0.5]$.
- *Treatment:* AUP-style penalty, $k=8$ auxiliary value heads (LoRA on the frozen SFT trunk) trained on random projections of the hidden state; penalty $\frac{1}{k}\sum_i|V_i(s_t)-V_i(s_t^{\varnothing})|$ where $s^\varnothing$ continues under $\pi_0$.
- *Held-out $S$ (fixed before running):* a pre-registered agentic side-effect suite — count of irreversible tool calls (file writes, deletions, sends) in a sandbox where the task never requires them — plus across-input diversity (distinct-3 across prompts). Neither is a distance from $\pi_0$, and neither is in the training reward.

**Deciding number.** Interpolate each arm's $\lambda$-sweep to the $\lambda$ giving $J=0.95\,J(\pi^\*)$ and read off $S$ there. The single number is $\Delta = S_{\mathrm{KL}}-S_{\mathrm{AUP}}$ at matched $J$, in irreversible calls per 100 episodes. $\Delta>0$ with a 95% bootstrap CI excluding zero across 3 seeds means optionality penalties buy something KL does not. $\Delta\approx 0$ means the LLM alignment tax is reward-model error, not impact, and the whole gridworld line does not transfer.

## 9. Key References

- **[Foundational]** Amodei, Olah, Steinhardt, Christiano, Schulman, Mané. *Concrete Problems in AI Safety.* 2016. — arXiv:1606.06565
- **[Foundational]** Armstrong, Levinstein. *Low Impact Artificial Intelligences.* 2017. — arXiv:1705.10720
- **[Foundational]** Krakovna, Orseau, Kumar, Martic, Legg. *Penalizing Side Effects using Stepwise Relative Reachability.* AI Safety Workshop (IJCAI), 2019. — arXiv:1806.01186
- **[SOTA]** Turner, Hadfield-Menell, Tadepalli. *Conservative Agency via Attainable Utility Preservation.* AIES, 2020. — arXiv:1902.09725
- **[SOTA]** Turner, Ratzlaff, Tadepalli. *Avoiding Side Effects in Complex Environments.* NeurIPS, 2020. — arXiv:2006.06547
- **[SOTA]** Krakovna, Orseau, Ngo, Martic, Legg. *Avoiding Side Effects By Considering Future Tasks.* NeurIPS, 2020. — arXiv:2010.07877
- **[SOTA]** Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[SOTA]** Ouyang et al. *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS, 2022. — arXiv:2203.02155
- **[Empirical]** Bai et al. *Training a Helpful and Harmless Assistant with RLHF.* Anthropic, 2022. — arXiv:2204.05862
- **[Empirical]** Kirk, Mediratta, Nalmpantis, Luketina, Grefenstette, Riedel, Rocktäschel. *Understanding the Effects of RLHF on LLM Generalisation and Diversity.* ICLR, 2024. — arXiv:2310.06452
- **[Theory]** Turner, Smith, Shah, Critch, Tadepalli. *Optimal Policies Tend to Seek Power.* NeurIPS, 2021. — arXiv:1912.01683
- **[Benchmark]** Wainwright, Eckersley. *SafeLife 1.0: Exploring Side Effects in Complex Environments.* SafeAI Workshop (AAAI), 2020. — arXiv:1912.01217
- **[Benchmark]** Leike, Martic, Krakovna, Ortega, Everitt, Lefrancq, Orseau, Legg. *AI Safety Gridworlds.* 2017. — arXiv:1711.09883
- **[Related]** Hase, Bansal, Kim, Ghandeharioun. *Does Localization Inform Editing?* NeurIPS, 2023. — arXiv:2301.04213
- **[Survey]** Ji et al. *AI Alignment: A Comprehensive Survey.* 2023. — arXiv:2310.19852

## 10. Worked Example

Take a 7B assistant, SFT reference $\pi_0$, helpfulness RM. Sweep the KL coefficient and record three columns: RM score, held-out per-token KL, and irreversible tool calls per 100 sandbox episodes where the task is read-only.

| $\lambda$ | RM score (norm.) | per-token KL | irreversible calls / 100 ep |
|---|---|---|---|
| 0.5 | 0.71 | 0.006 | 2 |
| 0.1 | 0.93 | 0.041 | 9 |
| 0.02 | 1.00 | 0.19 | 31 |
| 0.005 | 0.97 | 0.62 | 58 |

(Illustrative magnitudes consistent with the concave $R(\sqrt{\mathrm{KL}})$ shape measured by Gao et al., 2023.)

Read the frontier at $J=0.95 J(\pi^\*)$, which falls between $\lambda=0.1$ and $\lambda=0.02$: about 13 irreversible calls, roughly $0.42\times$ the unregularized rate — so $\gamma\approx 0.42$ at $\delta=0.05$, apparently passing the predicate.

Now the obstruction. Fit the same two columns against each other: irreversible calls scale close to linearly in $\sqrt{\mathrm{KL}}$ over this range ($2,9,31,58$ against $\sqrt{\mathrm{KL}}=0.077,0.20,0.44,0.79$; ratios $26,45,70,73$ per unit). The penalty is not selectively suppressing side effects — it is suppressing *all* deviation from $\pi_0$, and the side-effect count is a monotone readout of that same deviation. A regularizer that cut side effects at fixed KL would show a downward break in that ratio; none appears.

So the reported $\gamma=0.42$ is a restatement of "the policy moved less", available from any knob that shrinks KL, including simply lowering the learning rate. The number that would falsify this — side-effect count at *matched* KL and matched $J$ between two different penalty families — is the one nobody has published. That is the gap, and it is a measurement gap before it is a method gap.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*