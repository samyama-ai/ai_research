---
id: 17-reasoning/best-of-n-scaling-law
title: "Scaling Law for Best-of-N Sampling Returns"
topic: 17-reasoning
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Law for Best-of-N Sampling Returns

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/best-of-n-scaling-law` · **Status:** partially-solved

## 1. Problem Statement

Draw $N$ i.i.d. samples from a language model for a fixed prompt and return one of them, selected by a verifier. Accuracy rises with $N$, then flattens. The problem is to give a predictive law for that curve.

Three variants, routinely conflated:

- **Measurement.** Given accuracy at $N \in \{1, \dots, 10^2\}$, predict accuracy at $N = 10^4$ to within a stated error bar. Is the functional form power-law, exponentiated power-law, or floor-plus-power-law?
- **Method.** Given a total token budget $B$, choose $(\text{model size}, N, \text{verifier})$ to maximize accuracy. Solving this requires the measurement variant plus a verifier-cost model.
- **Theory.** Derive the observed form from properties of the per-problem success distribution and the verifier's error rate, rather than fitting it.

Solving it means: a parametric family, fit on cheap small-$N$ data, whose extrapolation to large $N$ is validated out-of-sample on held-out benchmarks and models — and which separates the *coverage* ceiling (does any sample succeed) from the *selection* ceiling (can the verifier find it).

## 2. Formal Setting

Let $p_\theta(\cdot \mid x)$ be the sampling policy at fixed temperature $T$ and nucleus $p$, $x \sim \mathcal{D}$ a problem drawn from a benchmark, and $y_1,\dots,y_N \stackrel{\text{iid}}{\sim} p_\theta(\cdot \mid x)$.

**Per-problem success rate** — measured by sampling $M \gg 1$ completions and scoring each with the benchmark's ground-truth checker:
$$p_x \;=\; \Pr_{y \sim p_\theta(\cdot\mid x)}\big[\mathrm{correct}(x,y)\big], \qquad \hat p_x = \tfrac{1}{M}\sum_{j=1}^M \mathbb{1}[\mathrm{correct}(x,y_j)].$$

**Coverage** (= pass@$N$ = oracle-verifier accuracy), estimated unbiasedly from $M \geq N$ samples by the Chen et al. (2021) combinatorial estimator rather than by resampling:
$$C(N) \;=\; \mathbb{E}_{x}\big[1 - (1-p_x)^N\big], \qquad \widehat{C}(N) = \frac{1}{|\mathcal{D}|}\sum_x \left[1 - \binom{M - c_x}{N}\Big/\binom{M}{N}\right].$$

**Best-of-$N$ accuracy** under a real verifier $v: (x,y)\to\mathbb{R}$ (reward model, PRM aggregate, unit tests, or majority vote):
$$A_v(N) \;=\; \mathbb{E}_x\Big[\Pr\big[\mathrm{correct}(x, y_{i^\star})\big]\Big], \qquad i^\star = \arg\max_i v(x,y_i).$$
Always $A_v(N) \le C(N)$; the *selection gap* is $G(N) = C(N) - A_v(N)$.

**Cost.** $B(N) = N \cdot \bar\ell \cdot 2P_{\text{gen}} + N \cdot \bar\ell \cdot 2P_{\text{ver}}$ FLOPs for a $P$-parameter generator and verifier at mean length $\bar\ell$. Any "scaling law" that plots accuracy against $N$ instead of $B$ is not comparable across model sizes.

**Distributional drift.** Best-of-$N$ shifts the output distribution; for continuous, a.s.-distinct rewards, $D_{\mathrm{KL}}(\pi_{\text{BoN}} \Vert p_\theta) = \log N - \frac{N-1}{N}$ (Beirami et al., 2024, which also proves this is an upper bound in the general case).

**Where the law comes from.** If the population density of $p_x$ near zero behaves as $f(p) \propto p^{a-1}$, then $1 - C(N) \sim \Gamma(a)\,N^{-a}$: a power law in $N$ whose exponent is a property of the *hard tail of the problem distribution*, not of the model.

Assumptions, with the ones known to fail marked:

1. Samples are i.i.d. — **violated** by sequential revision, tree search, and any adaptive-temperature scheme.
2. $p_x$ is stable across the run — **violated** across prompt-format and few-shot changes; $p_x$ can move by tens of points.
3. `correct` is exact — **violated**: MATH/GSM8K answer-matching admits false positives from guessed-then-unjustified answers, which inflate $C(N)$ precisely at large $N$.
4. $v$ is independent of correctness errors across samples — **violated**: verifier errors are strongly correlated within a problem, which is what creates a hard plateau rather than a slow one.

## 3. State of the Art

**Established (replicated, ablated).**
- Coverage grows smoothly and near-log-linearly over 4+ orders of magnitude in $N$. Brown et al. (*Large Language Monkeys*, 2024) fit $\log C(N) \approx -a N^{-b}$ ("exponentiated power law") across Llama-3, Gemma, and Pythia families on GSM8K, MATH, MiniF2F, CodeContests, and SWE-bench Lite.
- Coverage $\ne$ accuracy. In the same work, DeepSeek-Coder-V2-Instruct on SWE-bench Lite reaches 56% coverage at $N=250$ versus 15.9% at $N=1$, but the best available *selector* recovers only ~43%.
- Best-of-$N$ against a learned reward model overoptimizes: gold reward rises, peaks, then falls, and the proxy-vs-gold curve is well fit by $R(d) = d(\alpha - \beta d)$ with $d = \sqrt{D_{\mathrm{KL}}}$ (Gao, Schulman & Hilton, ICML 2023) — the single most reliable functional form in this literature.
- Under a fixed budget, sampling from a smaller model many times can beat one sample from a larger one, on the easier fraction of problems (Snell et al., 2024; Wu et al., 2024).

**Claimed but unablated.**
- That the exponentiated power law *extrapolates*. It is a fit over the measured range; published work does not report held-out extrapolation error from $N \le 10^2$ to $N = 10^4$.
- "Test-time compute beats 14× parameters" (Snell et al., 2024) is a benchmark number on MATH with PaLM 2-S\*, conditional on difficulty bin and on a PRM verifier of unreported cost. It is not a law.
- Reported gains at very large $N$ that rely on answer-string matching have no false-positive ablation. Yue et al. (2025) show pass@$k$ curves for RL-trained models cross their base models at large $k$, which is evidence that large-$N$ metrics measure something other than reasoning capability.

## 4. What Is Known

- **KL cost of Best-of-$N$ is exactly $\log N - (N-1)/N$** nats under distinct continuous rewards (Beirami et al., 2024). At $N=10^4$: $8.21$ nats. Verified numerically, not merely asserted.
- **Coverage numbers at scale.** Brown et al. (2024): Llama-3-8B-Instruct on MATH, ~15.9% → ~79% pass@$N$ from $N=1$ to $N=10^4$; Gemma-2B on GSM8K, ~0.8× improvement per decade sustained to $N=10^4$. Pythia-70M–12B show the same functional form across a 170× parameter range.
- **Selection is the binding constraint.** On SWE-bench Lite, coverage at $N=250$ (56%) exceeds achieved accuracy (43%) by 13 points with the best selector tried.
- **Verifier quality sets the plateau.** Stroebl, Kapoor & Narayanan (2024) show that with a verifier of fixed false-positive rate $\epsilon > 0$, accuracy under resampling is bounded and can *decline* in $N$; the plateau height depends on $\epsilon$, not on $N$.
- **PRMs beat ORMs at large $N$.** Lightman et al. (2023): process supervision reaches 78.2% on a MATH subset at $N = 1860$, versus a lower and earlier-saturating ORM curve — the earliest clean demonstration that the plateau is a property of the verifier.
- **Majority voting saturates earlier than reward-model BoN** on GSM8K/MATH, typically by $N \approx 64$ (Wang et al., 2023; reproduced in Wu et al., 2024).

## 5. What Is Not Known

- **Empirically open.** Whether any fitted form predicts $C(10^4)$ from data at $N \le 10^2$ within 2 accuracy points, out-of-sample across model families. The experiment is runnable today; nobody has published the held-out extrapolation error. This is the core gap.
- **Empirically open.** Whether the asymptotic floor $\pi = \lim_{N\to\infty}\big(1 - C(N)\big)$ is strictly positive on standard benchmarks at fixed $T$ — i.e. whether some problems have $p_x$ exactly $0$, or merely $10^{-6}$.
- **Theoretically open.** No derivation links the exponent $b$ to measurable properties of the model or the benchmark. $b$ is currently a fitted constant with no predictive content.
- **Theoretically open.** No tight lower bound on $G(N) = C(N) - A_v(N)$ as a function of verifier calibration error. The Stroebl et al. bound is an existence result, not a rate.
- **Methodologically blocked.** Coverage at $N \ge 10^3$ is not well defined under string-match grading, because the false-positive rate of the grader is itself an increasing function of $N$ and is unmeasured.

## 6. Why It Is Hard

**Non-identifiability of the tail.** The observable $C(N)$ is the Laplace-transform-like functional $\mathbb{E}_x[1-(1-p_x)^N]$. Measuring it at $N \le 10^2$ constrains only the part of the $p_x$ distribution above roughly $10^{-2}$. The extrapolation to $N=10^4$ is entirely determined by the density of $p_x$ in $[10^{-4}, 10^{-2}]$ — a region containing few problems and estimated from a handful of successes each. Two models of the tail (power law versus power-law-plus-floor) can agree within measurement noise over the measured range and diverge by 5+ accuracy points at $10^4$. This is not a compute problem; it is a statistical identifiability problem, and more samples at small $N$ do not fix it.

**Compounding this:** the grader's false-positive rate is confounded with the quantity being measured (Section 5), and per-problem cost is $O(N)$ so the decisive measurement is ~100× the cost of the fits people actually publish.

## 7. Current Research (as of 2026)

- **Compute-optimal inference allocation** — Snell/Kumar (Berkeley, Google DeepMind), Welleck/Yue (CMU): treating $N$, model size, and revision depth as a joint budget allocation.
- **Verification scaling** — the observation that scaling the *verifier's* compute (self-verification, sampled verification) moves the plateau, not just $N$; Google DeepMind's *Sample, Scrutinize and Scale* line, 2025 *(frontier — verify)*.
- **Overoptimization theory** — extensions of Gao et al. to BoN under bounded/heavy-tailed reward error; Beirami, Eisenstein and co-authors at Google Research.
- **Pass@$k$ as a capability metric under attack** — Yue et al. (2025) and follow-ups argue large-$k$ coverage reflects the base model, not post-training. *(frontier — verify)* whether this survives false-positive filtering.
- **Contamination of the premise** — reasoning models trained with RL on long chains-of-thought have much flatter pass@$k$ curves than instruct models; whether the same law family applies is untested at $N \ge 10^3$ *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does the fitted asymptotic floor $\pi$ differ from zero, and does small-$N$ extrapolation predict large-$N$ coverage?

**Scale.** One open 7–8B instruct model. $|\mathcal{D}| = 1000$ MATH-500 + AIME-style problems. $M = 10^4$ samples per problem at $T=0.8$: $10^7$ completions, ~$5\times10^9$ generated tokens, roughly 1.5k A100-hours — under \$5k on spot capacity.

**Arms.**
1. *Treatment:* fit three families — pure power law $1-C = A N^{-b}$; exponentiated power law $\log C = -aN^{-b}$; floor model $1-C = \pi + A N^{-b}$ — on $N \le 10^2$ only.
2. *Control arm:* the same three fits, on the full $N \le 10^4$ data. The difference between control and treatment fits is the extrapolation error.
3. *Grader control:* human or strong-model adjudication of a 500-completion stratified sample of *first-time successes at $N > 10^3$*, giving the false-positive rate $\epsilon_N$ used to correct $\widehat C$.

**Deciding number.** $|\widehat{C}_{\text{extrap}}(10^4) - \widehat{C}_{\text{measured}}(10^4)|$, after grader correction, in accuracy points. Under 2 points: the small-$N$ law is predictive and the measurement variant is solved for this regime. Over 5 points: it is a description, not a law, and the field should stop extrapolating it. Secondary readout: the 95% CI on $\pi$ in the floor model — if it excludes 0, coverage has a hard ceiling and unbounded resampling is a dead end.

## 9. Key References

- **[Foundational]** Cobbe, Kosaraju, Bavarian, et al. *Training Verifiers to Solve Math Word Problems.* 2021. — arXiv:2110.14168
- **[Foundational]** Chen, Tworek, Jun, et al. *Evaluating Large Language Models Trained on Code.* 2021. — arXiv:2107.03374 (unbiased pass@$k$ estimator)
- **[SOTA / theory]** Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML 2023. — arXiv:2210.10760
- **[SOTA / empirical]** Brown, Juravsky, Ehrlich, et al. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[SOTA]** Wu, Sun, Li, Welleck, Yue. *Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with LLMs.* 2024. — arXiv:2408.00724
- **[Theory]** Beirami, Agarwal, Berant, D'Amour, Eisenstein, Nagpal, Suresh. *Theoretical Guarantees on the Best-of-n Alignment Policy.* 2024. — arXiv:2401.01879
- **[Critique]** Stroebl, Kapoor, Narayanan. *Inference Scaling fLaws: The Limits of LLM Resampling with Imperfect Verifiers.* 2024. — arXiv:2411.17501
- **[Critique]** Yue, Chen, Lu, et al. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* 2025. — arXiv:2504.13837
- **[Related]** Lightman, Kosaraju, Burda, et al. *Let's Verify Step by Step.* ICLR 2024. — arXiv:2305.20050
- **[Related]** Wang, Wei, Schuurmans, et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023. — arXiv:2203.11171

## 10. Worked Example

Suppose a benchmark run on 500 problems gives coverage $\widehat C(1)=0.40$, $\widehat C(10)=0.72$, $\widehat C(100)=0.84$. The binomial standard error at $n=500$ is about $\pm 0.02$.

**Fit A — pure power law.** Anchor on $N=1$ and $N=100$: $1-C = 0.60\,N^{-b}$ with $0.60\cdot 100^{-b} = 0.16 \Rightarrow b = 0.287$. Predicted $C(10) = 1 - 0.60\cdot 10^{-0.287} = 0.69$, versus measured $0.72$ — a $1.5\sigma$ miss.

**Fit B — floor plus power law.** $1-C = \pi + A r^{\log_{10} N}$ fits all three points exactly: differences give $A(1-r)=0.32$, $Ar(1-r)=0.12$, so $r = 0.375$ ($b = 0.426$), $A = 0.512$, $\pi = 0.088$.

Both are defensible against 500-problem data. Their extrapolations are not:

| $N$ | Fit A (power law) | Fit B (floor $\pi=8.8\%$) |
|---|---|---|
| $10^2$ | 84.0% | 84.0% |
| $10^3$ | 89.0% | 88.4% |
| $10^4$ | **95.7%** | **90.2%** |
| $\infty$ | 100% | **91.2%** |

The two models differ by $0.6$ points at $N=10^3$ — undetectable at $n=500$, needing roughly $n \gtrsim 4000$ problems to separate at $2\sigma$ — and by $5.5$ points at $N=10^4$, where they imply opposite engineering conclusions: keep sampling, or stop and fix the model.

Now add the verifier. At $N=10^4$, $D_{\mathrm{KL}} = \log 10^4 - 0.9999 = 8.21$ nats, so $d = \sqrt{D_{\mathrm{KL}}} = 2.87$. Under a Gao-style proxy curve $R(d) = d(\alpha - \beta d)$, gold reward is already past its maximum $d^\star = \alpha/2\beta$ for any small reward model — so the realized $A_v(10^4)$ is *below* $A_v(10^3)$ even while $C(N)$ still climbs. And the 5.5-point coverage disagreement is of the same magnitude as the unmeasured grader false-positive rate at $N=10^4$.

The obstruction is now visible in one place: the quantity the law is supposed to predict is not identified by the data anyone collects, is truncated by a verifier whose plateau is set by a different parameter, and is measured by a grader whose error grows with the independent variable.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*