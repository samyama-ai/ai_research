---
id: 17-reasoning/best-of-n-reward-hacking-onset-scaling-law
title: "Scaling Law for Best-of-N Reward Hacking Onset"
topic: 17-reasoning
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Law for Best-of-N Reward Hacking Onset

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/best-of-n-reward-hacking-onset-scaling-law` · **Status:** empirically-open

## 1. Problem Statement

Best-of-$N$ (BoN) sampling draws $n$ candidate responses from a policy and returns the one scoring highest under a proxy verifier — a reward model, a process reward model, or a learned answer checker. True quality rises with $n$ up to a point, then falls: the argmax starts selecting responses that exploit the proxy rather than solve the task. Call the turning point $n^\star$.

The problem: **predict $n^\star$ from cheaply measurable quantities before spending the compute.**

Three variants, of different difficulty:

- **Measurement.** Given a policy $\pi$, a proxy reward $\hat r$, a gold reward $r$, and a task distribution, estimate $n^\star$ and the gold score at the peak, with confidence intervals. Currently doable but expensive; the difficulty is that gold reward is itself a model in every published study at scale.
- **Method.** Predict $n^\star$ for a held-out $(\pi, \hat r, \text{task})$ triple from a fit involving proxy-model size, proxy-training-data size, policy size, and observable statistics of the score distribution — without ever evaluating the gold reward at large $n$.
- **Theory.** Derive the functional form of gold score versus $n$ from a generative model of proxy error, and prove when $n^\star$ is finite versus infinite.

Solving it means: a formula $\hat n^\star(\cdot)$ whose predictions on unseen triples fall inside the measured confidence interval of $n^\star$, across at least one order of magnitude of proxy-model scale that was not used to fit it.

## 2. Formal Setting

Prompt $x \sim \mathcal{D}$. Policy $\pi(\cdot \mid x)$ over responses $y$. Draw $y_1,\dots,y_n \stackrel{iid}{\sim} \pi(\cdot\mid x)$ and select

$$y_{\mathrm{BoN}}(x,n) = \arg\max_{i \le n} \hat r(x, y_i).$$

Two rewards: proxy $\hat r$ (the deployed verifier) and gold $r$ (ground truth — a held-out human panel, an executable test suite, or a formal checker). Define

$$G(n) = \mathbb{E}_{x}\,\mathbb{E}\big[r(x, y_{\mathrm{BoN}}(x,n))\big], \qquad n^\star = \arg\max_{n \ge 1} G(n).$$

**Compute axis.** BoN induces a KL from the base policy that is closed-form and independent of the task:

$$\mathrm{KL}\!\left(\pi_{\mathrm{BoN}}^{(n)} \,\|\, \pi\right) = \log n - \frac{n-1}{n},$$

exact when ties have measure zero (Beirami et al., 2024; used as the $x$-axis by Gao et al., 2023). Write $d = \sqrt{\mathrm{KL}}$. Gao et al. fit

$$G(d) = d\,(\alpha_{\mathrm{bon}} - \beta_{\mathrm{bon}} d),$$

which peaks at $d^\star = \alpha_{\mathrm{bon}} / (2\beta_{\mathrm{bon}})$ and hence, using $\mathrm{KL} \approx \log n$ for $n \gg 1$,

$$n^\star \approx \exp\!\left(\frac{\alpha_{\mathrm{bon}}^2}{4\beta_{\mathrm{bon}}^2}\right).$$

**How each quantity is measured.**
- $G(n)$: sample $m$ prompts, $N_{\max}$ responses each, score all with $\hat r$ once, then compute BoN curves for every $n \le N_{\max}$ by bootstrap subsampling without re-generating. Gold scores are needed only for responses that win some subsample — in practice $O(m \log N_{\max})$ gold evaluations.
- $\alpha, \beta$: nonlinear least squares on $(d, G)$ pairs, weighted by the per-$n$ standard error.
- Proxy scale: parameter count $N_{\hat r}$ and preference-pair count $D_{\hat r}$.
- Verifier quality: pairwise accuracy of $\hat r$ against gold on the policy's own samples — not on a static preference benchmark.

**Assumptions, and which fail.**
1. *Gold reward is ground truth.* Violated: at scale the "gold" model is a larger reward model (Gao et al.) and is itself hackable. Only code and formal math give a non-model gold.
2. *Samples are i.i.d. and distinct.* Violated at large $n$ on reasoning tasks — duplicate answers dominate, and the KL formula overstates the true divergence.
3. *The quadratic-in-$d$ form.* Purely empirical, fit over $\mathrm{KL} \le 10$ nats; no derivation, and extrapolation to $n > 10^5$ is unsupported.
4. *A single scalar $r$.* Violated wherever gold quality is multi-attribute and hacking moves one attribute at another's expense.

## 3. State of the Art

**Established.**
- Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization* (ICML 2023). Proxy RMs from 3M to 3B parameters, a 6B gold RM, policies 1.2B–6B. The BoN form $d(\alpha - \beta d)$ fits well; $\beta_{\mathrm{bon}}$ shrinks systematically as proxy RM parameters and preference-data volume grow, i.e. $n^\star$ increases with proxy scale. Policy size shifted the whole curve up but changed the overoptimization onset in KL only weakly. This is the single reference result for the whole problem.
- Beirami et al. (2024): the $\log n - (n-1)/n$ expression, previously used as an approximation, is the exact KL under distinct outputs, and is an upper bound otherwise.

**Claimed but unablated, or benchmark-only.**
- Snell et al. (2024) report compute-optimal test-time scaling on MATH with a PRM verifier; the reported gains are benchmark numbers on one model family, and the onset of degradation is not fit as a law.
- Coste et al. (ICLR 2024) and Eisenstein et al. (COLM 2024) show reward-model ensembles push the peak later but do not remove it. Eisenstein et al. are explicit that ensembles *mitigate but do not eliminate* hacking; neither paper produces a predictive $n^\star$.
- Rafailov et al. (2024) fit overoptimization laws for direct alignment algorithms with the same functional family. Cross-family transfer of the coefficients is untested.
- Stroebl, Kapoor & Narayanan (2024) argue imperfect verifiers cap inference scaling; the argument is analytic plus small-scale demonstration, not a fitted onset law.

No published work predicts $n^\star$ out-of-sample.

## 4. What Is Known

- BoN and RLHF overoptimize on different curves: BoN fits $d(\alpha - \beta d)$, RLHF fits $d(\alpha - \beta \log d)$; at matched KL, BoN overoptimizes *less* (Gao et al., 2023, proxy RMs 3M–3B, gold 6B).
- Overoptimization coefficients depend on proxy RM size and data size smoothly and roughly log-linearly over the 3M–3B range measured; the RM-data effect saturates above roughly $10^4$–$10^5$ comparisons for small RMs (Gao et al.).
- Coverage (pass@$n$, an oracle selector) keeps rising as an approximate power law in $n$ out to $n = 10^4$ on GSM8K and MATH for Llama-3-8B-class models, while every automatic selector tested — majority vote, reward-model BoN — plateaus one to two orders of magnitude earlier (Brown et al., *Large Language Monkeys*, 2024). The gap between coverage and selected accuracy is the quantity $n^\star$ governs.
- Process reward models select better than outcome reward models at fixed $n$ on MATH (Lightman et al., *Let's Verify Step by Step*, 2023); whether they have a later $n^\star$ is not reported.
- Ensembling $K$ reward models raises the peak but leaves a peak, because members share pretraining-induced error directions (Eisenstein et al., 2024).

## 5. What Is Not Known

**Empirically open** (the dominant class here).
- Whether $n^\star$ follows a clean law in $(N_{\hat r}, D_{\hat r}, N_\pi)$ that extrapolates. Gao et al. stopped at 3B proxy RMs and pre-2023 policies; nobody has repeated the sweep with modern reasoning models, PRMs, or $n$ beyond $\sim 10^4$.
- Whether $n^\star$ on verifiable domains (code with tests, Lean) behaves like the model-gold case. Runnable today; unrun at scale.
- Whether $n^\star$ is task-conditional — i.e. whether the aggregate peak hides a mixture of easy prompts with $n^\star = \infty$ and hard prompts with $n^\star \approx 10$.

**Theoretically open.**
- No derivation of $d(\alpha - \beta d)$ from a model of proxy error. The natural candidate — extreme-value theory on $\hat r = r + \varepsilon$ with heavy-tailed $\varepsilon$ — predicts a family of shapes, and which member applies is unproven.
- Conditions on the joint law of $(r, \hat r)$ under $\pi$ for $n^\star < \infty$. Sufficient conditions are easy (correlated error with unbounded $\varepsilon$); necessary-and-sufficient ones are not established.

**Methodologically blocked.**
- "Gold reward" at frontier scale. With a model gold, measured $n^\star$ is a property of the *gold model's* blind spots as much as the proxy's, and there is no accepted way to separate them.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus absent ground truth**, not compute.

Measuring $G(n)$ out to $n = 10^4$ over $m = 500$ prompts is $5\times 10^6$ generations — a few thousand GPU-hours, affordable. The problem is what $G$ means. Every large-scale estimate uses a bigger reward model as gold, so the curve measures *proxy–gold disagreement*, which is bounded by the gold's own error. Two failure modes are indistinguishable from the data: (a) the proxy is being hacked, and (b) the proxy has found genuinely good responses the gold undervalues. Substituting a non-model gold (unit tests, a proof checker) fixes ground truth but changes the regime — verifiable domains are exactly where $\hat r$ can be near-perfect and $n^\star$ may be infinite, so the fitted law does not transfer back.

Second obstruction: **non-identifiability of the fit**. Near the peak $G$ is flat by construction ($G'(n^\star) = 0$), and $d = \sqrt{\log n}$ compresses a decade of $n$ into a small interval. The standard error on $\alpha/2\beta$ maps to a multiplicative interval on $n^\star = \exp(d^{\star2})$ that is routinely an order of magnitude wide.

## 7. Current Research (as of 2026)

- **Verifier-aware inference scaling.** Work following Snell et al. and Brown et al. on when to spend compute on more samples versus longer chains-of-thought. Groups at Google DeepMind, Stanford and Princeton. Predicting the selector plateau, not just observing it, is the open piece. *(frontier — verify)*
- **Reward-model robustness.** Ensembles, uncertainty-penalized selection, and reward-model debate as ways to push $n^\star$ out (Google DeepMind, Anthropic, academic RLHF groups).
- **RL-trained verifiers and self-verification** in reasoning models, where the verifier is a mode of the same policy — this makes the error correlation between $\pi$ and $\hat r$ maximal and should shrink $n^\star$ sharply. Largely unmeasured. *(frontier — verify)*
- **Theory of BoN alignment**: exact KL, win-rate bounds, and BoN-as-regularized-policy equivalences.

## 8. Concrete Next Experiment

**Question.** Does $\log n^\star$ scale linearly in $\log N_{\hat r}$, and does a fit on small proxies predict the peak for a large one?

**Scale.** One policy family, 8B, fixed. Proxy reward models at four sizes — 0.5B, 1.5B, 7B, 32B — all trained on the same $2\times10^5$ preference pairs. Task: 500 competition-math problems and 500 code problems with hidden test suites. Generate $N_{\max} = 10^4$ samples per prompt at temperature 1.0. Gold = executable tests (code) and a symbolic answer checker (math); no gold model anywhere. Score all samples with all four proxies once; recover $G(n)$ for all $n \le 10^4$ by bootstrap subsampling, 200 resamples per $n$ on a log grid.

**Control arm.** (i) Oracle selector — pass@$n$ coverage, upper bound; (ii) random selector — the no-verifier floor; (iii) a *label-shuffled* proxy trained on 10% corrupted preferences, which must show a markedly earlier $n^\star$ if the measurement is sensitive at all.

**Decision number.** Fit $\alpha, \beta$ on the 0.5B, 1.5B and 7B proxies only, extrapolate to 32B, then compare to the measured $n^\star_{32\text{B}}$. **The experiment succeeds if the predicted and measured $\log_{10} n^\star$ agree within 0.3 (a factor of 2) with non-overlapping-free bootstrap intervals.** Disagreement beyond a factor of 5 falsifies the single-exponent law and sends the problem back to the functional-form question in §5.

Cost estimate: $10^7$ generations at ~500 output tokens ≈ 5B tokens, plus $4\times10^7$ reward forward passes. Order $10^4$ A100-hours — one week on 64 GPUs.

## 9. Key References

- **[Foundational]** Leo Gao, John Schulman, Jacob Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[Foundational]** Karl Cobbe et al. *Training Verifiers to Solve Math Word Problems.* 2021. — arXiv:2110.14168
- **[Foundational]** Nisan Stiennon et al. *Learning to Summarize with Human Feedback.* NeurIPS, 2020. — arXiv:2009.01325
- **[SOTA]** Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[SOTA]** Hunter Lightman et al. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050
- **[Theory]** Ahmad Beirami, Alekh Agarwal, Jonathan Berant, Alexander D'Amour, Jacob Eisenstein, Chirag Nagpal, Ananda Theertha Suresh. *Theoretical Guarantees on the Best-of-n Alignment Policy.* 2024. — arXiv:2401.01879
- **[Theory]** Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, David Krueger. *Defining and Characterizing Reward Hacking.* NeurIPS, 2022. — arXiv:2209.13085
- **[Mitigation]** Thomas Coste, Usman Anwar, Robert Kirk, David Krueger. *Reward Model Ensembles Help Mitigate Overoptimization.* ICLR, 2024. — arXiv:2310.02743
- **[Mitigation]** Jacob Eisenstein et al. *Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking.* COLM, 2024. — arXiv:2312.09244
- **[Related]** Rafael Rafailov, Yaswanth Chittepu, Ryan Park, Harshit Sikchi et al. *Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms.* NeurIPS, 2024. — arXiv:2406.02900
- **[Critique]** Benedikt Stroebl, Sayash Kapoor, Arvind Narayanan. *Inference Scaling Flaws: The Limits of LLM Resampling with Imperfect Verifiers.* 2024. — arXiv:2411.17501

## 10. Worked Example

Take a proxy RM whose fitted coefficients give $\alpha_{\mathrm{bon}} = 1.0$, $\beta_{\mathrm{bon}} = 0.14$ (units: gold reward per $\sqrt{\text{nat}}$; the ratio is in the range Gao et al. report for mid-sized proxies).

Peak: $d^\star = \alpha/(2\beta) = 3.57$, so $\mathrm{KL}^\star = 12.8$ nats, and from $\log n - (n-1)/n = 12.8$,

$$n^\star \approx e^{13.8} \approx 9.8\times10^5.$$

Now the obstruction. Suppose the fit's standard errors are a modest 5% on each coefficient, uncorrelated. Then $d^\star$ has ~7% relative error, so $d^\star \in [3.32, 3.82]$ at one sigma. Squaring:

$$\mathrm{KL}^\star \in [11.0,\ 14.6] \ \Rightarrow\ n^\star \in [1.8\times10^5,\ 5.9\times10^6].$$

A 5% error on the fit becomes a **33× interval on $n^\star$**. To decide whether to deploy at $n = 10^3$ or $n = 10^5$, this is useless — and 5% is optimistic, because the data near the peak are flat and the gold evaluations are noisy.

Worse, check what the curve says about the cost of being wrong. At $d = 3.57$, $G = 1.79$; at $d = 5.0$ ($n \approx e^{26} \approx 2\times10^{11}$, unreachable) $G = 1.5$; but at $d = 2.0$ ($n \approx 150$) $G = 1.44$. So $G$ over the whole decade $n \in [10^3, 10^6]$ varies by under 10% of its peak value. **The quantity being optimized is nearly flat exactly where the answer must be read off.** That is the identifiability problem in one line: $n^\star$ is defined by a stationary point of a function whose curvature there, $G''(d^\star) = -2\beta = -0.28$, is small relative to the measurement noise on $G$ — which, with 500 prompts and a binary gold, is roughly $\sqrt{0.25/500} \approx 0.022$ per point before any gold-model bias is added.

The practical consequence: pinning $n^\star$ to a factor of 2 needs either far more prompts (noise scales as $m^{-1/2}$, so ~50× more prompts for a 7× tighter interval) or a different estimand — for instance the first $n$ at which $G$ falls 1% below its running maximum, which is a *threshold* rather than an argmax and is far better conditioned. Choosing that estimand is itself an open piece of the problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*