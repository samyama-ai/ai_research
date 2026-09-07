---
id: 10-scaling-laws/agentic-task-horizon-scaling
title: "Scaling Laws for Agentic Multi-Step Task Success"
topic: 10-scaling-laws
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Laws for Agentic Multi-Step Task Success

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/agentic-task-horizon-scaling` · **Status:** empirically-open

## 1. Problem Statement

Pretraining loss is predictable from compute to within a few percent across four orders of magnitude (Kaplan et al. 2020; Hoffmann et al. 2022). Agentic end-to-end task success is not. The problem: **find a law that predicts, from training compute, post-training compute, and inference compute, the length of multi-step task an agent completes at a fixed reliability level** — and say whether that law is a genuine scaling relation or an artifact of how task length is measured.

Three variants, of very different difficulty:

- **Measurement.** Define "task length" so that it is a property of the task, not of the agent or the scaffold, and so that the resulting horizon metric is stable under resampling the task suite. Currently open.
- **Method.** Given a fixed metric, fit $\text{horizon} = f(C_{\text{train}}, C_{\text{inf}})$ with held-out predictive error small enough to forecast one model generation ahead. Runnable now; unrun at the right scale.
- **Theory.** Derive the observed exponential-in-time trend from a per-step error model. Specifically: is horizon growth driven by rising per-step accuracy, by falling correlation between step failures, or by improved error recovery? These are not distinguishable from endpoint success rates alone.

A solution to the method variant is a fitted law with stated uncertainty that predicts the horizon of a model held out of the fit to within a factor of $1.5$.

## 2. Formal Setting

An agent $\pi_\theta$ interacts with an environment over an episode. A task $\tau$ is drawn from suite $\mathcal{D}$. Define:

- **Success** $S(\pi,\tau) \in \{0,1\}$: the suite's binary grader on the final state. Measured by running the episode $k$ times and reporting $\hat{p}(\tau) = \frac{1}{k}\sum_i S_i$; typically $k \in [4, 10]$, so the per-task standard error is $\ge 0.15$.
- **Task length** $T(\tau) \in \mathbb{R}_+$: measured as the *median wall-clock time taken by a qualified human baseliner*, in seconds. This is the METR convention (Kwa et al. 2025). Alternatives — reference-solution step count, minimum tool calls — give different orderings.
- **Compute** $C_{\text{train}}$ (FLOPs, from $6ND$ or reported), $C_{\text{inf}}$ (expected FLOPs per episode $= 2N \cdot \mathbb{E}[\text{tokens generated} + \text{tokens read}]$, including all rollouts under best-of-$n$ or tree search).

**Horizon.** Fit a logistic in log-length,
$$\Pr[S=1 \mid T] = \sigma\!\left(\alpha + \beta \log T\right),$$
and define the $q$-reliability horizon $H_q$ by $\Pr[S=1\mid T=H_q] = q$, i.e. $\log H_q = (\sigma^{-1}(q) - \alpha)/\beta$. The claimed scaling law is
$$\log H_{50}(t) = a t + b \quad \text{(calendar time)}, \qquad \text{or} \qquad \log H_{50} = a' \log C_{\text{train}} + b'.$$

**Step-level model.** Let the episode decompose into $n(\tau)$ steps with per-step failure indicators $Z_1,\dots,Z_n$. Under independent constant hazard $\Pr[Z_i=1]=1-p$,
$$\Pr[S=1] = p^{n}, \qquad H_q \propto \frac{\log q}{\log p},$$
so $\log H_{q_1}/\log H_{q_2}$ is a *fixed* ratio independent of $p$: $H_{50}/H_{80} = \log 0.5/\log 0.8 = 3.11$.

**Assumptions, and which are violated:**

| Assumption | Status |
|---|---|
| Step failures independent, constant hazard | **Violated.** Observed $H_{50}/H_{80} \approx 5$, not $3.11$ — failures are positively correlated. |
| Human time is agent-independent task difficulty | **Violated.** Agents are superhuman on lookup, subhuman on ambiguity; the map is not monotone. |
| Grader measures task completion | Partly violated. Reward hacking and under-specified graders inflate $S$ on long tasks. |
| Task suite is a stationary sample | **Violated.** Suites are curated post hoc; contamination grows with model recency. |
| Scaffold held fixed across models | Violated in practice; horizon moves with scaffold by a factor comparable to a model generation. |

## 3. State of the Art

**Empirical SOTA (established).** Kwa et al., *Measuring AI Ability to Complete Long Tasks* (METR, 2025, arXiv:2503.14499) fit $H_{50}$ across models from GPT-2 to Claude 3.7 Sonnet on a combined suite (HCAST, RE-Bench, and short software-atom tasks), and report an exponential trend with a **doubling time of about 7 months** over 2019–2025, with high fit quality on the log-horizon-versus-date regression. This is the only horizon-scaling result with an explicit reliability parameter and human-time calibration.

**Claimed but unablated.** (i) That the trend is *faster* in the 2024–2025 window (doubling time reported near 4 months in follow-up analysis) — this rests on few points and is sensitive to which models are included. (ii) That the trend is driven by capability rather than by scaffold and post-training-recipe improvements — no ablation holds the scaffold fixed across the model series. (iii) That $H_{50}$ extrapolates; the extrapolation is a straight-line fit, not a mechanistic law.

**Benchmark-number-only results.** SWE-bench Verified, OSWorld, WebArena, GAIA, and $\tau$-bench report single accuracies per model. They carry no length axis and cannot be converted into a horizon without re-baselining every task with human timings. Treat quoted "agentic performance improved from $X\%$ to $Y\%$" claims as points, not curves.

**Theory SOTA.** There is no derivation of horizon growth from a training-compute law. The nearest formal results are Ruan et al., *Observational Scaling Laws* (NeurIPS 2024), which shows downstream capabilities are low-dimensional functions of a few latent capability axes recoverable across model families, and Gao et al. (ICML 2023) on functional forms for overoptimization under KL budget. Neither addresses multi-step composition.

## 4. What Is Known

- **Horizon trend, measured.** METR (2025): GPT-4 (0314, March 2023) $H_{50} \approx 5$ minutes; Claude 3.7 Sonnet (Feb 2025) $H_{50} \approx 1$ hour as reported. Doubling time $\approx 7$ months, 2019–2025. Scale: $\sim$170 tasks with human baselines, $\le 8$ hours human time each.
- **Reliability costs a large constant factor.** $H_{80}$ is roughly $5\times$ shorter than $H_{50}$ in the same data — i.e., the same model is reliable over a task five times shorter than the one it can do half the time.
- **Repeated sampling buys coverage, not selection.** Brown et al. (2024, arXiv:2407.21787): DeepSeek-Coder-V2-Instruct on SWE-bench Lite goes from $15.9\%$ at one sample to $56\%$ coverage at 250 samples, exceeding the then-SOTA single-attempt agent — but only with an oracle verifier. Without one, realized pass@1 gains are far smaller. This makes $C_{\text{inf}}$ a real, but verifier-gated, axis.
- **Test-time compute trades against parameters.** Snell et al. (2024, arXiv:2408.03314) show compute-optimal test-time strategies can beat a $14\times$ larger model on some MATH slices — established for short, verifiable tasks only; not shown for long-horizon agentic tasks.
- **Metric choice manufactures shape.** Schaeffer et al. (NeurIPS 2023) show discontinuous "emergence" arises from thresholded metrics. Binary all-or-nothing episode success is exactly such a metric, so sharp horizon transitions are prior-suspect.

## 5. What Is Not Known

- **Methodologically blocked:** whether "task length" is a well-defined quantity. Human baseline time confounds task structure with human labor-market specialization; different valid length definitions reorder tasks and shift $H_{50}$ by an unquantified amount. No published sensitivity analysis of $H_{50}$ to the length definition exists.
- **Empirically open:** the compute law. Nobody has fit $H_q$ against $C_{\text{train}}$ *within a single model family, single scaffold, single tokenizer*, across $\ge 4$ scales. Every published horizon fit crosses families and dates, so compute, data, post-training recipe, and scaffold are collinear. The experiment is runnable by any lab with an existing model ladder.
- **Empirically open:** the exchange rate between $C_{\text{train}}$ and $C_{\text{inf}}$ for horizon. No isoquant $\{(C_{\text{train}}, C_{\text{inf}}) : H_{50} = h\}$ has been measured.
- **Theoretically open:** whether horizon growth is decomposable. Given only $\{S(\pi,\tau)\}$, the triple (per-step accuracy $p$, failure correlation $\rho$, recovery probability $r$) is **non-identifiable** — many $(p,\rho,r)$ produce the same success-versus-length curve.

## 6. Why It Is Hard

The binding obstruction is **non-identifiability from endpoint supervision**, compounded by **confounded measurement**.

Endpoint success is a scalar per episode. The generative model has at least three free parameters per model ($p$, $\rho$, $r$), and the observed curve $\Pr[S\mid T]$ is a two-parameter logistic. The system is under-determined: a model that doubled its horizon by improving per-step accuracy and one that doubled it by learning to notice and undo its own mistakes are indistinguishable in the data, yet extrapolate differently — the first is smooth in compute, the second may saturate at the point where errors become undetectable.

Second, the length axis is measured in *human* time, which is not an intrinsic property of the task. A task that takes a human 4 hours because it needs 400 mechanical edits and one that takes 4 hours because it needs one insight are the same point on the $x$-axis and behave completely differently under scaling.

Third, cost. Getting a horizon estimate with usable error bars needs $\sim$150 tasks $\times$ 8 rollouts $\times$ episodes of up to 8 human-hours, per model. At 4 model scales $\times$ 2 inference budgets that is thousands of long agent episodes plus the human baselining, which is the expensive half and cannot be amortized across changes to the suite.

## 7. Current Research (as of 2026)

- **METR** continues to extend the horizon series to new frontier releases and to refine the task suite; the live question is whether the 2024–2026 slope is a regime change or a small-sample artifact *(frontier — verify)*.
- **RL-for-agents groups at the major labs** (Anthropic, OpenAI, DeepMind, plus open efforts around agentic RL environments) are scaling outcome-supervised RL on long-horizon tool use. Whether post-training compute enters the horizon law as a separate term from pretraining compute is the open empirical question there *(frontier — verify)*.
- **Verifier scaling.** Work following Brown et al. on generative verifiers and process reward models targets the oracle-verifier gap that currently gates the $C_{\text{inf}}$ axis.
- **Length-metric reform.** Proposals to replace human time with reference-solution step counts or information-theoretic task descriptions circulate but nothing has displaced human baselining, because human time is the only unit that is comparable across environment types.

## 8. Concrete Next Experiment

**Question.** Does $H_{50}$ scale as a power law in $C_{\text{train}}$ within a controlled family, and is the exponent stable when inference compute is held fixed?

**Scale.** One model family, four pretraining scales spanning $\ge 100\times$ compute (e.g. 1B / 8B / 30B / 100B+ parameters at fixed tokens-per-parameter), identical tokenizer, identical data mix, identical post-training recipe. Task suite: 150 tasks with human baseline times spanning 1 second to 8 hours, log-uniformly binned. 8 rollouts per task per model. Two inference budgets: $C_{\text{inf}}$ and $8\times C_{\text{inf}}$ via best-of-$n$ with a *fixed, non-oracle* verifier. Cost estimate: $\sim$10k long episodes, plus reuse of an existing human-baselined suite.

**Control arm.** The same four models on the same tasks with the *scaffold frozen bit-for-bit* and, critically, a **step-count arm**: re-express every task's length as reference-solution tool-call count and refit $H_{50}$. The control tells you how much of the exponent is the model and how much is the length metric.

**Deciding number.** The **ratio of the fitted exponents $a'$ under the human-time length axis and the step-count length axis**, where $\log H_{50} = a' \log C_{\text{train}} + b'$. If that ratio lies in $[0.9, 1.1]$, horizon scaling is a property of the models and the law is real. If it falls outside $[0.7, 1.4]$, the published trend is substantially a property of the length metric, and the measurement variant must be solved first. Secondary readout: held-out prediction of the largest model's $H_{50}$ from a fit on the smaller three — a factor-$1.5$ error is the success bar.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Kwa, West, Becker, et al. (METR). *Measuring AI Ability to Complete Long Tasks.* 2025. — arXiv:2503.14499
- **[SOTA]** Brown, Juravsky, Ehrlich, Clark, Le, Ré, Mirhoseini. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[Method]** Ruan, Maddison, Hashimoto. *Observational Scaling Laws and the Predictability of Language Model Performance.* NeurIPS, 2024. — arXiv:2405.10938
- **[Method]** Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[Critique]** Schaeffer, Miranda, Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS, 2023. — arXiv:2304.15004
- **[Benchmark]** Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[Benchmark]** Wijk, Lin, Becker, et al. (METR). *RE-Bench: Evaluating Frontier AI R&D Capabilities of Language Model Agents against Human Experts.* 2024. — arXiv:2411.15114
- **[Benchmark]** Mialon, Fourrier, Swift, Wolf, LeCun, Scialom. *GAIA: A Benchmark for General AI Assistants.* ICLR, 2024. — arXiv:2311.12983
- **[Benchmark]** Xie, Zhang, Zhou, et al. *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments.* NeurIPS, 2024. — arXiv:2404.07972

## 10. Worked Example

Take the constant-hazard model at face value and check it against the published reliability ratio.

Suppose an agent has per-step success $p$ and independent step failures, and a task of human-length $T$ needs $n = cT$ steps. Then $\Pr[S=1\mid T] = p^{cT}$ and
$$H_q = \frac{\log q}{c \log p} \;\Rightarrow\; \frac{H_{50}}{H_{80}} = \frac{\log 0.5}{\log 0.8} = \frac{0.6931}{0.2231} = 3.11,$$
independent of $p$ and $c$. This is a parameter-free prediction.

Measured: METR reports $H_{80}$ about $5\times$ shorter than $H_{50}$. Take $H_{50} = 60$ min, so the model predicts $H_{80} = 60/3.11 = 19.3$ min, and the data say $\approx 12$ min. The shortfall is $38\%$ — long tasks fail more often than independent per-step hazard allows.

Now try to fix it. Two repairs both work:

1. **Correlated failures.** Let the hazard be drawn once per episode from a mixture: $p = 0.999$ with probability $0.75$ (agent is "on-track") and $p = 0.98$ otherwise. Then $\Pr[S\mid T] = 0.75\cdot 0.999^{cT} + 0.25\cdot 0.98^{cT}$, a heavy-tailed curve whose $H_{50}/H_{80}$ ratio exceeds 3.11 for suitable $c$.
2. **Length-metric compression.** Keep independence but let $n = cT^{\gamma}$ with $\gamma = 1.4$ — long human tasks contain disproportionately many agent steps, because human time undercounts routine mechanical work the human batches mentally. Then $H_{50}/H_{80} = 3.11^{1/\gamma}$... which moves the ratio the *wrong* way, so set $\gamma = 0.75$: the ratio becomes $3.11^{1/0.75} = 4.6$, matching the data with no correlation at all.

Both fits reproduce the observed curve. One says the model has a correlated-failure problem that better training could remove; the other says the horizon axis is mis-scaled and there is no correlation to fix. Endpoint success rates cannot separate them — that is the non-identifiability of Section 6, and it is why the step-count control arm in Section 8, not another frontier model, is what moves this problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*