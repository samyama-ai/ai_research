---
id: 05-retrieval-and-agents/non-stationary-drift-agent-benchmarks
title: "Non-Stationary Environment Drift in Agent Benchmarks"
topic: 05-retrieval-and-agents
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Non-Stationary Environment Drift in Agent Benchmarks

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/non-stationary-drift-agent-benchmarks` · **Status:** methodologically-blocked

## 1. Problem Statement

An agent benchmark scores a policy $\pi$ against an environment: a website, an API, a package index, a search engine, a simulated colleague. The environment changes between the day the benchmark was built and the day a score is reported. A reported delta between two systems evaluated months apart therefore mixes two causes: the policy got better, and the environment got easier or harder.

Three variants, with different difficulty:

- **Measurement.** Given two scores $S_1, S_2$ collected at times $t_1 < t_2$, estimate how much of $S_2 - S_1$ is attributable to the policy. This is the blocked variant: there is no agreed estimator and, for a frozen historical environment, no way to collect the missing counterfactual.
- **Method.** Build environments whose drift is bounded or measured — pinned container images, recorded network traces, rolling task windows. Partly solved, at real cost in realism.
- **Theory.** Under what conditions on the drift process is a ranking over policies identifiable from scores taken at different times? Open, and connected to non-stationary bandit theory.

Solving it means: a benchmark reports a policy effect with a drift-adjusted confidence interval, and re-running the same agent artifact one year later lands inside that interval.

## 2. Formal Setting

Let $E_t$ be the environment state at wall-clock time $t$: the tuple of external services, page content, tool schemas, dataset contents and grader code reachable by the agent. Let $\mathcal{T} = \{\tau_i\}_{i=1}^n$ be the task suite, $\pi$ a policy (model weights + scaffold + prompt + tool set, all pinned), and $r(\pi, \tau, E_t) \in \{0,1\}$ the grader's verdict on one rollout.

**Measured score.** With $m$ rollouts per task,
$$\hat{S}(\pi, E_t) = \frac{1}{nm}\sum_{i=1}^{n}\sum_{j=1}^{m} r(\pi, \tau_i, E_t^{(j)}),$$
where $E_t^{(j)}$ is the environment *as it actually was* during rollout $j$ — not a fixed object, since the two rollouts may hit different CDN nodes, different rate-limit states, or a page edited in between.

**Drift.** Define the environment-induced score change for a fixed anchor policy $\pi_0$:
$$\delta(t_1, t_2) = S(\pi_0, E_{t_2}) - S(\pi_0, E_{t_1}).$$
The naive policy effect is $\Delta = \hat{S}(\pi_2, E_{t_2}) - \hat{S}(\pi_1, E_{t_1})$; the anchor-adjusted effect is $\Delta_{\text{adj}} = \Delta - \hat{\delta}$.

**Variation budget.** Borrowing the non-stationary bandit formulation of Besbes, Gur and Zeevi (NeurIPS 2014), let
$$V_T = \sum_{t=1}^{T-1} \sup_{\pi \in \Pi} \left| S(\pi, E_{t+1}) - S(\pi, E_t) \right|.$$
Measuring $V_T$ requires evaluating a supremum over policies at every time step; in practice one substitutes a single anchor and reports a lower bound.

**Assumptions, and which are violated.**

1. *Additive separability*: $S(\pi, E_t) = f(\pi) + g(E_t)$, so $\Delta_{\text{adj}}$ is unbiased. **Violated.** Drift is task-local: a login flow that breaks zeroes a task for every policy (interaction $\approx 0$), but a search engine that starts returning a direct answer helps weak policies far more than strong ones.
2. *Grader stationarity*: $r$ depends on $E_t$ only through the agent's trajectory. **Violated** when the grader itself calls a live service or an LLM judge whose weights are updated behind an API.
3. *Task independence*: rollouts do not modify $E_t$. **Violated** in write-capable environments (a GitLab or CRM instance) unless state is reset per episode; reset scripts routinely leak state.
4. *No contamination path*: $\pi_2$'s training data is independent of $\mathcal{T}$. **Violated by construction** for public benchmarks after release.

## 3. State of the Art

**Established.**
- *Pinned, self-hosted environments.* WebArena (Zhou et al., ICLR 2024) ships Docker images of GitLab, Reddit-clone, an e-commerce site and a map service, with per-episode reset. This bounds drift in the sites but not in the model or scaffold dependencies.
- *Rolling time windows.* LiveCodeBench (Jain et al., ICLR 2025) collects problems continuously from LeetCode/AtCoder/Codeforces and reports scores restricted to problems published after a model's cutoff. This converts contamination drift into a measurable covariate rather than removing it.
- *Time-indexed QA.* StreamingQA (Liska et al., ICML 2022), RealTime QA (Kasai et al., NeurIPS 2023 Datasets) and FreshQA (Vu et al., 2023) make the answer key itself a function of $t$, so drift is explicit.
- *Cost-controlled reporting.* Kapoor et al., "AI Agents That Matter" (2024), showed agent leaderboards conflate accuracy with retry budget; their fix — report accuracy-vs-cost curves — is now partly adopted.

**Claimed but unablated.**
- That container pinning yields reproducible agent scores. No published study re-runs an identical pinned agent artifact against a pinned environment at a 6- or 12-month gap and reports the score difference. This is the missing control experiment for the whole field.
- That "held-out after cutoff" removes contamination. It removes verbatim leakage; it does not remove drift in problem difficulty or in the composition of contributed problems.

**Benchmark-number-only results.** Nearly all agent leaderboard entries (WebArena, GAIA, SWE-bench, τ-bench, TheAgentCompany) are single-timepoint numbers, with no anchor re-run and, until recently, no confidence intervals.

## 4. What Is Known

- **Rollout variance alone is large.** τ-bench (Yao et al., 2024) reports pass^1 far above pass^8 on the same tasks — e.g. GPT-4o in the retail domain drops from roughly 61% at $k{=}1$ to under 25% at $k{=}8$ over 115 tasks. Any drift signal below this envelope is unrecoverable without many rollouts.
- **Confidence intervals are usually absent and usually wide.** Miller, "Adding Error Bars to Evals" (2024), shows that on $n \approx 250$–$1000$ item benchmarks, standard errors of 1–2 points are typical, so 3-point leaderboard gaps are frequently non-significant.
- **Grader/task defects are common at benchmark scale.** SWE-bench: OpenAI's *Verified* subset retained 500 of 2,294 test instances after human screening (2024), i.e. ~78% of instances were dropped or flagged. Aleithan et al. (2024) report that a large share of "solved" SWE-bench instances had the fix leaked in the issue text or comments. Zhu et al., "Establishing Best Practices for Building Rigorous Agentic Benchmarks" (2025), find grader and task-validity defects across many popular agentic suites.
- **Temporal generalization decays measurably.** Lazaridou et al. (NeurIPS 2021) show LM perplexity degrades monotonically with the gap between training cutoff and test date on news corpora at up to ~1B parameters.
- **Headroom is huge, so drift is proportionally small — for now.** WebArena's original report: GPT-4 agent 14.41% vs. human 78.24% over 812 tasks. A 2-point drift is noise against a 60-point gap; it is decisive against the 1–3 point gaps at the top of a saturating leaderboard.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no accepted estimator for $\delta$, and no accepted definition of what "the same environment" means once a live dependency has changed. Without additive separability, $\Delta_{\text{adj}}$ has unquantified bias, and no benchmark reports the interaction term. The measurement is not well defined, not merely uncollected.
- **Empirically open.** Nobody has published the anchor re-run: one frozen agent artifact, one pinned suite, scored at $t$ and $t+12$ months, with $\hat{\delta}$ and its CI. It is cheap and runnable today.
- **Theoretically open.** No result gives conditions on $V_T$, $n$, $m$ under which the induced ranking over $\Pi$ is recoverable from asynchronous evaluations. The bandit analogue — $\tilde{\Theta}(V_T^{1/3}T^{2/3})$ regret under a variation budget (Besbes et al., 2014) — governs online decisions, not offline leaderboard identifiability.

## 6. Why It Is Hard

**Non-identifiability under an absent counterfactual.** Scoring $\pi_2$ against $E_{t_1}$ is impossible once $t_1$ has passed and the environment was live. The policy effect and the drift effect enter the observed score through the same channel, with only one observation per (policy, time) cell. The anchor design adds a second cell but still cannot estimate the policy×environment interaction, which needs $\pi_2$ evaluated at both times — the very thing that is unavailable.

Two aggravators, not the obstruction itself: rollout variance at the τ-bench scale means $\hat{\delta}$ needs hundreds of rollouts per timepoint to resolve 2 points; and drift is heavy-tailed and task-local — one broken selector can move a suite score by several points, so $\hat{\delta}$ is dominated by a handful of tasks and its variance is badly estimated by a binomial model.

## 7. Current Research (as of 2026)

- **Living/rolling benchmarks.** LiveCodeBench (Berkeley/CMU) and its descendants; SWE-bench-family refreshes drawn from post-cutoff commits.
- **Environment infrastructure.** BrowserGym/AgentLab (ServiceNow Research; de Chezelles et al., 2024) standardizes web-agent environments and seeds, making anchor re-runs technically cheap.
- **Benchmark hygiene.** Zhu, Kang et al. (UIUC) on agentic-benchmark validity checklists; Princeton's AI-agent-evaluation line (Kapoor, Narayanan) on cost control and reproducibility.
- **Statistical reporting.** Anthropic and others pushing per-eval standard errors and paired designs.
- *(frontier — verify)* Groups reportedly maintaining internal "canary" agent artifacts re-run on a fixed schedule to detect harness and API drift; results are largely unpublished.

## 8. Concrete Next Experiment

**The anchor re-run.**

- **Scale.** WebArena (812 tasks) plus τ-bench retail (115 tasks). One frozen agent artifact $\pi_0$: pinned open-weights model served locally (so the API cannot drift), pinned scaffold commit, pinned container digests. $m = 5$ rollouts per task, fixed seeds. Approx. 4,600 rollouts per timepoint; roughly one GPU-week plus environment hosting.
- **Timepoints.** $t_1$ = now, $t_2 = t_1 + 12$ months, identical hardware image.
- **Control arm.** The same artifact re-run at $t_1 + 1$ **day** against the same pinned images. This isolates pure stochastic + harness variance $\sigma_0$ from twelve-month drift.
- **Deciding number.** $\hat{\delta}(t_1,t_2)$ in percentage points, with a paired bootstrap CI over tasks. If $|\hat{\delta}| < 1$ pt and the CI excludes 2 pts, pinned environments are adequate and leaderboard deltas above 2 pts are safe. If $|\hat{\delta}| \geq 2$ pts — plausible, given how much a few broken tasks can move a suite — then every historical cross-time comparison on that suite is unresolved at leaderboard granularity, and anchor re-runs must become mandatory reporting.
- **Secondary output.** The per-task drift vector; its top-5 concentration ratio tells you whether drift is diffuse (poolable, binomial CIs fine) or task-local (needs robust CIs).

## 9. Key References

- **[Foundational]** Omar Besbes, Yonatan Gur, Assaf Zeevi. *Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards.* NeurIPS, 2014.
- **[Foundational]** Angeliki Lazaridou et al. *Mind the Gap: Assessing Temporal Generalization in Neural Language Models.* NeurIPS, 2021. — arXiv:2102.01951
- **[Foundational]** Adam Liska et al. *StreamingQA: A Benchmark for Adaptation to New Knowledge over Time in Question Answering Models.* ICML, 2022. — arXiv:2205.11388
- **[SOTA]** Shuyan Zhou et al. *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR, 2024. — arXiv:2307.13854
- **[SOTA]** Naman Jain et al. *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* ICLR, 2025. — arXiv:2403.07974
- **[SOTA]** Shunyu Yao et al. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024. — arXiv:2406.12045
- **[SOTA]** Carlos de Chezelles et al. *The BrowserGym Ecosystem for Web Agent Research.* 2024. — arXiv:2412.05467
- **[Methods]** Sayash Kapoor et al. *AI Agents That Matter.* 2024. — arXiv:2407.01502
- **[Methods]** Evan Miller. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* 2024. — arXiv:2411.00640
- **[Methods]** Carlos E. Jimenez et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[Survey]** Yuxuan Zhu et al. *Establishing Best Practices for Building Rigorous Agentic Benchmarks.* 2025.
- **[Context]** Jinsook Lee / Reza Aleithan et al. *SWE-Bench+: Enhanced Coding Benchmark for LLMs.* 2024. (verify author list before citing)

## 10. Worked Example

A vendor reports on a 812-task web suite: baseline agent $\pi_1$ at $t_1$ scores **14.4%**; new agent $\pi_2$ at $t_2 = t_1 + 9$ months scores **41.0%**. Naive $\Delta = 26.6$ pts.

The team does the right thing and re-runs the anchor $\pi_0 = \pi_1$ at $t_2$: it now scores **11.9%**. So $\hat{\delta} = -2.5$ pts and $\Delta_{\text{adj}} = 29.1$ pts. Comfortable — the drift is small next to the gain.

Now look at where the 2.5 points went. Per-task diffing shows 20 tasks flipped $1 \to 0$ and 0 flipped $0 \to 1$: $20/812 = 2.46$ pts. All 20 are on the shopping site, whose product-listing template gained a lazy-loading grid; the anchor's selector-based extraction fails on it. Of those 20, **18 were tasks $\pi_1$ solved at $t_1$** — i.e. drift is concentrated entirely in the solved set of the weak policy.

That is where the estimator breaks. $\hat{\delta}$ is computed only on tasks the anchor could solve. $\pi_2$ solves 333 tasks; the 20 damaged tasks are a 2.5% sample of the anchor's competence but an unknown fraction of $\pi_2$'s. If $\pi_2$ uses vision-based grounding, lazy loading may cost it nothing, and the true correction is $0$, not $+2.5$. If $\pi_2$ shares the DOM scaffold, the correction could be $20\times$ larger in absolute task count because $\pi_2$ attempts more of the shopping suite. Both stories fit the observed $\hat{\delta} = -2.5$.

Note also the sign trap: $\Delta_{\text{adj}} > \Delta$ says the environment got harder and the reported gain is an *under*-statement. That conclusion rests entirely on additive separability — assumption 1 of §2 — which the per-task diff has just falsified.

Recovering the answer needs $S(\pi_2, E_{t_1})$. $E_{t_1}$ is a container image nobody kept, fronted by a live payment sandbox that was decommissioned. The measurement is not expensive; it is unavailable. That is the obstruction.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*