---
id: 05-retrieval-and-agents/agentic-capability-independent-of-scaffold
title: "Measuring Agentic Capability Independent of Scaffold"
topic: 05-retrieval-and-agents
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Measuring Agentic Capability Independent of Scaffold

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/agentic-capability-independent-of-scaffold` · **Status:** methodologically-blocked

## 1. Problem Statement

An "agent" score is always a score for a *pair*: a model $m$ and a scaffold $s$ (the harness — prompt template, tool set, retry policy, context manager, verifier, step budget). Reported numbers name only the model. The problem is to recover a model-attributable quantity from pair-level observations.

- **Measurement variant (the live one).** Given a benchmark $\mathcal{T}$ and observations of $\mathrm{score}(m, s, \tau)$, define and estimate a scalar $C(m)$ such that ranking by $C$ predicts which model wins under a *held-out* scaffold not used in fitting. Solved = out-of-scaffold rank correlation stays high; today it is not even defined which scaffold family the expectation is over.
- **Method variant.** Build a *scaffold-neutral* protocol: a fixed minimal harness, or a per-model best-of-$K$ scaffold search with an equalized search budget, that makes cross-model comparison fair.
- **Theory variant.** Prove (or refute) identifiability: under what conditions do pair-level scores decompose as a model term plus a scaffold term plus bounded interaction, so a model term is recoverable at all?

Non-goal: making agents better. The claim under test is only that "GPT-X is a better agent than Claude-Y" is a well-formed statement.

## 2. Formal Setting

Let $m \in \mathcal{M}$ be a model (weights + decoding), $s \in \mathcal{S}$ a scaffold, $\tau \sim \mathcal{D}$ a task with a binary verifier $v_\tau \in \{0,1\}$ (SWE-bench: the hidden `FAIL_TO_PASS` test suite). A rollout consumes resources $R = (\text{tokens}, \text{tool calls}, \text{wall clock}, \$)$.

Measured quantity, per pair, at budget $B$:
$$\pi(m,s;B) \;=\; \mathbb{E}_{\tau\sim\mathcal{D}}\,\mathbb{E}_{\text{rollout}}\big[v_\tau \cdot \mathbb{1}[R \le B]\big]$$
estimated by $\hat\pi = \frac{1}{|\mathcal{T}|K}\sum_{\tau}\sum_{k} v_\tau(y_{\tau k})$ over $K$ seeds. Report $\hat\pi \pm 1.96\sqrt{\hat\pi(1-\hat\pi)/|\mathcal{T}|}$; on SWE-bench Verified ($|\mathcal{T}|=500$) that half-width is $\approx 4.4$ points at $\hat\pi=0.5$ — larger than most claimed model gaps.

Additive-with-interaction model on the logit scale:
$$\mathrm{logit}\,\pi(m,s) \;=\; \mu + \alpha_m + \beta_s + \gamma_{ms}, \qquad \sum_m \alpha_m = \sum_s \beta_s = 0 .$$
$C(m) := \alpha_m$ is the target. Scaffold-independence is the falsifiable hypothesis $\gamma \equiv 0$. Define the **interaction fraction**
$$\rho \;=\; \frac{\mathrm{Var}_{m,s}(\gamma_{ms})}{\mathrm{Var}_{m,s}(\alpha_m + \beta_s + \gamma_{ms})},$$
estimable only from a *filled* $|\mathcal{M}|\times|\mathcal{S}|$ grid. A rank-based restatement: for scaffold sets $S_1, S_2$, measure Kendall's $\tau_b$ between model rankings induced by each. Reliability under repetition uses $\mathrm{pass}^k$ (all $k$ i.i.d. trials succeed), not $\mathrm{pass}@k$.

Assumptions, with the violated ones flagged:
1. $v_\tau$ is a valid verifier. **Violated:** SWE-bench patches can pass hidden tests without fixing the issue; solutions leak via post-cutoff GitHub state.
2. Tasks are i.i.d. draws. **Violated:** benchmark instances cluster by repository, so effective $n \ll |\mathcal{T}|$.
3. Scaffolds are exchangeable across models. **Violated by construction:** scaffolds are tuned on the model that publishes them.
4. Cost is comparable at fixed $B$. **Violated:** token, latency and dollar budgets rank systems differently.

## 3. State of the Art

**Established.**
- Scaffold choice moves scores as much as model choice. On SWE-bench Lite, *Agentless* (Xia et al., 2024) — a fixed three-stage pipeline with no agentic control flow — reached 27.3% with GPT-4o, above the agentic SWE-agent result on the same model and split. Same weights, different harness, opposite conclusion about "agentic ability".
- Cost is a free axis. *AI Agents That Matter* (Kapoor, Stroebl et al., 2024) showed accuracy-only leaderboards reward unbounded retries, and that a trivial baseline (repeatedly calling GPT-4 and keeping the passing patch) matched then-SOTA agent architectures on HumanEval-style tasks at far lower complexity.
- Reliability collapses under repetition. On $\tau$-bench (Yao et al., 2024), frontier models' $\mathrm{pass}^1$ near 60% on the retail domain fall to roughly a third of that at $\mathrm{pass}^8$ — a scaffold-sensitive property invisible in single-run scores.

**Claimed but unablated.**
- Nearly every frontier SWE-bench Verified number ships with a vendor-specific harness (bash + edit tool, custom prompts, sometimes test-time compute). Cross-vendor comparisons are pair comparisons presented as model comparisons; the counterfactual "model A in vendor B's scaffold" is almost never run.
- METR's 50%-time-horizon (Kwa et al., 2025) — task length a model completes half the time, doubling roughly every 7 months — is a genuine attempt at a scaffold-robust scalar, but the horizon is fitted on one task family under METR's own harness. Its stability under an independent scaffold is unverified.

**Benchmark-number-only results.** GAIA, WebArena, OSWorld, Cybench and AgentBench headline figures exist as leaderboard entries; the underlying grids are sparse (each model measured under one or two scaffolds), so $\rho$ is not identified from published data.

## 4. What Is Known

- **Harness swaps are worth tens of points.** SWE-bench Verified (500 instances) scores for a single Claude 3.5-class model ranged from the low 30s under a general agent framework to roughly 49% under a minimal bash/edit harness — a ~15-point swing from the scaffold alone, larger than a model generation.
- **Human–agent gaps are scaffold-dominated at the low end.** WebArena (812 tasks): GPT-4 agent 14.4% vs 78.2% human. OSWorld (369 tasks): human 72.4% vs ~12% for the best system at release. When scores sit near the floor, scaffold changes multiply them rather than shift them.
- **GAIA** (466 questions, ICLR 2024): humans 92%, GPT-4 with plugins ~15%. Later top entries above 70% are *systems*, not models; the model is a component.
- **Verifier noise is measurable.** Audits of SWE-bench found a double-digit fraction of instances with under-specified issues or tests unsolvable from the stated problem — the motivation for the 500-instance human-filtered Verified split (OpenAI, 2024).
- **Ranking instability is documented.** Reordering of model ranks between scaffolds on the same benchmark has been reported by the Holistic Agent Leaderboard effort (Stroebl et al., 2025) *(frontier — verify magnitudes)*.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no accepted definition of the scaffold distribution $\mathcal{S}$ over which $C(m)$ is an expectation, and no accepted budget normalization. Without both, $\alpha_m$ is not a defined estimand — different reasonable choices of $\mathcal{S}$ give different signs for some model pairs.
- **Empirically open.** $\rho$ has never been estimated on a filled grid. A $6 \times 6$ model × scaffold design on SWE-bench Verified is runnable today for order $10^4$–$10^5$ USD; nobody has published it.
- **Theoretically open.** No identifiability result stating conditions (e.g. bounded $\|\gamma\|_\infty$, scaffold-exchangeability) under which $\alpha_m$ is recoverable from a sparse grid, and no sample-complexity bound for estimating it. Item-response-theory models are used for static benchmarks but have no agentic analogue that handles scaffold as a second facet with interaction.

## 6. Why It Is Hard

**Non-identifiability, plus confounded measurement.** With one scaffold per model, the observation is $\alpha_m + \beta_{s(m)} + \gamma_{m s(m)}$ — a single number for three unknowns. The design matrix has rank 1 per cell; no estimator recovers $\alpha_m$ without either a filled grid or a strong assumption ($\gamma = 0$) that the Agentless-vs-SWE-agent result already falsifies.

Three compounding obstructions:
1. **Adversarial scaffold selection.** Publishers optimize $s$ for their own $m$, so $\gamma_{m s(m)} > 0$ by construction — a selection bias with no known correction.
2. **Compute cost of the fix.** Filling a $6\times6$ grid at $K=5$ seeds on 500 tasks is 90,000 rollouts; at ~$0.4$ USD and 4 minutes per rollout that is tens of thousands of dollars and weeks of wall clock unless heavily parallelized.
3. **The verifier does not measure what it names.** A passing hidden test certifies "tests pass", not "issue fixed"; a scaffold that overfits tests gains score without gaining capability.

## 7. Current Research (as of 2026)

- **Cost-aware, multi-scaffold leaderboards.** Princeton/Stanford-affiliated work on the Holistic Agent Leaderboard evaluates model × scaffold × benchmark cells and reports Pareto frontiers over accuracy and dollars *(frontier — verify current coverage)*.
- **Standardized harnesses.** UK AISI's `Inspect` and OpenAI's SWE-bench-Verified reference harness push toward a shared minimal scaffold, making $\beta_s$ a constant rather than a free variable.
- **Horizon-style scalars.** METR continues time-horizon estimation as a scaffold-robust summary; the open question is cross-harness stability.
- **Reliability metrics.** $\mathrm{pass}^k$ and variance-reporting norms (Sierra's $\tau$-bench line, follow-on $\tau^2$-bench work) are spreading *(frontier — verify)*.
- **Contamination and verifier audits** on SWE-bench-family tasks continue to move the effective ceiling *(frontier — verify)*.

## 8. Concrete Next Experiment

**The scaffold-interaction grid.**

- **Scale.** $|\mathcal{M}| = 6$ frontier models × $|\mathcal{S}| = 6$ public scaffolds (minimal bash+edit; SWE-agent; Agentless; OpenHands; a ReAct baseline; a best-of-$n$ + test-selection wrapper) × 500 SWE-bench Verified instances × $K = 5$ seeds $= 90{,}000$ rollouts. Equalize budget at $B = 2\times10^6$ input-equivalent tokens per task and cap at 40 steps; log dollars.
- **Control arm.** The published pairing for each model (its vendor's own scaffold), scored identically. This is the arm that current leaderboards report; the grid is the treatment.
- **Deciding number.** The interaction fraction $\rho$, plus Kendall's $\tau_b$ between the model ranking from a random half of scaffolds and from the held-out half.
  - $\rho < 0.15$ and $\tau_b > 0.8$: a scaffold-independent $C(m)$ exists; publish $\hat\alpha_m$ with CIs and the problem downgrades to *empirically open*.
  - $\rho > 0.35$ or $\tau_b < 0.6$: model-only agentic rankings are not well defined; reporting must move to pair-level or Pareto-frontier form.
- **Cost estimate.** ~90k rollouts × ~$0.40 ≈ $36k; ~2 weeks at 200-way parallelism.

## 9. Key References

- **[Foundational]** Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[Foundational]** Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR, 2023. — arXiv:2210.03629
- **[SOTA]** Yang, Jimenez, Wettig, Lieret, Yao, Narasimhan, Press. *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.* NeurIPS, 2024. — arXiv:2405.15793
- **[SOTA]** Xia, Deng, Dunn, Zhang. *Agentless: Demystifying LLM-based Software Engineering Agents.* 2024. — arXiv:2407.01489
- **[SOTA]** Kwa, West, Becker et al. (METR). *Measuring AI Ability to Complete Long Tasks.* 2025. — arXiv:2503.14499
- **[Methodology]** Kapoor, Stroebl, Siegel, Nadgir, Narayanan. *AI Agents That Matter.* 2024. — arXiv:2407.01502
- **[Benchmark]** Mialon, Fourrier, Swift, Wolf, LeCun, Scialom. *GAIA: A Benchmark for General AI Assistants.* ICLR, 2024. — arXiv:2311.12983
- **[Benchmark]** Zhou, Xu, Zhu et al. *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR, 2024. — arXiv:2307.13854
- **[Benchmark]** Xie, Zhang, Zhou et al. *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments.* NeurIPS, 2024. — arXiv:2404.07972
- **[Benchmark]** Yao, Shinn, Razavi, Narasimhan. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024. — arXiv:2406.12045
- **[Survey]** Stroebl, Kapoor, Narayanan et al. *Holistic Agent Leaderboard.* 2025. *(identifier uncertain — omitted)*

## 10. Worked Example

Take one model $m_0$ (a Claude 3.5-class model) on SWE-bench Verified, $n = 500$.

| Scaffold | Score | 95% CI half-width |
|---|---|---|
| $s_1$ minimal bash+edit | 49.0% | ±4.4 pts |
| $s_2$ general agent framework | ~33% | ±4.1 pts |

The scaffold effect is $\approx 16$ points. Now suppose a competitor model $m_1$ publishes 45% under *its* vendor's harness $s_3$. The leaderboard prints $m_0 = 49\%$, $m_1 = 45\%$ and readers infer $\alpha_{m_0} > \alpha_{m_1}$.

Write the observations out:
$$\mathrm{logit}\,\hat\pi(m_0,s_1) = \mu + \alpha_{m_0} + \beta_{s_1} + \gamma_{m_0 s_1}, \qquad \mathrm{logit}\,\hat\pi(m_1,s_3) = \mu + \alpha_{m_1} + \beta_{s_3} + \gamma_{m_1 s_3}.$$
Subtracting: the 4-point gap equals $(\alpha_{m_0}-\alpha_{m_1}) + (\beta_{s_1}-\beta_{s_3}) + (\gamma_{m_0 s_1}-\gamma_{m_1 s_3})$. The $s_1$-vs-$s_2$ comparison already shows scaffold main effects on the order of 16 points — four times the observed model gap — and both $\gamma$ terms are positive by selection, since each vendor tuned its own harness. Statistical noise alone ($\pm 4.4$) covers the gap.

**The obstruction, made visible:** the reported 4-point difference is consistent with $\alpha_{m_1} > \alpha_{m_0}$ under any plausible assignment of $\beta$ and $\gamma$. Two off-diagonal cells — $m_1$ under $s_1$, $m_0$ under $s_3$ — would break the degeneracy, and they cost about $2\times 500 \times 5$ rollouts (~$2k). They are almost never run, which is why the problem is methodologically blocked rather than merely expensive.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*