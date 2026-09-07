---
id: 05-retrieval-and-agents/agent-benchmark-nondeterminism
title: "Environment Non-Determinism in Agent Benchmark Reproducibility"
topic: 05-retrieval-and-agents
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Environment Non-Determinism in Agent Benchmark Reproducibility

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/agent-benchmark-nondeterminism` · **Status:** methodologically-blocked

## 1. Problem Statement

An agent benchmark reports a single scalar — "54.2% resolved on SWE-bench Verified", "35.8% on WebArena". That number is the output of a stochastic pipeline with at least five independent noise sources: sampler randomness, floating-point/batching non-determinism in the serving stack, environment state drift (package registries, live websites, container images, wall-clock time), harness version, and grader stochasticity when the grader is itself a model. The reported number carries no error bar and no specification of which of these were held fixed.

Three variants, of sharply different difficulty:

- **Measurement variant.** Given an agent $A$, a benchmark $B$, and a run budget, produce an interval estimate for $A$'s score on $B$ that is *valid under re-execution six months later*. Solving it means: two labs running the same artifact get overlapping intervals, and the interval width is reported alongside the point estimate.
- **Method variant.** Engineer environments and inference stacks such that the residual variance is attributable and, where desired, removable — bit-exact replay of a trajectory given a seed and a pinned environment snapshot.
- **Theory variant.** Under what conditions is a benchmark score *identifiable* at all, when the environment is a non-stationary process rather than a fixed dataset? A dataset benchmark has a population parameter to estimate; a live-web agent benchmark may not.

The problem is filed as **methodologically blocked** because the estimand — "the score" — is not well defined until the environment distribution is specified, and no major agent benchmark specifies it.

## 2. Formal Setting

Let a task instance be $x \in \mathcal{X}$, $|\mathcal{X}| = n$ (e.g. $n = 500$ for SWE-bench Verified, $n = 812$ for WebArena, $n = 369$ for OSWorld). An agent is a policy $\pi_\theta$ with sampling temperature $\tau$ and seed $s$. An environment is a realization $e \sim \mathcal{E}_t$, where $\mathcal{E}_t$ is the environment distribution **at wall-clock time $t$** — it indexes the container image digest, the resolved dependency closure, the state of any live external service, and the harness commit.

A run produces a trajectory $\rho = \mathrm{Roll}(\pi_\theta, x, e, s, \omega)$, where $\omega$ collects execution-stack entropy not controlled by $s$: kernel selection, batch composition under continuous batching, atomic reduction order, tool-call latency and timeouts. Grader $g$ returns $Y = g(\rho, x, e) \in \{0,1\}$.

The reported score is
$$\hat{S} = \frac{1}{n}\sum_{i=1}^{n} g(\mathrm{Roll}(\pi_\theta, x_i, e, s_i, \omega_i), x_i, e).$$

The estimand people *believe* they are reporting is
$$S(t) = \mathbb{E}_{e \sim \mathcal{E}_t}\,\mathbb{E}_{s,\omega}\,\frac{1}{n}\sum_i g(\cdot).$$

Decompose the variance of $\hat S$:
$$\mathrm{Var}(\hat S) = \underbrace{\sigma^2_{\text{task}}/n}_{\text{finite }n} + \underbrace{\sigma^2_{\text{seed}}}_{\text{sampler}} + \underbrace{\sigma^2_{\omega}}_{\text{stack}} + \underbrace{\sigma^2_{\text{env}}(t)}_{\text{environment}} + \underbrace{\sigma^2_{g}}_{\text{grader}} + \text{cross terms}.$$

**How each is measured.** $\sigma^2_{\text{task}}/n$: binomial, $\hat S(1-\hat S)/n$ — $\approx 2.2$ pp at $\hat S = 0.5$, $n = 500$. $\sigma^2_{\text{seed}}$: $K$ repeats at fixed $e$, fixed stack, varying $s$. $\sigma^2_{\omega}$: $K$ repeats at fixed $e$ **and** fixed $s$, greedy decoding — anything non-zero here is stack entropy. $\sigma^2_{\text{env}}(t)$: replay the identical agent artifact against environment snapshots $e_1, e_2$ taken at different $t$. $\sigma^2_{g}$: re-grade fixed trajectories with fresh grader seeds; identically zero for exact-match unit-test graders, non-zero for LLM judges (WebArena's fuzzy-match checks, τ-bench's judge-assisted variants).

**Assumptions known to be violated.**
1. *Instances are i.i.d. draws from a population.* False — benchmark instances are curated, and SWE-bench instances cluster in 12 repositories, so failures are correlated within repo. The binomial bar understates spread.
2. *$\mathcal{E}_t$ is stationary.* False by construction for any benchmark touching PyPI/npm, a live website, or a hosted API.
3. *$g$ is a ground-truth oracle.* False — SWE-bench's fail-to-pass tests admit solutions that pass tests without fixing the issue; WebArena's string matchers admit format-lucky failures.
4. *$\omega \perp s$.* False under continuous batching, where a request's numerics depend on unrelated concurrent traffic.

## 3. State of the Art

**Established.**
- Bit-level non-determinism in LLM serving at temperature 0 is real and mechanistically explained. He et al., *Defeating Nondeterminism in LLM Inference* (Thinking Machines Lab, 2025), identify **batch-size-dependent reduction order** in RMSNorm/attention/matmul kernels as the dominant cause, not GPU atomics, and demonstrate batch-invariant kernels restoring bit-exactness across batch sizes at measurable throughput cost. This is a solution to $\sigma^2_\omega$ that most benchmark runs do not use.
- Atil et al., *LLM Stability: A Detailed Analysis with Some Surprises* (2024, arXiv:2408.04667), measure run-to-run disagreement at temperature 0 across models and tasks; no model was deterministic across repeats on all tasks.
- Kapoor, Stroebl et al., *AI Agents That Matter* (2024, arXiv:2407.01502), show agent leaderboards conflate accuracy with cost, that trivially-overfit "agents" can top HumanEval-style leaderboards, and that standardized, reproducible harnesses are largely absent. This motivated the Holistic Agent Leaderboard line of work (Princeton, 2025).
- Biderman et al., *Lessons from the Trenches on Reproducible Evaluation of Language Models* (2024, arXiv:2405.14782), document prompt-format and harness-version sensitivity large enough to reorder models on static benchmarks — a lower bound on the agentic case.

**Claimed but unablated.** Most agent-benchmark leaderboard entries report a single pass@1 with no seed count, no container digest, no harness commit. Where a variance figure is given (e.g. "±1.2%"), it is almost always the binomial bar from $n$, i.e. it measures only $\sigma^2_{\text{task}}$ and silently sets the other four terms to zero.

**Benchmark-number-only results.** SWE-bench Verified and OSWorld headline scores are, in the general case, single-run numbers whose environment snapshot is not published. They are comparable within a leaderboard's own harness and not obviously comparable across time.

## 4. What Is Known

- **Grader validity is a measured problem, not a hypothetical.** Aleithan et al., *SWE-Bench+: Enhanced Coding Benchmark for LLMs* (2024, arXiv:2410.06992), report solution leakage in the issue text and weak test suites in a substantial fraction of SWE-bench instances; when filtered, reported resolve rates drop sharply. OpenAI's SWE-bench Verified (2024) retained **500 of 2,294** instances after human screening — i.e. ~78% of the original set failed a basic validity screen.
- **Environment drift is documented at the harness level.** SWE-bench moved to per-instance Docker images with pinned digests precisely because pip-resolved environments made older results non-reproducible; results predating that change cannot be re-derived.
- **Interactive-environment scores are low and therefore noisy in relative terms.** WebArena (Zhou et al., ICLR 2024, $n=812$) reported ~14% for GPT-4-based agents against ~78% human; OSWorld (Xie et al., NeurIPS 2024, $n=369$) reported ~12% for the best agent against 72% human. At $\hat S \approx 0.12$, $n = 369$, the binomial bar alone is $\pm 1.7$ pp — comparable to the gaps separating adjacent leaderboard entries.
- **Long-horizon tasks amplify variance.** TheAgentCompany (Xu et al., 2024, arXiv:2412.14161) uses multi-hour, multi-tool workplace tasks with partial-credit checkpoints; per-task success is a product over many stochastic steps, so per-instance variance approaches its maximum near the middle of the score range.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted definition of the estimand for a benchmark whose environment is non-stationary. Reporting $S(t)$ for unrecorded $t$ is not a well-posed measurement. No major agent benchmark publishes an environment-snapshot identifier alongside scores, so $\sigma^2_{\text{env}}$ is not merely unmeasured — it is unmeasurable post hoc.
- **Empirically open.** The full variance decomposition of §2 has not been run at scale on any frontier agent/benchmark pair. Each term is cheaply measurable in isolation; nobody has published all five for one system.
- **Empirically open.** Whether batch-invariant kernels, applied end to end in an agent loop, collapse trajectory divergence — or merely delay the first divergent token, after which the environment re-injects entropy.
- **Theoretically open.** Conditions under which leaderboard *rankings* are stable even when scores are not. A ranking-stability guarantee under bounded per-instance noise and correlated instances (repo clustering) is not established.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the estimand under a moving environment**, compounded by **confounded variance attribution**.

Two runs of the same agent artifact six months apart differ in $\pi_\theta$ (hosted model silently updated), $e$ (dependency closure, live site), the harness, and the seed. A score difference of 4 pp cannot be assigned to any one of them, because the experiment that would separate them — freeze four factors, vary one — requires the environment to be freezable, which for live-web and hosted-API benchmarks it is not. This is not a compute problem; it is a design problem. Cost is a secondary obstruction: a single WebArena or OSWorld pass at frontier scale runs into hundreds of dollars, so the $K \geq 10$ repeats needed to estimate $\sigma^2_{\text{seed}}$ to useful precision are rarely funded.

Third: the grader does not measure what it names. "Resolved" means "the held-out tests pass", which is a proxy for "the issue is fixed" that is known to admit both false positives and false negatives.

## 7. Current Research (as of 2026)

- **Batch-invariant inference.** Thinking Machines Lab's batch-invariant kernel work (2025) is the clearest engineering path to $\sigma^2_\omega = 0$; integration into vLLM/SGLang-class serving is in progress *(frontier — verify current status)*.
- **Reproducible agent harnesses.** Princeton's AI-Snake-Oil / HAL group (Kapoor, Stroebl, Narayanan) runs standardized agent evaluation with cost reporting on a shared harness; the explicit design goal is cross-lab comparability.
- **Benchmark hardening.** SWE-bench Verified, SWE-bench Multimodal, and SWE-smith (Princeton NLP) continue to tighten instance validity and container pinning.
- **Snapshot-frozen web environments.** WebArena/VisualWebArena self-host their sites in containers rather than hitting the live web — the strongest existing mitigation, and the reason WebArena is more reproducible than any live-browsing benchmark.
- **Variance-aware reporting.** Proposals to require $K$-repeat reporting and confidence intervals in agent leaderboards are circulating but not yet standard *(frontier — verify adoption)*.

## 8. Concrete Next Experiment

**The five-way variance decomposition, run once, properly.**

- **Scale.** One frontier agent scaffold, two benchmarks: SWE-bench Verified ($n = 500$, deterministic unit-test grader, containerized) and WebArena ($n = 812$, self-hosted sites, partly fuzzy grader). $K = 10$ repeats per condition. Roughly $10 \times 1312 \approx 13{,}000$ trajectories per condition, four conditions — order $\$50$k–$\$150$k at 2026 frontier pricing.
- **Conditions.** (a) *Control arm:* fixed container digest, fixed seed, greedy decoding, batch-invariant kernels, dedicated single-tenant serving. Expected $\hat S$ variance across repeats: exactly $0$. (b) Vary $\omega$ only: same as (a) but standard continuous-batching serving under synthetic concurrent load. (c) Vary $s$ only: $\tau = 1$, control stack. (d) Vary $e$ only: re-resolve dependencies / rebuild images from tags rather than digests, fixed $s$, control stack.
- **Deciding number.** $\hat\sigma_\omega$ — the standard deviation of $\hat S$ across the 10 repeats in condition (b), expressed in percentage points. **If $\hat\sigma_\omega > 1.0$ pp**, then serving-stack entropy alone exceeds the margin separating adjacent frontier entries on both leaderboards, and every single-run agent leaderboard number is uninterpretable without a stack specification. **If $\hat\sigma_\omega < 0.2$ pp**, stack entropy is negligible and the field should spend its reproducibility budget entirely on $\sigma^2_{\text{env}}$ and grader validity instead.

## 9. Key References

- **[Foundational]** Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. — arXiv:2310.06770
- **[Foundational]** Zhou, Xu, Zhu, Zhou, Lo, Sridhar, Cheng, Ou, Bisk, Fried, Alon, Neubig. *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR 2024. — arXiv:2307.13854
- **[SOTA]** Xie, Zhang, Zhou, Yiheng Xu, et al. *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments.* NeurIPS 2024. — arXiv:2404.07972
- **[SOTA]** Kapoor, Stroebl, Siegel, Nadgir, Narayanan. *AI Agents That Matter.* 2024. — arXiv:2407.01502
- **[SOTA]** He, Thinking Machines Lab. *Defeating Nondeterminism in LLM Inference.* Thinking Machines Lab technical blog, 2025.
- **[Empirical]** Atil, Chittams, Fu, Ture, Xu, Baldwin. *LLM Stability: A Detailed Analysis with Some Surprises.* 2024. — arXiv:2408.04667
- **[Empirical]** Aleithan, Xue, Mohajer, Nnorom, Uddin, Wang. *SWE-Bench+: Enhanced Coding Benchmark for LLMs.* 2024. — arXiv:2410.06992
- **[Survey]** Biderman, Schoelkopf, Sutawika, et al. *Lessons from the Trenches on Reproducible Evaluation of Language Models.* 2024. — arXiv:2405.14782
- **[Benchmark]** Mialon, Fourrier, Swift, Wolf, LeCun, Scialom. *GAIA: A Benchmark for General AI Assistants.* ICLR 2024. — arXiv:2311.12983
- **[Benchmark]** Yao, Shinn, Razavi, Narasimhan. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024. — arXiv:2406.12045
- **[Benchmark]** Xu, Jain, Li, et al. *TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks.* 2024. — arXiv:2412.14161

## 10. Worked Example

Two labs report on SWE-bench Verified. Lab A: **54.2%**. Lab B, same open-weights model, same public scaffold, three weeks later: **51.6%**. Difference: **2.6 pp**, or 13 instances out of 500.

The reported uncertainty is the binomial bar: $\sqrt{0.542 \times 0.458 / 500} = 2.23$ pp, so each lab prints "$\pm 2.2$". The difference looks like a $\sim 1\sigma$ fluctuation and gets waved away.

Now add the terms the bar omits. Suppose the decomposition of §8 returns, for this setup:

| Source | $\hat\sigma$ (pp) | Controlled? |
|---|---|---|
| finite $n$ (binomial) | 2.23 | reported |
| $\sigma_\omega$ (batching/kernels) | 0.9 | no |
| $\sigma_{\text{seed}}$ ($\tau=0.2$) | 1.4 | no |
| $\sigma_{\text{env}}$ (image rebuild) | 1.1 | no |
| $\sigma_g$ (unit tests) | 0.0 | n/a |

Combined, assuming independence: $\sqrt{2.23^2 + 0.9^2 + 1.4^2 + 1.1^2} = 3.05$ pp. The honest interval is ~40% wider, and the 2.6 pp gap is well inside it.

**Where the obstruction becomes visible.** Repo clustering breaks the independence assumption that made that square-root legal. SWE-bench Verified draws heavily from a dozen repositories; if a `sympy` image rebuild picks up a newer transitive dependency that breaks 6 of the ~70 `sympy` instances, that is one correlated event moving the score 1.2 pp, not 6 independent Bernoulli flips. And the diagnosis is unavailable after the fact: Lab A did not publish an image digest, so nobody can rebuild Lab A's `sympy` environment to check. The 2.6 pp is not "noise" and it is not "a real difference" — it is **unattributable**, and it will stay unattributable no matter how many times either lab re-runs today. That is what methodologically blocked means here: the fix is a reporting convention adopted before the run, not an analysis performed after it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*