---
id: 05-retrieval-and-agents/agentic-generalization-vs-contamination
title: "Measuring Genuine Agentic Generalization vs. Benchmark Contamination"
topic: 05-retrieval-and-agents
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Measuring Genuine Agentic Generalization vs. Benchmark Contamination

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/agentic-generalization-vs-contamination` · **Status:** methodologically-blocked

## 1. Problem Statement

An agent scores 70% on SWE-bench Verified. How much of that score survives on tasks the agent's training pipeline could not have seen, and how much is recall of the fix, the repository, or the benchmark harness?

Three variants, of increasing difficulty:

- **Measurement.** Given a fixed agent $\pi$ and a fixed agentic benchmark $B$, estimate the fraction of $\pi$'s success rate on $B$ attributable to $B$ (or its solutions, or near-duplicates) appearing in $\pi$'s training data or its developers' iteration loop. Solving it means a contamination-corrected score with a calibrated error bar.
- **Method.** Construct an agentic evaluation whose score is provably insensitive to contamination — held-out-by-construction, continuously refreshed, or counterfactually controlled — without changing the task difficulty distribution.
- **Theory.** Decide whether the contamination correction is *identifiable at all* from black-box access to $\pi$ plus the benchmark, without retraining. Current evidence says no in the general case.

Static-benchmark contamination (MMLU, GSM8K) is a solved-ish measurement problem: you can string-match, or run a permutation test on likelihoods. Agentic contamination is not, because the leaked object is not a string — it is a repository, an API surface, a UI layout, or a solution pattern spread across thousands of documents, and the agent's success is mediated by a multi-step, stochastic, tool-using trajectory.

## 2. Formal Setting

A task is $\tau = (E_0, g, V)$: an initial environment state, a goal specification, and a verifier $V$ mapping a terminal state to $\{0,1\}$. An agent is a policy $\pi_\theta$ over actions given observation histories; a rollout produces trajectory $h \sim \pi_\theta(\cdot \mid E_0, g)$ and score $S(\pi_\theta, \tau) = V(\text{final}(h))$.

**Reported metric.** For benchmark $B = \{\tau_1,\dots,\tau_n\}$, the resolve rate is
$$\widehat{R}(\pi, B) = \frac{1}{n}\sum_{i=1}^{n} \frac{1}{k}\sum_{j=1}^{k} S_j(\pi, \tau_i),$$
with $k$ rollouts each — as measured, almost always $k=1$ and greedy/low-temperature decoding. Reliability is $\text{pass}^k = \frac{1}{n}\sum_i \prod_{j\le k} S_j$, the probability all $k$ independent rollouts succeed.

**Contamination, defined counterfactually.** Let $D$ be the training corpus and $L(\tau) \subseteq D$ the *leak set* of documents whose removal is intended to eliminate task-specific prior knowledge. The causal contamination effect on task $\tau$ is
$$\Delta(\tau) \;=\; \mathbb{E}\big[S(\pi_{\theta(D)}, \tau)\big] \;-\; \mathbb{E}\big[S(\pi_{\theta(D \setminus L(\tau))}, \tau)\big].$$
This is the quantity everyone means and nobody measures: it requires a retrain per leak set. Measured proxies substitute for it:

- **Lexical overlap.** $c_{\text{ngram}}(\tau) = \max_{d \in D} \text{LCS}_n(\text{repr}(\tau), d)/|\text{repr}(\tau)|$ — computable only with corpus access, and defeated by paraphrase.
- **Membership score.** Min-$K$% negative log-likelihood over the tokens of $\text{repr}(\tau)$, thresholded; reported as AUC against a known-nonmember control set.
- **Temporal split.** $c_{\text{time}}(\tau) = \mathbb{1}[t(\tau) < t_c]$, with $t(\tau)$ the task's creation date and $t_c$ the cutoff. Cheap, black-box, and the only proxy usable on closed models.
- **Observed gap.** $\widehat{G} = \widehat{R}(\pi, B_{\text{post}}) - \widehat{R}(\pi, B_{\text{pre}})$ over post- and pre-cutoff task splits.

**Assumptions, and which are violated.**

1. *$B_{\text{pre}}$ and $B_{\text{post}}$ are equal in difficulty.* **Violated.** Repository age, library churn, and issue-writing conventions drift; post-cutoff GitHub issues are not exchangeable with 2019 issues.
2. *$V$ is sound.* **Violated.** SWE-bench's fail-to-pass tests under-specify; patches that differ from the gold patch pass, and gold-patch-equivalent patches fail.
3. *$t_c$ is known and honest.* **Violated for closed models.** Cutoffs are self-reported, post-training data is continuously refreshed, and RL environments are built from the same repositories benchmarks are drawn from.
4. *The benchmark was held fixed during model development.* **Violated by construction.** Benchmarks are optimization targets; $\widehat{R}$ is a post-selection statistic, so even a leak-free corpus leaves adaptive-overfitting bias.
5. *Leakage flows only through task text.* **Violated.** For agents, the scaffold, the tool schemas, and the environment image are also learnable.

## 3. State of the Art

**Established.**
- Permutation/exchangeability testing (Oren et al., ICLR 2024) gives a provable false-positive guarantee for detecting whether an *ordered* test set was trained on, black-box, from log-likelihoods. It requires logprobs and a canonical ordering; it does not apply to trajectory-scored agentic tasks and gives no effect-size estimate.
- Membership inference on LLM pretraining data is near-chance: Duan et al. (COLM 2024) report AUC ≈ 0.5–0.55 across Pythia 160M–12B on MIMIR, attributing apparent successes to distribution shift between member and nonmember sets rather than memorization.
- Rephrasing defeats n-gram detection: Yang et al. (2023) trained a 13B model on rephrased test sets and reached GSM8K/MMLU/HumanEval scores comparable to GPT-4 while passing standard decontamination.
- SWE-bench solution leakage is real and quantified: Aleithan et al. (SWE-bench+, 2024) found that a large fraction of "resolved" instances in leading agent logs contained the fix in the issue text or comments, and that many remaining passes were weak-test artifacts.

**Claimed but unablated.**
- "Contamination-free" labels on LiveBench, LiveCodeBench, and rolling agentic suites rest on the temporal argument alone. They control $c_{\text{time}}$; they do not control difficulty drift, adaptive overfitting across releases, or leakage through public solution write-ups appearing within days.
- Frontier agent resolve rates on SWE-bench Verified (roughly 65–75% in 2025 vendor reports) exist **only as benchmark numbers**: no vendor has published a paired pre/post-cutoff arm with matched difficulty, and no vendor has published a retrained-without-leak-set control.

## 4. What Is Known

- **GSM1k (Zhang et al., NeurIPS 2024).** A held-out GSM8K clone, 1250 problems, human-authored to match the original distribution. Some model families dropped up to ~13 accuracy points versus GSM8K; frontier families (GPT-4, Claude, Gemini) showed near-zero gap. Scale: 8B–70B open models plus closed frontier. Interpretation: contamination effects are family-specific, not universal.
- **SWE-bench.** 2294 instances from 12 Python repos; SWE-bench Verified is a 500-instance human-filtered subset (OpenAI, 2024) created precisely because the verifier assumption failed on the full set. The filtering removed instances with under-specified issues and broken environments — an admission that $V$ was unsound at scale.
- **Reliability collapse.** On $\tau$-bench (Yao et al., 2024), GPT-4o's pass^1 in the retail domain (~61%) falls to pass^8 well under half that. Scale: ~115 retail / ~50 airline tasks. This is variance, not contamination, but it bounds how finely any contamination gap can be resolved.
- **Longitudinal effects.** Roberts et al. (ICLR 2024) found positive correlation between GitHub repository popularity/age before the cutoff and code-benchmark performance, on Codeforces and Project Euler splits — a temporal signal consistent with contamination but confounded with problem difficulty.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no accepted definition of the leak set $L(\tau)$ for an agentic task. A SWE-bench instance's "solution" is a commit, but the repository's later history, its docs, its Stack Overflow threads, and forks all encode the fix. Without a definition of $L(\tau)$, $\Delta(\tau)$ is not a well-posed quantity, so no estimator can be validated.
- **Theoretically open.** Whether $\Delta(\tau)$ is identifiable from black-box query access alone, under any nontrivial assumption set. The rephrasing result and the near-chance MIA results together suggest non-identifiability, but no impossibility theorem exists.
- **Empirically open.** The retrain-with-holdout experiment — train two otherwise identical models, one with $L(B)$ excised — has never been run at $\ge$10B scale on an agentic benchmark. It is runnable today; it costs a full pretraining run and nobody has paid for it.
- **Empirically open.** Whether agentic contamination transfers across repositories: does seeing `django` fixes improve `sympy` performance more than a matched non-code corpus does?

## 6. Why It Is Hard

Three named obstructions, in order of severity.

1. **Absent ground truth for $L(\tau)$.** Unlike a multiple-choice question, an agentic task has no canonical serialization to search for. The obstruction is definitional, not computational.
2. **Confounded measurement.** Every cheap proxy ($c_{\text{time}}$, popularity, repo age) is correlated with difficulty. A post-cutoff drop of 8 points is consistent with contamination, with harder recent issues, or with library-version drift the agent has not seen documentation for. These are not separable without an intervention.
3. **Non-identifiability under adaptive development.** Even a perfectly leak-free corpus leaves $\widehat{R}$ biased, because the benchmark was used for model selection across many candidate checkpoints. The bias scales with the number of evaluations and is unrecorded.

Compute is a secondary obstruction: the one clean experiment (paired retrain) costs a pretraining run per leak set, so it cannot be run per-benchmark, let alone per-task.

## 7. Current Research (as of 2026)

- **Rolling/temporal benchmarks.** LiveCodeBench (Jain et al., ICLR 2025) and LiveBench (White et al., Abacus.AI / NYU / Nvidia, 2024) refresh monthly. Agentic analogues — continuously scraped SWE-bench-style harvesters — are being built by several groups *(frontier — verify)*.
- **Synthetic-environment generation.** Procedurally generated agent tasks with no public provenance, so $L(\tau) = \emptyset$ by construction. The open question is whether synthetic tasks retain the difficulty structure of real ones.
- **Verifier hardening.** Test-suite strengthening and mutation testing for SWE-bench-style benchmarks, following SWE-bench+ and SWE-bench Verified.
- **Canary/watermark insertion** into benchmark releases to make future contamination detectable at release time rather than post hoc.
- **Pass^k and reliability reporting** as a standard alongside pass^1, following $\tau$-bench.

## 8. Concrete Next Experiment

**The paired-holdout pretraining ablation, restricted to make it affordable.**

- **Scale.** Two 7B models, identical architecture, data mixture, seed, and token budget (~1T tokens). Arm A trains on the full corpus. Arm B excises a defined leak set: all commits, issues, PRs, docs, forks, and Stack Overflow posts touching the 12 SWE-bench repositories, for all time. Cost: roughly $2 \times$ a 7B run.
- **Then post-train both identically** on the same agentic RL/SFT mixture, with SWE-bench repositories excluded from that mixture in both arms.
- **Control arm.** A third split of held-out tasks from *non-SWE-bench* repositories, matched on stars, age, test-suite size, and gold-patch line count. Both models are evaluated on it. This arm absorbs the general-code-ability difference caused by removing a large chunk of Python from Arm B.
- **The deciding number.**
$$\hat{\Delta} = \big[\widehat{R}_A(B_{\text{swe}}) - \widehat{R}_B(B_{\text{swe}})\big] - \big[\widehat{R}_A(B_{\text{ctrl}}) - \widehat{R}_B(B_{\text{ctrl}})\big]$$
the difference-in-differences, with $k=8$ rollouts per task for variance control. If $\hat{\Delta} \le 2$ points with a 95% CI excluding 5, temporal-split benchmarks are approximately sound and the field can stop worrying. If $\hat{\Delta} \ge 10$ points, every published SWE-bench number is uninterpretable and the correction is first-order.

Nothing smaller settles it, because every cheaper design substitutes a proxy for the intervention.

## 9. Key References

- **[Foundational]** Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. — arXiv:2310.06770
- **[Foundational]** Oren, Meister, Chatterji, Ladhak, Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR 2024. — arXiv:2310.17623
- **[SOTA]** Zhang, Da, Lee, Robinson, Wu, Song, Zhao, Raja, Slack, Lyu, Hendryx, Kaplan, Lundberg, Yue. *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* NeurIPS 2024. — arXiv:2405.00332
- **[SOTA]** Duan, Suri, Mireshghallah, Min, Shi, Zettlemoyer, Tsvetkov, Choi, Evans, Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM 2024. — arXiv:2402.07841
- **[SOTA]** Jain, Han, Gu, Li, Yan, Zhang, Wang, Solar-Lezama, Sen, Stoica. *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* ICLR 2025. — arXiv:2403.07974
- **[SOTA]** Yao, Shi, Cao, Chen, Chen, Liu, Wang, Yu. *$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024. — arXiv:2406.12045
- **[Analysis]** Aleithan, Xue, Mohajer, Nnorom, Uddin, Wang. *SWE-Bench+: Enhanced Coding Benchmark for LLMs.* 2024. — arXiv:2410.06992
- **[Analysis]** Yang, Chiang, Zheng, Gonzalez, Stoica. *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples.* 2023. — arXiv:2311.04850
- **[Analysis]** Roberts, Baral, White, Jain, Jain, Feizi, Goldstein. *To the Cutoff... and Beyond? A Longitudinal Perspective on LLM Data Contamination.* ICLR 2024.
- **[Survey]** Xu, Song, Feng, Wan, Foo, Ng, Joty. *Benchmark Data Contamination of Large Language Models: A Survey.* 2024. — arXiv:2406.04244
- **[Context]** Zhou, Xu, Zhu, Zhou, Lo, Sridhar, Cheng, Ou, Bisk, Fried, Alon, Neubig. *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR 2024. — arXiv:2307.13854

## 10. Worked Example

Take one SWE-bench instance: `django__django-11099` — a fix to `UsernameValidator` replacing regex `^` / `$` anchors with `\A` / `\Z`. The gold patch is two characters changed on two lines.

Walk the proxies:

- **N-gram check.** Serialize the issue text and grep the corpus. The issue title is generic; the *patch* is a two-token edit. A 13-gram overlap test on the patch returns hits in thousands of unrelated Django files. Lexical detection is uninformative here — signal-to-noise is near zero for a two-token fix.
- **Temporal split.** The commit predates every frontier model's cutoff. So $c_{\text{time}} = 1$ and the instance is labelled "contaminated". But the fix — use `\A`/`\Z` instead of `^`/`$` in Python regex — is *general Python knowledge* documented in the standard library docs and hundreds of blog posts. A model that has never seen this commit still solves it. The temporal label attributes to contamination what is actually transferable knowledge.
- **Counterfactual.** Under the definition of $L(\tau)$ as "the commit and its PR thread", removing it leaves the `re` module docs intact and $\Delta(\tau) \approx 0$. Under the definition "everything explaining `\A`/`\Z` anchoring", $\Delta(\tau)$ is large — but that leak set removes a general Python capability, and Arm B is now a worse Python model for reasons unrelated to this task.

**The obstruction, made visible.** The same instance is scored as fully contaminated, uncontaminated, or something in between depending on where you draw $L(\tau)$, and the two defensible definitions give answers that differ by the entire effect size. That is not estimator noise — it is the quantity being undefined. Aggregating over 500 Verified instances does not average the ambiguity away, because the leak-set choice biases every instance in the same direction. Any contamination-corrected score published today is reporting a number whose definition has not been fixed.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*