---
id: 26-code-generation/competitive-programming-to-real-software-transfer
title: "Transfer from Competitive Programming to Real Software Tasks"
topic: 26-code-generation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Transfer from Competitive Programming to Real Software Tasks

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/competitive-programming-to-real-software-transfer` · **Status:** empirically-open

## 1. Problem Statement

Competitive-programming (CP) ability is the headline number for frontier coding models: Codeforces Elo, IOI medals, LiveCodeBench pass@1. Real software work is issue resolution, refactoring across a repository, reading logs, and shipping a patch that survives review. The question is whether the first predicts or *causes* the second.

Three variants, with different difficulty:

- **Measurement variant.** Given a set of models $\mathcal{M}$, does CP score predict real-task score *after* controlling for general capability and training compute? This needs only careful regression on existing evaluations, but the covariates are largely unobservable for closed models.
- **Method variant.** Does *training* on CP — RL against hidden test suites, self-play on synthetic contests — improve real-task performance more per unit compute than training on repository-derived data? This is an intervention question and requires a controlled training run.
- **Theory variant.** Is there a structural reason CP transfer should saturate? CP tasks are specification-complete (the statement plus samples determine the function) and verifier-cheap. Real tasks are specification-incomplete and verifier-expensive. No formal model exists that predicts where the transfer curve bends.

Solving it means: a stated transfer coefficient with confidence intervals for the measurement variant, and a matched-compute ablation for the method variant.

## 2. Formal Setting

Let a model be $m$ with parameters $\theta$, pretraining compute $C(m)$ FLOPs, and post-training compute $C_{\text{RL}}(m)$.

**CP score.** $X(m) = \mathbb{E}_{p \sim \mathcal{D}_{\text{CP}}}[\,\mathbb{1}\{\text{all hidden tests pass}\}\,]$, measured as pass@1 at a fixed sampling budget $k$ and temperature, on a problem set $\mathcal{D}_{\text{CP}}$ released strictly after the model's data cutoff (LiveCodeBench-style windowing). Reported alternatively as an Elo $R(m)$ obtained by simulating contest submissions under real penalty rules.

**Real-task score.** $Y(m) = \mathbb{E}_{t \sim \mathcal{D}_{\text{SWE}}}[\,\mathbb{1}\{\text{FAIL\_TO\_PASS} \cup \text{PASS\_TO\_PASS} \text{ all green}\}\,]$ under a fixed agent scaffold $\mathcal{S}$, fixed step limit $L$, and fixed token budget $B$. $Y$ is a property of the pair $(m, \mathcal{S})$, not of $m$.

**Transfer coefficient.** For a model family, fit

$$Y(m) = \alpha + \beta\, X(m) + \gamma\, Z(m) + \varepsilon,$$

where $Z(m)$ is a general-capability control (e.g. MMLU-Pro or GPQA-Diamond, itself contamination-windowed). The quantity of interest is $\beta$ — CP score's marginal predictive power. $\beta \approx 0$ with $\gamma > 0$ means CP is a proxy for general capability, not a coding-specific signal.

**Causal transfer.** For an intervention, take a base model $\theta_0$ and two matched-FLOP post-training runs: $\theta_{\text{CP}}$ (RL on contest problems) and $\theta_{\text{repo}}$ (RL on repository tasks). Define

$$\tau = Y(\theta_{\text{CP}}) - Y(\theta_0), \qquad \Delta = Y(\theta_{\text{repo}}) - Y(\theta_{\text{CP}}).$$

$\tau > 0$ establishes transfer; $\Delta$ measures its opportunity cost.

**Assumptions, and which are violated.**
1. *$X$ and $Y$ measure the same latent skill up to noise.* Violated: SWE-bench instances are resolvable by hard-coded reproduction of a known upstream patch, and CP instances are not.
2. *Test cutoff removes contamination.* Partly violated — Codeforces problems are frequently restatements of standard techniques present in pretraining, so a "post-cutoff" problem is not novel in solution space.
3. *$\mathcal{S}$ is held fixed across models.* Routinely violated: leaderboard entries use different scaffolds, so cross-model $Y$ differences confound model and harness.
4. *Hidden tests are a sound oracle for $Y$.* Violated at a measured rate: a nontrivial fraction of SWE-bench "resolved" patches are false positives under stricter human review.

## 3. State of the Art

**Empirical SOTA, CP side (established as benchmark numbers).** AlphaCode (Li et al., *Science* 2022) reached a simulated Codeforces rating around the 54th percentile of participants using $\sim 10^6$ samples per problem with filtering and clustering. OpenAI's reasoning-model report (El-Kishky et al., 2025) reports o3 at a simulated Codeforces rating near 2700 and gold-medal-level IOI 2024 performance under relaxed submission conditions. These are benchmark numbers: the o3 result is not accompanied by a public ablation isolating which post-training component produced it.

**Empirical SOTA, real-task side.** SWE-bench Verified (Jimenez et al., ICLR 2024; Verified subset curated by OpenAI, 2024) is the field's default. Frontier agents in 2025 report roughly 70–80% resolved. This is a benchmark number with a known ceiling problem: the subset is 500 Python instances from 12 repositories with executable tests, i.e. the easiest measurable slice of software work.

**Established, not merely claimed.** Code data in *pretraining* transfers to non-code reasoning: Aryabumi et al. (Cohere, 2024) show controlled pretraining mixtures where adding code improves natural-language reasoning at fixed token budget. That is genuine transfer, but from code *corpora*, not from CP *RL*.

**Claimed but unablated.** That RL on verifiable competition tasks is the driver of agentic software ability. DeepSeek-R1 (DeepSeek-AI, *Nature* 2025) shows RL on verifiable math/code raises reasoning benchmarks, but does not isolate a CP→repo transfer term.

**Contrary evidence.** METR's 2025 randomized controlled trial with 16 experienced open-source developers over 246 tasks found developers using early-2025 AI tools were **19% slower**, while forecasting 24% faster. Benchmark CP/SWE gains did not convert into measured throughput on the developers' own repositories.

## 4. What Is Known

- **Scale of CP gains.** AlphaCode: ~54th percentile on 10 Codeforces contests, requiring up to $10^6$ samples/problem (2022). Reasoning models reach comparable or better with $\sim 10^0$–$10^1$ samples (2025). Sample efficiency improved by roughly 5 orders of magnitude in three years.
- **Real-task ceiling.** SWE-bench full set: 2,294 instances; original 2023 baselines with retrieval + GPT-4 resolved under 2%. SWE-bench Verified (500 instances): 70%+ in 2025.
- **Benchmark validity is contested.** SWE-Bench+ (Aleithan et al., 2024) reports that a large share of apparently-passing patches on the original SWE-bench are "solution leakage" (the fix appears in the issue text or comments) or pass weak tests; their re-audit drops headline resolution rates by more than half on the subsets examined.
- **Economic-value tasks are much harder than issue-resolution.** SWE-Lancer (Miserendino et al., OpenAI, 2025) scores models on 1,400+ real freelance tasks worth \$1M total; frontier models in early 2025 earned a minority fraction of the total payout, with individual-contributor task pass rates far below SWE-bench Verified rates on the same models.
- **Horizon length, not task type, tracks failure.** METR's time-horizon analysis (Kwa et al., 2025) finds the task length a model completes at 50% success grows roughly exponentially with release date; CP problems sit at the ~1-hour-human end, real repository tasks at the multi-hour to multi-day end.
- **Contamination is real and measurable.** LiveCodeBench (Jain et al., ICLR 2025) shows several models' pass rates drop sharply on problems released after their cutoff relative to before — direct evidence that raw CP scores overstate ability.

## 5. What Is Not Known

- **Empirically open (the main gap).** No published matched-compute ablation of $\tau$ and $\Delta$: take one base model, spend equal RL FLOPs on contest problems versus repository tasks, and measure SWE-bench Verified and a held-out repo suite. Runnable today at 7B–32B scale for well under \$1M. Nobody has published it.
- **Empirically open.** The value of $\beta$ in the regression of §2. Public leaderboards have enough models to fit it, but not enough with disclosed compute, scaffold, and contamination controls.
- **Methodologically blocked.** Whether $Y$ measures "real software ability" at all. With false-positive patches and solution leakage unquantified per-instance, the dependent variable is mis-specified; a null $\beta$ could be measurement noise.
- **Theoretically open.** Whether the specification-complete → specification-incomplete gap admits a formal separation — e.g. whether any policy trained solely on verifier-complete tasks has bounded performance on tasks where the objective must be inferred from ambiguous natural language.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by an evaluation that does not measure the thing it names**.

- CP score and real-task score both rise with the same underlying variable — post-training compute on verifiable rewards — so the observational correlation is nearly guaranteed and nearly uninformative. Separating them requires the intervention, not more leaderboard rows.
- $Y$ is a joint property of model and scaffold. Two labs reporting 72% on SWE-bench Verified with different harnesses are not measuring the same quantity; the between-scaffold variance is comparable to the between-model variance being studied.
- The oracle is unsound in both directions: weak hidden tests admit wrong patches, and strict `PASS_TO_PASS` sets reject correct patches that differ from upstream. Neither error rate is published per-instance.
- Confounder $Z$ is unobservable for exactly the models with the highest CP scores — closed frontier models with undisclosed pretraining compute and data mix.

## 7. Current Research (as of 2026)

- **Repository-native RL environments.** SWE-Gym (Pan et al., 2024) and SWE-smith-style synthetic-task generation build training environments from real repositories, making the $\Delta$ arm of the ablation feasible. Groups: UIUC/Berkeley (OpenHands lineage), Princeton NLP (SWE-agent). *(frontier — verify current scale)*
- **Contamination-hardened CP evaluation.** LiveCodeBench continues rolling windows; CodeElo (Quan et al., Alibaba, 2025) submits to Codeforces' real judge to obtain Elo under authentic penalties.
- **Economic-validity benchmarks.** SWE-Lancer (OpenAI) and multi-repo/multi-language successors push $Y$ toward paid work rather than curated issues.
- **Human-in-the-loop RCTs.** METR is the only group publishing randomized measurement of developer throughput; replication at larger $n$ is the obvious follow-up. *(frontier — verify)*
- **Long-horizon agent training.** Anthropic, OpenAI, and Google DeepMind all report agentic-coding post-training; none has published the CP-versus-repo compute split. *(frontier — verify)*

## 8. Concrete Next Experiment

**The matched-compute transfer ablation.**

- **Scale.** One open base model at 32B parameters (e.g. Qwen2.5-Coder-32B class). Three arms, each $2\times10^{21}$ training FLOPs of RL — roughly a few thousand H100-hours per arm, about \$100–300k total including evaluation.
- **Arms.**
  - **A (CP):** RL with verifiable rewards on ~10k contest problems (CodeContests + post-2024 Codeforces), hidden-test reward.
  - **B (repo):** RL on ~10k automatically-mined repository tasks with executable tests, from repositories disjoint from all evaluation sets.
  - **C (control):** identical RL on a *format-matched but reward-shuffled* task set, isolating the effect of RL machinery from task content. This arm is what most published comparisons omit.
- **Evaluation.** SWE-bench Verified (500 instances, single fixed scaffold, fixed 50-step / 200k-token budget, 5 seeds) plus a held-out 200-instance suite mined from repositories created after all cutoffs, human-audited for solution leakage.
- **The deciding number.** $\tau = Y(A) - Y(C)$ on the held-out post-cutoff suite. With $n = 200$ and 5 seeds, the 95% CI on a proportion near 0.3 is about $\pm 6$ points. **If $\tau < 5$ points, CP RL does not transfer to real software tasks at this scale, and the field's headline metric is decorative.** Report $\Delta = Y(B) - Y(A)$ alongside as the opportunity cost.

## 9. Key References

- **[Foundational]** Yujia Li et al. *Competition-Level Code Generation with AlphaCode.* Science, 2022. — arXiv:2203.07814
- **[Foundational]** Mark Chen et al. *Evaluating Large Language Models Trained on Code.* 2021. — arXiv:2107.03374
- **[SOTA / benchmark]** Carlos E. Jimenez et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[SOTA]** Ahmed El-Kishky et al. *Competitive Programming with Large Reasoning Models.* OpenAI, 2025. — arXiv:2502.06807
- **[SOTA / benchmark]** Naman Jain et al. *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* ICLR, 2025. — arXiv:2403.07974
- **[Benchmark]** Samuel Miserendino et al. *SWE-Lancer: Can Frontier LLMs Earn \$1 Million from Real-World Freelance Software Engineering?* OpenAI, 2025. — arXiv:2502.12115
- **[Critique]** Reem Aleithan et al. *SWE-Bench+: Enhanced Coding Benchmark for LLMs.* 2024. — arXiv:2410.06992
- **[Critique / RCT]** Joel Becker, Nate Rush, Beth Barnes, David Rein. *Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity.* METR, 2025.
- **[Measurement]** Thomas Kwa et al. *Measuring AI Ability to Complete Long Tasks.* METR, 2025. — arXiv:2503.14499
- **[Transfer evidence]** Viraat Aryabumi et al. *To Code, or Not To Code? Exploring Impact of Code in Pre-training.* 2024. — arXiv:2408.10914
- **[Training environment]** Jiayi Pan et al. *Training Software Engineering Agents and Verifiers with SWE-Gym.* 2024. — arXiv:2412.21139

## 10. Worked Example

Take a single model reported at Codeforces Elo $\approx 2700$ (top ~0.2% of human contestants) and 72% on SWE-bench Verified. Naively: near-superhuman at CP, near-expert at software.

Now decompose the 72%, using published per-instance audits as rates:

| Component | Rate | Instances (of 500) |
|---|---|---|
| Reported resolved | 0.72 | 360 |
| Minus solution-leakage instances (issue text contains the fix) | −0.10 | 310 |
| Minus weak-test false positives (patch passes, semantics wrong) | −0.07 | 275 |
| **Audited resolved** | **≈0.55** | **≈275** |

The corrections are order-of-magnitude estimates from the SWE-Bench+ audit applied to the Verified subset; the point is their size, not their precision. A ±10-point uncertainty in the *dependent variable* swamps the between-model CP spread being used as the predictor: across the five public models with both scores, Codeforces Elo ranges over ~1,000 points while audited SWE-bench Verified ranges over maybe 15 points. Fitting $\beta$ on that gives a standard error larger than the coefficient.

Then the METR RCT supplies the sign check. Its developers, using tools built on exactly these models, were 19% *slower* on their own repositories — while believing they were 20% faster. So the mapping runs: Elo 2700 → reported 72% → audited ~55% → measured throughput −19%. Every step loses information, and no step has a published error bar.

That chain is the obstruction. The transfer question is not unanswered because the experiment is expensive; it is unanswered because the dependent variable has an unquantified error rate comparable to the effect size, and the only intervention that would fix it — the three-arm ablation in §8, with the reward-shuffled control — has not been run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*