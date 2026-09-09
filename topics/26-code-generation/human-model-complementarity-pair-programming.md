---
id: 26-code-generation/human-model-complementarity-pair-programming
title: "Human-Model Complementarity in Pair Programming Throughput"
topic: 26-code-generation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Human-Model Complementarity in Pair Programming Throughput

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/human-model-complementarity-pair-programming` · **Status:** empirically-open

## 1. Problem Statement

A developer and a code model work the same task together. The question is whether the pair beats the better of its two parts, and if so on which tasks.

- **Input:** a task distribution $\mathcal{T}$ (issues, features, refactors), a developer population $\mathcal{H}$, a model $M$ with an interaction harness (inline completion, chat, or agentic loop).
- **Output:** a scalar **complementarity gap** $\Delta$ and its sign, plus a per-task predictor of that sign.
- **Decision predicate:** is $\Delta > 0$ for some non-trivial subset of $\mathcal{T}$, and can that subset be identified *before* the task is attempted?

Three variants, of very different difficulty:

- **Measurement variant.** Define a throughput metric that is quality-adjusted, not gameable by generated line count, and estimable within a realistic sample. Currently the weakest link.
- **Method variant.** Build an allocation or interaction policy that realizes positive $\Delta$ — routing, deferral, confidence-gated suggestion, interface design.
- **Theory variant.** State conditions on the human's and model's error distributions under which a team can strictly dominate both members. Complementarity requires *conditional* error independence plus a usable signal for who is right; both are assumptions, not facts.

Solving it means: a preregistered field experiment showing $\Delta > 0$ with quality held fixed, replicated across at least two independent organizations, plus a task-level predictor that beats the marginal rate.

## 2. Formal Setting

Task $t \sim \mathcal{T}$, developer $h \sim \mathcal{H}$. Let $Y \in \{0,1\}$ indicate the task is completed to spec and $C > 0$ the wall-clock cost in developer-hours. Define quality-adjusted throughput for arm $a \in \{H, M, HM\}$:

$$T_a = \mathbb{E}_{t,h}\!\left[\frac{Y_a(t,h)\,\cdot\,q_a(t,h)}{C_a(t,h)}\right]$$

where $q \in [0,1]$ is a quality weight. **As measured:** $Y$ is "PR merged by a reviewer blind to arm"; $C$ is IDE-active time from editor telemetry plus attributed review time, not self-report; $q$ is $1 - \hat{d}$ where $\hat{d}$ is the defect-escape rate over a fixed 90-day post-merge window (reverts, hotfixes, linked bug issues per merged PR).

Complementarity gap:

$$\Delta = T_{HM} - \max(T_H,\; T_M)$$

$T_M$ is the *autonomous* arm — model given the issue, no human in the loop, PR judged by the same blind reviewer. Almost no published study measures $T_M$; without it $\Delta$ is undefined and only the weaker $T_{HM} - T_H > 0$ is being tested.

**Oracle routing bound.** With a per-task allocator $\pi: t \mapsto \{H, M, HM\}$,

$$T^\star = \mathbb{E}_t\big[\max_a T_a(t)\big] \;\ge\; \max_a T_a$$

and the *routable* headroom is $T^\star - \max_a T_a$. This is the quantity worth chasing: it is positive whenever the arms' per-task advantages cross, even when the marginal $\Delta \le 0$.

**Interaction accounting.** Following the CUPS formulation (Mozannar et al., CHI 2024), a session decomposes into states $s \in \{\text{write}, \text{prompt}, \text{verify suggestion}, \text{edit suggestion}, \text{debug}, \text{think}\}$ with dwell times $\tau_s$. Then $C_{HM} = \sum_s \tau_s$, and the model helps only if the reduction in $\tau_{\text{write}}$ exceeds the sum $\tau_{\text{verify}} + \tau_{\text{edit}}$ it creates. Acceptance rate is *not* a proxy for this; an accepted suggestion that is later rewritten has negative value.

**Assumptions, with the violated ones flagged:**

1. Tasks are i.i.d. draws — **violated**: real backlogs are ordered, and easy tasks are pulled first.
2. Quality is observable within the window — **violated**: security and architectural defects surface far beyond 90 days (Perry et al., CCS 2023, found assistant users wrote more insecure code while believing it more secure).
3. No learning or Hawthorne effect — **violated**: METR (2025) found developers' expectations diverged from their measured outcome by ~40 percentage points, and skill with the tool changes over weeks.
4. $C$ is fungible across arms — **violated**: review of model output loads different cognitive resources than authoring, so hours are not exchangeable.

## 3. State of the Art

**Established (randomized, published):**

- Peng, Kalliamvakou, Cihon, Demirer (2023): a controlled experiment, $n = 95$ recruited developers, single task (HTTP server in JavaScript). Copilot group finished **55.8% faster** (1.61 h vs 2.41 h). Real and clean, but one greenfield task with an unambiguous spec — the far tail of $\mathcal{T}$, and no quality-adjusted endpoint.
- METR (Becker et al., 2025): 16 experienced open-source maintainers, 246 real issues on repos they already owned, randomized per issue. AI-allowed tasks took **19% longer**. Developers *forecast* 24% speedup and *reported* 20% speedup after the fact. This is currently the strongest evidence against naive complementarity on mature codebases.

**Claimed but unablated:**

- Cui et al. (2024), three field experiments at Microsoft, Accenture, and a Fortune 100 firm, ~4,800 developers: ~**26%** more completed pull requests for the treated group. The endpoint is PR count, not quality-adjusted throughput; the mechanism is not ablated, and PR count is exactly the metric most sensitive to task splitting.
- Vendor and industry dashboards (DORA reports, GitClear code-churn analyses) report directionally opposed results — self-reported productivity up, delivery throughput and code duplication worse. Not randomized; treat as descriptive only.

**Benchmark-only numbers.** SWE-bench Verified pass rates (agentic systems in the 70–80% range through 2025–2026) measure $T_M$ on a curated Python issue set with hidden tests available as ground truth. They do not transfer to $T_M$ as defined above: the tasks are pre-filtered for solvability and the acceptance oracle does not exist in production.

**Cross-domain SOTA on the general question.** Vaccaro, Almaatouq, Malone (*Nature Human Behaviour*, 2024), meta-analysis of 106 experiments: human–AI combinations on average performed **worse than the best of human alone or AI alone** ($\Delta < 0$), with positive $\Delta$ concentrated in creation tasks and negative in decision tasks. Code work spans both.

## 4. What Is Known

- Effect sizes are not merely uncertain, they have **opposite signs** across credible randomized studies: $+55.8\%$ (Peng et al., $n=95$, one synthetic task) versus $-19\%$ (METR, $n=16$ developers / 246 tasks, mature repos, >1M LOC, high prior familiarity).
- **Self-report is unreliable at scale.** METR: participants reported a 20% speedup on tasks where they were measured 19% slower. Any study whose endpoint is a survey is measuring belief, not throughput.
- **Acceptance ≠ retention.** Telemetry work on assistant use finds a substantial fraction of accepted completions are subsequently edited or deleted; CUPS-style time accounting (Mozannar et al., CHI 2024) shows verification time is a first-order cost, not a rounding error.
- **Task-type moderation is real.** Greenfield, well-specified, boilerplate-heavy tasks show the largest gains; work on large, unfamiliar-to-the-model, high-context codebases shows the smallest or negative gains.
- **Quality is not free.** Perry et al. (CCS 2023, $n=47$) found assistant users produced less secure code across several tasks while reporting higher confidence in its security.
- **The complementarity prior is bad.** Bansal et al. (CHI 2021) showed AI explanations increase human acceptance of AI answers whether or not the AI is right — improving team accuracy less than the AI alone in several conditions.

## 5. What Is Not Known

- **Methodologically blocked (the binding constraint).** There is no agreed, non-gameable, quality-adjusted throughput metric. PR count, lines shipped, and story points all move under task-splitting; defect escape needs a long window; "developer-hours" is not comparable across authoring and reviewing. Until $T_a$ is pinned down, $\Delta$ cannot be estimated, only argued about.
- **Empirically open.** No published study measures all three arms — $T_H$, $T_M$ (autonomous), $T_{HM}$ — on the same task distribution with the same blind acceptance criterion. So the headline quantity $\Delta$ has never been computed for software engineering. Runnable today; nobody has run it.
- **Empirically open.** The routable headroom $T^\star - \max_a T_a$ is unmeasured. Nor is it known whether a pre-task predictor of the right arm exists that beats the base rate.
- **Empirically open.** Duration. Every randomized study is days to weeks; the sign of $\Delta$ after 12 months of tool-adapted workflow, and after model-authored code accumulates in the repo, is unmeasured.
- **Theoretically open.** No characterization of when a human–model team strictly dominates both members for *sequential, partially observable* tasks. Existing learning-to-defer theory (Mozannar & Sontag, ICML 2020) covers one-shot classification with a clean loss; program synthesis with iterative repair does not fit that model.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by an absent $T_M$ baseline**.

- Every field study measures $T_{HM} - T_H$ and calls it complementarity. It is not: if the model alone would ship the same PR unaided, the observed gain is *substitution*, not complementarity, and the correct policy is to remove the human, not to pair.
- The confounder is task selection. Developers with an assistant choose different tasks, split PRs differently, and accept different scopes. Randomizing the *tool* does not randomize the *task*, so the treatment effect on PR count absorbs a scope change of unknown size.
- Ground truth for quality is absent at the horizon that matters. A 90-day defect window catches crashes and misses architectural debt and latent CVEs — the failure modes assistant-heavy code is most suspected of.
- Non-identifiability: $\tau_{\text{verify}}$ (time spent checking a suggestion) and $\tau_{\text{think}}$ (time spent reasoning about the problem) are not separable from keystroke telemetry alone. They have opposite signs in the throughput ledger.
- Statistical power. With per-task variance in completion time typically exceeding the mean, detecting a 10% effect on real issues needs hundreds of tasks per arm; METR needed 246 tasks to resolve 19%.

## 7. Current Research (as of 2026)

- **METR** is extending the 2025 RCT design to longer horizons and more repositories, and separately publishing the time-horizon scaling of autonomous agent capability — the natural source of a credible $T_M$ curve.
- **Microsoft Research / GitHub Next** continue telemetry-based interaction modeling (CUPS lineage, Mozannar and colleagues) and large-scale field deployment analysis (Cui, Demirer, Peng and coauthors).
- **DORA / Google Cloud** run the largest observational panel linking AI adoption to delivery metrics; observational, but the only source with throughput and stability measured together.
- **Economics of AI groups (MIT, Harvard, Stanford Digital Economy Lab)** are extending the "jagged frontier" framing (Dell'Acqua et al., HBS WP, 2023) toward task-level ability-crossing, which is exactly the routable-headroom question. *(frontier — verify)*
- **Agentic-loop vendors** are shifting the interaction unit from completion to delegated task, which changes $C_{HM}$ from authoring-plus-verification to specification-plus-review. No randomized measurement of this regime exists yet. *(frontier — verify)*

## 8. Concrete Next Experiment

**The three-arm blind-review issue trial.**

- **Scale.** 2 organizations, 60 developers, 900 real backlog issues (450 per org), randomized at the issue level, 300 issues per arm. Powered for a 10% difference in median completion time at $\alpha = 0.05$ given a log-normal duration model with $\sigma \approx 1.0$.
- **Arms.**
  1. $H$ — **control arm**: developer, no model assistance, tool use logged and enforced.
  2. $M$ — agent given the issue text and repo, no human intervention; the resulting PR goes straight to review.
  3. $HM$ — developer with unrestricted model assistance.
- **Instrumentation.** IDE-active time (not wall clock), CUPS state labeling on a 10% subsample via screen recording, and blind review: reviewers see the diff with no arm label and no telltale commit metadata.
- **Endpoint.** 90-day defect-escape tracking on every merged PR.
- **The deciding number.** $\hat{\Delta} = \hat{T}_{HM} - \max(\hat{T}_H, \hat{T}_M)$ with a bootstrap 95% CI. If the CI excludes 0 from below, complementarity is real at the margin. If it does not, report the routable headroom $\hat{T}^\star - \max_a \hat{T}_a$ from the per-issue arm outcomes — a positive headroom with a null $\hat{\Delta}$ is the informative result, and it says the problem is allocation, not capability.
- **Cost.** Roughly 6 person-months of developer time per arm plus agent inference; feasible for one mid-size engineering org over two quarters.

## 9. Key References

- **[Foundational]** Sida Peng, Eirini Kalliamvakou, Peter Cihon, Mert Demirer. *The Impact of AI on Developer Productivity: Evidence from GitHub Copilot.* 2023. — arXiv:2302.06590
- **[SOTA]** Joel Becker, Nate Rush, Beth Barnes, David Rein (METR). *Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity.* METR technical report, 2025.
- **[SOTA]** Zheyuan (Kevin) Cui, Mert Demirer, Sonia Jaffe, Leon Musolff, Sida Peng, Tobias Salz. *The Effects of Generative AI on High-Skilled Work: Evidence from Three Field Experiments with Software Developers.* Working paper, 2024.
- **[Foundational]** Michelle Vaccaro, Abdullah Almaatouq, Thomas Malone. *When Combinations of Humans and AI Are Useful: A Systematic Review and Meta-Analysis.* Nature Human Behaviour, 2024.
- **[Foundational]** Gagan Bansal, Tongshuang Wu, Joyce Zhou, Raymond Fok, Besmira Nushi, Ece Kamar, Marco Tulio Ribeiro, Daniel S. Weld. *Does the Whole Exceed Its Parts? The Effect of AI Explanations on Complementary Team Performance.* CHI 2021.
- **[Method]** Hussein Mozannar, Gagan Bansal, Adam Fourney, Eric Horvitz. *Reading Between the Lines: Modeling User Behavior and Costs in AI-Assisted Programming.* CHI 2024.
- **[Theory]** Hussein Mozannar, David Sontag. *Consistent Estimators for Learning to Defer to an Expert.* ICML 2020.
- **[Quality]** Neil Perry, Megha Srivastava, Deepak Kumar, Dan Boneh. *Do Users Write More Insecure Code with AI Assistants?* ACM CCS 2023.
- **[Qualitative]** Shraddha Barke, Michael B. James, Nadia Polikarpova. *Grounded Copilot: How Programmers Interact with Code-Generating Models.* OOPSLA 2023.
- **[Survey]** Mark Steyvers, Aakriti Kumar. *Three Challenges for AI-Assisted Decision-Making.* Perspectives on Psychological Science, 2024.
- **[Context]** Fabrizio Dell'Acqua et al. *Navigating the Jagged Technological Frontier.* Harvard Business School Working Paper 24-013, 2023.

## 10. Worked Example

One issue, three arms, real magnitudes drawn from the studies above.

Task: add a rate-limiting middleware to a 400k-LOC Python service, with tests.

| Arm | $C$ (dev-hours) | $Y$ (merged) | $q$ (1 − defect escape) | $T = Yq/C$ |
|---|---|---|---|---|
| $H$ | 4.0 | 1 | 0.95 | 0.238 |
| $M$ (autonomous) | 0.4 review-hours | 0.55 | 0.85 | 1.169 |
| $HM$ | 3.4 | 1 | 0.90 | 0.265 |

Read naively: $T_{HM} - T_H = +0.027$, an 11% gain — the number a field study would publish. But $T_M = 1.169$ dominates both, because the autonomous arm consumes only review time and succeeds 55% of the time. So

$$\Delta = T_{HM} - \max(T_H, T_M) = 0.265 - 1.169 = -0.904 < 0.$$

The pair is worse than the better part on this task. Now the obstruction becomes visible in three places.

1. **The $M$ arm's cost is not comparable.** Its 0.4 hours are review hours; the human arm's 4.0 are authoring hours. Assumption 4 is violated, and the ratio $T_M/T_H \approx 4.9$ is an artifact of that incomparability as much as of capability. Fixing it needs a conversion factor nobody has measured.
2. **The 45% failure mass has no cost assigned.** When the agent fails, someone still has to do the task. Charging those failures back at $H$'s rate gives $C_M = 0.4 + 0.45 \times 4.0 = 2.2$ and $T_M = 0.55 \times 0.85 / 2.2 = 0.213$ — now *below* $T_H$, and $\Delta$ flips positive to $+0.052$. **The sign of the answer is determined entirely by an accounting choice that no published study states.**
3. **$q$ is the least trustworthy column.** A 5-point difference in escape rate between arms (0.95 vs 0.90) is roughly 1 extra escaped defect per 20 PRs; distinguishing that from noise needs on the order of 300 PRs per arm, which is precisely the scale of §8 and larger than every randomized study run to date except Cui et al., which did not measure $q$ at all.

The example is not evidence about the real sign of $\Delta$. It is evidence that with today's measurement conventions, two defensible accounting choices on the same data produce opposite conclusions — which is why this problem is classified methodologically blocked before it is empirically open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*