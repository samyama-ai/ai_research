---
id: 19-evaluation/human-baseline-estimation-benchmarks
title: "Human Baseline Estimation for Superhuman-Claimed Benchmarks"
topic: 19-evaluation
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Human Baseline Estimation for Superhuman-Claimed Benchmarks

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/human-baseline-estimation-benchmarks` · **Status:** methodologically-blocked

## 1. Problem Statement

A benchmark reports a model score and a "human score", and the model exceeds it. The claim "superhuman" is then made. The problem is that the human number is almost never estimated in a way that supports the comparison.

- **Input:** a task distribution $D$, an item set, a scoring function, a model $M$, and a human population $P$.
- **Output:** an estimate of human performance $\hat A_H$ with a stated population, a stated protocol, and a calibrated interval.
- **Decision predicate:** does $A_M > A_H$ hold on the *same* items, under the *same* affordances, against the *same* gold labels, with the difference outside the joint uncertainty band?

Three variants, with very different difficulty:

- **Measurement variant (the blocked one).** Define and collect $A_H$ so that the comparison is a comparison. Currently blocked: the target estimand is under-specified before any data is collected.
- **Method variant (empirically open).** Given a definition, collect it cheaply — adaptive item selection, item-response models, cross-population transfer.
- **Theory variant (partly open).** Under what noise and coverage conditions is a benchmark gap identifiable at all, and what is the minimum number of (human, item) observations for a given interval width?

Solving it means: for any benchmark on which superhumanity is claimed, a reproducible procedure returns $\hat A_H$ with a 95% interval, and the claim is stated as a signed, bounded difference on adjudicated items rather than as two numbers from two different studies.

## 2. Formal Setting

Items $x \sim D$ with true answers $y^*(x)$ and recorded labels $\tilde y(x)$. Label error rate

$$\epsilon = \Pr_{x\sim D}\big[\tilde y(x) \neq y^*(x)\big].$$

Affordance vector $c$: time limit, tool and web access, reference materials, number of attempts, aggregation rule (single response vs. majority of $k$), and prompt/instruction text. Model accuracy under affordances $c$ and labels $\tilde y$:

$$\hat A_M(c) = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}\{M_c(x_i) = \tilde y(x_i)\}.$$

Human population $P$ with per-person accuracy $\alpha_h(c) = \mathbb{E}_{x\sim D}[\mathbb{1}\{h_c(x)=\tilde y(x)\}]$. The *human baseline* is a functional of the distribution of $\alpha_h$, and which functional is chosen changes the answer by tens of points:

$$A_H^{\text{mean}} = \mathbb{E}_{h\sim P}[\alpha_h], \qquad A_H^{(q)} = F_{\alpha}^{-1}(q), \qquad A_H^{\text{maj}} = \mathbb{E}_x\big[\mathbb{1}\{\mathrm{maj}_k(h^{(1..k)}(x)) = \tilde y(x)\}\big].$$

The superhuman predicate is a *paired* quantity on shared items:

$$\Delta(c,q) = \hat A_M(c) - \hat A_H^{(q)}(c), \qquad \text{claim valid} \iff \mathrm{LB}_{95}(\Delta) > 0,$$

with the interval bootstrapped over **both** items and subjects (a crossed random-effects design; item-only bootstrap understates variance).

Label noise bounds what any estimate can mean: for any predictor, $|A^{\tilde y} - A^{y^*}| \le \epsilon$, so a $\Delta$ smaller than $\epsilon$ is not interpretable without re-adjudication.

Assumptions, with those known violated in practice marked:

1. Humans and model face the same items. **Violated** — human baselines are routinely run on a small subsample (SuperGLUE, GPQA) while models are scored on the full set.
2. $\tilde y = y^*$. **Violated** — see §4.
3. Affordances matched. **Violated** — humans get minutes and no web; models get chain-of-thought, retrieval, and best-of-$k$.
4. Items are novel to both. **Violated** — training-set contamination makes the model's $D$ not the human's $D$.
5. $P$ is exchangeable and pre-registered. **Violated** — crowdworkers, paper authors, and "95th-percentile test takers" are all used interchangeably.
6. The functional is fixed in advance. **Violated** — mean, max, and majority-vote baselines are reported without distinction.

Under violations of (1), (3) and (5), $A_H$ is **not identified**: the observed data are compatible with a range of $\Delta$ whose width is set by unmeasured protocol differences, not by sample size.

## 3. State of the Art

**Established (protocol-level).**
- Nangia & Bowman (*Human vs. Muppet*, ACL 2019) is the reference standard: crowdworkers trained on the actual task, paid, with a held-out qualification round, majority-of-5 aggregation, and per-task documentation. It produced GLUE human average 87.1 — and showed the earlier informal numbers were too low.
- METR's HCAST / RE-Bench line (2024–2025) runs human experts and agents on the *same* tasks with matched time budgets and logs time-to-completion, not just accuracy. This is currently the best-instrumented human-vs-model comparison in the field.
- GPQA (Rein et al., COLM 2024) separates two human arms by design: domain experts and non-expert validators *with unrestricted web access*, which makes the affordance dimension explicit.

**Claimed but unablated.**
- MMLU's "expert-level 89.8%" is not a measured human baseline. It is an estimate built from 95th-percentile scores of human test takers on the source exams, on different item sets, under exam conditions. Every "GPT-4 is superhuman on MMLU" claim rests on it.
- ImageNet's 5.1% top-5 human error is a **single annotator** (Karpathy) reported in Russakovsky et al. (IJCV 2015), widely cited as a species-level baseline.
- Most "human baseline" rows on leaderboards exist only as a benchmark number: no subject count, no interval, no protocol, no per-item data release. This is the modal case.

**Critique SOTA.** Bowman & Dahl (NAACL 2021), Raji et al. (*AI and the Everything in the Whole Wide World Benchmark*, NeurIPS D&B 2021), and Tedeschi et al. (*What's the Meaning of Superhuman Performance in Today's NLU?*, ACL 2023) each show the comparison fails on construct grounds; none supplies a replacement estimator.

## 4. What Is Known

- **Label noise is large enough to swallow the gap.** Northcutt et al. (NeurIPS D&B 2021) found 5.8% label errors in the ImageNet validation set (50,000 items) and 3.4% average across 10 common test sets. Gema et al. (*Are We Done with MMLU?*, NAACL 2025) hand-re-annotated ~5,700 MMLU questions across 30 subjects and estimated ~6.5% error overall, with 57% of the sampled Virology questions containing errors.
- **The choice of functional dominates.** On SuperGLUE (Wang et al., NeurIPS 2019), the human baseline of 89.8 uses trained, aggregated annotators; individual untrained crowdworkers score far lower. On WNLI in GLUE, humans reach 95.9 where majority-class is 65.1 — a 30-point spread that depends entirely on annotator training.
- **Expertise spread is enormous within one benchmark.** On MATH (Hendrycks et al., NeurIPS D&B 2021), a three-time IMO gold medalist scored 90% and a computer-science PhD student scored 40% on the same 5,000-problem test set. There is no single "human" number.
- **Non-experts with web access are far below experts.** GPQA: expert accuracy 65% (74% after retrospective correction of expert mistakes), non-expert validators with unrestricted web and >30 min/question at 34% (546 questions, hundreds of validators).
- **Sampling error is not the binding constraint.** MMLU has 14,042 test items; at $p=0.86$ the binomial standard error is 0.29 pp. Label noise (6.5 pp) is ~22× larger.
- **Time-matched comparisons behave differently from accuracy-matched ones.** METR's RE-Bench (arXiv:2411.15114) found agents ahead of human experts at short time budgets and behind at long ones on the same ML research tasks — a crossover invisible to any single accuracy number.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no agreed estimand. "Human performance" is not one quantity: the population, the functional (mean / quantile / majority-vote / best-expert), and the affordance vector must all be fixed before an estimate is even defined, and no benchmark currently pre-registers all three. Until they are fixed, $\Delta$ is not identifiable and no amount of data resolves it.
- **Empirically open.** For essentially every benchmark with a superhuman claim — MMLU, SuperGLUE, ImageNet, HellaSwag, SQuAD 2.0 — nobody has run the matched-affordance, shared-item, adjudicated-label experiment at $n \ge 500$ items and $\ge 30$ subjects. It is runnable; it costs money, not new science.
- **Theoretically open.** No characterisation exists of the minimum $(\text{items} \times \text{subjects})$ budget needed for a target interval width under a crossed random-effects model with item-level label noise $\epsilon$ and per-subject ability variance $\sigma^2$. Item-response theory (Lalor et al., EMNLP 2016; Rodriguez et al., ACL 2021) supplies the machinery; the sample-size theorem for the *difference* estimand has not been written.
- **Open:** whether contamination-adjusted model accuracy can be point-identified at all when the training corpus is unreleased.

## 6. Why It Is Hard

Three specific obstructions, in order of severity:

1. **Non-identifiability from protocol mismatch.** Human and model scores are generated by different data-generating processes (different items, different time, different tools, different aggregation). The difference of two such numbers is not an estimate of anything; there is no estimand it converges to as $n \to \infty$. This is a structural failure, not a precision failure.
2. **Absent ground truth.** With $\epsilon \approx 6\%$ on MMLU and $\approx 6\%$ on ImageNet, both arms are scored against labels that are wrong often enough to flip the sign of a 3-point gap. Fixing this requires re-adjudication by domain experts — expensive, and on frontier benchmarks (FrontierMath, HLE) the adjudicators are the scarce resource.
3. **Ceiling compression plus selection.** Benchmarks near saturation have most of their remaining signal in their noisiest items. Selecting the humans who can do those items ("experts") is the same act as selecting the items, so ability and difficulty are confounded — the classic IRT non-separability, made worse because the human sample is typically $n \le 10$.

Cost is real but secondary: 30 subjects × 100 items at expert rates is roughly $30$–$80$k, not a compute-scale barrier.

## 7. Current Research (as of 2026)

- **METR** — time-horizon methodology (Kwa et al., *Measuring AI Ability to Complete Long Tasks*, arXiv:2503.14499; HCAST, arXiv:2503.17354). Replaces accuracy comparison with "the human time budget at which the agent succeeds 50% of the time". This sidesteps the functional problem by changing the estimand to a duration.
- **Benchmark re-adjudication** — MMLU-Redux (Gema et al., NAACL 2025), Cleanlab-style label auditing, PlatinumBench-style manually verified subsets. Directly attacks obstruction (2).
- **Uncertainty reporting** — Miller, *Adding Error Bars to Evals* (Anthropic, arXiv:2411.00640) pushed clustered standard errors and paired tests into eval practice; adoption of the *subject-side* random effect remains rare. *(frontier — verify: whether any 2026 major model card reports a crossed human/item interval.)*
- **Expert-panel benchmarks with published human data** — ARC-AGI-2 (Chollet et al., arXiv:2505.11831) reports per-task human panel results rather than a single ceiling; GPQA and FrontierMath publish expert-arm details. This is the direction that makes §8 cheap.
- *(frontier — verify)* Item-response-theory-based adaptive human baselining, which would cut subject-hours by selecting maximally informative items per subject.

## 8. Concrete Next Experiment

**Question:** is the reported MMLU superhuman gap real once items, labels, and affordances are matched?

- **Scale.** 600 items stratified across 10 MMLU subjects, drawn from the MMLU-Redux re-annotated pool; each item re-adjudicated to consensus gold by 3 domain experts with a fourth as tiebreak. 40 human subjects with a pre-registered qualification (graduate-level coursework in the subject), each answering 150 items in a Latin-square design so every item gets 10 independent human responses. Total: 6,000 human item-responses.
- **Arms.**
  - *Human, matched:* closed-book, 90 s/item median, single attempt.
  - *Human, tooled (control arm):* unrestricted web, no time limit. Isolates the affordance term.
  - *Model, matched:* single sample, temperature 0, no tools, no chain-of-thought beyond the human's think-time analogue.
  - *Model, tooled:* retrieval + best-of-8. Isolates the same term on the model side.
  - *Label control:* every arm scored twice — once against original MMLU labels, once against adjudicated gold. The difference between these two scorings is the $\epsilon$ correction, measured rather than assumed.
- **Deciding number.** $\mathrm{LB}_{95}\big(\hat A_M^{\text{matched}} - \hat A_H^{(0.95),\text{matched}}\big)$ on adjudicated gold, from a bootstrap crossed over items and subjects. **The claim survives only if this lower bound exceeds 0.** Target interval half-width ≤ 2 pp, which 600 items × 10 responses supports for per-subject ability SD up to ~8 pp.
- **Secondary readout:** the shift in $\Delta$ between original and adjudicated labels. If that shift is larger than $\Delta$ itself, every prior MMLU superhuman claim is uninterpretable, independent of this experiment's sign.

## 9. Key References

- **[Foundational]** Nangia, N., Bowman, S. R. *Human vs. Muppet: A Conservative Estimate of Human Performance on the GLUE Benchmark.* ACL, 2019. — arXiv:1905.10425
- **[Foundational]** Wang, A., Pruksachatkun, Y., Nangia, N., Singh, A., Michael, J., Hill, F., Levy, O., Bowman, S. R. *SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems.* NeurIPS, 2019. — arXiv:1905.00537
- **[Foundational]** Russakovsky, O., et al. *ImageNet Large Scale Visual Recognition Challenge.* IJCV, 2015. — arXiv:1409.0575
- **[SOTA]** Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., Bowman, S. R. *GPQA: A Graduate-Level Google-Proof Q&A Benchmark.* COLM, 2024. — arXiv:2311.12022
- **[SOTA]** Wijk, H., et al. *RE-Bench: Evaluating Frontier AI R&D Capabilities of Language Model Agents Against Human Experts.* METR, 2024. — arXiv:2411.15114
- **[SOTA]** Kwa, T., et al. *Measuring AI Ability to Complete Long Tasks.* METR, 2025. — arXiv:2503.14499
- **[SOTA]** Gema, A. P., et al. *Are We Done with MMLU?* NAACL, 2025. — arXiv:2406.04127
- **[SOTA]** Northcutt, C. G., Athalye, A., Mueller, J. *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2103.14749
- **[SOTA]** Shankar, V., Roelofs, R., Mania, H., Fang, A., Recht, B., Schmidt, L. *Evaluating Machine Accuracy on ImageNet.* ICML, 2020.
- **[Method]** Miller, E. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* Anthropic, 2024. — arXiv:2411.00640
- **[Method]** Rodriguez, P., Barrow, J., Hoyle, A., Lalor, J. P., Jia, R., Boyd-Graber, J. *Evaluation Examples Are Not Equally Informative: How Should That Change NLP Leaderboards?* ACL, 2021.
- **[Survey]** Bowman, S. R., Dahl, G. E. *What Will It Take to Fix Benchmarking in Natural Language Understanding?* NAACL, 2021. — arXiv:2104.02145
- **[Survey]** Raji, I. D., Bender, E. M., Paullada, A., Denton, E., Hanna, A. *AI and the Everything in the Whole Wide World Benchmark.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2111.15366
- **[Survey]** Tedeschi, S., Bos, J., Declerck, T., Hajič, J., Hershcovich, D., Hovy, E., Koller, A., Krek, S., Schockaert, S., Sennrich, R., Shutova, E., Navigli, R. *What's the Meaning of Superhuman Performance in Today's NLU?* ACL, 2023.
- **[Context]** Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., Steinhardt, J. *Measuring Massive Multitask Language Understanding.* ICLR, 2021. — arXiv:2009.03300
- **[Context]** Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., Steinhardt, J. *Measuring Mathematical Problem Solving With the MATH Dataset.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2103.03874

## 10. Worked Example

**Instance: "GPT-4 at 86.4 on MMLU beats the 89.8 human expert baseline" — the arithmetic.**

Take the two published numbers at face value: model 86.4, "expert-level" 89.8. The reported gap is $-3.4$ pp, so the strict reading is *not* superhuman; later models reported above 89.8 make the gap positive by 1–4 pp. Now price each error source.

| Source | Magnitude on MMLU | Direction |
|---|---|---|
| Binomial sampling error, $n=14{,}042$, $p=0.86$ | 0.29 pp (1 SE) | symmetric |
| Label error $\epsilon$ (Gema et al., 30 subjects) | 6.5 pp | bounds interpretability |
| Population functional: mean crowdworker (34.5) vs. 95th-pct exam taker (89.8) | 55.3 pp | choice-dependent |
| Item-set mismatch: human number from source exams, not MMLU items | unmeasured | unbounded |
| Affordance mismatch: exam conditions vs. CoT + best-of-$k$ | unmeasured | favours model |

The decisive line is the third and fourth. The 89.8 figure is a 95th-percentile score of a different population on a different item set under different conditions. Even granting it, the 6.5 pp label-error band is 22× the sampling SE and roughly 2× the largest claimed gap.

Concretely: suppose 6.5% of items have wrong labels, and on those items the model — trained on the same web text the labels came from — reproduces the erroneous label 60% of the time while a domain expert answers correctly. Then on the 913 mislabelled items the model gains $0.6 \times 913 = 548$ spurious credits and the expert gains 0, a swing of $548/14{,}042 = 3.9$ pp in the model's favour. That single, plausible, unmeasured effect is larger than any superhuman gap reported on MMLU.

**The obstruction made visible:** the experiment has not been run that would tell you the sign of $\Delta$. It is not that the interval is wide — it is that the two numbers being subtracted estimate different quantities, so there is no interval to widen. That is why the status here is *methodologically blocked* and not merely *empirically open*.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*