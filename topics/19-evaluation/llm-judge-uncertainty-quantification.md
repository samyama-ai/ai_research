---
id: 19-evaluation/llm-judge-uncertainty-quantification
title: "Uncertainty Quantification for LLM-Judge Aggregate Scores"
topic: 19-evaluation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Uncertainty Quantification for LLM-Judge Aggregate Scores

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/llm-judge-uncertainty-quantification` · **Status:** partially-solved

## 1. Problem Statement

An LLM judge scores a model's outputs on a benchmark and reports an aggregate: a win rate, a mean 1–10 rating, an Elo. The number is then used for a decision — ship or don't, model A over model B. The question is what error bar belongs on that number, where "error" covers everything that would move it under a re-run that should not have moved it.

Three variants, with different difficulty:

- **Measurement variant.** Given a fixed judge, a fixed prompt and a fixed item set, estimate the sampling uncertainty of the aggregate. Largely solved: bootstrap over items, clustered standard errors, paired differences.
- **Method variant.** Produce an interval for the *human-consensus* quantity the judge is a proxy for, using few human labels. Partially solved by prediction-powered inference (PPI); the open part is how much it actually buys at realistic judge accuracy.
- **Theory variant.** Characterise when a judge-based interval can be valid at all, given that judge error is correlated with the property being scored (verbosity, style, self-preference). Open.

Solving it means: an interval $[\hat\theta_L,\hat\theta_R]$ with stated coverage of the human-consensus target $\theta^\star$, whose coverage is empirically verified on held-out human labels across judges, prompts and model pairs — not just on the item-resampling distribution.

## 2. Formal Setting

Items $i=1,\dots,n$ drawn i.i.d. from a task distribution $P_X$. For candidate model $M$, output $A_i \sim M(\cdot\mid x_i)$. A judge $J$ with prompt template $t$ and decoding temperature $\tau$ returns a score $f_{J,t}(x_i,A_i) \in \mathcal{S}$, where $\mathcal{S}=\{0,1\}$ (pairwise win), $\{1,\dots,10\}$ (Likert), or $[0,1]$ (probability read off logits).

Target:
$$\theta^\star = \mathbb{E}_{x\sim P_X,\,A\sim M}\big[\bar Y(x,A)\big],\qquad \bar Y(x,A)=\mathbb{E}_{h\sim P_H}\big[Y_h(x,A)\big]$$
with $Y_h$ the label of human annotator $h$ from population $P_H$. $\theta^\star$ is *defined by* $P_H$: change the rater pool and the estimand changes.

Judge estimate: $\hat\theta_J=\frac1n\sum_i f_{J,t}(x_i,A_i)$. Decompose
$$\hat\theta_J-\theta^\star = \underbrace{\tfrac1n\textstyle\sum_i(\bar Y_i-\theta^\star)}_{\text{item sampling}} + \underbrace{\tfrac1n\textstyle\sum_i(\mathbb{E}_\tau f_i-\bar Y_i)}_{b(J,t)\ \text{judge bias}} + \underbrace{\tfrac1n\textstyle\sum_i(f_i-\mathbb{E}_\tau f_i)}_{\text{decoding noise}}.$$

Measured as: item sampling by nonparametric bootstrap over $i$; decoding noise by $R$ independent judge calls per item at the deployed $\tau$; **prompt/design variance** by evaluating over a set $T$ of admissible templates and position orders, $\sigma^2_{\text{design}}=\mathrm{Var}_{t\in T}(\hat\theta_{J,t})$; judge bias only against human labels on a subsample $\mathcal{L}$, $|\mathcal{L}|=m\ll n$.

PPI estimator (Angelopoulos et al., *Science* 2023):
$$\hat\theta_{\mathrm{PPI}}=\frac1n\sum_{i=1}^n f_i \;-\; \frac{\lambda}{m}\sum_{i\in\mathcal{L}}(f_i-Y_i),\qquad
\mathrm{Var}\approx \frac{\lambda^2\sigma_f^2}{n}+\frac{\lambda^2\sigma^2_{f-Y}}{m},$$
with $\lambda$ tuned (PPI++). The interval is valid for $\theta^\star$ for any $f$; its *width* depends entirely on $\sigma^2_{f-Y}$.

Assumptions and their status:

| Assumption | Status in practice |
|---|---|
| Items i.i.d. | **Violated** — benchmarks are clustered by source document, task family, prompt seed. |
| Judge calls independent across items | **Violated** — a shared prompt template induces a common bias term $b(J,t)$ that no item-bootstrap sees. |
| $\bar Y$ well defined | **Weakly violated** — human agreement on MT-Bench-style pairwise judgments runs ~80%, so $\theta^\star$ is rater-pool-dependent. |
| $\mathcal{L}$ drawn uniformly from the same $P_X$ | Usually **satisfied by construction**, but often ignored (labels come from a convenience subset). |
| $n$ large enough for the CLT | **Violated** at $n<300$ with skewed per-item scores (Bowyer et al., ICML 2025). |

## 3. State of the Art

**Established.**
- Item-level bootstrap / clustered SEs for benchmark means and paired model differences; Miller (2024, Anthropic) formalises the recipe: cluster by question group, use the paired difference $\hat\theta_A-\hat\theta_B$ on shared items, and report the SE of that difference rather than of each arm.
- PPI (*Science* 2023), PPI++ , cross-prediction (Zrnic & Candès, *PNAS* 2024), stratified PPI for hybrid LM evaluation (Fisch et al., NeurIPS 2024). These give **provably valid** intervals for $\theta^\star$ regardless of judge quality. Validity is a theorem; the efficiency gain is not.
- Chatbot Arena reports bootstrap CIs on Bradley–Terry coefficients (Chiang et al., ICML 2024) — this is the measurement variant done properly for a ranking.
- Chaganty, Mussmann & Liang (ACL 2018) proved the control-variate framing for automatic NLG metrics and showed the gain is capped by metric–human correlation. The modern PPI literature rediscovers this bound.

**Claimed but unablated.**
- Judge self-reported confidence, verbalised probability, or token logprobs used as a per-item uncertainty signal. Calibration is reported on the judge's own agreement task, not propagated to a coverage-checked aggregate interval.
- Ensembling judges (multi-model panels) to "reduce variance". Reduces decoding noise; there is no published evidence it reduces the shared $b(J,t)$ term, which is the dominant one.
- Bayesian win-rate calibration with an explicit judge-error model (Gao et al., EMNLP 2024) — a real method, but coverage is validated on a small number of model pairs.

**Benchmark-number-only.** Length-controlled AlpacaEval (Dubois et al., 2024) raises Spearman correlation with Chatbot Arena from 0.94 to 0.98 on ~20 models. That is a point estimate on a tiny model sample with no interval; it is evidence about bias correction, not about uncertainty.

## 4. What Is Known

- **Position bias is large.** Wang et al. (ACL 2024) report GPT-4 pairwise verdicts flipping on a substantial fraction of pairs under candidate-order swap; on their evaluation sets order alone moves aggregate win rates by several points — larger than typical bootstrap CIs at $n\approx 500$ ($\pm 4$ points).
- **Judge–human agreement ceiling.** GPT-4 agrees with human pairwise preference at ~80–85% on MT-Bench, about the human–human rate (Zheng et al., NeurIPS D&B 2023, $n\approx 3{,}000$ judgments). So $\sigma^2_{f-Y}$ is not small.
- **Debiasing has a hard ceiling.** Dorner et al. (ICLR 2025, "Limits to scalable evaluation at the frontier") show that under judge error correlated with the target, using a judge plus ground-truth labels cannot beat roughly **twice** the number of ground-truth labels used alone. Combined with Dorner & Hardt (ICML 2024) on label budgets, this bounds what any PPI-style scheme can deliver.
- **Self-preference is real.** Panickssery et al. (NeurIPS 2024) show LLM judges score their own generations higher, with the effect tracking the judge's ability to recognise its own text — a bias that is systematic per judge, so invisible to item bootstrap.
- **Small-$n$ CLT failure.** Bowyer, Aitchison & Kaddour (ICML 2025) show CLT intervals undercover on eval sets below a few hundred items; Bayesian/exact alternatives are needed.
- **Judge scores are low-entropy and clumped.** Stureborg et al. (2024) find Likert judges concentrate on 2–3 values, inflating apparent agreement and deflating item variance.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted definition of the *design* reference class $T$. Prompt wording, rubric, order, temperature and judge version all move the aggregate, but nobody has specified which perturbations the interval should cover. Without $T$, "the uncertainty of an LLM-judge score" is not a well-posed quantity — this is the core blockage.
- **Empirically open.** Whether PPI-style estimators deliver a useful width reduction at realistic judge accuracy on real leaderboards. The experiment is cheap; the systematic sweep over judges × tasks × $m$ with coverage checked against held-out humans has not been published at scale.
- **Empirically open.** Whether judge-emitted confidence carries information *beyond* the score itself for aggregate interval construction.
- **Theoretically open.** Sharp minimax rates for estimating $\theta^\star$ from $n$ judge calls and $m$ labels when judge bias is an arbitrary function of $x$ with bounded correlation to $Y$. Dorner et al. give a factor-2 style bound under specific conditions; the general characterisation is unproven.
- **Theoretically open.** Identifiability: given only judge scores and $m$ labels, $b(J,t)$ and true model quality are separable only under assumptions no one has tested.

## 6. Why It Is Hard

**Confounded measurement plus non-identifiability, not compute.** Judge error is not noise — it is a function of the same output features the score is supposed to reward. A verbose, confidently-worded answer raises both the judge's score and the judge's error in the same direction. So the error term does not average away over items, and the item-resampling bootstrap — the one estimator everyone runs — is provably blind to it: it estimates $\mathrm{Var}(\frac1n\sum \bar Y_i)$, not $b(J,t)$.

Second: the sample size that matters is $|T|$ and the number of judges, both typically **1**. An interval computed over $n=800$ items has $n=1$ prompt. Third: correcting the bias needs human labels, and Dorner et al.'s bound says those labels buy at most a factor ~2 over using them directly — so the label budget, not the judge, sets the achievable precision.

## 7. Current Research (as of 2026)

- **PPI for evaluation.** Berkeley (Angelopoulos, Jordan, Zrnic) and Google DeepMind (Fisch et al.) on stratified and cross-prediction variants; post-hoc regression rectifiers with very small $m$ (Eyre & Madras, 2024).
- **Judge-bias correction as estimand redefinition.** Length-controlled and style-controlled regressions (Tatsu Lab; LMArena style control). *(frontier — verify)* Style control in Arena is applied to the BT regression, so its CIs cover the style-controlled estimand, not the raw one.
- **Eval statistics hygiene.** Anthropic's error-bar note, Bowyer et al.'s small-$n$ position paper, tinyBenchmarks-style IRT subsampling (Polo et al., ICML 2024) which reports its own approximation error.
- **Judge robustness audits.** Thakur et al. (2024) "Judging the Judges"; ongoing work on adversarial/prompt-sensitivity stress tests.
- **Under-explored:** treating prompt template as a random effect and reporting a variance component for it. No standard tooling. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Do published LLM-judge intervals cover the human-consensus target, and does PPI meaningfully narrow them at realistic judge accuracy?

**Scale.** 12 model pairs on an AlpacaEval-style instruction set. $n=1{,}000$ items per pair judged. Human labels on $m=200$ items per pair, 3 annotators each, majority vote — 7,200 human judgments, roughly 120 annotator-hours. Judge grid: 3 judge models × 4 prompt templates × 2 candidate orders = 24 configurations per pair (~288k judge calls; under \$3k at 2026 pricing).

**Arms.**
- *Control:* item bootstrap CI on one judge, one template — the current standard practice.
- *Arm A:* CI widened by the measured design variance $\sigma^2_{\text{design}}$ over the 24 configurations.
- *Arm B:* PPI++ using the $m=200$ labels.
- *Reference:* human-only interval from the same 200 labels.

**Deciding number.** Empirical coverage of $\theta^\star$ (estimated from the full human labels, held out per pair via cross-fitting) across the 12 pairs, at nominal 95%. **If control coverage is below 80%, item bootstrap is inadequate and must be reported as such.** Secondary number: median width ratio $W_{\text{PPI}}/W_{\text{human-only}}$. If that ratio exceeds 0.71 ($=1/\sqrt{2}$), PPI is not beating "just collect twice the labels", consistent with Dorner et al.'s bound.

## 9. Key References

- **[Foundational]** Chaganty, Mussmann & Liang. *The Price of Debiasing Automatic Metrics in Natural Language Evaluation.* ACL, 2018. — arXiv:1807.02202
- **[Foundational]** Zheng, Chiang, Sheng, Zhuang, Wu, Zhuang, Lin, Li, Li, Xing, Zhang, Gonzalez & Stoica. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2306.05685
- **[SOTA]** Angelopoulos, Bates, Fannjiang, Jordan & Zrnic. *Prediction-Powered Inference.* Science 382(6671), 2023. — arXiv:2301.09633
- **[SOTA]** Zrnic & Candès. *Cross-prediction-powered inference.* PNAS 121(15), 2024. — arXiv:2309.16598
- **[SOTA]** Fisch, Maynez, Hofer, Dhingra, Globerson & Cohen. *Stratified Prediction-Powered Inference for Hybrid Language Model Evaluation.* NeurIPS, 2024. — arXiv:2406.04291
- **[SOTA]** Boyeau, Angelopoulos, Yosef, Malik, Jordan & Yosef. *AutoEval Done Right: Using Synthetic Data for Model Evaluation.* 2024. — arXiv:2403.07008
- **[Theory]** Dorner, Nastl & Hardt. *Limits to scalable evaluation at the frontier: LLM as judge won't beat twice the data.* ICLR, 2025. — arXiv:2410.13341
- **[Theory]** Dorner & Hardt. *Don't Label Twice: Quantity Beats Quality When Comparing Binary Classifiers on a Budget.* ICML, 2024.
- **[Method]** Miller. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* 2024. — arXiv:2411.00640
- **[Method]** Bowyer, Aitchison & Kaddour. *Position: Don't use the CLT in LLM evals with fewer than a few hundred datapoints.* ICML, 2025. — arXiv:2503.01747
- **[Bias]** Wang, Li, Chen, Zhu, Lin, Cao, Liu, Liu & Sui. *Large Language Models are not Fair Evaluators.* ACL, 2024. — arXiv:2305.17926
- **[Bias]** Panickssery, Bowman & Feng. *LLM Evaluators Recognize and Favor Their Own Generations.* NeurIPS, 2024. — arXiv:2404.13076
- **[Bias]** Dubois, Galambosi, Liang & Hashimoto. *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators.* 2024. — arXiv:2404.04475
- **[Method]** Gao, Xu, Wang, et al. *Bayesian Calibration of Win Rate Estimation with LLM Evaluators.* EMNLP, 2024.
- **[Survey]** Gu, Jiang, Han, et al. *A Survey on LLM-as-a-Judge.* 2024. — arXiv:2411.15594
- **[Survey]** Li, Wang, Zhu, et al. *From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge.* 2024. — arXiv:2411.16594

## 10. Worked Example

A pairwise win-rate eval: $n=500$ items judged, $m=100$ human-labelled. Judge win rate $\hat\theta_J=0.62$; judge–human agreement 80%; human win rate on the labelled subset 0.58.

**Reported interval (control).** $\mathrm{SE}=\sqrt{0.62\cdot0.38/500}=0.0217$, CI $=0.62\pm0.043$. Published as "62% ± 4.3%".

**PPI interval.** $\sigma_f^2=0.2356$, and with 20% disagreement and bias $\approx 0.04$, $\sigma^2_{f-Y}\approx 0.20-0.04^2=0.198$.
$$\mathrm{Var}\approx \frac{0.2356}{500}+\frac{0.198}{100}=0.000471+0.00198=0.00245,\quad \mathrm{SE}=0.0495.$$
**Human-only interval.** $\sqrt{0.58\cdot0.42/100}=0.0494$.

PPI SE 0.0495 vs human-only 0.0494. **The 500 judge calls bought nothing.** Width ratio 1.00, far above the 0.71 threshold. At 95% agreement instead, $\sigma^2_{f-Y}\approx0.0475$, SE $=0.0308$, ratio 0.62 — worth having, but that agreement level is above the human–human ceiling on this task.

**The obstruction, made visible.** The control interval is $\pm 0.043$ and centred on 0.62. The target is near 0.58. Swapping candidate order moves the judge win rate by roughly 3–5 points (Wang et al.). So the honest interval must include a design term: with $\sigma_{\text{design}}\approx 0.02$, total SE $=\sqrt{0.0217^2+0.02^2}=0.0295$, and the interval $0.62\pm0.058$ finally covers 0.58 — but only because it was widened by a variance component the standard bootstrap never estimates, and whose reference class $T$ nobody has agreed on. The number that decides ship-or-not is dominated by a term that is not currently measured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*