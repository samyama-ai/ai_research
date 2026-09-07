---
id: 27-multilingual/annotation-budget-allocation-across-languages
title: "Cost-Optimal Annotation Budget Allocation Across Languages"
topic: 27-multilingual
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cost-Optimal Annotation Budget Allocation Across Languages

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/annotation-budget-allocation-across-languages` · **Status:** open

## 1. Problem Statement

Given a fixed budget (money, annotator-hours, or both), a task, a multilingual base model, and a set of $L$ target languages with different per-label costs and different amounts of existing data, decide **how many labels to buy in each language**.

Three variants, of very different difficulty:

- **Measurement variant.** Estimate, for each language $\ell$, the function mapping labels bought to task performance *after* joint multilingual fine-tuning — i.e. the marginal value of one more Wolof sentence, holding the rest of the allocation fixed. Open because the function is coupled across languages and cheap estimators of it are biased.
- **Method variant.** Produce an allocation policy that beats fixed heuristics (uniform-per-language, uniform-per-dollar, proportional-to-speakers, temperature-sampled) on a held-out aggregate objective, with the pilot cost of running the policy charged against the same budget. Open: no policy has been shown to win net of its own estimation cost at realistic $L$.
- **Theory variant.** Characterise when the allocation problem is concave/submodular (so greedy has a guarantee) and when transfer coupling makes it non-concave, and give a label-complexity bound for the multi-task case. Open.

Solved would mean: a policy that, at $L \geq 20$ and a budget of $10^4$–$10^5$ labels, beats every fixed heuristic on a pre-registered aggregate objective across at least two tasks and two model families, with the pilot cost included.

## 2. Formal Setting

Languages $\ell \in \{1,\dots,L\}$. Allocation $\mathbf{n} = (n_1,\dots,n_L)$, $n_\ell \in \mathbb{Z}_{\geq 0}$ labels bought.

**Cost.** $c_\ell$ = marginal dollars per accepted label, measured as (total payments to annotators and reviewers for language $\ell$) / (labels surviving adjudication). Fixed costs $\kappa_\ell$ (recruiting, guideline translation, pilot rounds) are real and often dominate at small $n_\ell$:
$$\text{Cost}(\mathbf{n}) = \sum_{\ell} \big( \kappa_\ell \mathbb{1}[n_\ell > 0] + c_\ell n_\ell \big) \le B .$$

**Performance.** One model $\theta(\mathbf{n})$ trained jointly on all purchased data plus existing data $m_\ell$. $f_\ell(\mathbf{n}) = $ task metric (F1, chrF, exact match) for language $\ell$ on a *held-out test set annotated independently of the training pool*. This is the quantity actually measured; everything else is a fit to it.

**Objective.** A scalarisation $U$ with weights $w_\ell$:
$$\mathbf{n}^\star = \arg\max_{\mathbf{n}} \; U\big(f_1(\mathbf{n}),\dots,f_L(\mathbf{n})\big), \quad U_{\text{mean}} = \sum_\ell w_\ell f_\ell, \quad U_{\min} = \min_\ell f_\ell .$$
Choice of $U$ is a policy decision, not an empirical one; $w_\ell \propto$ speakers, $\propto$ demand, or uniform give different optima. Blasi et al. (ACL 2021) formalise the speaker-weighted version.

**Working parametric model.** Residual error below a per-language ceiling, power law in labels:
$$f_\ell(\mathbf{n}) = A_\ell - b_\ell\,\big(m_\ell + n_\ell + \textstyle\sum_{k\neq\ell} \tau_{\ell k} n_k\big)^{-\alpha_\ell},$$
with $\tau_{\ell k} \in [0,1]$ the **effective transfer coefficient**: how many own-language labels one label of $k$ is worth for $\ell$. Measured only by ablation (train with and without $n_k$), $O(L^2)$ runs.

Interior optimum (ignoring $\kappa_\ell$, $\tau=0$) equalises gain per dollar:
$$\frac{1}{c_\ell}\frac{\partial U}{\partial n_\ell} = \lambda \;\;\forall \ell \quad\Longrightarrow\quad n_\ell \propto \big(w_\ell \alpha_\ell b_\ell / c_\ell\big)^{1/(1+\alpha_\ell)} .$$

**Assumptions, and which are violated.**
1. *Separability ($\tau_{\ell k}=0$)* — violated: cross-lingual transfer is the reason multilingual models work.
2. *Constant $c_\ell$* — violated: costs are step functions with large $\kappa_\ell$ and annotator-supply ceilings.
3. *Label quality independent of language* — violated: Kreutzer et al. (TACL 2022) audited 205 multilingual corpora and found many low-resource subsets with under 50% correct-language content.
4. *Test sets comparable across languages* — violated: translated test sets ("translationese") inflate transfer scores relative to natively-authored ones.
5. *Single stationary $\alpha_\ell$* — violated at small $n$, where the curve has not entered its power-law regime.

## 3. State of the Art

**Established.**
- *Few-shot beats zero-shot by a large margin, cheaply.* Lauscher et al. (EMNLP 2020) show tens of annotated target-language examples recover much of the gap for POS and NER, and that zero-shot transfer correlates with target-language pretraining size. Reproduced across benchmarks.
- *Sampling weights matter for a fixed data pool.* UniMax (Chung et al., ICLR 2023) beats temperature sampling for multilingual pretraining under an epoch-budget constraint. This is *reweighting*, not *purchasing* — the closest well-ablated neighbour.
- *Transfer-source selection is learnable.* LangRank (Lin et al., ACL 2019) ranks transfer languages from typological and data features better than single-feature baselines.
- *Joint multilingual scaling laws exist.* Fernandes et al. (ICML 2023) fit scaling laws for multilingual NMT that include the language-weighting vector and predict held-out mixtures.

**Claimed but unablated.** That per-language active learning (Chaudhary et al., TACL 2021, for POS) composes across languages into a budget-optimal global policy. Per-language gains are real; the cross-language allocation on top of them has not been ablated against uniform-per-dollar with pilot cost charged.

**Benchmark-number-only.** XTREME / XTREME-R (Hu et al., ICML 2020; Ruder et al., EMNLP 2021) report per-language scores that are routinely used to argue which languages "need data", but the benchmarks contain no cost model and mostly use translated test sets — the number does not license the allocation conclusion.

## 4. What Is Known

- **Very small budgets buy a lot.** Garrette & Baldridge (NAACL 2013) built usable POS taggers from **2 hours** of annotation per language (Kinyarwanda, Malagasy) — the steep part of the curve is real and cheap.
- **Real annotation costs are documented, and they differ by an order of magnitude.** MasakhaNER (Adelani et al., TACL 2021; MasakhaNER 2.0, EMNLP 2022, 20 African languages) report the participatory pipelines; recruiting cost, not per-label cost, dominates for the rarest languages.
- **Pretraining-size effects at scale.** XLM-R (Conneau et al., ACL 2020) at 100 languages, 2.5 TB CommonCrawl: adding languages helps low-resource ones up to a capacity point, then hurts — the "curse of multilinguality". Allocation interacts with model capacity, not just data.
- **Small, clean, in-language beats large and noisy.** Ogueji et al. (2021) train AfriBERTa on ~1 GB across 11 African languages and match/beat far larger multilingual models on those languages.
- **Cost asymmetry is systematic.** Ahia et al. (Findings of EMNLP 2021) document the "low-resource double bind": the languages with least data also face the tightest compute/deployment constraints.
- **Greedy has a guarantee only under submodularity.** Nemhauser, Wolsey & Fisher (1978): $1-1/e$ for monotone submodular maximisation under a cardinality constraint. Whether $f$ here is submodular is unproven.

## 5. What Is Not Known

- **Theoretically open.** Whether $\mathbf{n} \mapsto U(f(\mathbf{n}))$ is submodular under a fixed multilingual architecture. Nobody has proved or disproved it, so greedy allocation runs without a guarantee. Also open: a label-complexity bound for active learning in the multi-task/multilingual setting analogous to single-task agnostic bounds.
- **Empirically open.** Whether *any* learned allocation policy beats uniform-per-dollar at $L\ge 20$ once its pilot cost is charged. Runnable today: $\sim$20 languages $\times$ 2 tasks $\times$ a few dozen fine-tuning runs. Nobody has run it with cost accounting.
- **Methodologically blocked.** The per-language objective itself. Reported gains are measured on test sets whose provenance (translated vs native, annotator overlap with training pool) differs across languages, so $f_\ell$ is not comparable across $\ell$ — and the whole problem is a comparison across $\ell$. Until natively-authored, cost-annotated parallel test suites exist for the same task in all $L$ languages, the objective is under-specified.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the marginal-value curve from any affordable pilot**. The optimum depends on $\alpha_\ell$ through the exponent $1/(1+\alpha_\ell)$, and $\alpha_\ell$ must be estimated from a pilot small enough to leave the budget intact. A 200–500 label pilot sits *before* the power-law regime, so the fitted $\alpha_\ell$ has a confidence interval wide enough to swing the dollar share of a language by tens of percent (Section 10). Compounding it: the cross-language coupling $\tau_{\ell k}$ needs $O(L^2)$ ablation runs to measure, and it changes when the allocation changes — the quantity you must know to choose the allocation is a function of the allocation.

Secondary obstruction: **evaluation that does not measure what it names**. "Performance on language $\ell$" measured on translated test data measures transfer-from-English partly, so allocations tuned on it are tuned on the wrong target.

## 7. Current Research (as of 2026)

- **Data-mixing laws applied to language weights** — DoReMi (Xie et al., NeurIPS 2023) and follow-on mixing-law work fit proxy-model weights and transfer them to large runs. Extension from domains to *purchasable* languages is active but mostly on pool reweighting, not procurement *(frontier — verify)*.
- **Participatory / community annotation economics** — Masakhane, Aya (Singh et al., ACL 2024, 65 languages via volunteer contribution). These change $c_\ell$ and $\kappa_\ell$ qualitatively; almost no published cost curves.
- **LLM-assisted pre-annotation with human adjudication**, lowering $c_\ell$ unevenly across languages (more for high-resource ones), which *worsens* the allocation asymmetry rather than fixing it *(frontier — verify)*.
- **Native-authored multilingual benchmarks** (e.g. Global-MMLU-style native subsets, Include-style exams) partially unblock Section 5's measurement problem *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does a curve-fitting allocation policy beat uniform-per-dollar, net of pilot cost, at realistic $L$?

- **Scale.** $L=20$ languages from MasakhaNER 2.0 (NER) plus the overlapping UD treebanks (POS). Base model: one 0.5–1 B multilingual encoder/decoder, held fixed. Budget $B$ = 40,000 label-units, denominated in *annotator-minutes* measured per language from a timed pilot, not assumed.
- **Arms.** (1) *Uniform-per-dollar* — control. (2) *Uniform-per-language*. (3) *Speaker-weighted*. (4) *Fitted policy*: spend 10% of $B$ on a staged pilot (4 points per language, $n \in \{100,300,900,2700\}$), fit $\alpha_\ell, b_\ell$, allocate the remaining 90% by the equal-marginal-gain rule, re-fit once at 50% spend.
- **Everything downstream identical**: same joint fine-tuning recipe, same 5 seeds, evaluated on **natively-authored** test sets (no translated splits).
- **Deciding number.** $\Delta = U_{\text{mean}}(\text{arm 4}) - U_{\text{mean}}(\text{arm 1})$ in F1, with the pilot's 10% already spent. Pre-register the threshold: **$\Delta \geq 1.0$ F1 with a 95% bootstrap CI excluding 0** counts as the fitted policy winning. Report $U_{\min}$ alongside; a policy that gains on the mean while losing $>2$ F1 on the worst language should be recorded as a fairness regression, not a win.

## 9. Key References

- **[Foundational]** Dan Garrette, Jason Baldridge. *Learning a Part-of-Speech Tagger from Two Hours of Annotation.* NAACL 2013.
- **[Foundational]** G. L. Nemhauser, L. A. Wolsey, M. L. Fisher. *An analysis of approximations for maximizing submodular set functions—I.* Mathematical Programming, 1978.
- **[Foundational]** Burr Settles. *Active Learning Literature Survey.* University of Wisconsin–Madison Technical Report 1648, 2009.
- **[SOTA]** Hyung Won Chung, Xavier Garcia, Adam Roberts, Yi Tay, Orhan Firat, Sharan Narang, Noah Constant. *UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining.* ICLR 2023.
- **[SOTA]** Patrick Fernandes, Behrooz Ghorbani, Xavier Garcia, Markus Freitag, Orhan Firat. *Scaling Laws for Multilingual Neural Machine Translation.* ICML 2023.
- **[SOTA]** Yu-Hsiang Lin, Chian-Yu Chen, Jean Lee, Zirui Li, Yuyan Zhang, Mengzhou Xia, Shruti Rijhwani, Junxian He, Zhisong Zhang, Xuezhe Ma, Antonios Anastasopoulos, Patrick Littell, Graham Neubig. *Choosing Transfer Languages for Cross-Lingual Learning.* ACL 2019.
- **[SOTA]** Aditi Chaudhary, Antonios Anastasopoulos, Zaid Sheikh, Graham Neubig. *Reducing Confusion in Active Learning for Part-Of-Speech Tagging.* TACL 2021.
- **[Empirical]** David Ifeoluwa Adelani et al. *MasakhaNER: Named Entity Recognition for African Languages.* TACL 2021. — and *MasakhaNER 2.0: Africa-centric Transfer Learning for Named Entity Recognition.* EMNLP 2022.
- **[Empirical]** Anne Lauscher, Vinit Ravishankar, Ivan Vulić, Goran Glavaš. *From Zero to Hero: On the Limitations of Zero-Shot Language Transfer with Multilingual Transformers.* EMNLP 2020.
- **[Empirical]** Alexis Conneau et al. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020. — arXiv:1911.02116
- **[Empirical]** Julia Kreutzer et al. *Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets.* TACL 2022.
- **[Position]** Damián Blasi, Antonios Anastasopoulos, Graham Neubig. *Systematic Inequalities in Language Technology Performance across the World's Languages.* ACL 2021.
- **[Survey]** Michael A. Hedderich, Lukas Lange, Heike Adel, Jannik Strötgen, Dietrich Klakow. *A Survey on Recent Approaches for Natural Language Processing in Low-Resource Scenarios.* NAACL 2021.
- **[Survey]** Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, Monojit Choudhury. *The State and Fate of Linguistic Diversity and Inclusion in the NLP World.* ACL 2020.

## 10. Worked Example

Three languages, NER, budget $B = \$6{,}000$. Measured per-sentence costs: Swahili $c=\$0.10$, Yoruba $\$0.25$, Wolof $\$0.60$. Fit $e_\ell(n) = b_\ell n^{-\alpha}$ (F1 points below ceiling), $\alpha = 0.5$ for all, $b = (1200, 2000, 2800)$.

Equal-marginal-gain rule, $n_\ell \propto (b_\ell/c_\ell)^{2/3}$:

| | $b/c$ | $n_\ell$ | dollars | share | residual $e_\ell$ |
|---|---|---|---|---|---|
| Swahili | 12,000 | 9,836 | \$984 | 16% | 12.1 |
| Yoruba | 8,000 | 7,496 | \$1,874 | 31% | 23.1 |
| Wolof | 4,667 | 5,239 | \$3,143 | 52% | 38.7 |

Mean residual error **24.6**. Uniform-per-dollar control (\$2,000 each → 20,000 / 8,000 / 3,333 sentences) gives residuals 8.5 / 22.4 / 48.5, mean **26.5**. The fitted policy wins by **1.9 F1** on the mean and by 9.8 F1 on the worst language.

**Now the obstruction.** $\alpha_{\text{wo}} = 0.5$ came from a 200-sentence pilot. Re-fit the same pilot point with $\alpha_{\text{wo}} = 0.3$ — inside the CI a 200-sentence pilot can produce, and matched to give the identical error at $n=2{,}000$ ($b' = 612$). Re-solving at the same shadow price $\lambda = 6.15\times10^{-3}$:
$$n_{\text{wo}} = \Big(\tfrac{\alpha b'}{\lambda c}\Big)^{1/(1+\alpha)} = \Big(\tfrac{0.3 \times 612}{6.15\times10^{-3}\times 0.60}\Big)^{0.769} \approx 4{,}090 .$$
Wolof's dollar share falls from **52% to ~42%** — a \$600 swing on a \$6,000 budget — driven entirely by an exponent the pilot cannot resolve, because 200 sentences is below the power-law regime. The gap between the two candidate allocations (\$600) is comparable to the entire measured advantage of optimising at all. That is the problem: the policy's edge over uniform-per-dollar is the same size as the error in estimating the policy.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*