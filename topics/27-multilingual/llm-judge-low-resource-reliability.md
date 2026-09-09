---
id: 27-multilingual/llm-judge-low-resource-reliability
title: "LLM-as-Judge Reliability in Low-Resource Languages"
topic: 27-multilingual
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# LLM-as-Judge Reliability in Low-Resource Languages

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/llm-judge-low-resource-reliability` · **Status:** methodologically-blocked

## 1. Problem Statement

LLM-as-judge — using a strong model to score or rank generations in place of human raters — is now the default evaluation substrate for instruction following, chat, and RLHF reward modelling. It was validated almost entirely on English. The question is whether a judge's verdicts remain trustworthy in a language the judge itself generates badly.

Three variants, with very different difficulty:

- **Measurement.** Given a judge $J$, a language $\ell$, and a task, estimate how well $J$'s verdicts track the verdicts of a competent speaker of $\ell$. Solving this means producing a per-language reliability estimate with a calibrated confidence interval that is *not* an artefact of the reference annotators' own noise.
- **Method.** Build a judging procedure whose reliability in Swahili, Amharic, Sindhi, or Quechua matches its reliability in English at fixed cost. Solving this means closing the per-language agreement gap without simply importing English-centric preferences.
- **Theory.** Characterise when a model that cannot *produce* fluent $\ell$ can nonetheless *rank* text in $\ell$ — i.e. whether evaluation is strictly easier than generation, and under what conditions the generation–evaluation gap is provably non-negative.

The measurement variant is the blocker: the other two cannot be scored without it. This page is filed **methodologically blocked** for that reason.

## 2. Formal Setting

Let $\ell$ index a language, $x$ a prompt, and $(y_A, y_B)$ two candidate responses. A latent quality relation defines the true label $z \in \{A, B, \text{tie}\}$ — the verdict a fully competent, culturally situated speaker of $\ell$ would give. $z$ is never observed.

A judge is a stochastic map $J_\ell:(x,y_A,y_B)\mapsto \hat z$. A human rater pool for $\ell$ gives $h \sim H_\ell$. Both are noisy channels on $z$, with confusion matrices
$$\Pi^{J}_{\ell}[k,j]=\Pr(\hat z = j \mid z = k),\qquad \Pi^{H}_{\ell}[k,j]=\Pr(h = j \mid z = k).$$

**What is actually measured** is the judge–human agreement rate on $n_\ell$ items,
$$\hat A_\ell=\frac{1}{n_\ell}\sum_{i=1}^{n_\ell}\mathbb{1}[\hat z_i = h_i],$$
usually reported chance-corrected as Cohen's $\kappa_\ell$ or Krippendorff's $\alpha_\ell$. Under conditional independence of judge and human given $z$,
$$\Pr(\hat z = h)=\sum_{k}\pi_k \left(\Pi^{J}_{\ell}\Pi^{H\top}_{\ell}\right)[k,k],$$
so $\hat A_\ell$ is a **product** of two accuracies. The quantity people want is judge fidelity $F_\ell = \sum_k \pi_k \Pi^J_\ell[k,k]$; the quantity they compute is a bilinear form in $F_\ell$ and human fidelity. The cross-language contrast reported in papers,
$$\Delta_\ell = \hat A_{\text{eng}} - \hat A_{\ell},$$
therefore confounds judge degradation with annotator-pool degradation.

**Identifiability.** With two raters and a latent categorical label, the Dawid–Skene model (Dawid & Skene, *Applied Statistics*, 1979) is not identified: the likelihood is invariant to trading accuracy between the two channels. Identifiability requires three conditionally independent raters, via Kruskal's three-way array uniqueness theorem as applied to latent-class models by Allman, Matias & Rhodes (*Annals of Statistics*, 2009). Almost no multilingual judge study fields three independent channels per low-resource language.

**Assumptions, and where they break:**

| Assumption | Status in practice |
|---|---|
| Judge $\perp$ human given $z$ | **Violated.** Both often see the same MT-produced or English-templated prompt; human raters are frequently the same people who curated the data. |
| Human pool is competent in $\ell$ | **Often violated.** Low-resource annotator pools are small, sometimes L2 speakers, sometimes crowdworkers paid per item. |
| $z$ exists and is language-invariant | **Violated for culture-laden prompts.** Preferred register, directness, and honorifics differ; "helpfulness" is not language-invariant. |
| Items are i.i.d. | Violated: parallel test sets translate one English item into 50 languages, so errors are correlated across languages. |
| Positions/verbosity are neutral | Violated; see §4. |

## 3. State of the Art

**Empirical SOTA (established).** GPT-4-class judges reach usable but degraded agreement outside English. Hada et al. (*Are Large Language Model-based Evaluators the Solution to Scaling Up Multilingual Evaluation?*, Findings of EACL 2024) evaluated a GPT-4 judge against native speakers across eight languages and found agreement highest for English and systematically lower for non-Latin-script and lower-resource languages. METAL (Hada et al., Findings of NAACL 2024) extended this to a purpose-built meta-evaluation set across ten languages and showed the judge's *reasoning traces* are frequently wrong even when the final score matches.

**Meta-evaluation benchmarks (numbers only, no mechanism).** MM-Eval (Son et al., 2024) supplies a multilingual judge/reward-model meta-evaluation suite spanning >100 languages; PARIKSHA (Watts et al., EMNLP 2024) collected on the order of $10^5$ human judgements over ~30 models in 10 Indic languages. Both report per-language agreement tables. **These are benchmark numbers, not identified estimates of $F_\ell$** — they inherit the two-rater confound of §2.

**Method SOTA (claimed, largely unablated).** Fine-tuned open judges — Prometheus 2 (Kim et al., EMNLP 2024) and its multilingual successors — and cross-lingual judging, where the judge reasons in English over a translated response (Doddapaneni et al., *Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs*, 2024). Cross-lingual judging improves correlation on some low-resource languages, but the ablation that matters — does it improve fidelity, or merely align the judge with English-speaking annotators' preferences? — has not been run.

**Theory SOTA.** Essentially absent. There is no theorem separating ranking competence from generation competence for autoregressive models in a target language. The nearest formal result is the identifiability boundary above, which is a negative one.

## 4. What Is Known

- **Judge biases replicate cross-lingually.** Position bias (Wang et al., *Large Language Models are not Fair Evaluators*, ACL 2024) — verdict flips on swapping $(y_A,y_B)$ at rates of tens of percent for weaker judges — persists in non-English settings, and self-preference (Panickssery et al., NeurIPS 2024) is measurable at the scale of a few percentage points of win rate on English; it has not been decomposed by language.
- **Agreement falls with resource level.** Across EACL-2024/NAACL-2024 multilingual judge studies at the scale of $10^2$–$10^3$ human-annotated items per language, English judge–human agreement is the ceiling and low-resource, non-Latin-script languages sit well below it. The direction is reproduced independently; the magnitude varies by paper because the annotator pools differ.
- **Underlying generation quality is genuinely poor.** MEGA and MEGAVERSE (Ahuja et al., EMNLP 2023; NAACL 2024) and IrokoBench (Adelani et al., NAACL 2025) show large GPT-4/Claude-class deficits on African and low-resource Asian languages relative to English, on tasks with gold labels. A judge is being asked to grade a domain where it scores badly itself.
- **Translated benchmarks carry artefacts.** Global MMLU (Singh et al., 2024/ACL 2025) found a substantial fraction of MMLU items are culturally or translation-sensitive, and rankings shift when those items are separated out.
- **Structural inequality is quantified.** Blasi, Anastasopoulos & Neubig (ACL 2021) give a utility-vs-speakers measure showing performance concentrated in a handful of languages; Joshi et al. (ACL 2020) give the resource taxonomy the field still uses.

## 5. What Is Not Known

- **Methodologically blocked.** No accepted estimator separates $F_\ell$ from human-pool fidelity. Every published per-language "judge reliability" number is a two-channel product reported as if it were a one-channel quantity. Until a third independent channel (or an anchoring set with defensible gold) is standard, the field cannot say whether a low $\kappa_\ell$ means a bad judge or a thin annotator pool.
- **Methodologically blocked.** Whether $z$ is well defined for open-ended prompts in $\ell$ when preference norms are culture-dependent. If raters within $\ell$ legitimately disagree, the target of estimation is a distribution, not a label, and $\kappa$ is the wrong statistic.
- **Empirically open.** Per-language decomposition of position bias, verbosity bias, and self-preference. Runnable today: swap-consistency needs no human labels at all. Nobody has published a $\geq$50-language swap-consistency sweep.
- **Empirically open.** Whether cross-lingual judging (translate-then-judge-in-English) beats native-language judging on *fidelity* rather than on agreement with English-normed labels.
- **Theoretically open.** Whether evaluation in $\ell$ is provably easier than generation in $\ell$ for a model with limited $\ell$ pretraining mass — no proof either way, and no formal statement of the conjecture.

## 6. Why It Is Hard

The obstruction is **non-identifiability compounded by absent ground truth**, not compute.

For a two-rater latent-label model the likelihood surface has a continuum of maxima trading judge accuracy against human accuracy (Dawid & Skene 1979; identifiability restored only at three conditionally independent raters, Allman–Matias–Rhodes 2009). In English this is masked: annotator pools are large and cheap, so human fidelity is high enough to treat as known, and $\hat A_\ell \approx F_\ell \cdot \text{const}$. In Amharic or Quechua, hiring three independent competent annotators per item costs an order of magnitude more per item and is often infeasible at all — exactly the regime where the approximation fails.

Two aggravations. First, **conditional independence itself fails**: judge and human commonly both condition on a machine-translated prompt, so their errors correlate and $\hat A_\ell$ is biased *upward*, in the optimistic direction. Second, the evaluation does not measure what it names: a benchmark labelled "judge reliability in Yoruba" whose reference labels came from bilingual annotators reading English-sourced prompts measures agreement with an English-normed preference function, which is a different quantity.

## 7. Current Research (as of 2026)

- **Multilingual meta-evaluation suites.** Microsoft Research India (Hada, Ahuja, Watts, Sitaram) on METAL and PARIKSHA; the MM-Eval line on judge/reward-model meta-evaluation at >100 languages. Direction: more languages, more items.
- **Open multilingual judges.** Prometheus-family and M-Prometheus-style fine-tuned evaluators; Cohere Labs' Aya line supplies multilingual preference data (Dang et al., EMNLP 2024) that doubles as judge-training data. *(frontier — verify current model versions and per-language coverage.)*
- **Cross-lingual and pivot judging.** AI4Bharat and collaborators on judging translated responses in a high-resource pivot.
- **Community-sourced native evaluation.** Masakhane and the African NLP community (Adelani and collaborators) building natively authored, not translated, test data — the only route that attacks the ground-truth problem directly.
- **Rater-model statistics.** Reintroduction of Dawid–Skene / item-response models into LLM evaluation to separate rater and item effects. *(frontier — verify; adoption in multilingual judge papers is still thin.)*

## 8. Concrete Next Experiment

**Three-channel identified estimate of judge fidelity for six languages.**

- **Scale.** Six languages spanning the resource spectrum: English and German (high), Swahili and Bengali (mid), Amharic and Yoruba (low). 400 pairwise items per language, all **natively authored** prompts (no translation from English), 2,400 items total. Three *independent* annotation channels per item: (a) two disjoint pools of native-speaker annotators recruited through different providers, (b) the LLM judge. Cost: ~7,200 human judgements.
- **Control arm.** Two controls. (i) The standard single-pool protocol run on the same items, giving the conventional $\hat A_\ell$ for direct comparison. (ii) The same 400 items per language machine-translated from an English source set, annotated identically — isolating the translation artefact.
- **Estimator.** Fit the three-rater Dawid–Skene model per language; the third channel makes $\Pi^J_\ell$ identified (Allman–Matias–Rhodes conditions). Report $\hat F_\ell$ with bootstrap CIs.
- **The deciding number.** The gap between the identified judge fidelity and the naive agreement, in points, as a function of resource level:
$$G_\ell = \hat F_\ell - \hat A_\ell .$$
If $G_\ell$ is roughly constant across the six languages, then existing per-language agreement tables are a valid *ordinal* measure of judge quality and the field can keep using them. If $G_\ell$ grows by more than 10 points from English to Amharic/Yoruba — the outcome the identifiability argument predicts — then every published multilingual judge ranking is confounded by annotator quality and must be recomputed. One number, one decision.

Cheap precursor, runnable this week with no humans: swap-consistency $C_\ell = \Pr(\hat z(y_A,y_B) = \text{flip}\,\hat z(y_B,y_A))$ over 50+ languages. $C_\ell$ upper-bounds judge fidelity without any labels, so a language where $C_\ell \approx$ chance is one where no agreement table can be trusted.

## 9. Key References

- **[Foundational]** A. P. Dawid, A. M. Skene. *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.* Journal of the Royal Statistical Society Series C (Applied Statistics), 1979.
- **[Foundational]** E. S. Allman, C. Matias, J. A. Rhodes. *Identifiability of Parameters in Latent Structure Models with Many Observed Variables.* Annals of Statistics, 2009.
- **[Foundational]** L. Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2306.05685
- **[Foundational]** P. Joshi, S. Santy, A. Budhiraja, K. Bali, M. Choudhury. *The State and Fate of Linguistic Diversity and Inclusion in the NLP World.* ACL, 2020.
- **[SOTA]** R. Hada, V. Gumma, A. de Wynter, H. Diddee, M. Ahmed, M. Choudhury, K. Bali, S. Sitaram. *Are Large Language Model-based Evaluators the Solution to Scaling Up Multilingual Evaluation?* Findings of EACL, 2024. — arXiv:2309.07462
- **[SOTA]** R. Hada et al. *METAL: Towards Multilingual Meta-Evaluation.* Findings of NAACL, 2024. — arXiv:2404.01667
- **[SOTA]** I. Watts et al. *PARIKSHA: A Large-Scale Investigation of Human-LLM Evaluator Agreement on Multilingual and Multi-Cultural Data.* EMNLP, 2024. — arXiv:2406.15053
- **[SOTA]** S. Doddapaneni et al. *Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs.* 2024. — arXiv:2410.13394
- **[SOTA]** S. Kim et al. *Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models.* EMNLP, 2024. — arXiv:2405.01535
- **[SOTA]** G. Son et al. *MM-Eval: A Multilingual Meta-Evaluation Benchmark for LLM-as-a-Judge and Reward Models.* 2024. — arXiv:2410.17578
- **[Established bias results]** P. Wang et al. *Large Language Models are not Fair Evaluators.* ACL, 2024. — arXiv:2305.17926
- **[Established bias results]** A. Panickssery, S. R. Bowman, S. Feng. *LLM Evaluators Recognize and Favor Their Own Generations.* NeurIPS, 2024. — arXiv:2404.13076
- **[Survey / landscape]** K. Ahuja et al. *MEGA: Multilingual Evaluation of Generative AI.* EMNLP, 2023. — arXiv:2303.12528; and *MEGAVERSE.* NAACL, 2024.
- **[Survey / landscape]** D. I. Adelani et al. *IrokoBench: A New Benchmark for African Languages in the Age of Large Language Models.* NAACL, 2025. — arXiv:2406.03368
- **[Survey / landscape]** S. Singh et al. *Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation.* ACL, 2025. — arXiv:2412.03304
- **[Survey / landscape]** D. Blasi, A. Anastasopoulos, G. Neubig. *Systematic Inequalities in Language Technology Performance across the World's Languages.* ACL, 2021.
- **[Context]** J. Dang et al. *RLHF Can Speak Many Languages: Unlocking Multilingual Preference Optimization for LLMs.* EMNLP, 2024. — arXiv:2407.02552

## 10. Worked Example

Take Yoruba. Suppose a judge study reports $\hat A_{\text{yor}} = 0.62$ raw agreement on a binary A/B comparison, against $\hat A_{\text{eng}} = 0.80$, and concludes the judge is "much less reliable in Yoruba."

Model both channels as symmetric binary with accuracies $f_J$ and $f_H$, ties dropped. Conditional independence gives
$$\Pr(\hat z = h) = f_J f_H + (1-f_J)(1-f_H).$$

For English, take the annotator pool as strong, $f_H = 0.90$. Then $0.80 = 0.9 f_J + 0.1(1-f_J) \Rightarrow f_J = 0.875$.

For Yoruba, the observed $0.62$ admits a family of solutions:

| Assumed $f_H$ | Implied judge fidelity $f_J$ |
|---|---|
| 0.90 | 0.65 |
| 0.80 | 0.70 |
| 0.70 | 0.80 |
| 0.65 | 0.90 |

Every row is an exact fit to the same data. If the Yoruba pool is as good as the English one, the judge collapsed from 0.875 to 0.65. If the pool is at $f_H=0.65$ — plausible for a thin, partly-L2, per-item-paid pool on culture-laden prompts — the judge is *better* in Yoruba (0.90) than in English (0.875), and the headline conclusion inverts. Two raters cannot tell these apart: the likelihood is flat along the curve $f_Jf_H+(1-f_J)(1-f_H)=0.62$.

Now break independence. If judge and human both read the same machine-translated prompt and both mis-parse a fraction $\rho=0.15$ of items in the same direction, the shared-error term inflates observed agreement by roughly $\rho$, so a true $\hat A_{\text{yor}}$ of $0.47$ reads as $0.62$ — and $0.47$ is barely above the $0.50$ chance floor.

The obstruction is visible in the table: the reported number is one equation in two unknowns, and the direction of the bias from the broken independence assumption is optimistic. Adding a second disjoint annotator pool — the §8 experiment — adds the second equation and closes the system. Adding more languages, more items, or a better judge does not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*