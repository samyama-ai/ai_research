---
id: 06-data-pipeline/aggregation-with-biased-raters
title: "Annotation Aggregation With Systematically Biased Raters"
topic: 06-data-pipeline
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Annotation Aggregation With Systematically Biased Raters

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/aggregation-with-biased-raters` · **Status:** partially-solved

## 1. Problem Statement

**Input.** A sparse label matrix: $n$ items, $m$ raters, observed labels $y_{ij}$ for $(i,j)$ in an assignment set $\Omega$ with $|\Omega| \ll nm$; optionally rater covariates (demographics, prior task history) and item features.

**Output.** An estimate $\hat{z}_i$ of the item's target label, or a distribution $\hat{p}_i \in \Delta^{K-1}$, plus per-rater error parameters.

**Objective.** Minimise error against the target under a budget of $|\Omega|$ labels — but the target is contested. Three variants, routinely conflated:

- **Measurement variant.** Given a label set, decide whether observed rater disagreement is *noise* (independent, zero-mean, aggregable) or *systematic bias* (rater-group-correlated, non-zero-mean, not aggregable). This is a hypothesis test, and it is the variant that is furthest from settled.
- **Method variant.** Given that bias exists, build an aggregator that beats majority vote on a held-out target. Solved for independent-noise models; open when bias is shared across a majority of the rater pool.
- **Theory variant.** Under what conditions on the rater-bias structure is the latent truth $z_i$ *identifiable* from $\{y_{ij}\}$ alone, with no anchor raters and no gold items? Largely negative.

Solving it means: an aggregator with a stated identifiability condition, an estimator that meets the minimax rate under that condition, and a diagnostic that says when the condition fails on a real dataset.

## 2. Formal Setting

Items $i \in [n]$ carry latent labels $z_i \in [K]$ drawn i.i.d. from prior $\pi \in \Delta^{K-1}$. Rater $j$ is described by a confusion matrix $C^{(j)} \in [0,1]^{K \times K}$ with $C^{(j)}_{kl} = \Pr(y_{ij} = l \mid z_i = k)$. The Dawid–Skene likelihood is

$$\mathcal{L}(\pi, C) = \prod_{i=1}^{n} \sum_{k=1}^{K} \pi_k \prod_{j:(i,j)\in\Omega} \prod_{l=1}^{K} \big(C^{(j)}_{kl}\big)^{\mathbb{1}[y_{ij}=l]}.$$

**Measured as.** $C^{(j)}$ is never observed; it is estimated by EM from $\ge 20$–$50$ labels per rater, or seeded spectrally from third-order moments of co-occurring rater triples. Rater "competence" is reported as $\bar{c}_j = \frac{1}{K}\sum_k \hat{C}^{(j)}_{kk}$; "bias" as the off-diagonal asymmetry $\hat{C}^{(j)}_{kl} - \hat{C}^{(j)}_{lk}$.

**Systematic bias.** Partition raters into latent groups $g(j) \in [G]$ (e.g. by demographic or ideological covariate). Write

$$C^{(j)} = \bar{C} + \Delta^{(g(j))} + E^{(j)}, \qquad \mathbb{E}[E^{(j)}] = 0,$$

where $\Delta^{(g)}$ is the group-level *systematic* offset and $E^{(j)}$ is idiosyncratic noise. Let $w_g$ be the share of the recruited pool in group $g$. The **pool bias** is $\Delta^\star = \sum_g w_g \Delta^{(g)}$.

Measured quantities that matter:
- **Group-conditional error gap** $\; \gamma = \max_g \Pr(\hat{z} \neq z \mid g\text{-relevant items}) - \min_g(\cdot)$, estimated on an adjudicated subset.
- **Disagreement decomposition** — the fraction of total label variance explained by rater group, e.g. an ICC or a mixed-effects $R^2$ with rater-group as random effect.

**Assumptions and their status in practice.**
1. *Conditional independence of raters given $z_i$* — **violated**: shared instructions, shared training, and shared cultural priors correlate errors. This is the assumption that systematic bias breaks by definition.
2. *A single latent $z_i$ exists* — **violated** on subjective tasks (toxicity, NLI, helpfulness), where disagreement is irreducible rather than error.
3. *$\Delta^\star = 0$ (unbiased pool on average)* — **violated whenever recruitment is non-representative**, which is the normal case on crowd platforms.
4. *Item difficulty is rater-independent* — **violated**; GLAD-style models add per-item difficulty precisely because it is not.
5. *Labels are missing at random given $\Omega$* — **often violated** by self-selected task-taking and by rater churn.

Under assumption 3 failing, the aggregate converges to $\arg\max_k (\bar{C} + \Delta^\star)$-corrupted posterior: **more labels do not help**.

## 3. State of the Art

**Theory SOTA (established).**
- Dawid & Skene (1979) EM remains the base estimator. Zhang, Chen, Zhou & Jordan (NeurIPS 2014) give a spectral method-of-moments initialiser plus one EM step that attains the **minimax rate** for one-coin and general D&S under conditional independence. Gao & Zhou (2013/2016) characterise the minimax error as $\exp(-\Theta(\bar{I}\cdot \ell))$ where $\ell$ is labels per item and $\bar{I}$ an average Chernoff information between rater confusion rows.
- Karger, Oh & Shah (Operations Research, 2014) give a message-passing estimator for binary tasks with error decaying exponentially in $\ell q$ ($q$ = rater-quality parameter), order-optimal, and strictly better than majority vote at equal budget.
- Identifiability of D&S holds only up to label permutation and requires at least three conditionally independent raters per item (Kruskal-rank condition). **No positive identifiability result exists when $\Delta^\star \neq 0$.**

**Empirical SOTA (established).** MACE (Hovy et al., NAACL 2013) — an item-response model with a per-rater "spamming" latent — reliably beats majority vote on NLP annotation and is widely reproduced. Passonneau & Carpenter (TACL 2014) show on word-sense annotation that a Bayesian annotation model changes the resulting gold standard on a non-trivial fraction of items relative to majority vote.

**Claimed but unablated.** Bias-aware and perspectivist aggregators — multi-annotator multi-task heads (Davani, Díaz & Prabhakaran, TACL 2022), jury learning (Gordon et al., CHI 2022), annotator-disagreement modelling for subjective tasks (Fleisig et al., EMNLP 2023) — report gains on their own splits. What is generally *not* ablated: whether the gain survives matched annotation budget, whether it comes from the bias model or from the extra capacity, and whether it holds when the evaluation gold is itself produced by the same biased pool. Several results exist **only as a benchmark number on a single dataset**.

**LLM-as-rater.** Model raters inherit and amplify pool bias (position, verbosity and self-preference effects documented in Zheng et al., NeurIPS 2023 Datasets & Benchmarks). Treating them as extra conditionally independent raters is not supported.

## 4. What Is Known

- **Aggregating independent noise works.** Snow et al. (EMNLP 2008): on five NLP tasks, averaging roughly **4 non-expert crowd labels** matched a single expert annotator; on affect recognition, non-expert correlation rose from ~0.45 (one rater) to expert level by ~10 raters. Scale: ~22,000 labels, 7 tasks.
- **Disagreement is often irreducible, not noise.** ChaosNLI (Nie, Zhou & Bansal, EMNLP 2020): **100 fresh annotations each for 4,645 NLI items (~464,500 labels)**. Majority-label entropy is high on a large minority of items; SOTA models' accuracy against majority label overstates agreement with the human distribution. Pavlick & Kwiatkowski (TACL 2019) independently found multi-modal, reproducible disagreement distributions in NLI.
- **Rater identity moves labels.** Sap et al. (ACL 2019): tweets in African-American English were up to **~1.5–2x** more likely to be labelled offensive/abusive across two hate-speech corpora. Sap et al. (NAACL 2022, "Annotators with Attitudes"): rater political identity and racial attitudes predicted toxicity ratings; priming raters with dialect/race information shifted ratings measurably. These are group effects, not per-rater noise.
- **Bias is not fixed by more raters.** Direct consequence of the decomposition in §2 and confirmed in practice: majority vote over a pool sharing $\Delta^\star$ converges to the pool's consensus, not the target.
- **Spectral + one EM step is rate-optimal** under conditional independence (Zhang et al. 2014) — i.e. the *noise* problem is theoretically closed.

## 5. What Is Not Known

- **Theoretically open.** No identifiability theory for $z_i$ when errors are group-correlated with $\Delta^\star \neq 0$. Partial conjecture: identifiability requires either (a) a subset of anchor items with external ground truth, (b) a known bound $\|\Delta^\star\|$, or (c) a rater group known to be unbiased on a subtask. No proof that these are necessary, and no minimax rate under a bounded-bias constraint.
- **Empirically open.** Whether any bias-aware aggregator beats majority vote **at matched annotation budget** against an *externally adjudicated* gold standard, across more than one dataset. The experiment is runnable; the cost is an expert adjudication set, not compute.
- **Methodologically blocked.** Separating "systematic bias" from "legitimate perspective" is not a well-posed measurement without a stated target construct. The same group offset $\Delta^{(g)}$ is a defect if the construct is "what a trained legal annotator would say" and signal if the construct is "what an affected community would say." Most datasets do not state the construct, so the quantity is undefined before it is unestimated.

## 6. Why It Is Hard

**Non-identifiability plus absent ground truth, compounded.** The model $\big(z, \bar{C}+\Delta^\star\big)$ and the model $\big(z', \bar{C}\big)$ with $z'$ the shifted labels can produce identical label distributions. No amount of data from the same pool distinguishes them — this is a structural symmetry, not a sample-size problem. The usual break is a gold set, but a gold set produced by the same recruitment channel inherits $\Delta^\star$, so the "validation" confirms the bias it was meant to detect. Add the second obstruction: **the evaluation does not measure what it names.** Reporting "accuracy against the aggregated label" scores an aggregator against its own inductive bias when the aggregator also produced the reference. Compute is irrelevant here; the binding constraint is an independently constructed target.

## 7. Current Research (as of 2026)

- **Perspectivist / label-variation modelling.** Plank (EMNLP 2022) reframes disagreement as human label variation to be modelled, not removed; the Perspectivist Data Manifesto community and the NLPerspectives workshop series continue this line. Active: Bocconi (Dirk Hovy), Bar-Ilan/Copenhagen (Barbara Plank), Google Research (Vinodkumar Prabhakaran, Mark Díaz), UW/AI2 (Maarten Sap).
- **Distribution-targeting evaluation.** Training and scoring against the full human label distribution rather than the majority label, following ChaosNLI; adoption in safety benchmarks is partial.
- **Preference data.** RLHF preference pools have the same structure — pool bias becomes a reward-model bias, then a policy bias. Length and sycophancy preferences are the documented cases. *(frontier — verify)* Work on group-robust or distributionally-robust reward aggregation is active but the matched-budget ablation against plain Bradley–Terry is mostly missing.
- **LLM raters as bias amplifiers or as cheap volume.** *(frontier — verify)* Hybrid human/LLM aggregation with explicit correlation terms between the model rater and the human pool is being tried; no established identifiability result.

## 8. Concrete Next Experiment

**Question.** Does any bias-aware aggregator beat majority vote against an *externally adjudicated* target at matched labelling budget?

**Scale.** $n = 2{,}000$ items from a subjective-but-adjudicable task (e.g. platform-policy violation, where a written policy exists). Recruit $m = 120$ raters, stratified $40/40/40$ across a covariate known to shift labels (e.g. self-reported political identity, or in-group vs out-group membership for the targeted attribute). Each item gets $\ell = 15$ labels, balanced across strata: **30,000 crowd labels total**. Separately, **3 trained adjudicators** working to the written policy label all 2,000 items with a documented dispute-resolution step: 6,000 expert labels, the target $z^\star$. Total budget ≈ 36,000 labels, roughly $15$–$25$k at typical rates.

**Arms.**
- *Control:* majority vote over the 15 crowd labels.
- *A:* Dawid–Skene (spectral init + EM), per-rater confusion only.
- *B:* group-aware aggregator with an explicit $\Delta^{(g)}$ term, fitted with **no** access to $z^\star$.
- *C (oracle upper bound):* $\Delta^{(g)}$ estimated on 200 held-out items with $z^\star$ revealed, applied to the remaining 1,800.

Budget-match by subsampling: every arm may use at most 30,000 crowd labels; report each arm's curve at $\ell \in \{3, 5, 9, 15\}$.

**Deciding number.** **Worst-stratum-relevant-item error against $z^\star$ at $\ell = 9$.** Arm B is a win only if it cuts this by $\ge 5$ absolute percentage points versus control while not increasing overall error. If B $\approx$ control but C $\ll$ control, the conclusion is sharp and useful: the bias is real and correctable, but **only with external ground truth** — i.e. the unsupervised problem is empirically as well as theoretically blocked.

## 9. Key References

- **[Foundational]** A. P. Dawid, A. M. Skene. *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.* Journal of the Royal Statistical Society: Series C, 1979.
- **[Foundational]** R. Snow, B. O'Connor, D. Jurafsky, A. Y. Ng. *Cheap and Fast — But is it Good? Evaluating Non-Expert Annotations for Natural Language Tasks.* EMNLP, 2008.
- **[SOTA – theory]** Y. Zhang, X. Chen, D. Zhou, M. I. Jordan. *Spectral Methods Meet EM: A Provably Optimal Algorithm for Crowdsourcing.* NeurIPS, 2014. — arXiv:1406.3824
- **[SOTA – theory]** C. Gao, D. Zhou. *Minimax Optimal Convergence Rates for Estimating Ground Truth from Crowdsourced Labels.* 2013. — arXiv:1310.5764
- **[SOTA – theory]** D. R. Karger, S. Oh, D. Shah. *Budget-Optimal Task Allocation for Reliable Crowdsourcing Systems.* Operations Research, 2014.
- **[SOTA – method]** D. Hovy, T. Berg-Kirkpatrick, A. Vaswani, E. Hovy. *Learning Whom to Trust with MACE.* NAACL-HLT, 2013.
- **[SOTA – method]** J. Whitehill, P. Ruvolo, T. Wu, J. Bergsma, J. Movellan. *Whose Vote Should Count More: Optimal Integration of Labels from Labelers of Unknown Expertise.* NIPS, 2009.
- **[SOTA – method]** V. C. Raykar et al. *Learning From Crowds.* JMLR, 2010.
- **[Bias evidence]** M. Sap, D. Card, S. Gabriel, Y. Choi, N. A. Smith. *The Risk of Racial Bias in Hate Speech Detection.* ACL, 2019.
- **[Bias evidence]** M. Sap, S. Swayamdipta, L. Vianna, X. Zhou, Y. Choi, N. A. Smith. *Annotators with Attitudes: How Annotator Beliefs And Identities Bias Toxic Language Detection.* NAACL, 2022.
- **[Disagreement]** E. Pavlick, T. Kwiatkowski. *Inherent Disagreements in Human Textual Inferences.* TACL, 2019.
- **[Disagreement]** Y. Nie, X. Zhou, M. Bansal. *What Can We Learn from Collective Human Opinions on Natural Language Inference Data?* EMNLP, 2020.
- **[Perspectivist]** A. M. Davani, M. Díaz, V. Prabhakaran. *Dealing with Disagreements: Looking Beyond the Majority Vote in Subjective Annotations.* TACL, 2022.
- **[Perspectivist]** M. L. Gordon, M. S. Lam, J. S. Park, K. Patel, J. Hancock, T. Hashimoto, M. S. Bernstein. *Jury Learning: Integrating Dissenting Voices into Machine Learning Models.* CHI, 2022.
- **[Perspectivist]** E. Fleisig, R. Abebe, D. Klein. *When the Majority is Wrong: Modeling Annotator Disagreement for Subjective Tasks.* EMNLP, 2023.
- **[Survey/position]** B. Plank. *The "Problem" of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation.* EMNLP, 2022.
- **[Survey/position]** L. Aroyo, C. Welty. *Truth Is a Lie: Crowd Truth and the Seven Myths of Human Annotation.* AI Magazine, 2015.
- **[Applied]** R. Passonneau, B. Carpenter. *The Benefits of a Model of Annotation.* TACL, 2014.
- **[Model raters]** L. Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2306.05685

## 10. Worked Example

Binary toxicity, $K=2$. One item: a post using a reclaimed slur, target construct "would a member of the referenced group judge this as an attack?" — expert-adjudicated answer $z^\star = 0$ (not toxic).

Pool: 15 raters, 12 out-group, 3 in-group. Group false-positive rates on reclaimed-slur items, taken from the magnitudes in Sap et al. (2019, 2022):

| group | $w_g$ | $\Pr(y=1 \mid z^\star=0)$ | expected votes for "toxic" |
|---|---|---|---|
| out-group | 0.80 | 0.70 | $12 \times 0.70 = 8.4$ |
| in-group | 0.20 | 0.15 | $3 \times 0.15 = 0.45$ |

**Majority vote.** Expected toxic votes $= 8.85 / 15 = 0.59 > 0.5$. Label = toxic. Wrong.

**Does more data help?** By the CLT the vote share concentrates at $0.59$; with $\ell = 101$ raters drawn from the same pool the probability of the correct majority is $\Pr\big(\text{Bin}(101, 0.59) \le 50\big) \approx 0.03$. **Raising $\ell$ from 15 to 101 — a 6.7x budget increase — makes the error more certain, not less.** This is $\Delta^\star \neq 0$ in one number.

**Does Dawid–Skene help?** D&S fits per-rater $C^{(j)}$ against its own latent posterior. The 12 out-group raters agree with each other; the 3 in-group raters dissent. EM assigns the majority the high-competence confusion matrix and the dissenters $\hat{C}^{(j)}_{00} \approx 0.3$ — it labels the *correct* raters as the unreliable ones and downweights them. Posterior $\Pr(z=1)$ rises from $0.59$ toward $\approx 0.95$. **D&S makes the answer worse and more confident.**

**What breaks the tie.** Only an exogenous signal: an anchor set with adjudicated $z^\star$, or a prior asserting that in-group raters are authoritative on this item class. Both are external inputs. Nothing in $\{y_{ij}\}$ distinguishes "12 raters are biased" from "3 raters are noisy" — the likelihoods are identical under a relabelling of $z$. That is the non-identifiability of §6, visible in 15 labels.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*