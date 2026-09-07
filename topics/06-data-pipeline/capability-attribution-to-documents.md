---
id: 06-data-pipeline/capability-attribution-to-documents
title: "Causal Attribution of Capability to Training Documents"
topic: 06-data-pipeline
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Causal Attribution of Capability to Training Documents

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/capability-attribution-to-documents` · **Status:** open

## 1. Problem Statement

Given a trained language model, a held-out capability (say, three-digit multiplication, or Python `asyncio` usage, or Bengali summarization), and the corpus it was trained on, identify the subset of training documents that *caused* the capability — in the counterfactual sense that removing them and retraining destroys it.

Three variants, of sharply different difficulty:

- **Measurement.** Define the target. "Capability" is not a scalar: it is a distribution over eval instances, and the causal effect of a document set on it is a random variable over training seeds. Without a stable target there is nothing to attribute to.
- **Method.** Given a target functional $f$, compute or approximate $\tau(S) = \mathbb{E}[f(\theta_{D})] - \mathbb{E}[f(\theta_{D \setminus S})]$ for candidate sets $S$, cheaply enough to search over a $10^{9}$-document corpus.
- **Theory.** Establish when $\tau$ is *identifiable* at all — when a group effect can be decomposed into per-document contributions, and when it cannot because the capability is produced by document redundancy or by an interaction with no additive part.

Solved would mean: a procedure that, for a named capability, returns a document set of size $\ll |D|$ whose ablation-and-retrain drops the capability's score by a predicted amount, with the prediction validated out-of-sample against actual retrains.

## 2. Formal Setting

Corpus $D = \{d_1,\dots,d_N\}$. Training is a stochastic map $\mathcal{A}: 2^{D} \times \Xi \to \Theta$, with $\xi \in \Xi$ the seed (init, data order, dropout). Capability measured by $f: \Theta \to \mathbb{R}$, e.g. mean accuracy on eval set $E$:
$$f(\theta) = \frac{1}{|E|}\sum_{e \in E} \mathbf{1}\!\left[\arg\max_y p_\theta(y \mid x_e) = y_e\right].$$
In practice $f$ is measured with a fixed prompt template, fixed decoding, and $|E|$ typically $10^2$–$10^4$, giving binomial noise $\sigma_E \approx \sqrt{p(1-p)/|E|}$.

**Causal effect of a set** $S \subseteq D$ under a weighting $w \in \{0,1\}^N$:
$$\tau(S) \;=\; \mathbb{E}_{\xi}\big[f(\mathcal{A}(D,\xi))\big] \;-\; \mathbb{E}_{\xi}\big[f(\mathcal{A}(D\setminus S,\xi))\big].$$
Measured by retraining $k$ times per arm and differencing sample means; the standard error is $\sqrt{2(\sigma_\xi^2 + \sigma_E^2)/k}$, where $\sigma_\xi$ is the seed-to-seed spread of $f$ — nonzero and often larger than $\sigma_E$ at small scale.

**Additive surrogate (datamodel).** Fit $\hat f(w) = \beta_0 + \sum_i \beta_i w_i$ over random subsets. Attribution quality is the **linear datamodeling score** (LDS): Spearman correlation between $\hat f(w^{(j)})$ and measured $f(\mathcal{A}(D_{w^{(j)}}))$ over held-out masks $w^{(j)}$.

**Gradient surrogate.** Influence of $d_i$ on eval point $e$:
$$\mathcal{I}(d_i, e) = \nabla_\theta \ell(e;\theta)^\top H_\theta^{-1} \nabla_\theta \ell(d_i;\theta), \quad H_\theta = \nabla^2_\theta \mathcal{L}(D;\theta),$$
approximated with EK-FAC or a random-projection sketch (TRAK).

**Assumptions, and which fail.**
- *Convexity / unique minimizer* (needed for the influence-function derivation): false for deep nets.
- *Additivity in $w$*: false when the same fact appears in $10^3$ documents — removing any one has zero effect, removing all has a large one. This is the redundancy regime and it breaks per-document attribution by construction.
- *Retraining determinism*: false; $\sigma_\xi$ is nonzero, so single-retrain ablations are unreliable.
- *Capability is a fixed target*: false; $f$ moves with prompt format and decoder, sometimes by more than the ablation effect.

## 3. State of the Art

**Established (validated against actual retrains):**
- **Datamodels** (Ilyas et al., ICML 2022) — linear surrogates fit on $\sim3\times10^5$ CIFAR-10 retrains predict held-out subset performance well. Ground truth is real, but the cost scales with retrains, not with corpus size.
- **TRAK** (Park et al., ICML 2023) — gradient-projection attribution reaching datamodel-comparable LDS on CIFAR-10/ImageNet at $\sim10^2$ models rather than $10^5$. Validated on classifiers; the LM results are on much smaller settings than pretraining.
- **DsDm** (Engstrom et al., ICML 2024) and **LESS** (Xia et al., ICML 2024) — attribution used *for selection*, with downstream gains measured. These validate the ranking's usefulness, not the counterfactual claim about any individual document.

**Claimed but unablated at scale:**
- **EK-FAC influence functions for LLMs** (Grosse et al., 2023) scale to a 52B-parameter model and surface qualitatively plausible influential sequences. There is no retrain-based validation at that scale; the evidence is inspection plus small proxies.
- **OLMoTrace** (Liu et al., ACL 2025 demos) returns verbatim training-corpus matches for model outputs across trillions of tokens. This is *provenance by string match*, not causal effect — a matched document may be one of thousands of duplicates.

**Benchmark-number-only:** FTRACE-TREx (Akyürek et al., EMNLP Findings 2022) gives a fact-tracing benchmark where gradient-based tracers are compared to BM25 retrieval; BM25 is competitive with or better than the influence methods. That is a benchmark result, not a validated causal claim.

## 4. What Is Known

- **Influence estimates are brittle.** Basu et al. (ICLR 2021) show deep-net influence estimates change substantially with network depth/width, weight decay, and stochastic training. Bae et al. (NeurIPS 2022) show influence functions estimate the *proximal Bregman response function*, not leave-one-out retraining — a different quantity.
- **Single-document effects are usually below noise.** Removing one document from a corpus of $10^9$ shifts eval accuracy by far less than $\sigma_\xi$, so leave-one-out is unmeasurable and only group effects are.
- **Frequency drives capability, measurably.** Razeghi et al. (Findings of EMNLP 2022) show few-shot arithmetic accuracy in GPT-J-6B correlates with pretraining term frequency of the operands — an average gap of tens of accuracy points between frequent and rare terms.
- **Knowledge acquisition needs repetition.** Allen-Zhu & Li (2024, "Knowledge Capacity Scaling Laws") report roughly 2 bits of stored knowledge per parameter at ~1000 exposures per fact, dropping to ~1 bit/param at ~100 exposures, on synthetic biography corpora with GPT-2-class models up to ~1B params.
- **Memorization scales with duplication.** Carlini et al. (ICLR 2023) show extractable memorization grows log-linearly with model size, example duplication count, and prompt context length, measured on GPT-Neo models up to 6B.
- **Attribution scores are useful for selection even when the counterfactual claim is unverified** — DsDm and LESS both beat random and heuristic-filter baselines on downstream tasks.

## 5. What Is Not Known

- **Methodologically blocked:** what "the capability" is. Eval scores move with prompt format, and the choice of metric can create or erase apparent discontinuities (Schaeffer et al., NeurIPS 2023). Until $f$ is specified so that $\sigma_\xi$ and format sensitivity are both smaller than the effect being attributed, the attribution target is undefined.
- **Theoretically open:** identifiability under redundancy. No characterization exists of when a group effect $\tau(S)$ admits a per-document decomposition. Shapley values (Ghorbani & Zou, ICML 2019) give a unique *axiomatic* split, but the axioms assign near-zero value to every member of a redundant set, which is not what the practitioner wants.
- **Empirically open:** whether any cheap attribution method has nonzero LDS against *pretraining-scale* retrains. The experiment is runnable — 100–1000 retrains at 1B params on a 30B-token corpus is $\sim10^5$ GPU-hours — but has not been published.
- **Empirically open:** whether attributed sets transfer across scale. If a set found at 1B params predicts effects at 70B, attribution becomes cheap; if not, every model needs its own study.

## 6. Why It Is Hard

The primary obstruction is **non-identifiability under redundancy**, compounded by **effect size below retraining noise**.

A fact appearing in $m = 10^4$ documents has per-document leave-one-out effect $\approx 0$ and group effect large. Any method whose output is a per-document score $\beta_i$ estimated from small random ablations is estimating a derivative that is genuinely zero in a neighbourhood, so the score carries no information about the group. The additive surrogate is not merely inaccurate; the quantity it targets is the wrong one.

Second, the ground truth is retraining, and retraining costs the training run. Validating $10^3$ candidate sets at 1B params is $10^3$ pretraining runs. This is why the field validates on CIFAR-10 and infers upward.

Third, gradient-based methods measure a different estimand (Bae et al. 2022): the proximal Bregman response, which coincides with leave-one-out only under convexity and exact optimization. Neither holds.

## 7. Current Research (as of 2026)

- **Open-corpus tracing.** AI2's OLMo/Dolma line makes the corpus queryable (OLMoTrace, `infini-gram`), enabling frequency-conditioned analyses that were impossible on closed models.
- **Scalable influence.** Anthropic's EK-FAC line (Grosse et al.) and follow-ups on cheaper Hessian sketches; MIT's Madry lab on TRAK/datamodel-based selection.
- **In-run attribution.** In-run Data Shapley (Wang et al., NeurIPS 2024) computes contributions during a single training run rather than by retraining — the most promising cost route, but its agreement with retrain ground truth at pretraining scale is unestablished *(frontier — verify)*.
- **Synthetic-corpus causal control.** The "Physics of Language Models" programme (Allen-Zhu & Li) builds corpora where ground-truth provenance is known by construction, trading realism for identifiability.
- **Knowledge-acquisition dynamics.** Chang et al. (NeurIPS 2024) track when facts are acquired and forgotten during pretraining, giving a per-injection effect size.

## 8. Concrete Next Experiment

**Question:** does any cheap attribution method have nonzero predictive power against real retrain counterfactuals at pretraining scale?

- **Scale.** 1.4B-parameter decoder, 30B tokens from a deduplicated open corpus (Dolma slice). Base run: 3 seeds. Then 200 retrains, each with an independent random 5% of *documents* held out (mask $w^{(j)}$), 1 seed each. At ~600 GPU-hours per run this is ~1.2$\times10^5$ A100-hours.
- **Target $f$.** Fix five capabilities, each scored as mean log-likelihood margin (not accuracy) over $\ge 2000$ instances, averaged over 5 prompt templates, to push $\sigma_E + \sigma_{\text{format}}$ below 0.5% of the mean. Measure $\sigma_\xi$ from the 3 base seeds first; if $\sigma_\xi$ exceeds the expected 5%-ablation effect, the experiment is underpowered and the sample size must rise, not the conclusion.
- **Arms.** (a) TRAK/EK-FAC influence scores from the base model; (b) BM25 lexical overlap with the eval set; (c) raw n-gram frequency of eval-relevant terms; (d) **control arm: random document scores**, which fixes LDS $= 0$ in expectation.
- **Deciding number.** Held-out **LDS** on 40 reserved masks. If arm (a) does not exceed arm (b) by $\ge 0.10$ LDS with a 95% CI excluding 0, gradient-based attribution buys nothing over lexical retrieval at pretraining scale — the current implicit claim of the field fails its first real test.
- **Second readout.** Run the redundancy diagnostic: for the top-1000 attributed documents, measure duplication multiplicity $m$ of their key facts via `infini-gram`. If median $m > 100$, per-document scores are provably near-zero causal effects and the group-attribution formulation is mandatory.

## 9. Key References

- **[Foundational]** Pang Wei Koh, Percy Liang. *Understanding Black-box Predictions via Influence Functions.* ICML, 2017. — arXiv:1703.04730
- **[Foundational]** Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry. *Datamodels: Predicting Predictions from Training Data.* ICML, 2022. — arXiv:2202.00622
- **[SOTA]** Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, Aleksander Madry. *TRAK: Attributing Model Behavior at Scale.* ICML, 2023. — arXiv:2303.14186
- **[SOTA]** Roger Grosse, Juhan Bae, Cem Anil, et al. *Studying Large Language Model Generalization with Influence Functions.* Anthropic technical report, 2023. — arXiv:2308.03296
- **[SOTA]** Logan Engstrom, Axel Feldmann, Aleksander Madry. *DsDm: Model-Aware Dataset Selection with Datamodels.* ICML, 2024. — arXiv:2401.12926
- **[SOTA]** Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen. *LESS: Selecting Influential Data for Targeted Instruction Tuning.* ICML, 2024. — arXiv:2402.04333
- **[Critique]** Samyadeep Basu, Philip Pope, Soheil Feizi. *Influence Functions in Deep Learning Are Fragile.* ICLR, 2021. — arXiv:2006.14651
- **[Critique]** Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, Roger Grosse. *If Influence Functions are the Answer, Then What is the Question?* NeurIPS, 2022. — arXiv:2209.05364
- **[Benchmark]** Ekin Akyürek, Tolga Bolukbasi, Frederick Liu, Binbin Xiong, Ian Tenney, Jacob Andreas, Kelvin Guu. *Towards Tracing Knowledge in Language Models Back to the Training Data.* Findings of EMNLP, 2022. — arXiv:2205.11482
- **[Empirical]** Yasaman Razeghi, Robert L. Logan IV, Matt Gardner, Sameer Singh. *Impact of Pretraining Term Frequencies on Few-Shot Numerical Reasoning.* Findings of EMNLP, 2022. — arXiv:2202.07206
- **[Empirical]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Empirical]** Zeyuan Allen-Zhu, Yuanzhi Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* 2024. — arXiv:2404.05405
- **[Theory]** Amirata Ghorbani, James Zou. *Data Shapley: Equitable Valuation of Data for Machine Learning.* ICML, 2019. — arXiv:1904.02868
- **[Systems]** Jiacheng Liu, Taylor Blanton, Yanai Elazar, et al. *OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens.* ACL (System Demonstrations), 2025.
- **[Measurement]** Rylan Schaeffer, Brando Miranda, Sanmi Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS, 2023. — arXiv:2304.15004

## 10. Worked Example

**Capability:** answering "What is the boiling point of water in Fahrenheit?" and 199 similar single-fact questions.

**Step 1 — provenance.** OLMoTrace-style string search over a 3T-token corpus returns $m \approx 41{,}000$ documents containing the string "212 degrees Fahrenheit" near "boiling".

**Step 2 — per-document effect.** Remove the single highest-influence document. Expected change in $f$: the fact is still present 40,999 times. Under Allen-Zhu & Li's exposure curve, capability saturates well below $10^3$ exposures, so $\tau(\{d_1\}) \approx 0$. The measured change will be dominated by seed noise. At 1.4B params, $\sigma_\xi$ on a 200-item eval is roughly 1–2 accuracy points; the true effect is $\ll 0.01$ points. **Signal-to-noise $< 10^{-2}$.**

**Step 3 — what influence reports anyway.** EK-FAC returns a finite, nonzero, rank-ordered score for $d_1$ — say $\mathcal{I} = 3.7\times10^{-4}$, the largest in the corpus. Nothing in the method flags that the counterfactual effect is zero. The score measures gradient alignment, not removability.

**Step 4 — group effect.** Remove all 41,000. Now $f$ on the boiling-point item drops from ~0.98 to near chance. But 41,000 documents is 0.004% of the corpus, and the *ablate-all* experiment costs a full retrain per capability studied.

**The obstruction, visible:** the quantity that is cheap to compute (per-document influence) has true value $\approx 0$ and is therefore pure ranking noise as a causal estimate; the quantity that is causally meaningful (group effect over the redundancy class) requires knowing the class in advance and costs one pretraining run to verify. Attribution methods are validated on the first and marketed as the second.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*