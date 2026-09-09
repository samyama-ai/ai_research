---
id: 23-privacy-memorization/memorization-generalization-capacity-separation
title: "Provable Separation Between Memorization and Generalization Capacity"
topic: 23-privacy-memorization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Provable Separation Between Memorization and Generalization Capacity

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/memorization-generalization-capacity-separation` · **Status:** open

## 1. Problem Statement

A trained model's parameters carry two kinds of information about its training set: **generalizing structure** (statistics that transfer to fresh samples from the same distribution) and **memorized specifics** (per-example content recoverable only because that example was in the set). The question is whether these are separable *in capacity*, i.e. whether a model can be forced to keep the first and drop the second without paying a loss penalty.

- **Theory variant.** For a task family $\mathcal{F}$, sample size $n$, and target excess risk $\varepsilon$, is there a nonzero lower bound on per-example memorization for *every* learner reaching risk $\varepsilon$? Equivalently, is the memorization–risk frontier strictly interior, or is memorization always removable at zero cost?
- **Method variant.** Given a fixed architecture and compute budget, construct a training procedure achieving the non-private loss at memorization below the proven lower bound for naive training — or show that no such procedure exists in that budget.
- **Measurement variant.** Given only a released checkpoint, decide how many bits about a specific training example it retains, separating this from what a model trained without that example would also predict.

**Solved** would mean: a task family plus a theorem of the form *any learner with excess risk $\le \varepsilon$ has memorization $\ge M(n,d,\varepsilon)$*, together with a matching learner achieving $O(M)$ — and an empirical demonstration that the frontier is real at language-model scale rather than an artifact of the memorization definition used.

## 2. Formal Setting

Let $\mathcal{D}$ be a distribution over $\mathcal{X}\times\mathcal{Y}$, $S=(z_1,\dots,z_n)\sim\mathcal{D}^n$, and $A$ a (randomized) learner producing $\hat{f}=A(S)$. Write $S^{\setminus i}$ for $S$ with $z_i$ removed.

**Leave-one-out memorization** (Feldman 2020; Feldman & Zhang 2020), measured by retraining:

$$\mathrm{mem}(A,S,i)=\Pr_{A}\big[A(S)(x_i)=y_i\big]-\Pr_{A}\big[A(S^{\setminus i})(x_i)=y_i\big].$$

Measured in practice by training $K$ models on random subsets of size $\alpha n$ and comparing the in-set and out-of-set empirical means for each $i$; the Feldman–Zhang estimator uses $K\approx 2000$ ResNet runs on CIFAR-100.

**Total memorization budget:** $\mathrm{Mem}(A,n)=\mathbb{E}_S\sum_{i=1}^n \mathrm{mem}(A,S,i)$, or in bits, the conditional mutual information $\mathrm{CMI} = I(A(S);S\mid \tilde{S})$ over a supersample $\tilde{S}$.

**Excess risk:** $\mathrm{exc}(A)=\mathbb{E}_S[L_\mathcal{D}(A(S))]-\inf_f L_\mathcal{D}(f)$, with $L_\mathcal{D}$ 0–1 loss or per-token cross-entropy in nats.

**The frontier** is the object of study:

$$\mathcal{S}_{\mathcal{F}}(n,\varepsilon)=\inf\{\ \mathrm{Mem}(A,n)\ :\ \mathrm{exc}_{\mathcal{D}}(A)\le\varepsilon\ \ \forall \mathcal{D}\in\mathcal{F}\ \}.$$

A **provable separation** is a proof that $\mathcal{S}_{\mathcal{F}}(n,\varepsilon)=\omega(1)$ for a natural $\mathcal{F}$ (lower bound) plus a learner matching it (upper bound). Cast as capacities: if a model of $P$ parameters holds $C$ bits total and generalizing structure needs $G$ bits, the question is whether $C-G$ is *forced* to hold example-specific bits.

Assumptions, with the ones violated in practice flagged:

- **i.i.d. sampling** — violated: web corpora contain near-duplicates at rates of a few percent, and duplication is the single strongest predictor of extraction.
- **Fixed learner, retrainable $K$ times** — violated above ~1B parameters; the retraining estimator is unaffordable, so proxies (loss gap, extraction rate, LiRA-style membership scores) are substituted, and they measure different things.
- **A single ground-truth label per $x$** — violated for next-token prediction, where $\mathrm{mem}$ must be replaced by a continuous counterfactual log-likelihood gap (Zhang et al. 2023).
- **Task distribution is fixed and known** — violated: for pretraining, $\mathcal{F}$ is whatever the crawl contains, so no lower bound can be instantiated on it.

## 3. State of the Art

**Theory SOTA (established).**
- Feldman (STOC 2020) proves that under a long-tailed prior over subpopulations, *label memorization is necessary* for near-optimal generalization: any learner that does not fit singleton subpopulations loses $\Omega(\text{fraction of tail mass})$ accuracy. This is a genuine lower bound, but it bounds *label* memorization only — one bit per example — not verbatim content.
- Brown, Bun, Feldman, Smith, Talwar (STOC 2021) give the strongest existing separation: learning problems (a subpopulation task and a next-symbol prediction task) where every accurate learner must encode $\Omega(nd)$ bits about the sample, including data *irrelevant* to the target function. This is the closest thing to the theorem this problem asks for; the tasks are constructed, not natural.
- Attias, Dziugaite, Haghifam, Livni, Roy (ICML 2024) extend this to stochastic convex optimization: any learner attaining the optimal rate must have information complexity growing with dimension, so memorization is unavoidable in a canonical convex problem, not just a hand-built one.
- Bun, Livni, Moran (FOCS 2020) and Alon, Livni, Malliaris, Yehudayoff (STOC 2019) give the dual: private (hence low-memorization) PAC learnability is equivalent to finite Littlestone dimension, so *some* classes are learnable only with memorization.

**Empirical SOTA (established).** Carlini et al. (ICLR 2023) show extractable memorization grows log-linearly in model size, in example duplication count, and in prompt prefix length across the GPT-Neo family up to 6B. Morris et al. (2025) estimate raw storage capacity of GPT-style transformers at $\approx 3.6$ bits per parameter and show that memorization saturates before generalization begins.

**Claimed but unablated.** That "deduplication removes memorization without cost" is supported by loss curves but has no controlled arm isolating deduplication from the accompanying data-quality change. Extraction-rate figures on production models (Nasr et al. 2023) are benchmark numbers under a specific attack, not lower bounds on retained information — a failed attack does not establish absence.

## 4. What Is Known

- **Networks can memorize arbitrary labels.** Zhang et al. (ICLR 2017): Inception on CIFAR-10 reaches 0% training error on fully randomized labels while test accuracy stays at chance. Capacity is not the binding constraint.
- **Memorized examples carry real test accuracy.** Feldman & Zhang (NeurIPS 2020), CIFAR-100 / ResNet-50, ~2000 retrained models: removing the highest-memorization-score examples costs several points of test accuracy, materially more than removing an equal-sized random subset. The marginal value concentrates in a long tail of atypical examples.
- **Memorization precedes overfitting.** Tirumala et al. (NeurIPS 2022): models up to 1B memorize training sequences faster than they overfit, and larger models memorize more before any validation-loss turn.
- **Scaling is log-linear and duplication-dominated.** Carlini et al. (ICLR 2023), GPT-Neo 125M→6B on the Pile: extraction rate roughly doubles for a $10\times$ increase in parameters; sequences duplicated $\ge 100$ times are extracted at orders of magnitude higher rates than singletons.
- **The privacy price is measurable.** Anil et al. (2022): DP-BERT pretraining reaches 60.5% MLM accuracy at $\varepsilon=5.36$ against ~70% non-private. De et al. (2022): 81.4% on CIFAR-10 from scratch at $\varepsilon=8$. So a strong upper bound on memorization currently costs roughly 10 accuracy points at pretraining scale, and near-zero on some fine-tuning tasks (Li et al. 2022; Yu et al. 2022).
- **Small memorization is predictable.** Biderman et al. (NeurIPS 2023): which sequences a Pythia model memorizes is only weakly predicted from smaller models in the same family — low-recall forecasting, so pre-hoc filtering is not currently viable.

## 5. What Is Not Known

- **Theoretically open.** No lower bound on $\mathcal{S}_{\mathcal{F}}$ for any *naturally occurring* $\mathcal{F}$ — natural-language next-token prediction, or an image distribution defined by a real dataset. All existing separations use constructed tasks. Also open: whether the $\Omega(nd)$ bound of Brown et al. survives when the learner is restricted to gradient descent on a fixed architecture.
- **Theoretically open.** Whether the frontier $\mathcal{S}$ is convex or has a threshold: is there an $\varepsilon^\ast$ below which memorization jumps discontinuously?
- **Empirically open.** Whether the DP accuracy gap at pretraining scale is intrinsic or an artifact of DP-SGD's per-example clipping. Runnable: train matched 1–7B models with and without a memorization-suppressing objective at fixed token budget. Nobody has published the matched pair.
- **Methodologically blocked.** Content memorization in generative models has no agreed measurement. Verbatim extraction, counterfactual log-likelihood gap (Zhang et al. 2023), and adversarial-compression ratio (Schwarzschild et al. 2024) disagree on which sequences count, and none is calibrated against a retraining ground truth above ~1B parameters.

## 6. Why It Is Hard

The binding obstruction is **non-identifiability under a compute wall**. The definition of memorization is counterfactual — it needs $A(S^{\setminus i})$ — and that counterfactual costs one full pretraining run per example. At 7B parameters, one run is $\sim 10^{22}$ FLOPs; the Feldman–Zhang estimator's ~2000 runs is therefore $\sim 10^{25}$ FLOPs for a single dataset. Every scalable proxy replaces the counterfactual with a *correlate* (loss gap, extraction success), and each correlate is confounded by exactly the quantity being separated: a sequence that is easy to reproduce may be memorized or may be predictable. The evaluation does not measure what it names. Compounding this, no lower bound can be stated without fixing $\mathcal{F}$, and the pretraining distribution is not a specifiable object.

## 7. Current Research (as of 2026)

- **Information-complexity lower bounds** for convex and online learning — Livni, Attias, Dziugaite, Haghifam, Roy; the most likely route to a bound on a non-constructed task class.
- **Long-tail theory and its extension to sequence models** — Feldman and collaborators (Apple/Google); the open item is a next-token analogue of the subpopulation argument.
- **Measurement standardization** — Carlini, Tramèr, Ippolito, Lee, Jagielski and the extraction community; counterfactual and compression-based definitions are converging but not reconciled.
- **Capacity accounting** — Morris et al.'s bits-per-parameter line, being extended to separate "storage" from "grammar" bits *(frontier — verify)*.
- **Unlearning as an operational separation** — certified-removal work aims to delete example-specific bits post hoc; current guarantees hold only for convex or last-layer settings.

## 8. Concrete Next Experiment

**Goal.** Estimate the frontier $\mathcal{S}$ at a scale where the counterfactual is still affordable, and check whether it is strictly positive.

**Scale.** A 410M-parameter decoder (Pythia-410M configuration), trained on a 10B-token deduplicated slice of the Pile. Build a canary set of 5,000 held-in sequences of 128 tokens each, spanning duplication counts $\{1,2,8,64\}$. Train $K=32$ models on random 50% shards of the canary set (each canary is in ~16 models and out of ~16), giving a per-canary retraining ground truth. Cost: 32 runs $\times \approx 10^{20}$ FLOPs $\approx 3\times10^{21}$ FLOPs — days on 64 A100s, not a national-lab budget.

**Arms.**
1. Standard AdamW training (control).
2. DP-SGD at $\varepsilon\in\{8,\ 30,\ \infty\}$, same token budget.
3. Non-DP memorization penalty: an auxiliary loss penalizing the per-example log-likelihood gap against an EMA reference model.

**Deciding number.** For each arm, plot validation cross-entropy (nats/token) against measured counterfactual memorization $\overline{\mathrm{mem}}=\frac{1}{5000}\sum_i[\log p_{\text{in}}(z_i)-\log p_{\text{out}}(z_i)]$ in nats. **The decisive quantity is the validation loss of arm 3 at the $\overline{\mathrm{mem}}$ achieved by DP-SGD at $\varepsilon=8$.** If arm 3 matches DP's memorization at $\le 0.01$ nats/token worse validation loss than the control, the DP gap is a mechanism artifact and the frontier is near-flat there. If every arm reaching that $\overline{\mathrm{mem}}$ loses $\ge 0.05$ nats/token, the frontier is strictly interior and the separation is real at this scale — publishable either way. Secondary output: the calibration curve between $\overline{\mathrm{mem}}$ and extraction rate, which is the missing link that would license using extraction as a proxy at 7B+.

## 9. Key References

- **[Foundational]** Vitaly Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC, 2020. — arXiv:1906.05271
- **[Foundational]** Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals. *Understanding Deep Learning Requires Rethinking Generalization.* ICLR, 2017. — arXiv:1611.03530
- **[SOTA-theory]** Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, Kunal Talwar. *When Is Memorization of Irrelevant Training Data Necessary for High-Accuracy Learning?* STOC, 2021. — arXiv:2012.06421
- **[SOTA-theory]** Idan Attias, Gintare Karolina Dziugaite, Mahdi Haghifam, Roi Livni, Daniel M. Roy. *Information Complexity of Stochastic Convex Optimization: Applications to Generalization and Memorization.* ICML, 2024. — arXiv:2402.09327
- **[Measurement]** Vitaly Feldman, Chiyuan Zhang. *What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation.* NeurIPS, 2020. — arXiv:2008.03703
- **[SOTA-empirical]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Empirical]** Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, Nicholas Carlini. *Counterfactual Memorization in Neural Language Models.* NeurIPS, 2023. — arXiv:2112.12938
- **[Empirical]** Kushal Tirumala, Aram H. Markosyan, Luke Zettlemoyer, Armen Aghajanyan. *Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models.* NeurIPS, 2022. — arXiv:2205.10770
- **[Empirical]** Stella Biderman et al. *Emergent and Predictable Memorization in Large Language Models.* NeurIPS, 2023. — arXiv:2304.11158
- **[Capacity]** John X. Morris et al. *How Much Do Language Models Memorize?* arXiv preprint, 2025. — arXiv:2505.24832
- **[Upper bound]** Soham De, Leonard Berrada, Jamie Hayes, Samuel L. Smith, Borja Balle. *Unlocking High-Accuracy Differentially Private Image Classification through Scale.* arXiv preprint, 2022. — arXiv:2204.13650
- **[Upper bound]** Rohan Anil, Badih Ghazi, Vineet Gupta, Ravi Kumar, Pasin Manurangsi. *Large-Scale Differentially Private BERT.* Findings of EMNLP, 2022.
- **[Equivalence]** Mark Bun, Roi Livni, Shay Moran. *An Equivalence Between Private Classification and Online Prediction.* FOCS, 2020.
- **[Measurement]** Avi Schwarzschild, Zhili Feng, Pratyush Maini, Zachary C. Lipton, J. Zico Kolter. *Rethinking LLM Memorization through the Lens of Adversarial Compression.* NeurIPS, 2024. — arXiv:2404.15146

## 10. Worked Example

Take a single canary: a 128-token address block appearing **once** in a 10B-token corpus.

**Capacity accounting.** At 3.6 bits/parameter (Morris et al. 2025), a 410M model holds $\approx 1.5\times10^{9}$ bits $= 185$ MB. The canary at ~4.5 bits/token of residual entropy is ~576 bits — $4\times10^{-7}$ of capacity. Storing it is free in capacity terms, so no counting argument can forbid it. This is why the problem is about the *frontier*, not about capacity exhaustion.

**Measurement.** Train 32 shard models. Suppose in-set mean log-likelihood is $-31$ nats for the sequence and out-of-set is $-58$ nats. Then $\mathrm{mem}=27$ nats $\approx 39$ bits — the model retains about 7% of the canary's content, and 93% of what it "knows" is generic English structure it would have had anyway.

**The obstruction, made visible.** Now run the standard scalable proxy on the same canary: greedy decoding from a 50-token prefix. The continuation is not reproduced verbatim, so extraction reports **zero memorization**. Two measurements of the same object on the same checkpoint disagree by 39 bits. Change the canary to a common street name and the *out*-of-set likelihood rises to $-34$ nats; $\mathrm{mem}$ collapses to 3 nats while extraction now *succeeds*, because the sequence is predictable, not stored. The proxy and the counterfactual are anti-correlated on this pair.

At 410M the counterfactual arbitrates. At 70B the 32 retrains cost $\sim 10^{24}$ FLOPs and nobody runs them, so the field reports the proxy — which, as this pair shows, can be wrong in both directions. Any claimed separation at frontier scale currently rests on a measurement that has never been calibrated against its own definition.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*