---
id: 10-scaling-laws/memorization-generalization-scaling
title: "Scaling Laws for Memorization Versus Generalization"
topic: 10-scaling-laws
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Laws for Memorization Versus Generalization

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/memorization-generalization-scaling` · **Status:** open

## 1. Problem Statement

Loss scaling laws predict aggregate test loss $L(N, D)$ from parameters $N$ and tokens $D$. They say nothing about how that loss decomposes into *storage of specific training examples* versus *acquisition of transferable structure*. The problem: give a predictive law for that decomposition.

Three variants, of different difficulty:

- **Measurement.** Define a per-example quantity $m_i \in [0, \infty)$ in bits that is (a) attributable to example $i$, (b) additive or near-additive over a dataset, and (c) computable at frontier scale without retraining. Verbatim-extraction rates fail (c) is easy but (a) and (b) badly — a sequence can be emitted because it was stored or because it is predictable.
- **Method.** Given a compute budget $C$, choose $(N, D)$, epoch count, and deduplication threshold to minimize memorization at fixed test loss. Currently done by folklore ("dedupe, one epoch").
- **Theory.** Prove or refute that for a fixed architecture family there exists a capacity constant $\kappa$ (bits per parameter) such that total unintended memorization is $\min(\kappa N, H(D))$, and that generalization gains begin only once $H(D) > \kappa N$.

Solving it means: a fitted law that predicts, on a held-out scale at least $10\times$ larger than anything used to fit it, both test loss and a memorization measure, with the memorization prediction accurate to within a factor of 2.

## 2. Formal Setting

Model $f_\theta$, $\theta \in \mathbb{R}^N$, trained on $S = \{x_1, \dots, x_n\}$ drawn i.i.d. from $\mathcal{D}$, total tokens $D$. Cross-entropy loss $L(x) = -\frac{1}{|x|}\sum_t \log p_\theta(x_t \mid x_{<t})$.

**Extraction memorization** (Carlini et al., 2023), as measured: a sequence $x$ of length $\ell$ is $k$-extractable if greedy decoding from the first $k$ tokens reproduces the remaining $\ell - k$ exactly. Reported quantity is the empirical rate
$$\hat{M}_{k,\ell} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}\big[\text{greedy}(x_i^{\le k}) = x_i^{>k}\big].$$
Measured by sampling $10^4$–$10^6$ training sequences and one forward pass each. Cheap; not attributable.

**Counterfactual memorization** (Zhang et al., 2023), as measured: train $R$ models on random subsets, and take
$$\mathrm{mem}(x_i) = \mathbb{E}_{S \ni x_i}\big[\text{perf}(f_S, x_i)\big] - \mathbb{E}_{S \not\ni x_i}\big[\text{perf}(f_S, x_i)\big].$$
Attributable and near-additive; costs $R$ training runs ($R \approx 400$ in published work). Infeasible above ~1B parameters.

**Bit-level memorization** (Morris et al., 2025), as measured: with a reference model $\hat{f}$ trained without $x$,
$$\mathrm{mem}_U(x) \;=\; \underbrace{\mathrm{HK}(x \mid \hat{f})}_{\text{code length under ref.}} - \underbrace{\mathrm{HK}(x \mid f_\theta)}_{\text{code length under target}},$$
approximated by the arithmetic-coding length difference $\sum_t [-\log_2 \hat{p}(x_t) + \log_2 p_\theta(x_t)]$. This is the only measure with units that let capacity claims be stated: $\kappa = \sup_D \sum_i \mathrm{mem}_U(x_i) / N$.

**Capacity hypothesis.** $\sum_i \mathrm{mem}_U(x_i) = \min(\kappa N,\, H(S))$, with generalization $\partial L_{\text{test}}/\partial D$ turning sharply negative at the crossing point $H(S) = \kappa N$.

**Assumptions known to be violated in practice.** (i) *I.i.d. sampling* — web corpora contain near-duplicates at $10^2$–$10^4$ multiplicity, and duplication is the strongest single driver of memorization. (ii) *A clean reference model* — any $\hat{f}$ trained on web data has likely seen $x$ or a paraphrase, so $\mathrm{mem}_U$ is a lower bound of unknown tightness. (iii) *Verbatim identity* — style transfer and paraphrase leak content that $\hat{M}_{k,\ell}$ scores as zero (Ippolito et al., 2023). (iv) *Single-epoch training* — frontier runs now repeat high-quality data, breaking the $H(S)$ bookkeeping.

## 3. State of the Art

**Established (replicated, ablated).**
- Memorization grows log-linearly in model size, in the number of duplicates of a sequence, and in prompt-prefix length $k$ — GPT-Neo 125M→6B, Pythia 70M→12B (Carlini et al., ICLR 2023; Biderman et al., NeurIPS 2023).
- Deduplication cuts verbatim emission by roughly $10\times$ at matched loss, and *improves* test loss (Lee et al., ACL 2022; Kandpal et al., ICML 2022). This is the one intervention with a clean ablation on both sides of the trade-off.
- Memorization precedes overfitting: memorization of a training batch rises while validation loss is still falling (Tirumala et al., NeurIPS 2022).

**Claimed but not independently ablated.**
- $\kappa \approx 3.6$ bits/parameter for GPT-2-style transformers in bfloat16 (Morris et al., 2025), fitted over $N \in [5\times10^5, 1.5\times10^9]$ on synthetic uniform-random sequences plus a FineWeb arm. The synthetic arm is where the constant is identified; the transfer of $\kappa$ to natural text is asserted, not demonstrated.
- $\approx 2$ bits/parameter of *factual* knowledge capacity, robust across depth/width and int8 quantization (Allen-Zhu & Li, 2024). Measured on synthetic biography data; no natural-corpus replication.
- The claim that these two constants describe the same physical quantity is unsupported.

**Benchmark-number-only.** Extraction rates from production models (Nasr et al., 2023) — a divergence attack recovered several megabytes from ChatGPT — establish that memorization is exploitable at frontier scale but give no scaling exponent, since the training set is unknown.

## 4. What Is Known

- **Duplication dominates.** In Carlini et al. (ICLR 2023) on Pythia/GPT-Neo, sequences duplicated $\sim10^2$ times are memorized at rates one to two orders of magnitude above singletons at fixed $N$.
- **Repeated data has a damage threshold.** Repeating 0.1% of the corpus 100 times degraded an 800M-parameter model to the test loss of a 400M-parameter model — a full halving of effective capacity (Hernandez et al., 2022, $N$ up to 800M).
- **Repetition is nearly free up to 4 epochs.** Loss from 4 epochs of repeated data is within noise of fresh data; marginal value is essentially zero by ~40 epochs (Muennighoff et al., NeurIPS 2023; up to 9B parameters, 900B tokens).
- **Small models do not predict large-model memorization.** Pythia 70M memorization of a specific sequence predicts 12B memorization with low recall — aggregate rates extrapolate, per-sequence identity does not (Biderman et al., NeurIPS 2023).
- **Memorization partially decays.** Examples memorized early are forgotten during later training, with a heavy tail that never decays (Jagielski et al., ICLR 2023).
- **Some memorization is necessary.** For long-tailed label distributions, near-optimal generalization requires memorizing singleton examples (Feldman, STOC 2020); there are learning tasks where any accurate learner must store $\Omega(n)$ bits of irrelevant training data (Brown et al., STOC 2021).

## 5. What Is Not Known

- **Theoretically open.** Whether $\kappa$ is architecture-determined or dataset-determined. Feldman's lower bound says memorization must be nonzero; there is no matching upper bound tying the *amount* to $N$ for transformers on natural language. No proof that the capacity-saturation point coincides with the onset of generalization, despite the empirical coincidence reported by Morris et al.
- **Empirically open.** The single decisive run — fit $\kappa$ at $N \le 10^9$ and test the prediction at $N = 10^{11}$ on the same corpus — is runnable today by any frontier lab and has not been published. Also open: whether $\kappa$ shifts under modern recipes (MoE, multi-epoch curricula, distillation, RL post-training).
- **Methodologically blocked.** Per-example attribution at frontier scale. Counterfactual memorization needs hundreds of runs; the compression estimator needs an uncontaminated reference model, which does not exist for web-scale corpora. Until one of these is fixed, "memorization" at $N > 10^{10}$ is measured only by verbatim proxies that are known to both over- and under-count.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the measurement**, not compute. A model emits a training string for two reasons — it stored the string, or the string is predictable from structure the model learned. Every scalable estimator conflates them:

- $\hat{M}_{k,\ell}$ counts "1 + 1 = 2" and a memorized SSN identically.
- $\mathrm{mem}_U$ subtracts a reference model to separate them, but the reference is contaminated; the correction is a lower bound with no known slack.
- Counterfactual memorization identifies the quantity correctly and costs $O(R)$ training runs, so it exists only below 1B parameters.

Consequently the memorization axis of any scaling law is fitted on a proxy whose bias itself scales with $N$ — larger models are better at *predicting* text, so the fraction of emissions falsely scored as memorization changes with the independent variable. Compounding this: membership inference on LLMs performs near chance under proper controls (Duan et al., COLM 2024), and privacy-defense evaluations that average over examples systematically understate tail risk (Aerni et al., CCS 2024). The field lacks ground truth, not FLOPs.

## 7. Current Research (as of 2026)

- **Capacity measurement in bits.** Meta FAIR / Cornell (Morris, Chaudhuri, Mahloujifar) on unintended-memorization capacity; Microsoft Research (Allen-Zhu, Li) on knowledge-bit capacity. Both push toward a single $\kappa$; the two constants have not been reconciled. *(frontier — verify)*
- **Memorization taxonomy.** EleutherAI and collaborators decompose memorization into recitation (duplicated text), reconstruction (templatic), and recollection (rare) — different scaling behavior per class (Prashanth et al., 2024), which implies a single scalar law is misspecified.
- **Data-constrained allocation.** Extending Muennighoff-style repetition laws into the $\ge 5$-epoch regime now that high-quality web text is a binding constraint. *(frontier — verify)*
- **Interpretability-side attribution.** Locating memorized content in specific parameter subsets to make $\mathrm{mem}$ structurally identifiable rather than behaviorally inferred. Early and unvalidated. *(frontier — verify)*

## 8. Concrete Next Experiment

**Test whether $\kappa$ is a constant or a fitting artifact.**

- **Scale.** Train a ladder at $N \in \{0.5, 1, 2, 4, 8\} \times 10^8$ and one held-out point at $N = 8\times10^9$ (a $10\times$ extrapolation), fixed decoder-only architecture, on a *fully deduplicated* natural corpus with an injected canary set: $10^4$ synthetic sequences of exactly 256 uniformly random bits each, at controlled multiplicities $\{1, 4, 16, 64\}$. Total: about $6\times10^{22}$ FLOPs, ~2,000 H100-days.
- **Control arm.** The identical ladder with the canary set removed, and a second control where canaries are replaced by natural sequences of matched perplexity. The random-bit canaries are the ground truth the field lacks: their entropy is exactly 256 bits, so $\mathrm{mem}_U$ has a known ceiling and the reference-contamination problem vanishes for them. The natural-text control measures how much the estimator's bias moves with $N$.
- **Deciding number.** Fit $\kappa$ on the five small points; predict total canary bits stored at $N = 8\times10^9$. **The problem moves if the predicted value is within a factor of 2 of measured.** A miss by more than $2\times$ falsifies the constant-$\kappa$ hypothesis in one run. Secondary readout: the gap between the canary-derived $\kappa$ and the natural-text-derived $\kappa$ at each $N$ quantifies the estimator bias directly — if that gap grows monotonically with $N$, every published natural-text capacity number is scale-dependent and the measurement problem, not the theory, is the blocker.

## 9. Key References

- **[Foundational]** Vitaly Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC, 2020. — arXiv:1906.05271
- **[Foundational]** Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, Kunal Talwar. *When is Memorization of Irrelevant Training Data Necessary for High-Accuracy Learning?* STOC, 2021. — arXiv:2012.06421
- **[Foundational]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[SOTA]** Jack Morris, Chawin Sitawarin, Chuan Guo, Narine Kokhlikyan, G. Edward Suh, Alexander M. Rush, Kamalika Chaudhuri, Saeed Mahloujifar. *How Much Do Language Models Memorize?* 2025. — arXiv:2505.24832
- **[SOTA]** Zeyuan Allen-Zhu, Yuanzhi Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* 2024. — arXiv:2404.05405
- **[SOTA]** Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, Colin Raffel. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- **[SOTA]** Stella Biderman, USVSN Sai Prashanth, Lintang Sutawika, Hailey Schoelkopf, Quentin Anthony, Shivanshu Purohit, Edward Raff. *Emergent and Predictable Memorization in Large Language Models.* NeurIPS, 2023. — arXiv:2304.11158
- Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, Nicholas Carlini. *Counterfactual Memorization in Neural Language Models.* NeurIPS, 2023. — arXiv:2112.12938
- Kushal Tirumala, Aram H. Markosyan, Luke Zettlemoyer, Armen Aghajanyan. *Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models.* NeurIPS, 2022. — arXiv:2205.10770
- Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, et al. *Scaling Laws and Interpretability of Learning from Repeated Data.* 2022. — arXiv:2205.10487
- Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini. *Deduplicating Training Data Makes Language Models Better.* ACL, 2022. — arXiv:2107.06499
- Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[Survey]** Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, Katherine Lee. *Scalable Extraction of Training Data from (Production) Language Models.* 2023. — arXiv:2311.17035

## 10. Worked Example

Take a 1.4B-parameter model trained Chinchilla-optimally on $D = 28\times10^9$ tokens.

**Capacity side.** At $\kappa = 3.6$ bits/parameter, storage capacity is
$$3.6 \times 1.4\times10^9 = 5.0\times10^9 \text{ bits} \approx 630\ \text{MB}.$$

**Data side.** A well-trained model reaches roughly $2.0$ nats/token $\approx 2.9$ bits/token on held-out web text, so the corpus carries about
$$28\times10^9 \times 2.9 = 8.1\times10^{10} \text{ bits}.$$

Ratio: $8.1\times10^{10} / 5.0\times10^{9} \approx 16$. The corpus exceeds capacity by $16\times$, so the capacity hypothesis says this model is deep in the generalization regime and total unintended memorization is pinned at $5.0\times10^9$ bits regardless of how much more data you add.

**Where the obstruction becomes visible.** That 630 MB is an *aggregate*. It is consistent with radically different allocations:

| Allocation | Sequences stored | Extraction rate $\hat{M}$ | Privacy harm |
|---|---|---|---|
| Spread thin: 1 bit each over $5\times10^9$ sequences | $0$ verbatim | $\approx 0$ | none |
| Concentrated: 256 bits each on $2\times10^7$ rare sequences | $2\times10^7$ | $\approx 0.07\%$ of corpus | severe |

Both saturate $\kappa N$ exactly. The aggregate law cannot distinguish them, and the measured $\hat{M}_{k,\ell}$ on this model — around $0.1$–$1\%$ at $k=50$ for models in this size class — is *also* consistent with both, because the concentrated case's memorized sequences may be rare enough never to appear in a $10^4$-sequence sample, while the thin case's near-misses can still be greedily completed from structure alone.

A single canary tells you which world you are in, and costs nothing to inject: 256 random bits, duplicated 4 times, is $10^{-8}$ of the corpus and $2\times10^{-7}$ of capacity. If it comes back verbatim, the allocation is concentrated. Nobody publishes this alongside their scaling law. That omission, not the FLOPs, is why the problem is open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*