---
id: 06-data-pipeline/coresets-for-deep-training
title: "Coreset Guarantees for Deep Network Training"
topic: 06-data-pipeline
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Coreset Guarantees for Deep Network Training

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/coresets-for-deep-training` · **Status:** open

## 1. Problem Statement

A **coreset** is a small weighted subset of a training set that provably substitutes for the whole set. For convex problems the guarantee is uniform over the hypothesis class: every parameter vector gets its loss preserved to $1\pm\epsilon$. For deep networks no such guarantee is known, and the practice — "data pruning", "subset selection" — has drifted away from the word's original meaning.

Three variants, routinely conflated:

- **Theory variant.** Does there exist, for a fixed architecture family $\mathcal{F}$ of depth $\ge 2$ with ReLU activations, a weighted subset of size $\mathrm{poly}(d,1/\epsilon)$ — independent of $n$ — that $\epsilon$-approximates the empirical loss uniformly over $\mathcal{F}$? Open, and the known negative results for logistic regression suggest the honest answer is no without data-dependent complexity parameters.
- **Method variant.** Given a budget $b = \beta n$, pick $S$ so that SGD run on $S$ reaches test accuracy within $\delta$ of SGD run on the full set. This is an *optimization-trajectory* claim, not a function-approximation claim, and it is what every practical method actually targets.
- **Measurement variant.** What counts as "the same training run"? Equal epochs, equal gradient steps, equal FLOPs and equal wall-clock give different rankings of the same methods. Absent a fixed convention, published speedups are not comparable.

Solving the theory variant means a theorem with a size bound and a matching lower bound. Solving the method variant means a selection rule that beats uniform random sampling at a *fixed compute budget including selection cost*, across at least two modalities, with the margin surviving seed variance.

## 2. Formal Setting

Data $D=\{z_i\}_{i=1}^n$, $z_i=(x_i,y_i)$. Parameters $\theta\in\mathbb{R}^p$. Loss $\ell(\theta,z)\ge 0$. Empirical risk

$$L(\theta) = \frac{1}{n}\sum_{i=1}^n \ell(\theta,z_i),\qquad L_S(\theta)=\sum_{i\in S} w_i\,\ell(\theta,z_i).$$

**Strong (uniform) coreset.** $S$ with weights $w$ is an $\epsilon$-coreset for $\mathcal{F}$ if
$$\sup_{\theta\in\mathcal{F}}\ \bigl|L_S(\theta)-L(\theta)\bigr| \le \epsilon\, L(\theta).$$
Measured by: exhaustive evaluation is impossible, so in practice one reports $\max$ over a finite probe set of checkpoints — an *upper-bounded-below* estimate that can only understate the violation.

**Sensitivity.** $s_i = \sup_{\theta\in\mathcal{F}} \frac{\ell(\theta,z_i)}{\sum_j \ell(\theta,z_j)}$, total sensitivity $\mathfrak{S}=\sum_i s_i$. Feldman–Langberg sampling with probabilities $\propto s_i$ and weights $w_i = \mathfrak{S}/(|S| s_i)$ gives size $\tilde O(\mathfrak{S}\cdot \mathrm{vcdim}/\epsilon^2)$. Measured by: $s_i$ is a supremum over a non-convex landscape; every deep-network implementation replaces it with a bound computed at one checkpoint.

**Gradient-matching surrogate (the practical object).** Choose $S,w$ minimizing
$$\Bigl\|\ \sum_{i=1}^n \nabla_\theta \ell(\theta_t,z_i) \;-\; \sum_{i\in S} w_i \nabla_\theta \ell(\theta_t,z_i)\ \Bigr\|_2$$
at the current iterate $\theta_t$. Measured by: last-layer gradients only (full $\nabla_\theta$ is $p$-dimensional, $p\sim10^8$–$10^{11}$), re-solved every $R$ epochs.

**Scoring surrogates.** EL2N score $\|\,\mathrm{softmax}(f_{\theta_t}(x_i)) - y_i\|_2$ averaged over seeds; forgetting count $\\#\{t: \text{correct}\to\text{incorrect}\}$; datamodel influence $\hat\tau_i$ from linear regression of held-out loss on random subset masks.

**Compute accounting.** $C_{\text{total}} = C_{\text{select}} + C_{\text{train}}(S)$. Fair comparison fixes $C_{\text{total}}$, not $|S|$.

**Assumptions known to be violated.** (i) Convexity / a fixed hypothesis class — deep training visits a data-dependent region, so $\mathcal{F}$ is not fixed in advance. (ii) Loss-value approximation implies trajectory equivalence — false: SGD on a reweighted objective follows a different path with different implicit regularization. (iii) Score stability across seeds — EL2N and forgetting scores have rank correlation well below 1 between seeds. (iv) Score transfer across scale — scores from a small proxy do not preserve ranking for a larger target model. (v) Clean labels — sensitivity-style scores concentrate on mislabeled and atypical points, which is exactly wrong under label noise.

## 3. State of the Art

**Theory SOTA (established).** Feldman–Langberg sensitivity sampling (STOC 2011) and the Cohen-Addad–Saulpic–Schwiegelshohn framework (STOC 2021) give $n$-independent coresets for clustering and shape-fitting. Munteanu et al. (NeurIPS 2018) is the sharp boundary case: logistic regression admits coresets of size $\mathrm{poly}(\mu,d,1/\epsilon)$ where $\mu$ measures label-complexity of the data, **and** no sublinear coreset exists for general instances ($\mu$ unbounded). Tukan et al. (NeurIPS 2020) extend to near-convex functions. No uniform-coreset theorem exists for a ReLU network of depth $\ge 2$.

**Empirical SOTA (established by benchmark, not by theorem).** CRAIG (Mirzasoleiman et al., ICML 2020) and GRAD-MATCH (Killamsetty et al., ICML 2021) do submodular / orthogonal-matching-pursuit gradient matching. EL2N/GraNd (Paul et al., NeurIPS 2021) score early. CCS (Zheng et al., ICLR 2023) fixes the high-pruning-rate failure by stratifying over the score range. DsDm (Engstrom et al., ICML 2024) selects by datamodel-estimated target-task influence. At LLM scale the field runs quality/dedup filters (SemDeDup; DataComp, NeurIPS 2023) and domain reweighting (DoReMi, DSIR, NeurIPS 2023), not coresets.

**Claimed but unablated.** Reported speedups usually exclude selection cost, and often score with the same architecture as the target — a leak that flatters transferability. The DeepCore benchmark (Guo, Zhao, Bai, 2022) is the main independent re-run and finds most methods do not beat random uniform sampling at aggressive pruning on CIFAR-10/ImageNet. Sorscher et al. (NeurIPS 2022) claim data pruning can beat power-law scaling; the exponent-breaking claim rests on a theory model plus ImageNet ResNet-50 curves, and has not been independently reproduced at another modality/scale.

## 4. What Is Known

- **Lower bound, convex case.** No $o(n)$-size coreset for logistic regression on arbitrary data (Munteanu et al., NeurIPS 2018). Deep networks strictly generalize logistic regression, so the negative result inherits upward.
- **Random is a hard baseline.** DeepCore (2022): across 10+ methods on CIFAR-10 and ImageNet, at retention $\le 10\%$ no method reliably beats uniform random; at 30–70% retention margins over random are typically 1–3 accuracy points.
- **Score direction flips with pruning rate.** Zheng et al. (ICLR 2023): keeping only *hard* examples is best at mild pruning and catastrophic at severe pruning; keeping easy/prototypical points is better in the low-data regime. Same dataset, opposite rule.
- **Early scores work.** EL2N at epoch ~10 on CIFAR-10 identifies a prunable ~50% with near-baseline final accuracy (Paul et al., NeurIPS 2021, ResNet-18/CIFAR-10, CIFAR-100).
- **Noise sensitivity.** Score-based selection preferentially retains mislabeled points; robustness needs an explicit correction (Mirzasoleiman et al., NeurIPS 2020).
- **Curation is compute-dependent.** Goyal et al. (CVPR 2024): the optimal filtering aggressiveness changes with the training-compute budget — a fixed "good subset" does not exist independent of the budget it will be trained under.

## 5. What Is Not Known

- **Theoretically open.** Whether any $n$-independent uniform coreset exists for depth-$\ge 2$ ReLU networks under a bounded data-complexity parameter analogous to $\mu$. No proof either way. Also open: whether trajectory-equivalence (not loss-equivalence) admits any nontrivial size bound.
- **Empirically open.** Whether *any* selection method beats random uniform sampling at matched total compute $C_{\text{select}}+C_{\text{train}}$ at LLM pretraining scale ($\ge 10^{10}$ tokens). Runnable; not run as a controlled head-to-head.
- **Empirically open.** Cross-scale score transfer: does a ranking computed with a 100M-parameter proxy retain its advantage for a 7B target? Selection-via-proxy (Coleman et al., ICLR 2020) shows transfer within a narrow band; the wide band is untested.
- **Methodologically blocked.** "Equivalent training run" has no agreed definition — equal epochs, equal steps, or equal FLOPs. Until fixed, the field's headline speedups are not comparable across papers.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the target**. A coreset guarantee is uniform over a hypothesis class; deep training does not visit a fixed class, it visits a trajectory that *depends on the subset chosen*. Selecting $S$ changes $\theta_t$, which changes the gradients that justified selecting $S$. Every practical method breaks this circularity by freezing $\theta_t$ at a checkpoint — so the guarantee holds for a model that the run will no longer produce.

Second obstruction: **the evaluation does not measure what it names**. "Coreset" implies a uniform bound; the reported number is single-run final test accuracy at a fixed retention ratio, with selection cost excluded. Those are different claims, and the second can be satisfied by a method that provably violates the first.

## 7. Current Research (as of 2026)

- Datamodel/influence-based selection scaled by approximate estimators (Engstrom et al.; MadryLab lineage). *(frontier — verify current scale)*
- Compute-aware curation: selection policy as a function of the training budget, following Goyal et al. (CVPR 2024).
- Coverage/stratification-first selection (CCS descendants) as the pragmatic answer to the score-direction flip.
- Coreset theory pushing outward from near-convex classes and from clustering to kernel/NTK-linearized regimes — the honest statement is that NTK-regime results do not transfer to feature-learning regimes. *(frontier — verify)*
- LLM-side: dedup, quality classifiers, domain mixture optimization. Explicitly not coresets, and rarely evaluated as such.

## 8. Concrete Next Experiment

**Question.** Does any selection method beat random uniform sampling at *matched total compute*, once selection cost is charged?

**Scale.** 1.4B-parameter decoder LM, Chinchilla-style budget, ~30B training tokens drawn from a 300B-token pool. Three seeds per arm.

**Arms.**
1. **Control:** uniform random 30B-token sample. Selection cost = 0.
2. Score-based: EL2N-style early-loss scoring with a 160M proxy trained on 3B tokens; charge the proxy run to the budget (reduce arm 2's training tokens by the proxy FLOPs equivalent, ~2–3%).
3. Gradient-matching (GRAD-MATCH, last-layer, re-solved every 2B tokens); charge the matching solves.
4. Dedup-only (SemDeDup) at equal token count.

**Decision number.** Mean held-out log-likelihood on a fixed 5-domain suite, arm $k$ minus control, in nats/token. Decide **yes** only if the gap exceeds $3\times$ the across-seed standard deviation of the control arm (expect $\sigma\approx 0.003$ nats/token at this scale; threshold $\approx 0.01$). A single number, one comparison, no per-benchmark cherry-picking.

**Why it settles something.** Every existing positive result is at $\le$ ImageNet scale with selection cost excluded. If no arm clears $3\sigma$, the field's claim of practical coresets for large-scale deep training is empirically refuted at the scale that matters; if one does, it is the first compute-honest positive.

## 9. Key References

- **[Foundational]** Dan Feldman, Michael Langberg. *A Unified Framework for Approximating and Clustering Data.* STOC, 2011.
- **[Foundational]** Sariel Har-Peled, Soham Mazumdar. *On Coresets for k-Means and k-Median Clustering.* STOC, 2004.
- **[Foundational / lower bound]** Alexander Munteanu, Chris Schwiegelshohn, Christian Sohler, David P. Woodruff. *On Coresets for Logistic Regression.* NeurIPS, 2018.
- **[Theory]** Vladimir Braverman, Vincent Cohen-Addad, Shaofeng H.-C. Jiang, Robert Krauthgamer, Chris Schwiegelshohn, Mads Bjerregaard Toftrup, Xuan Wu. *The Power of Uniform Sampling for Coresets.* FOCS, 2022.
- **[Theory]** Murad Tukan, Alaa Maalouf, Dan Feldman. *Coresets for Near-Convex Functions.* NeurIPS, 2020.
- **[SOTA — method]** Baharan Mirzasoleiman, Jeff Bilmes, Jure Leskovec. *Coresets for Data-Efficient Training of Machine Learning Models.* ICML, 2020.
- **[SOTA — method]** Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, Abir De, Rishabh Iyer. *GRAD-MATCH: Gradient Matching Based Data Subset Selection for Efficient Deep Model Training.* ICML, 2021.
- **[SOTA — scoring]** Mansheej Paul, Surya Ganguli, Gintare Karolina Dziugaite. *Deep Learning on a Data Diet: Finding Important Examples Early in Training.* NeurIPS, 2021.
- **[SOTA — scaling claim]** Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, Ari Morcos. *Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning.* NeurIPS, 2022.
- **[SOTA — coverage]** Haizhong Zheng, Rui Liu, Fan Lai, Atul Prakash. *Coverage-Centric Coreset Selection for High Pruning Rates.* ICLR, 2023.
- **[SOTA — influence]** Logan Engstrom, Axel Feldmann, Aleksander Mądry. *DsDm: Model-Aware Dataset Selection with Datamodels.* ICML, 2024.
- **[Benchmark]** Chengcheng Guo, Bo Zhao, Yanbing Zhao. *DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning.* DEXA, 2022.
- **[Benchmark]** Samir Yitzhak Gadre et al. *DataComp: In Search of the Next Generation of Multimodal Datasets.* NeurIPS Datasets & Benchmarks, 2023.
- **[Compute-awareness]** Sachin Goyal, Pratyush Maini, Zachary C. Lipton, Aditi Raghunathan, J. Zico Kolter. *Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic.* CVPR, 2024.
- **[Robustness]** Baharan Mirzasoleiman, Kaidi Cao, Jure Leskovec. *Coresets for Robust Training of Deep Neural Networks against Noisy Labels.* NeurIPS, 2020.
- **[Proxy]** Cody Coleman, Christopher Yeh, Stephen Mussmann, Baharan Mirzasoleiman, Peter Bailis, Percy Liang, Jure Leskovec, Matei Zaharia. *Selection via Proxy: Efficient Data Selection for Deep Learning.* ICLR, 2020.
- **[Survey]** Dan Feldman. *Core-Sets: Updated Survey.* In *Sampling Techniques for Supervised or Unsupervised Tasks*, Springer, 2020.

## 10. Worked Example

CIFAR-10, $n=50{,}000$, ResNet-18, target retention $\beta=0.1$ ($|S|=5{,}000$).

**Step 1 — selection cost.** EL2N needs a proxy run of ~10 epochs, averaged over 10 seeds: $10\times10 = 100$ proxy epochs. Full training is 200 epochs. Training on $S$ for 200 epochs costs $0.1\times200 = 20$ epoch-equivalents. So

$$C_{\text{total}} = 100 + 20 = 120 \text{ epoch-equivalents},$$

against 200 for the full run. The advertised "$10\times$ less data" is a **1.67× speedup**, and collapses to ~1.0× if the score is computed with a single seed at 20 epochs instead.

**Step 2 — the accuracy flip.** Full-data ResNet-18 reaches ~95% on CIFAR-10. At $\beta=0.1$, top-EL2N selection lands in the 60s–70s — *below* uniform random at the same budget — because the 5,000 highest-score points are dominated by ambiguous and mislabeled images and cover only part of the class-conditional support. CCS-style stratification over the score range recovers most of the gap. The rule that wins at $\beta=0.7$ loses at $\beta=0.1$.

**Step 3 — where the obstruction becomes visible.** Take the gradient-matching objective at $\theta_{t}$ (epoch 10) and evaluate the same weighted subset at $\theta_{t'}$ (epoch 120) of the run it produced. The matched residual $\|\sum_i \nabla\ell - \sum_{i\in S} w_i\nabla\ell\|$ is small at $\theta_t$ by construction and large at $\theta_{t'}$, because the subset-trained model has moved to a region the selection never saw. The selection was justified by a checkpoint the run then destroyed. That is the circularity, not an implementation defect: no fixed $S$ can be gradient-optimal for the trajectory it itself induces, and this is exactly the property a uniform coreset would supply and no known deep-network method has.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*