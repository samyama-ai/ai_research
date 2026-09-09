---
id: 06-data-pipeline/label-noise-without-clean-validation
title: "Detecting Label Noise Without Clean Validation Data"
topic: 06-data-pipeline
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Detecting Label Noise Without Clean Validation Data

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/label-noise-without-clean-validation` · **Status:** partially-solved

## 1. Problem Statement

**Input.** A dataset $\tilde{D} = \{(x_i, \tilde{y}_i)\}_{i=1}^n$ whose labels $\tilde{y}_i$ were produced by a noisy process (crowd workers, web scraping, weak supervision, a teacher model). No verified-clean subset is available, and no second annotation pass is budgeted.

**Output.** Either (a) a per-example score $s_i \in \mathbb{R}$ ranking examples by probability that $\tilde{y}_i \neq y_i^\star$ (the unobserved clean label), or (b) a hard set $\hat{E} \subseteq [n]$ of flagged examples, or (c) a scalar estimate $\hat{\rho}$ of the dataset-level noise rate.

**Decision predicate.** For (a): does $s$ achieve area under the precision–recall curve (AUPRC) against the true error mask that matches what a detector tuned on a clean held-out set achieves? For (c): is $|\hat{\rho} - \rho^\star|$ below a stated tolerance without any clean anchor?

Three variants, routinely conflated:

- **Measurement.** Score and rank corrupted examples. Partially solved for class-conditional noise; unresolved for instance-dependent noise.
- **Method.** Train a model that is *robust* to the noise without detecting it. Substantially solved — robust losses and sample-selection methods recover most clean-data accuracy. Robustness is not detection: a robust learner can ignore errors it cannot name, which does not fix the dataset.
- **Theory.** Is the clean posterior $p(y^\star \mid x)$ identifiable from $p(x, \tilde{y})$ alone? Provably **no** in general. Identifiability holds only under structural assumptions (anchor points, irreducibility, clusterability, part-dependence).

Solving it means: a procedure with no clean data whose detection AUPRC on real human-annotated noise is statistically indistinguishable from a clean-validation-tuned oracle, plus a stated assumption under which it is consistent.

## 2. Formal Setting

Clean joint $P(X, Y)$ over $\mathcal{X} \times [K]$. Corruption is a Markov kernel
$$T(x)_{jk} = \Pr[\tilde{Y} = k \mid Y = j, X = x],$$
giving observed posterior $p(\tilde{y} = k \mid x) = \sum_j T(x)_{jk}\, p(y^\star = j \mid x)$.

- **Class-conditional noise (CCN):** $T(x) = T$, a $K \times K$ row-stochastic matrix, constant in $x$.
- **Symmetric noise:** $T = (1-\rho)I + \frac{\rho}{K-1}(\mathbf{1}\mathbf{1}^\top - I)$.
- **Instance-dependent noise (IDN):** $T(x)$ varies; this is the real case.

**Measured quantities.**

- Noise rate $\rho^\star = \frac{1}{n}\sum_i \mathbb{1}[\tilde y_i \neq y_i^\star]$ — measurable only by re-annotating a sample, e.g. Mechanical Turk consensus over 5 workers.
- Detector score $s_i$; evaluated as $\mathrm{AUPRC}(s, e)$ with $e_i = \mathbb{1}[\tilde y_i \neq y_i^\star]$, or F1 at a fixed flag budget $b = |\hat E|/n$.
- **Confident learning** counts: with out-of-sample predicted probabilities $\hat p(k \mid x_i)$ from $k$-fold cross-validation, per-class thresholds $t_k = \frac{1}{|\tilde D_k|}\sum_{i \in \tilde D_k} \hat p(k\mid x_i)$, and confident joint $C_{jk} = |\{i : \tilde y_i = j,\ \hat p(k \mid x_i) \ge t_k,\ k = \arg\max_{l: \hat p(l|x_i)\ge t_l} \hat p(l \mid x_i)\}|$.
- **AUM** (area under the margin): $\mathrm{AUM}_i = \frac{1}{E}\sum_{t=1}^{E} \big(z^{(t)}_{i,\tilde y_i} - \max_{k \neq \tilde y_i} z^{(t)}_{i,k}\big)$, logits $z$ averaged over $E$ training epochs.
- **Anchor point:** $x$ with $p(y^\star = j \mid x) = 1$; then $T_{jk} = p(\tilde y = k \mid x)$ directly.

**Assumptions, and which break.**

| Assumption | Status in practice |
|---|---|
| $T$ independent of $x$ | **Violated.** Real errors concentrate on ambiguous instances (fine-grained classes, blurred images, borderline sentiment). |
| Anchor points exist and are findable | **Violated/unverifiable** — the anchor is identified by $\max_x \hat p(\tilde y = j \mid x)$, which is itself noise-contaminated. |
| One true label per example | **Violated.** ImageNet has multi-object images with no single correct label (Beyer et al., 2020). |
| Clean class posterior is learnable | Holds only up to model capacity; early-learning is empirical, not guaranteed. |
| Errors are a minority ($\rho < 0.5$, diagonally dominant $T$) | Usually holds; fails for adversarial or systematically miscoded label schemas. |

## 3. State of the Art

**Theory SOTA (established).** Under mutual irreducibility, the mixture proportions — hence CCN rates — are identifiable and estimable consistently (Blanchard, Lee & Scott, JMLR 2010; Scott, Blanchard & Handy, COLT 2013). Natarajan et al. (NeurIPS 2013) give unbiased loss correction given known $T$, with excess-risk bounds scaling as $1/(1-\rho_{+1}-\rho_{-1})$. Ghosh et al. (AAAI 2017) prove symmetric losses (MAE) are noise-tolerant under symmetric noise. Liu, Cheng & Zhang (ICML 2023) show the transition matrix is *not* identifiable from noisy data alone in the general instance-dependent case, and characterize when it is.

**Empirical SOTA (established, reproduced).** Confident Learning / cleanlab (Northcutt, Jiang & Chuang, JAIR 2021) — the practical default; no clean data needed, only cross-validated probabilities. AUM ranking (Pleiss et al., NeurIPS 2020) — training-dynamics based, adds a threshold class of deliberately mislabeled samples. SimiFeat (Zhu, Dong & Liu, ICML 2022) — $k$-NN vote in a pretrained feature space, no training at all.

**Claimed but unablated.** Instance-dependent transition estimators (part-dependent $T(x)$, Xia et al. NeurIPS 2020; end-to-end anchor-free estimation, Li et al. ICML 2021) report gains almost entirely on *synthesized* IDN, where the generator's parametric form matches the estimator's. Detection performance under real human noise is rarely reported. LLM-as-annotator-auditor pipelines report high agreement with human review on curated subsets — benchmark numbers only, no controlled ablation isolating the LLM's own label prior.

**Benchmark-number-only results.** DivideMix's 74.76% on Clothing1M (ICLR 2020) is a *classification* number; it is routinely cited as evidence of detection quality, which it is not — Clothing1M has no per-example clean mask for its training set.

## 4. What Is Known

- **Real datasets carry percent-level test-set noise.** Northcutt, Athalye & Mueller (NeurIPS D&B 2021) found an average 3.3% label errors across 10 benchmark *test* sets; ImageNet validation ≈5.8%, QuickDraw ≈10.1%, MNIST test 15 errors (0.15%). Scale: 10 datasets, ~2.6M examples audited by MTurk consensus.
- **Real human noise is heavy and asymmetric.** CIFAR-10N (Wei et al., ICLR 2022; 3 independent MTurk labels per image, 50k train images): 9.03% aggregate, 17.23% single-annotator, 40.21% worst-case; CIFAR-100N 40.20%. Errors cluster on confusable class pairs — direct evidence against CCN.
- **Early learning is real.** Networks fit clean examples before noisy ones (Arpit et al., ICML 2017), yet can fit fully random labels given enough epochs (Zhang et al., ICLR 2017). The gap is the detection signal, and its width is unbounded in theory.
- **Confident learning is consistent under CCN** with imperfect probability estimates, given per-class error bounds (Northcutt et al., JAIR 2021, Thm. 1–2); empirically it finds ~90%+ of injected symmetric errors on CIFAR-10 at 40% noise.
- **Correcting labels helps benchmark validity.** On ImageNet with corrected multi-label evaluation, a large share of "errors" by top models are correct (Beyer et al., 2020; Vasudevan et al., NeurIPS 2022) — model ranking changes when the mask changes.
- **Training-free detection is competitive.** SimiFeat (ICML 2022) matches or beats training-based detectors on CIFAR-10N/100N at a fraction of the compute.

## 5. What Is Not Known

- **Theoretically open.** Necessary and sufficient conditions for identifying instance-dependent $T(x)$ from $p(x,\tilde y)$ alone, beyond the sufficient conditions already known (anchor points, clusterability, part-dependence). Also open: whether any clean-data-free estimator attains a minimax rate for $\rho^\star$ under IDN, or whether the rate is bounded away from zero.
- **Empirically open.** Nobody has run a like-for-like comparison of clean-free detectors against clean-validation-tuned oracles on **real** noise at web scale ($\ge 10^7$ examples, e.g. LAION or Clothing1M) with a re-annotated ground-truth mask on a stratified sample. The experiment is runnable; the annotation cost (~$10^4–10^5) is the barrier, not the method.
- **Methodologically blocked.** "Label error" is not well defined where multiple labels are correct or the taxonomy is genuinely ambiguous. Reported detection F1 mixes *errors* with *ambiguity*, and there is no accepted protocol separating them. Klie, Webber & Gurevych (*Computational Linguistics*, 2023) document exactly this in NLP annotation-error detection: incomparable metrics, incomparable datasets.

## 6. Why It Is Hard

**Primary obstruction: non-identifiability.** Fix any observed $p(x, \tilde y)$. For instance-dependent noise the decomposition into (clean posterior, corruption kernel) has a continuum of solutions — a confident-and-clean model and a less-confident-but-noisy model produce identical observables. Nothing in the data selects between "the label is wrong" and "the label is right and the concept is hard". Clean validation data resolves this by pinning one point of the decomposition; without it, an assumption must supply the missing constraint, and every such assumption (anchors, clusterability, low-rank $T$) is empirically violated in the exact regions where errors concentrate.

**Secondary: confounded measurement.** Detection is scored against a "ground truth" mask that is itself crowd consensus — i.e. the same noise process, averaged. Where an example is genuinely ambiguous, consensus is arbitrary, so measured precision is bounded by annotator agreement, not by the detector.

**Tertiary: circularity of the scoring model.** Detectors use a model trained on the noisy labels. Its errors correlate with the label errors, so detected errors are biased toward *easy* corruptions and away from systematic ones — exactly the ones that matter for downstream bias.

## 7. Current Research (as of 2026)

- **Clean-free transition estimation:** Yang Liu's group (UC Santa Cruz / Docta.ai) on clusterability (HOC), training-free detection (SimiFeat), and identifiability limits (ICML 2023).
- **Tooling:** Cleanlab (Northcutt et al.) — confident learning productized, extended to token classification, object detection, multi-label, and multi-annotator settings.
- **Foundation-model priors as pseudo-clean signal:** using CLIP/LLM zero-shot agreement as a noise-free-ish reference. Displaces the clean-data requirement onto the pretraining corpus rather than removing it *(frontier — verify)*.
- **Annotation-error detection in NLP** (Gurevych's UKP Lab) — standardizing evaluation so numbers become comparable.
- **Ambiguity-aware relabeling:** soft/multi-label ground truth (ImageNet ReaL lineage) replacing binary error masks *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** How much detection quality is actually lost by removing clean validation data, on *real* noise?

**Scale.** CIFAR-10N and CIFAR-100N (50k train images each; real human labels with the clean CIFAR label available as ground truth) plus Clothing1M's 47,570-image clean-train subset used only as ground truth. One ResNet-50 / ViT-B backbone family, 5 seeds.

**Arms.**
1. *Clean-free detectors:* confident learning, AUM, SimiFeat, ELR-loss ranking — all hyperparameters fixed a priori, no clean data touched.
2. **Control arm (oracle):** the identical detectors with thresholds, epoch counts, and score temperature tuned on a held-out clean 5% split, and an upper-bound arm tuned on 100% of the clean mask.
3. *Ambiguity split:* partition ground-truth errors into unanimous (3/3 annotators agree with clean label) vs. contested; report metrics separately.

**Deciding number.** $\Delta = \mathrm{AUPRC}_{\text{oracle-tuned}} - \mathrm{AUPRC}_{\text{clean-free}}$ on the unanimous subset, at fixed flag budget $b = 2\rho^\star$.

- $\Delta \le 0.02$ (within seed variance) on both CIFAR-10N and CIFAR-100N → clean validation data buys nothing for detection; status moves toward *solved for this regime*.
- $\Delta \ge 0.10$ on CIFAR-100N (high real noise, 40.2%) → the clean-free assumption set fails precisely where noise is instance-dependent, and the theory gap in §5 is the binding one.

Cost: order 100 GPU-hours. No new annotation needed — the ground truth already exists.

## 9. Key References

- **[Foundational]** Angluin, D. & Laird, P. *Learning from Noisy Examples.* Machine Learning 2(4), 1988.
- **[Foundational]** Natarajan, N., Dhillon, I., Ravikumar, P. & Tewari, A. *Learning with Noisy Labels.* NeurIPS, 2013.
- **[Foundational]** Blanchard, G., Lee, G. & Scott, C. *Semi-Supervised Novelty Detection.* JMLR 11, 2010.
- **[Theory]** Scott, C., Blanchard, G. & Handy, G. *Classification with Asymmetric Label Noise: Consistency and Maximal Denoising.* COLT, 2013.
- **[Theory]** Ghosh, A., Kumar, H. & Sastry, P. S. *Robust Loss Functions under Label Noise for Deep Neural Networks.* AAAI, 2017. — arXiv:1712.09482
- **[Theory]** Liu, Y., Cheng, H. & Zhang, K. *Identifiability of Label Noise Transition Matrix.* ICML, 2023.
- **[SOTA]** Northcutt, C., Jiang, L. & Chuang, I. *Confident Learning: Estimating Uncertainty in Dataset Labels.* JAIR 70, 2021. — arXiv:1911.00068
- **[SOTA]** Pleiss, G., Zhang, T., Elenberg, E. & Weinberger, K. Q. *Identifying Mislabeled Data using the Area Under the Margin Ranking.* NeurIPS, 2020. — arXiv:2001.10528
- **[SOTA]** Zhu, Z., Dong, Z. & Liu, Y. *Detecting Corrupted Labels Without Training a Model to Predict.* ICML, 2022.
- **[SOTA]** Zhu, Z., Song, Y. & Liu, Y. *Clusterability as an Alternative to Anchor Points When Learning with Noisy Labels.* ICML, 2021.
- **[Benchmark]** Wei, J., Zhu, Z., Cheng, H., Liu, T., Niu, G. & Liu, Y. *Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations.* ICLR, 2022. — arXiv:2110.12088
- **[Benchmark]** Northcutt, C., Athalye, A. & Mueller, J. *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2103.14749
- **[Benchmark]** Beyer, L., Hénaff, O., Kolesnikov, A., Zhai, X. & van den Oord, A. *Are We Done with ImageNet?* 2020. — arXiv:2006.07159
- **[Empirical]** Arpit, D. et al. *A Closer Look at Memorization in Deep Networks.* ICML, 2017. — arXiv:1706.05394
- **[Empirical]** Liu, S., Niles-Weed, J., Razavian, N. & Fernandez-Granda, C. *Early-Learning Regularization Prevents Memorization of Noisy Labels.* NeurIPS, 2020. — arXiv:2007.00151
- **[Survey]** Song, H., Kim, M., Park, D., Shin, Y. & Lee, J.-G. *Learning from Noisy Labels with Deep Neural Networks: A Survey.* IEEE TNNLS, 2022.
- **[Survey]** Frénay, B. & Verleysen, M. *Classification in the Presence of Label Noise: A Survey.* IEEE TNNLS 25(5), 2014.
- **[Survey]** Klie, J.-C., Webber, B. & Gurevych, I. *Annotation Error Detection: Analyzing the Past and Present for a More Coherent Future.* Computational Linguistics 49(1), 2023.

## 10. Worked Example

Two classes, cat/dog, $n = 10{,}000$, balanced. Suppose the true generative story is: 8,000 images are unambiguous and correctly labeled; 1,000 are unambiguous but flipped by a careless annotator; 1,000 are genuinely ambiguous (cat-like dogs) where the annotator effectively coin-flips.

Run confident learning. Cross-validated $\hat p(\text{cat} \mid x)$ gives, for each group:

| Group | $n$ | $\hat p(\tilde y \mid x)$ typical | Flagged at threshold $t = 0.5$ |
|---|---|---|---|
| Clean, unambiguous | 8,000 | 0.97 | ~0 |
| Flipped, unambiguous | 1,000 | 0.06 | ~940 |
| Ambiguous | 1,000 | 0.52 / 0.48 | ~500 |

Detector output: 1,440 flagged. Precision against the "true error" mask, if we define errors as the 1,000 flips plus the ~500 ambiguous images whose coin-flip disagreed with a consensus relabel: $\approx 940/1440 = 0.65$ measured, but the 500 ambiguous flags are *half correct by construction* — which half is unknowable from the data.

Now the obstruction. Consider a second generative story fitting the *same* observed $p(x,\tilde y)$: those 1,000 ambiguous images have clean labels exactly as given, and the model is simply underconfident on a hard region. Observationally identical. Under story A the correct action is to relabel; under story B relabeling injects 500 fresh errors. The estimated noise rate differs by $1000/10000 - 500/10000 = 5$ percentage points depending on which story you assume — a 50% relative error in $\hat\rho$, with no data-only test to choose.

A single clean pass over 200 stratified ambiguous images resolves it: measure the fraction where expert consensus disagrees with $\tilde y$; the binomial standard error at $p=0.5$, $n=200$ is 3.5pp, enough to separate the two stories. That is the exact quantity the clean-data-free setting forbids you from buying — which is why the problem is *partially* solved: detection works where labels are unambiguous, and degenerates into an unidentifiable mixture exactly where they are not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*