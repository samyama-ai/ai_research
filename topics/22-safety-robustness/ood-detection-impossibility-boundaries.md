---
id: 22-safety-robustness/ood-detection-impossibility-boundaries
title: "Out-of-Distribution Detection Impossibility Boundaries"
topic: 22-safety-robustness
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Out-of-Distribution Detection Impossibility Boundaries

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/ood-detection-impossibility-boundaries` · **Status:** partially-solved

## 1. Problem Statement

Given a model trained on data from an in-distribution (ID) source, decide for each new input whether it came from that source. The open question is not "which detector wins a benchmark" but **where the boundary of possibility lies**: for which pairs (ID distribution, admissible OOD family, hypothesis class) does *any* learning algorithm achieve low error, and for which is the task information-theoretically impossible regardless of compute or data.

Three variants, routinely conflated:

- **Theory variant.** Characterize the (ID, OOD, hypothesis-class) triples for which OOD detection is PAC-learnable. Partially settled — see §4.
- **Method variant.** Build a detector that attains a target false-positive rate at fixed true-positive rate on realistic near-OOD. Empirically open; progress is real but slow.
- **Measurement variant.** Decide whether a given benchmark result reflects the detector or the benchmark's construction. Methodologically blocked — the label "OOD" is not a property of a sample.

A solution to the theory variant is a theorem naming necessary and sufficient conditions with a matching algorithm. A solution to the measurement variant is a protocol whose OOD labels are reproducible by independent annotators and invariant to the input coordinate system.

## 2. Formal Setting

Input space $\mathcal{X}$, label space $\mathcal{Y}=\{1,\dots,K\}$. ID distribution $D_{XY}^{\text{ID}}$ over $\mathcal{X}\times\mathcal{Y}$ with marginal $D_X^{\text{ID}}$. An OOD distribution $D_X^{\text{out}}$ is drawn from an admissible family $\mathcal{D}_{\text{out}}$. Test data is the mixture
$$D_X^{\text{test}} = (1-\pi)\,D_X^{\text{ID}} + \pi\,D_X^{\text{out}},\qquad \pi\in(0,1).$$

A detector is $g:\mathcal{X}\to\{\text{in},\text{out}\}$, usually thresholded from a score $s:\mathcal{X}\to\mathbb{R}$: $g_\tau(x)=\text{out}$ iff $s(x)<\tau$. The risk used in the learnability literature augments the label space with an OOD symbol $K{+}1$ and measures
$$R(f) = (1-\pi)\,\mathbb{E}_{D^{\text{ID}}}\!\left[\ell(f(x),y)\right] + \pi\,\mathbb{E}_{D^{\text{out}}}\!\left[\mathbb{1}\{f(x)\neq K{+}1\}\right].$$

**How each quantity is actually measured.**

- $s(x)$: a scalar the code emits — max softmax probability, $-\!\max_k \mathrm{logit}_k$ (energy $-T\log\sum_k e^{z_k/T}$), Mahalanobis distance in penultimate features, or $k$-NN distance to the training set. Fully observable.
- **AUROC**: rank statistic of $s$ on $N_{\text{in}}$ held-out ID samples versus $N_{\text{out}}$ samples from a *hand-chosen* set. Reported to $\pm$0.3 points at $N\approx 10^4$; the choice of the OOD set moves it by 10–40 points.
- **FPR@95TPR**: fraction of OOD samples scored above the threshold at which 95% of ID passes. More sensitive than AUROC to the score's left tail; its standard error at $N_{\text{out}}=5{,}000$ is roughly $\pm1.5$ points.
- $\pi$: never measured. Every benchmark fixes it by construction (often $\pi=0.5$), so reported numbers are threshold-free by design and say nothing about deployment error rates.
- $D_X^{\text{out}}$: not sampled from a distribution at all — it is a curated image or text set. This is the load-bearing fiction.

**Assumptions, and which are violated.**

1. *$D_X^{\text{out}}$ is a fixed distribution independent of the detector.* Violated: adversarial and adaptive OOD exists, and benchmark curation is itself detector-informed.
2. *Supports are separable, $\mathrm{supp}(D_X^{\text{ID}})\cap\mathrm{supp}(D_X^{\text{out}})=\emptyset$.* Violated on near-OOD: an ImageNet class and a semantically novel class share low-level image statistics and often share objects outright.
3. *ID training labels are clean and the ID class list is closed.* Violated: benchmark OOD sets contain ID-class objects (§4).
4. *Density is a meaningful ordering on $\mathcal{X}$.* Violated: for continuous $\mathcal{X}$, densities are defined relative to a base measure, and a diffeomorphism $T$ reweights $p$ by $|\det J_T|^{-1}$, which can reverse any density ranking.

## 3. State of the Art

**Theory SOTA — established.** Fang, Li, Lu, Xie, Ye, Zhang, *Is Out-of-Distribution Detection Learnable?* (NeurIPS 2022, outstanding paper). Gives necessary and sufficient conditions for PAC learnability of OOD detection under a domain-space model, proves impossibility in the total space (all possible $D_X^{\text{out}}$), and proves learnability in the separate space and in finite-ID-distribution and density-based spaces under stated conditions. This is a real theorem, not a benchmark claim.

**Theory SOTA — established.** Le Lan and Dinh, *Perfect Density Models Cannot Guarantee Anomaly Detection* (Entropy, 2021): even with the exact density, likelihood-based detection is not invariant to reparameterization, so "low likelihood $\Rightarrow$ anomalous" is not well posed without fixing a reference measure. Zhang, Goldstein, Ranganath, *Understanding Failures in Out-of-Distribution Detection with Deep Generative Models* (ICML 2021) show no single-model likelihood rule succeeds uniformly over OOD families.

**Empirical SOTA — benchmark numbers only.** OpenOOD v1.5 (Zhang et al., NeurIPS 2023 Datasets & Benchmarks track) evaluates ~40 methods under one protocol. On ImageNet-1K near-OOD, the best post-hoc scores cluster near AUROC 0.76–0.80; far-OOD reaches ~0.90. On CIFAR-10 near-OOD (CIFAR-100 / TinyImageNet) the leaders sit near 0.90. Strong post-hoc entries: KNN (Sun et al., ICML 2022), ViM (Wang et al., CVPR 2022), ASH (Djurisic et al., ICLR 2023), energy (Liu et al., NeurIPS 2020).

**Claimed but unablated.** Most method papers report gains against a fixed OOD suite without varying the suite; OpenOOD's re-ranking shows several published leads shrink to under 1 AUROC point or invert once the protocol is held constant. Claims that a score "detects semantic novelty" are architecture-and-benchmark-coupled, not ablated against a matched-covariate-shift control.

## 4. What Is Known

- **Total-space impossibility.** With no restriction on $\mathcal{D}_{\text{out}}$ and overlapping supports, no algorithm PAC-learns OOD detection (Fang et al., NeurIPS 2022). The escape hatches are all *restrictions on the OOD family* or *separability*, both unverifiable at deployment.
- **Density order is coordinate-dependent.** Glow trained on CIFAR-10 assigns SVHN *higher* likelihood — about 2.39 bits/dim versus 3.46 bits/dim on CIFAR-10 test (Nalisnick et al., ICLR 2019). Thresholding likelihood gives an AUROC well below 0.5, i.e. anti-detection at $N \approx 26{,}000$ SVHN test images.
- **Benchmark labels are contaminated.** Bitterwolf et al., *In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation* (ICML 2023), found large fractions of ID-class objects in widely used ImageNet OOD sets and released NINCO — 5,879 manually verified samples over 64 OOD classes. Method rankings change once contamination is removed.
- **Closed-set accuracy carries much of the signal.** Vaze et al., *Open-Set Recognition: A Good Closed-Set Classifier Is All You Need?* (ICLR 2022): open-set AUROC rises monotonically with closed-set accuracy across five standard benchmarks, and a well-trained MSP baseline matches or beats specialized detectors. Much apparent OOD progress is representation progress.
- **Near/far gap is large and stable.** Across OpenOOD v1.5, the same detector loses roughly 10–15 AUROC points moving from far-OOD to near-OOD at ImageNet-1K scale.
- **Outlier exposure helps when the exposure set matches.** Hendrycks, Mazeika, Dietterich (ICLR 2019) show large gains from training on auxiliary outliers; gains attenuate when the test OOD family is disjoint from the exposure family.

## 5. What Is Not Known

- **Theoretically open.** The learnability conditions are stated over abstract domain spaces. Nobody has shown whether the conditions hold, or fail, for any *concrete* realistic pair such as (ImageNet-1K, NINCO) with a ResNet or ViT hypothesis class. The theory does not yet certify or refute a single deployed system.
- **Theoretically open.** No lower bound on achievable near-OOD AUROC as a function of a measurable overlap quantity. There is no ImageNet analogue of a Bayes error for OOD.
- **Empirically open.** Whether the near-OOD ceiling around 0.80 AUROC at ImageNet-1K is a property of representations or of the benchmark. Runnable: hold representation fixed, vary the OOD set along a controlled semantic-distance axis. Not run at scale.
- **Methodologically blocked.** "OOD" has no observer-independent definition. Semantic novelty, covariate shift, and label-set extension are three different predicates measured by one number. Farquhar and Gal, *What 'Out-of-Distribution' Is and Is Not* (NeurIPS ML Safety Workshop, 2022), make the case that the term names several disjoint problems.

## 6. Why It Is Hard

**Absent ground truth compounded by non-identifiability.** The OOD label is a curator's decision, not a measurable attribute. Two annotators disagree on near-OOD membership, and NINCO shows the disagreement is large enough to reorder published methods. On top of that, the underlying quantity is non-identifiable: for any invertible reparameterization $T$ of $\mathcal{X}$, densities transform as $p_T(z) = p(T^{-1}z)\,|\det J_{T^{-1}}(z)|$, so a likelihood ranking of two points can be inverted by a change of variables that leaves every ID classification decision intact. There is no privileged coordinate system on pixels or token embeddings that the theory can point to.

Consequence: the evaluation does not measure what it names. AUROC on a curated OOD set measures *separability of two hand-picked sample collections under one coordinate system*, and reports it as *detection of the unknown*.

## 7. Current Research (as of 2026)

- **Learnability under structure.** Extensions of Fang et al. to conditions on the hypothesis class and to open-set domain adaptation; groups at HKBU, RIKEN AIP, and the University of Melbourne are the visible line.
- **Benchmark repair.** OpenOOD maintenance and NINCO-style curated, contamination-audited suites (Tübingen; NUS/Wisconsin OpenOOD contributors).
- **Foundation-model detection.** CLIP- and VLM-based zero-shot OOD scoring (MCM, Ming et al., NeurIPS 2022) and its LLM analogues — whether pretraining breadth makes the ID/OOD boundary less well defined rather than easier is unresolved *(frontier — verify)*.
- **LLM-side variants.** Hallucination gating and refusal calibration recast as OOD detection over prompts; no accepted formalization of what the "in-distribution" prompt set is *(frontier — verify)*.
- **Conformal and PU-learning framings.** Treating detection as testing with a controlled false-alarm rate, which sidesteps the density-ordering problem but requires exchangeability that near-OOD violates.

## 8. Concrete Next Experiment

**Question:** is the near-OOD ranking of detectors a property of the distributions or of the coordinate system?

- **Scale.** ImageNet-1K ID; one fixed backbone (ViT-B/16, ImageNet-21k pretrained) and one ResNet-50, so the representation is held constant. Eight post-hoc detectors: MSP, energy, ODIN, Mahalanobis, ViM, KNN, ASH, ReAct. OOD sets: NINCO (5,879 samples) as near-OOD, iNaturalist subset as far-OOD.
- **Treatment.** Construct $M=20$ invertible input warps $T_m$ — per-channel monotone spline warps fitted so the ID marginal statistics and top-1 ID accuracy change by less than 0.2 points. Retrain nothing; apply $T_m$ to both ID and OOD inputs at test time, re-fit only the detector's ID statistics. Each $T_m$ preserves the ID distribution's decision structure by construction.
- **Control arm.** $M=20$ random seeds of detector-fitting on untransformed inputs, giving the noise floor of AUROC from fitting variance alone.
- **Deciding number.** The median across detectors of $\Delta = \mathrm{range}_m(\mathrm{AUROC}_m) - \mathrm{range}_{\text{seed}}(\mathrm{AUROC})$ on NINCO. If $\Delta > 5$ AUROC points, near-OOD detection scores are substantially a coordinate artifact and current leaderboards do not measure a distributional property. If $\Delta < 1$ point, the impossibility results from density reparameterization are real but empirically inert at ImageNet scale — a genuinely informative negative.
- **Cost.** Inference only: about 200 forward passes over ~55k images per backbone, under 300 GPU-hours on A100s.

## 9. Key References

- **[Foundational]** Zhen Fang, Yixuan Li, Jie Lu, Jiahua Dong, Bo Han, Feng Liu. *Is Out-of-Distribution Detection Learnable?* NeurIPS, 2022.
- **[Foundational]** Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, Balaji Lakshminarayanan. *Do Deep Generative Models Know What They Don't Know?* ICLR, 2019.
- **[Foundational]** Dan Hendrycks, Kevin Gimpel. *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR, 2017.
- **[Theory]** Charline Le Lan, Laurent Dinh. *Perfect Density Models Cannot Guarantee Anomaly Detection.* Entropy 23(12), 2021.
- **[Theory]** Lily Zhang, Mark Goldstein, Rajesh Ranganath. *Understanding Failures in Out-of-Distribution Detection with Deep Generative Models.* ICML, 2021.
- **[SOTA]** Jingyang Zhang et al. *OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection.* NeurIPS Datasets & Benchmarks, 2023.
- **[SOTA]** Julian Bitterwolf, Maximilian Müller, Matthias Hein. *In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation.* ICML, 2023.
- **[SOTA]** Yiyou Sun, Yifei Ming, Xiaojin Zhu, Yixuan Li. *Out-of-Distribution Detection with Deep Nearest Neighbors.* ICML, 2022.
- **[SOTA]** Weitang Liu, Xiaoyun Wang, John D. Owens, Yixuan Li. *Energy-based Out-of-distribution Detection.* NeurIPS, 2020.
- **[Related]** Sagar Vaze, Kai Han, Andrea Vedaldi, Andrew Zisserman. *Open-Set Recognition: A Good Closed-Set Classifier Is All You Need?* ICLR, 2022.
- **[Related]** Dan Hendrycks, Mantas Mazeika, Thomas Dietterich. *Deep Anomaly Detection with Outlier Exposure.* ICLR, 2019.
- **[Survey]** Jingkang Yang, Kaiyang Zhou, Yixuan Li, Ziwei Liu. *Generalized Out-of-Distribution Detection: A Survey.* IJCV, 2024.

## 10. Worked Example

Take a Glow model trained on CIFAR-10. Test bits-per-dimension: CIFAR-10 $\approx 3.46$, SVHN $\approx 2.39$. Convert: $\log p$ per dimension differs by $(3.46-2.39)\times\ln 2 \approx 0.74$ nats, over $D = 3072$ dimensions, so
$$\log p(x_{\text{SVHN}}) - \log p(x_{\text{CIFAR}}) \approx 0.74 \times 3072 \approx 2.3\times10^{3}\ \text{nats}.$$

The out-of-distribution data is more likely by a factor of $e^{2300}$. A detector using "flag the low-likelihood tail" does not merely fail; it is anti-correlated with the truth.

Now make the obstruction explicit. Suppose we fix this by any monotone rescoring of $\log p$. Choose an invertible, smooth per-pixel map $T$ — say a monotone spline on $[0,1]$ per channel. Under $T$, the density transforms as
$$\log p_T(T x) = \log p(x) - \sum_{i=1}^{D}\log T'(x_i).$$
Pick $T$ with $T' > 1$ on the mid-grey band where SVHN's flat digit backgrounds live and $T' \approx 1$ elsewhere. If SVHN images place, say, 40% of their $3072$ pixels in that band and CIFAR-10 places 15%, a warp with $\log T' = 2$ there subtracts $0.40\times3072\times2 \approx 2458$ nats from SVHN and $0.15\times3072\times2 \approx 922$ from CIFAR — a net swing of about $1{,}536$ nats, enough to erase and reverse the original $2{,}300$-nat gap with a mild extra push.

Both $p$ and $p_T$ are *exact* densities of the same data, related by a bijection that changes no semantic content and, fed through a classifier retrained on the warped inputs, changes ID accuracy negligibly. The ordering "SVHN is more likely than CIFAR" is therefore not a fact about the distributions — it is a fact about the pixel coordinate system. That is Le Lan and Dinh's result made numeric, and it is why the impossibility boundary cannot be pushed by better density estimation. It moves only by fixing a reference measure or restricting $\mathcal{D}_{\text{out}}$ — and no benchmark currently states which one it assumes.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*