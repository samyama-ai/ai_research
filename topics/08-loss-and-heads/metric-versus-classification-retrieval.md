---
id: 08-loss-and-heads/metric-versus-classification-retrieval
title: "Metric Losses Versus Classification Pretraining for Retrieval"
topic: 08-loss-and-heads
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Metric Losses Versus Classification Pretraining for Retrieval

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/metric-versus-classification-retrieval` · **Status:** empirically-open

## 1. Problem Statement

Retrieval systems need an embedding $f_\theta: \mathcal{X} \to \mathbb{S}^{d-1}$ whose cosine geometry ranks same-class or same-entity items above others. Two families of objective are used to train it:

- **Metric losses** operate directly on the geometry of embedding pairs or tuples: contrastive, triplet, multi-similarity, InfoNCE, supervised contrastive.
- **Classification pretraining** trains a discarded linear (or margin-modified) head over $C$ training classes with softmax cross-entropy, then throws the head away and retrieves with the penultimate features.

**The question.** Under a matched compute budget, matched architecture, and equal hyperparameter search effort, does either family produce better retrieval on *held-out classes* — and if so, at what $(C, n, d)$ does the ordering flip?

Three variants, with different difficulty:

- **Measurement.** Does the reported metric-loss advantage survive a fair protocol? Largely answered: no (§3).
- **Method.** Is there a regime — very large $C$, very few examples per class, open-set queries — where a metric loss wins by a margin larger than seed noise? Empirically open.
- **Theory.** Softmax cross-entropy and pairwise losses have been shown to optimize related mutual-information objectives; whether the *generalization* to unseen classes differs is not proved either way.

Solving it means: a stated predicate over $(C, n, d, B)$ that predicts which family wins, validated out of sample.

## 2. Formal Setting

Training set $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, $y_i \in \{1,\dots,C\}$, with $n = N/C$ mean examples per class. Encoder $f_\theta$ produces $z = f_\theta(x)/\|f_\theta(x)\|_2 \in \mathbb{S}^{d-1}$.

**Classification arm.** Head $W \in \mathbb{R}^{d \times C}$ with normalized columns $w_c$, scale $s$, angular margin $m$ (ArcFace form):

$$\mathcal{L}_{\text{cls}} = -\log \frac{e^{s\cos(\theta_{y}+m)}}{e^{s\cos(\theta_{y}+m)} + \sum_{c \neq y} e^{s\cos\theta_c}}, \quad \cos\theta_c = z^\top w_c.$$

$m=0$ recovers normalized softmax; $s\to$ unnormalized $W$ recovers vanilla cross-entropy.

**Metric arm.** InfoNCE over in-batch positives with temperature $\tau$:

$$\mathcal{L}_{\text{nce}} = -\log \frac{e^{z^\top z^+/\tau}}{e^{z^\top z^+/\tau} + \sum_{k} e^{z^\top z_k^-/\tau}},$$

or triplet with margin $\alpha$: $\mathcal{L}_{\text{tri}} = [\|z-z^+\|^2 - \|z-z^-\|^2 + \alpha]_+$.

**Evaluation, as actually measured.** Disjoint class split: query set $\mathcal{Q}$ and gallery $\mathcal{G}$ contain only classes unseen in training. For query $q$ with $R_q$ relevant gallery items,

$$\text{R@}K = \frac{1}{|\mathcal{Q}|}\sum_q \mathbb{1}[\exists\, \text{relevant item in top-}K], \qquad \text{MAP@R} = \frac{1}{|\mathcal{Q}|}\sum_q \frac{1}{R_q}\sum_{k=1}^{R_q} P(k)\,\mathbb{1}[\text{rel}(k)].$$

MAP@R is preferred because R@1 saturates and is insensitive to the tail of the ranking (Musgrave et al., ECCV 2020).

**Budget.** $B$ = total training FLOPs, not epochs. The classification arm pays $O(Cd)$ extra per step for the head; at $C=10^6$, $d=512$ this is $5\times10^8$ MACs/sample and is *not* negligible — budget matching must account for it.

**The decision quantity.**

$$\Delta(C,n,d,B) = \mathbb{E}_{\text{seeds}}\!\left[\text{MAP@R}_{\text{metric}}\right] - \mathbb{E}_{\text{seeds}}\!\left[\text{MAP@R}_{\text{cls}}\right],$$

each arm tuned by an identical search protocol (same trial count, same search space size).

**Assumptions, and which are violated.**

- *Equal tuning effort.* Violated routinely: metric losses in papers get bespoke mining, the classification baseline gets defaults.
- *Disjoint train/test classes.* Holds for CUB/Cars/SOP; violated for MTEB and for face benchmarks with identity overlap.
- *Class labels are clean and mutually exclusive.* Violated for web-scale entity retrieval, where "negatives" are frequently false negatives.
- *A single $\tau$ or $(s,m)$ transfers across scale.* Violated — both optima move with $C$ and batch size.

## 3. State of the Art

**Established (fair-protocol, reproduced).**

- Musgrave, Belongie, Lim, *A Metric Learning Reality Check* (ECCV 2020, arXiv:2003.08505): with a fixed backbone, fixed embedding size, and Bayesian hyperparameter search applied equally to all losses, thirteen years of claimed improvement collapse to roughly a 1–2 point band on MAP@R. Normalized softmax classification sits inside the band with the best pairwise losses.
- Zhai & Wu, *Classification is a Strong Baseline for Deep Metric Learning* (BMVC 2019, arXiv:1811.12649): layer-normalized, class-balanced softmax matches or beats contemporaneous metric losses on CUB-200, Cars196, SOP, In-Shop.
- Boudiaf et al., *A Unifying Mutual Information View of Metric Learning* (ECCV 2020, arXiv:2003.08983): cross-entropy is an approximate bound-optimizer of the same mutual-information objective that pairwise losses target; the label-smoothed variant is competitive on all four standard DML benchmarks.

**Established (large-$C$ regime).** Margin softmax dominates open-set face recognition: SphereFace (CVPR 2017), CosFace (CVPR 2018), ArcFace (CVPR 2019). At $C \approx 10^5$ identities, ArcFace on IJB-C reports TAR@FAR$=10^{-4}$ in the mid-90s, above triplet-trained FaceNet-style systems trained on comparable data.

**Established (web-scale, no usable class labels).** CLIP (Radford et al., ICML 2021) and text dual-encoders (DPR, EMNLP 2020; Contriever, TMLR 2022; GTR, EMNLP 2022; E5, arXiv:2212.03533) use InfoNCE because $C$ is effectively $N$ — a classification head over $10^9$ "classes" is not trainable. This is an engineering constraint, not evidence that the metric loss generalizes better.

**Claimed but unablated.** That hard-negative mining is what makes metric losses work at scale; that large in-batch negative counts in InfoNCE are equivalent to a large softmax denominator. The second is a folk-theorem: InfoNCE with $B$ negatives *is* a softmax over $B$ instance classes, but the equivalence to a fixed $C$-way head under the same FLOPs has not been measured with matched budgets.

**Benchmark-number-only.** Most MTEB leaderboard entries. Models differ in data, scale, distillation, and instruction tuning simultaneously; no loss-controlled arm exists.

## 4. What Is Known

- **CUB-200-2011, BN-Inception, 512-d concatenated, 4-fold, 10 seeds** (Musgrave et al. 2020): contrastive MAP@R $\approx 26.5$, normalized softmax $\approx 24.9$, ProxyNCA $\approx 24.2$, multi-similarity $\approx 24.3$, triplet $\approx 23.7$. Spread $\approx 2.8$ points; between-seed confidence intervals are of order $\pm 0.3$–$0.5$. Reported R@1 improvements in the original papers were 5–10 points; under the fair protocol they are ~1–3.
- **Reproducibility of the ordering.** Roth et al., *Revisiting Training Strategies and Generalization Performance in Deep Metric Learning* (ICML 2020, arXiv:2002.08473): batch composition, sampling, and regularization move performance by more than the choice of loss. Same conclusion, independent codebase.
- **Scale flip evidence.** Face recognition at $C \sim 10^5$–$10^6$ moved from triplet (FaceNet, CVPR 2015) to margin softmax and stayed. This is the strongest evidence for a $C$-dependent flip — but it is confounded by a simultaneous ~$100\times$ increase in training-set size.
- **Loss shape affects transfer.** Kornblith et al., *Why Do Better Loss Functions Lead to Less Transferable Features?* (NeurIPS 2021): label smoothing and larger softmax temperature raise ImageNet top-1 while *reducing* linear-transfer and $k$-NN quality of the penultimate features, by collapsing within-class variance. Directly relevant: the classification arm's retrieval quality is not monotone in its classification accuracy.
- **Geometry decomposition.** Wang & Isola (ICML 2020, arXiv:2005.10242): contrastive loss decomposes into alignment and uniformity; both correlate with downstream retrieval. Margin softmax also increases uniformity but through class prototypes rather than sampled negatives.

## 5. What Is Not Known

- **Empirically open (the main gap).** $\Delta(C,n,d,B)$ has never been mapped. Nobody has run a compute-matched sweep over $C \in \{10^2, 10^3, 10^4, 10^5, 10^6\}$ at fixed $N$, with equal tuning, on a single dataset with disjoint held-out classes. Every existing comparison varies $C$, $N$, architecture, and augmentation together. The experiment is runnable today for well under $10^4$ GPU-hours.
- **Empirically open.** Whether the classification arm's advantage is a *head* effect or an *optimization* effect — proxies are fixed targets with low-variance gradients, whereas sampled negatives inject batch noise. Proxy-Anchor (CVPR 2020) and ProxyNCA++ (ECCV 2020) sit between the families and are the natural interpolation, but were not evaluated as a controlled interpolation.
- **Theoretically open.** No generalization bound separates the families on *unseen* classes. Boudiaf et al. equate the training objectives, not the open-set risk. There is no theorem of the form "for $C > g(n,d)$, the prototype-based estimator has lower excess open-set risk."
- **Methodologically blocked.** "Equal tuning effort" is not a well-defined quantity. Search spaces differ in dimension (triplet: margin, mining strategy, sampler; softmax: $s$, $m$, weight decay on head). Until tuning budget is normalized in a principled way — e.g. equal expected-improvement under a fixed surrogate — reported gaps are not comparable across papers.

## 6. Why It Is Hard

**The specific obstruction is confounded measurement compounded by tuning asymmetry.** The loss is never varied alone. Changing to a classification head changes: the sampler (class-balanced vs. random), the effective negative count per step (all $C$ prototypes vs. batch size $B$), the gradient variance, and the parameter count. A raw A/B therefore measures a bundle, not the loss.

Second obstruction: **the benchmark does not measure what it names.** CUB-200 and Cars196 have ~5,900 training images across 100 classes. At that scale differences are within backbone-initialization noise, and the reported metric is R@1, which saturates. The benchmarks that would resolve the question ($C \ge 10^5$) are face-identity or product datasets with licensing and de-duplication problems, and known train/test identity leakage.

Third: **compute cost of the tuning arm, not the training arm.** A credible answer needs $\ge 30$ trials per (loss, $C$) cell $\times$ 5 values of $C$ $\times$ 2 families $\times$ 3 seeds $\approx 900$ runs.

## 7. Current Research (as of 2026)

- **Text retrieval** has effectively settled on InfoNCE with hard negatives plus distillation from cross-encoders (E5, GTE, BGE lineages; Microsoft, BAAI, Alibaba groups). No loss-controlled ablation accompanies these releases; the wins come from data curation.
- **Self-supervised backbones as the real baseline.** DINOv2 (Oquab et al., TMLR 2024) features are competitive at $k$-NN retrieval with no retrieval-specific loss, which makes "which retrieval loss" partly moot for image search — the pretraining dominates. *(frontier — verify: whether fine-tuning DINOv2 with a metric loss beats fine-tuning it with a classification head, on disjoint classes.)*
- **Prototype/parametric heads at extreme $C$** — partial-FC style sampled-softmax to make $C \sim 10^7$ tractable (InsightFace group). This directly attacks the FLOPs asymmetry in §2 and is the enabling technology for the missing experiment.
- **Reality-check maintenance.** `pytorch-metric-learning` and the `powerful-benchmarker` protocol (Musgrave) remain the reference fair-comparison harness.

## 8. Concrete Next Experiment

**Scale.** iNaturalist-2021 or a de-duplicated Google Landmarks v2 subset: fix $N = 2{\times}10^6$ images, ViT-B/16 initialized identically, $d=512$, fixed 200-epoch-equivalent FLOP budget $B$. Construct nested label granularities giving $C \in \{10^2, 10^3, 10^4, 10^5\}$ with $N$ held constant (so $n$ falls from $2{\times}10^4$ to $20$). Hold out 20% of the finest-grained classes entirely for evaluation, at every granularity.

**Arms.** (a) ArcFace head, partial-FC sampling so head FLOPs are constant across $C$; (b) InfoNCE with class-positive sampling, batch 4096; (c) control arm: **frozen DINOv2 features, no fine-tuning** — this is the arm that is almost always missing and that decides whether either loss earns its compute.

**Tuning.** Identical protocol: 30 ASHA trials per cell, search spaces of equal dimension (3 hyperparameters each). 3 seeds at the best config.

**The deciding number.** $\Delta(C) = \text{MAP@R}_{\text{nce}}(C) - \text{MAP@R}_{\text{arc}}(C)$ on held-out classes. The question is settled if $\Delta$ crosses zero with $|\Delta| > 4\sigma_{\text{seed}}$ on both sides of the crossing. Prediction to be falsified: $\Delta > 0$ at $C=10^2$ and $\Delta < 0$ at $C = 10^5$, with the crossing near $n \approx 200$. A flat $\Delta$ within noise across three decades of $C$ would be an equally publishable null and would retire the debate.

## 9. Key References

- **[Foundational]** Schroff, Kalenichenko, Philbin. *FaceNet: A Unified Embedding for Face Recognition and Clustering.* CVPR, 2015. — arXiv:1503.03832
- **[Foundational]** Deng, Guo, Xue, Zafeiriou. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR, 2019. — arXiv:1801.07698
- **[SOTA / protocol]** Musgrave, Belongie, Lim. *A Metric Learning Reality Check.* ECCV, 2020. — arXiv:2003.08505
- **[SOTA]** Zhai, Wu. *Classification is a Strong Baseline for Deep Metric Learning.* BMVC, 2019. — arXiv:1811.12649
- **[Theory]** Boudiaf, Rony, Ziko, Granger, Pedersoli, Piantanida, Ben Ayed. *A Unifying Mutual Information View of Metric Learning: Cross-Entropy vs. Pairwise Losses.* ECCV, 2020. — arXiv:2003.08983
- **[Theory]** Wang, Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML, 2020. — arXiv:2005.10242
- **[Ablation]** Roth, Milbich, Sinha, Gupta, Ommer, Cohen. *Revisiting Training Strategies and Generalization Performance in Deep Metric Learning.* ICML, 2020. — arXiv:2002.08473
- **[Ablation]** Kornblith, Chen, Lee, Norouzi. *Why Do Better Loss Functions Lead to Less Transferable Features?* NeurIPS, 2021.
- **[Related]** Khosla, Teterwak, Wang, Sarna, Tian, Isola, Maschinot, Liu, Krishnan. *Supervised Contrastive Learning.* NeurIPS, 2020. — arXiv:2004.11362
- **[Related]** Kim, Kim, Cho, Kwak. *Proxy Anchor Loss for Deep Metric Learning.* CVPR, 2020. — arXiv:2003.13911
- **[Related]** Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML, 2021. — arXiv:2103.00020
- **[Survey/benchmark]** Muennighoff, Tazi, Magne, Reimers. *MTEB: Massive Text Embedding Benchmark.* EACL, 2023. — arXiv:2210.07316

## 10. Worked Example

Take the CUB-200 cell from the reality check: 100 train classes, ~5,900 images, $n \approx 59$, BN-Inception, $d = 128$ per model, 4-model concatenation to 512.

Reported MAP@R: contrastive $\approx 26.5$, normalized softmax $\approx 24.9$. Naive reading: metric loss wins by 1.6 points.

Now put a number on the noise. With $|\mathcal{Q}| \approx 5{,}900$ test images and per-query MAP@R roughly Bernoulli-like with $p \approx 0.25$, the sampling standard error is

$$\sigma_{\text{eval}} \approx \sqrt{\frac{0.25 \times 0.75}{5900}} \approx 0.0056 = 0.56 \text{ points},$$

and the across-seed standard deviation the paper reports is of comparable size. The 1.6-point gap is about $2\sigma$ — real, but small.

Now apply the loss-to-scale extrapolation people actually make. At $C = 10^5$ face identities the ordering is reversed and the gap is much larger: margin softmax vs. triplet at that scale is a difference of several points of TAR@FAR$=10^{-4}$, not fractions. So the evidence base contains one 2$\sigma$ result favouring metric losses at $C=10^2$ and one large-margin result favouring classification at $C=10^5$ — measured on different data, different backbones, different metrics, different decades.

**Where the obstruction becomes visible.** To interpolate you need $\Delta(C)$ on *one* dataset. Try it on CUB: coarsening 100 classes into 20 super-classes changes $n$ from 59 to 295 and simultaneously changes the *test* task, because held-out evaluation is at species granularity. The relevance judgments move with $C$. There is no way to vary $C$ on a small benchmark while holding the evaluation fixed — which is exactly why §8 requires a dataset with nested labels and $N \sim 10^6$, and why the question is empirically open rather than merely unanswered.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*