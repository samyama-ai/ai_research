---
id: 08-loss-and-heads/negative-count-batch-size-confound
title: "Negative Sample Count Versus Batch Size Confound"
topic: 08-loss-and-heads
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Negative Sample Count Versus Batch Size Confound

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/negative-count-batch-size-confound` · **Status:** open

## 1. Problem Statement

In-batch contrastive losses (InfoNCE, CLIP's symmetric cross-entropy, SimCSE, dense-retrieval duals) draw negatives from the same minibatch that supplies the gradient. One knob, batch size $B$, therefore sets at least four things at once:

1. the number of negatives per anchor, $N = B-1$ (or $2B-2$ with two views);
2. the gradient noise scale, which falls as $1/B$;
3. the number of optimizer updates at fixed epoch budget, $S = ED/B$;
4. the loss's own normalization — the softmax partition is over $B$ terms, so the effective temperature and the negative–positive coupling term both shift with $B$.

The near-universal claim "contrastive learning needs large batches because it needs many negatives" attributes the observed effect entirely to (1). No published experiment at ImageNet or LAION scale cleanly separates (1) from (2)–(4).

- **Measurement variant.** Estimate $\partial(\text{downstream metric})/\partial \log N$ at fixed $B$, fixed step count, fixed compute. Solved when this partial derivative is reported with a control arm, not inferred from a batch-size sweep.
- **Method variant.** Build a training recipe whose quality at $N$ negatives is invariant to how those negatives were obtained (in-batch, queue, cached, subsampled). Solved when in-batch and decoupled arms match within noise.
- **Theory variant.** Give a generalization or excess-risk bound in which $N$ and the optimization terms ($B$, $S$, $\eta$) appear as separable factors. Currently every bound holds $N$ fixed and ignores the optimizer entirely.

## 2. Formal Setting

Data $x \sim \mathcal{D}$ over $\mathcal{X}$, augmentation/pairing kernel $\mathcal{A}(\cdot \mid x)$ giving views $(x^+, x)$, encoder $f_\theta : \mathcal{X} \to \mathbb{S}^{d-1}$, temperature $\tau$. For anchor $i$ in a batch $\mathcal{B}$ of size $B$, with negative index set $\mathcal{N}_i \subseteq \mathcal{B}\setminus\{i\}$, $|\mathcal{N}_i| = N$:

$$\ell_i = -\log \frac{\exp(f_i^\top f_i^+/\tau)}{\exp(f_i^\top f_i^+/\tau) + \sum_{j\in\mathcal{N}_i}\exp(f_i^\top f_j/\tau)}.$$

**Quantities as measured.**

- $N$ — count of terms in the denominator sum, excluding the positive. Instrumented by counting mask entries, not inferred from $B$.
- $B$ — number of anchors contributing to one parameter update, i.e. accumulated microbatches included. Under gradient caching, $B$ for the loss and $B$ for the memory footprint differ; report the loss one.
- $S$ — optimizer steps to convergence criterion. $C \approx 6PBS$ FLOPs for a $P$-parameter transformer encoder; hold $C$ fixed across arms or report it.
- Gradient noise scale $\mathcal{B}_{\text{noise}} = \mathrm{tr}(\Sigma)/|G|^2$ (McCandlish et al., 2018), measured by the two-batch-size estimator, not assumed.
- Effect size: $\Delta = \mathrm{Acc}(N_1) - \mathrm{Acc}(N_0)$ on a fixed downstream probe, with seed variance $\sigma$ from $\geq 3$ seeds. A claim is only meaningful at $|\Delta| > 2\sigma$.

**Assumptions, and which fail.**

- *Negatives are i.i.d. from the marginal.* False for in-batch negatives under any non-uniform sampler — shard-local batches, curriculum, deduplicated web crawls, and hard-negative mining all correlate $\mathcal{N}_i$ with $x_i$.
- *No false negatives.* False: at $B = 32{,}768$ on ImageNet-1k, expected same-class negatives per anchor is $\approx N/1000 \approx 33$. The class-collision term in Arora et al. (2019) grows linearly in $N$.
- *The gradient of $\ell_i$ w.r.t. the positive term is independent of $N$.* False. InfoNCE has a negative–positive coupling multiplier that scales the positive gradient by $1 - q_{i,i^+}$; at small $B$ this multiplier is systematically small, which Yeh et al. (2022) identify as an *optimization* pathology, not an information deficit.
- *Downstream linear probe accuracy is monotone in representation quality.* Weakly false; probe accuracy is sensitive to feature norm and to the layer probed.

## 3. State of the Art

**Empirical SOTA (established).** SimCLR (Chen et al., ICML 2020) reports ImageNet linear-probe top-1 rising with batch size at short schedules and the gap largely closing at 1000 epochs — established, and the paper itself attributes part of the effect to step count. MoCo (He et al., CVPR 2020) decouples $N$ from $B$ with a momentum queue: $B=256$, $K=65{,}536$ negatives, 60.6% top-1 at 200 epochs. This is the single strongest existing evidence that $N$ and $B$ are separable, but MoCo also changes the encoder (momentum target) and the negative staleness, so it is not a clean control.

**Claimed but unablated.** "Large batch is required for contrastive learning" appears as motivation in dozens of papers with no $N$-controlled arm. CLIP (Radford et al., ICML 2021) trained at $B=32{,}768$ and reports no batch-size ablation at all; the number is a hyperparameter choice, not a measured optimum.

**Benchmark-number-only.** SigLIP (Zhai et al., ICCV 2023) reports sigmoid-loss ImageNet zero-shot peaking near $B \approx 32$k with $B=98$k *worse* — a single benchmark curve, no decomposition into negatives versus optimization. OpenCLIP scaling laws (Cherti et al., CVPR 2023) hold batch size near-fixed while scaling data and compute, so they say nothing about this axis.

**Theory SOTA.** Arora et al. (ICML 2019): bound degrades with $N$ through class collision. Ash et al. (AISTATS 2022): bound and experiments give a **U-shape**, optimum near the number of latent classes. Awasthi, Dikkala, Kamath (ICML 2022): under a different decomposition, more negatives do *not* hurt. The three are not contradictory — they hold different things fixed — which is itself the diagnosis.

## 4. What Is Known

- SimCLR, ResNet-50, ImageNet-1k: batch $256 \to 8192$ gives several points of top-1 at 100 epochs; at 1000 epochs the spread across $256$–$8192$ is roughly a point. Scale: 1.28M images, 100–1000 epochs.
- MoCo v1 queue ablation, ResNet-50, ImageNet, 200 epochs: accuracy rises monotonically with $K$ but saturates — the $K = 16{,}384 \to 65{,}536$ step is under one point. So $256\times$ more negatives at fixed $B$ buys far less than the SimCLR batch sweep implies.
- BYOL (Grill et al., NeurIPS 2020) reaches 74.3% top-1 with **zero** negatives, and degrades only modestly from $B=4096$ to $B=256$ after re-tuning. SimSiam (Chen & He, CVPR 2021) works at $B=256$ with no negatives and no momentum encoder. Negative count cannot be the sole mechanism.
- Decoupled Contrastive Learning (Yeh et al., ECCV 2022): removing the positive term from the denominator recovers most of the large-batch gain at small batch on ImageNet — evidence that a chunk of the "negatives help" effect is the coupling artifact.
- MoCo v3 (Chen, Xie, He, ICCV 2021): ViT training at $B=4096$ is *less* stable and can score below $B=1024$. Larger $B$ is not monotone once optimization is the binding constraint.
- GradCache (Gao, Zhang, Han, Callan, RepL4NLP 2021) makes $B$ up to ~$10^5$ feasible on limited memory with mathematically identical gradients — the tool for the decoupling experiment exists.
- Critical batch size (McCandlish et al., 2018; Shallue et al., JMLR 2019): past $\mathcal{B}_{\text{noise}}$, extra batch buys no step-time reduction. This effect alone predicts diminishing returns in $B$ with no reference to negatives.

## 5. What Is Not Known

- **Empirically open (primary).** $\partial \mathrm{Acc}/\partial \log N$ at fixed $B$, fixed $S$, fixed $\tau$, at CLIP scale ($\geq 400$M pairs). Runnable today with GradCache plus negative masking; nobody has published it. Cost is the only barrier.
- **Empirically open.** The interaction $N \times \tau$. Temperature is almost always re-tuned per batch size, so published $N$-sweeps silently vary two knobs.
- **Theoretically open.** No bound in which the optimization terms ($B, S, \eta$, noise scale) and the statistical term in $N$ appear separably. Reconciling Arora's degradation, Ash's U-shape, and Awasthi's non-degradation is open — they may all be tight under their own assumptions.
- **Methodologically blocked.** "Effective number of negatives" is not well defined once negatives are stale (queue), reweighted (hard-negative mining), or correlated within a shard. A queue entry from 500 steps ago and a fresh in-batch entry are both counted as "1" and are plainly not equivalent. Until an effective-$N$ has an estimator, cross-method comparisons of $N$ are not comparisons.

## 6. Why It Is Hard

**Confounded measurement by construction, plus non-identifiability.** The loss is written so that a single integer $B$ enters four distinct mechanisms. Any sweep over $B$ produces a curve that is a sum of four partial effects, and the design admits no way to read off one of them. Two of the four have opposite signs at scale: more negatives raises false-negative collision (hurts) while lower gradient noise stabilizes training (helps), so a flat measured curve is consistent with two large cancelling effects — the parameters are not identified from the batch sweep alone.

Second obstruction: **compute**. The decoupling arm must hold step count fixed, so the small-$N$/large-$B$ arm costs the same as the large-$N$ arm — no savings. At CLIP scale a 3-seed, 4-point sweep is $\sim 12$ full pretraining runs.

Third: the downstream metric is a probe, not the objective. A change that moves linear-probe top-1 by 0.5 points may move retrieval recall by 3 points, so "the number that decides it" must be fixed in advance.

## 7. Current Research (as of 2026)

- **Loss redesign that removes the coupling.** SigLIP-style pairwise sigmoid losses (Google DeepMind, Zurich) sidestep the batch-wide partition entirely; SigLIP 2 continues the line. The relevant open question — does sigmoid loss make quality flat in $N$? — is stated but not ablated at fixed $S$.
- **Decoupled / cached negatives in retrieval.** GradCache, cross-batch memory, and ANCE-style asynchronous indices are standard in dense retrieval (CMU, Baidu, Meta). These groups routinely run $N \gg B$, which makes them the cheapest place to run the clean ablation. *(frontier — verify)* Several 2025–2026 retrieval papers report near-flat quality from $N=2^{13}$ to $2^{16}$ at fixed $B$; the individual claims are scattered across appendices and not aggregated.
- **Negative-free methods.** BYOL/SimSiam/VICReg/DINO lines continue to close the gap without any $N$, which bounds how large the true $N$-effect can be.
- **Theory.** Follow-ups to Awasthi et al. on when collision is benign, and spectral/kernel analyses of InfoNCE. No separable optimization-plus-statistics bound yet.

## 8. Concrete Next Experiment

**Scale.** LAION-400M subset of 100M image–text pairs, ViT-B/16 + text transformer, 1 epoch, AdamW, $\tau$ learned, GradCache so memory is decoupled from $B$. Cost per arm $\approx$ a few hundred A100-days; 6 arms + 3 seeds on the decisive pair.

**Arms.** All fix $B = 16{,}384$, hence identical step count $S = 6{,}104$, identical FLOPs, identical LR schedule, identical $\tau$ initialization.

| Arm | $N$ per anchor | Negative source |
|---|---|---|
| A (control) | 16,383 | full in-batch |
| B | 1,023 | uniform random mask of the same batch |
| C | 255 | uniform random mask |
| D | 1,023 | disjoint fixed partition (blocked) |
| E (reference) | 1,023 | true $B=1024$, $16\times$ more steps |
| F | 16,383 | full in-batch, DCL (coupling term removed) |

Arms A–D isolate $N$ with everything else nailed down. E reproduces the classic confounded comparison. F measures how much of any A-vs-C gap is the coupling artifact rather than information.

**Deciding number.** $\Delta_{\text{AC}} = $ ImageNet zero-shot top-1(A) $-$ top-1(C), reported with 3-seed $\sigma$.

- $\Delta_{\text{AC}} < 0.5$ pt while $\Delta_{\text{AE}} > 2$ pt $\Rightarrow$ the batch-size benefit is optimization, not negatives. The field's standard justification is wrong.
- $\Delta_{\text{AC}} > 2$ pt $\Rightarrow$ negative count is genuinely doing the work, and memory-efficient large-$N$ methods are the right investment.
- $\Delta_{\text{AC}} > 2$ pt but $\Delta_{\text{FC}} < 0.5$ pt $\Rightarrow$ the effect is the coupling term, removable by loss surgery at no compute cost.

## 9. Key References

- **[Foundational]** van den Oord, Li, Vinyals. *Representation Learning with Contrastive Predictive Coding.* 2018. — arXiv:1807.03748
- **[Foundational]** Arora, Khandeparkar, Khodak, Plevrakis, Saunshi. *A Theoretical Analysis of Contrastive Unsupervised Representation Learning.* ICML 2019. — arXiv:1902.09229
- **[SOTA]** Chen, Kornblith, Norouzi, Hinton. *A Simple Framework for Contrastive Learning of Visual Representations.* ICML 2020. — arXiv:2002.05709
- **[SOTA]** He, Fan, Wu, Xie, Girshick. *Momentum Contrast for Unsupervised Visual Representation Learning.* CVPR 2020. — arXiv:1911.05722
- **[SOTA]** Zhai, Mustafa, Kolesnikov, Beyer. *Sigmoid Loss for Language Image Pre-Training.* ICCV 2023. — arXiv:2303.15343
- Grill et al. *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.* NeurIPS 2020. — arXiv:2006.07733
- Chen, Xie, He. *An Empirical Study of Training Self-Supervised Vision Transformers.* ICCV 2021. — arXiv:2104.02057
- Yeh, Hong, Hsu, Liu, Chen, LeCun. *Decoupled Contrastive Learning.* ECCV 2022. — arXiv:2110.06848
- Ash, Goel, Krishnamurthy, Misra. *Investigating the Role of Negatives in Contrastive Representation Learning.* AISTATS 2022. — arXiv:2106.09943
- Awasthi, Dikkala, Kamath. *Do More Negative Samples Necessarily Hurt in Contrastive Learning?* ICML 2022. — arXiv:2205.01789
- Nozawa, Sato. *Understanding Negative Samples in Instance Discriminative Self-supervised Representation Learning.* NeurIPS 2021. — arXiv:2102.06866
- Gao, Zhang, Han, Callan. *Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup.* RepL4NLP @ ACL 2021. — arXiv:2101.06983
- Wang, Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML 2020. — arXiv:2005.10242
- **[Survey]** McCandlish, Kaplan, Amodei, OpenAI Dota Team. *An Empirical Model of Large-Batch Training.* 2018. — arXiv:1812.06162
- **[Survey]** Shallue, Lee, Antognini, Sohl-Dickstein, Frostig, Dahl. *Measuring the Effects of Data Parallelism on Neural Network Training.* JMLR 20(112), 2019. — arXiv:1811.03600

## 10. Worked Example

ImageNet-1k, $D = 1{,}281{,}167$ images, $E = 100$ epochs, SimCLR. Compare the two arms the field actually runs:

| | $B=256$ | $B=8192$ | ratio |
|---|---|---|---|
| negatives per anchor $N=2B-2$ | 510 | 16,382 | $32\times$ |
| optimizer steps $S=ED/B$ | 500,456 | 15,639 | $1/32$ |
| expected same-class negatives ($N/1000$) | 0.51 | 16.4 | $32\times$ |
| gradient noise scale (fixed, measured) | — | — | $1\times$ |

The measured top-1 gap of a few points is reported as evidence for the negatives. But the arms differ by $32\times$ in *both* $N$ and $S$, in opposite directions, and by $32\times$ in false-negative load. Three effects, one observation, one equation: not identifiable.

Now bound the negatives channel from data that already exists. MoCo holds $B=256$ and raises $K$ from $2^{14}$ to $2^{16}$ — a $4\times$ increase in $N$ landing entirely in the range the SimCLR sweep covers — and gains under one point of top-1. Extrapolating that slope, $32\times$ more negatives is worth roughly $\log_4(32) \approx 2.5$ such steps, i.e. a couple of points at most, and only if the queue's stale negatives count as full-value negatives — which is exactly the effective-$N$ measurement that section 5 flags as undefined.

Meanwhile BYOL at $N=0$ scores 74.3%, above SimCLR at $N=16{,}382$. The obstruction is visible here: the $N$-axis explanation must simultaneously account for a large batch-size gap, a small queue-size gap, and a zero-negative method that beats both. No single value of $\partial \mathrm{Acc}/\partial \log N$ fits all three, because the batch sweep is not measuring that derivative.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*