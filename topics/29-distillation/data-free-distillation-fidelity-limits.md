---
id: 29-distillation/data-free-distillation-fidelity-limits
title: "Data-Free Distillation Fidelity Limits"
topic: 29-distillation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Data-Free Distillation Fidelity Limits

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/data-free-distillation-fidelity-limits` · **Status:** open

## 1. Problem Statement

Data-free distillation trains a student $f_S$ to imitate a teacher $f_T$ without any sample from the teacher's training distribution $P$. The only interface is query access to $f_T$ (logits, or in some threat models gradients and BatchNorm buffers). Synthetic inputs are produced by inversion or by a generator, labelled by the teacher, and used as the transfer set.

Three distinct questions get called "the data-free distillation problem":

- **Measurement.** How should fidelity be scored? *Accuracy* on $P$ (does the student classify correctly) and *fidelity* on $P$ (does the student reproduce the teacher's decision, right or wrong) are different objectives, and off-distribution agreement is different again. Most data-free papers report accuracy and call it distillation quality.
- **Method.** Given a query budget $B$ and no real data, what is the best achievable agreement with $f_T$ on $P$? Current methods leave a persistent gap on anything harder than CIFAR.
- **Theory.** Is there a *limit*? Does there exist a teacher, a student class, and a query budget such that no data-free procedure can reach agreement $1-\epsilon$ on $P$, while a real-data procedure with the same budget can? No such separation has been proven for realistic architectures.

**Solved** would mean: a bound on attainable fidelity as a function of the query budget and of how far the synthetic distribution sits from $P$, plus a method that provably or reproducibly closes the gap to that bound.

## 2. Formal Setting

Teacher $f_T:\mathcal{X}\to\mathbb{R}^K$, softmax $\sigma$, prediction $h_T(x)=\arg\max_k f_T(x)_k$. Student $f_S\in\mathcal{H}$. Data distribution $P$ over $\mathcal{X}$, unavailable. Synthetic transfer distribution $Q$, from which $n$ samples are drawn under a query budget $B$ (total teacher forward/backward passes, including those spent *inside* the inversion loop — the usual accounting error).

**Fidelity (measured on a held-out real set $S\sim P^m$, $m$ named explicitly):**

$$\mathcal{F}_P(f_S)=\frac{1}{m}\sum_{i=1}^{m}\mathbf{1}[h_S(x_i)=h_T(x_i)], \qquad \mathcal{K}_P(f_S)=\frac{1}{m}\sum_{i=1}^m \mathrm{KL}\!\left(\sigma(f_T(x_i)/\tau)\,\|\,\sigma(f_S(x_i)/\tau)\right).$$

$\mathcal{F}_P$ is top-1 agreement; $\mathcal{K}_P$ is distributional fidelity at temperature $\tau$. Accuracy $\mathcal{A}_P$ uses the true label and is *not* a fidelity measure.

**Training objective (what is optimized):** $\min_{f_S}\ \mathbb{E}_{x\sim Q}\,\mathrm{KL}(\sigma(f_T(x)/\tau)\|\sigma(f_S(x)/\tau))$, with $Q$ produced by DeepInversion-style objectives: match per-channel BatchNorm statistics, $\sum_l \|\mu_l(x)-\hat\mu_l\|_2^2+\|\sigma^2_l(x)-\hat\sigma^2_l\|_2^2$, plus a confidence term $-\log \sigma(f_T(x))_{y}$ for a sampled target $y$, plus image priors (TV, $\ell_2$).

**The gap that matters.** Writing disagreement as a loss, standard covariate-shift decomposition (Ben-David et al., 2010) gives

$$1-\mathcal{F}_P(f_S)\ \le\ \underbrace{(1-\mathcal{F}_Q(f_S))}_{\text{trainable}}\ +\ \underbrace{d_{\mathcal{H}\Delta\mathcal{H}}(Q,P)}_{\text{not measurable data-free}},$$

where $d_{\mathcal{H}\Delta\mathcal{H}}$ is the disagreement discrepancy over the student class. The second term is the object of interest and cannot be estimated without samples from $P$ — the definitional core of the problem.

**Assumptions, and which break.**
1. *BN statistics identify the input distribution.* False: infinitely many $Q$ match the first two moments per channel; inversion recovers a texture-like mode, not $P$.
2. *Teacher logits are meaningful off-manifold.* Violated. Off-distribution logits are arbitrary and adversarially sensitive; the student fits noise the teacher does not "believe".
3. *Realizability* ($\exists f_S\in\mathcal{H}$ with $f_S\equiv f_T$). Usually false when the student is smaller; then even infinite data caps $\mathcal{F}_P<1$.
4. *Synthetic samples are i.i.d.* False: generator-based batches are strongly correlated and mode-collapse across rounds.
5. *Budget accounting is comparable.* Routinely violated — inversion consumes $10^2$–$10^4$ teacher passes per synthetic image, rarely counted against the baseline.

## 3. State of the Art

**Empirical SOTA (established).**
- **DeepInversion / Adaptive DeepInversion** (Yin et al., CVPR 2020) — BN-statistics inversion made ImageNet-scale data-free distillation and pruning work at all; the standard generator objective since.
- **DAFL** (Chen et al., ICCV 2019) and **Data-Free Adversarial Distillation** (Fang et al., 2019) — GAN-style generators trained against the teacher–student disagreement; the adversarial variant remains the strongest family on CIFAR.
- **CMI** (Fang et al., IJCAI 2021) adds contrastive diversity; **FastDFKD** (Fang et al., AAAI 2022) reports up to $100\times$ speedup via meta-learned reusable common features.
- **Data-free quantization**: DFQ (Nagel et al., ICCV 2019) and ZeroQ (Cai et al., CVPR 2020) — the one deployed success case, because 8-bit quantization needs only calibration statistics, not a transfer set.
- **Data-free model stealing**: MAZE (Kariyappa et al., CVPR 2021) and DFME (Truong et al., CVPR 2021) — the same machinery under a hard/soft-label query budget, and the only literature that reports *agreement* rather than accuracy as the headline.

**Claimed but unablated.** Nearly all CIFAR gains are reported without matching the teacher-query budget against a real-data or even a random-natural-image control. Generator-diversity claims are supported by downstream accuracy, not by any direct measurement of coverage of $P$.

**Benchmark-number-only.** ImageNet data-free results exist as single top-1 numbers per architecture pair, typically one seed, no agreement metric, no OOD or calibration reporting.

**Theory SOTA.** No fidelity limit for data-free distillation exists. The nearest results are (i) exact extraction of ReLU networks with finite queries — Carlini, Jagielski & Mironov (CRYPTO 2020), Rolnick & Kording (ICML 2020) — which shows the *information* is present in the query oracle; and (ii) Ben-David-style shift bounds, which are vacuous here because the discrepancy term is unmeasurable.

## 4. What Is Known

- **Query oracles carry enough information in principle.** Carlini et al. (CRYPTO 2020) extract a 2-layer, 4096-neuron ReLU network on MNIST to $\sim2^{-25}$ worst-case error with roughly $2^{21.5}$ queries. This is an existence result for arbitrary chosen inputs, not for gradient-descent distillation.
- **Accuracy and fidelity are separate axes.** Jagielski et al. (USENIX Security 2020) construct high-*accuracy* extractions that are low-*fidelity* and vice versa, at ImageNet scale.
- **Even with real data, fidelity plateaus.** Stanton et al. (NeurIPS 2021) show self-distilled students on CIFAR-100 and ImageNet fail to reach high teacher agreement even when the student can represent the teacher and the full training set is used — an optimization, not a data, failure. This bounds how much of the data-free gap can be blamed on the synthetic data.
- **The data-free gap is small on CIFAR and large above it.** ResNet-34→ResNet-18 CIFAR-10: teacher $\approx 95.6\%$, DAFL $\approx 92.2\%$, adversarial variants $\approx 93$–$95\%$. ImageNet-scale data-free students remain several points below their real-data counterparts, and no paper reports $\mathcal{F}_P$ there.
- **Statistics-only transfer works when the task is calibration.** DFQ reaches $71.2\%$ INT8 top-1 for MobileNetV2 against $71.7\%$ FP32 — no data at all. This is the boundary case where the data-free assumption is genuinely sufficient.

## 5. What Is Not Known

- **Theoretically open.** Whether a query-budget separation exists between data-free and data-using distillation for standard architectures: no lower bound saying $\mathcal{F}_P\le 1-\epsilon$ for all data-free procedures at budget $B$, and no matching upper bound. Also open: whether BN statistics plus logit access identify $P$ up to a class that preserves the teacher's decision boundary on $P$.
- **Empirically open.** The controlled comparison at ImageNet scale — data-free vs. real-data vs. *unrelated-natural-images* distillation at equal teacher-query budget, scored by $\mathcal{F}_P$. Runnable today on ~$10^3$ GPU-hours; nobody has published it.
- **Methodologically blocked.** There is no accepted measurement of how far a synthetic $Q$ is from $P$ that predicts fidelity. FID needs real data. Coverage/diversity proxies are computed in the teacher's own feature space, which is exactly the space the generator was optimized against — circular.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of $P$ from the teacher's exposed statistics, compounded by an evaluation that does not measure fidelity.** BN buffers fix per-channel means and variances — a measure-zero constraint on the space of input distributions. Every distribution matching those moments is an admissible $Q$, and the teacher's logits on the ones inversion actually finds are off-manifold outputs with no defined semantics. Secondary: the field's headline metric is accuracy on $P$, which can be high while agreement is low, so the gap the theory is about is not being recorded. Third: honest budget accounting is expensive — inversion costs $10^2$–$10^4$ teacher passes per image, so an equal-budget control arm at ImageNet scale is a real compute line item, which is why it is skipped.

## 7. Current Research (as of 2026)

- **LLM-era data-free distillation** — synthetic-prompt generation from the teacher itself, then distillation on teacher responses. Effectively data-free transfer with a language prior replacing image priors; fidelity is measured as win-rate, not agreement. *(frontier — verify)*
- **Diffusion priors as the generator**, replacing per-image inversion with sampling from a pretrained generative model conditioned on teacher gradients. Cheaper per sample; introduces an external data dependency that arguably breaks the data-free premise. *(frontier — verify)*
- **Extraction-defence work** (Kariyappa, Qureshi, and others) — perturbing logits to break data-free stealing. This community measures agreement properly and is the best source of honest fidelity numbers.
- **Data-free quantization beyond 8-bit** — INT4 without data remains unreliable; active in the efficient-inference community.

## 8. Concrete Next Experiment

**Question.** At an equal, fully-accounted teacher-query budget, how much fidelity does removing real data actually cost?

**Scale.** Teacher: ResNet-50 on ImageNet-1k ($76.1\%$ top-1). Student: ResNet-18. Budget $B=10^{8}$ teacher forward-pass-equivalents, counted for *all* arms including every pass inside the inversion loop (one inversion step = one forward + one backward = 2 units). ~1,500 A100-hours total.

**Arms.**
1. *Data-free*: DeepInversion + adversarial diversity, budget $B$.
2. **Control**: real-data distillation on ImageNet train images, same $B$, same student, same schedule, same augmentation.
3. *Proxy-data control*: distillation on an equal number of unlabeled, non-ImageNet natural images (e.g. Places365), same $B$. This is the arm that decides whether the synthetic generator adds anything over "any natural images at all".

**Deciding number.** Top-1 agreement with the teacher on the ImageNet validation set, $\mathcal{F}_P$, $m=50{,}000$, 3 seeds, report mean ± s.d.

- If $\mathcal{F}_P(\text{arm 1}) \ge \mathcal{F}_P(\text{arm 3})$ by more than 2 points, inversion is doing real work and the fidelity limit is a method problem.
- If arms 1 and 3 are within 2 points while arm 2 leads both by more than 5 points, the gap is about $P$-coverage, not generator sophistication, and the field's generator research is misdirected.

Secondary readout: $\mathcal{K}_P$ at $\tau=1$ and agreement restricted to teacher-misclassified validation points — the cleanest fidelity signal, since matching the teacher's *errors* cannot be achieved by generic learning.

## 9. Key References

- **[Foundational]** Hinton, Vinyals & Dean. *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[Foundational]** Lopes, Fenu & Starner. *Data-Free Knowledge Distillation for Deep Neural Networks.* NIPS LLD Workshop, 2017. — arXiv:1710.07535
- **[SOTA]** Yin, Molchanov, Alvarez, Li, Mallya, Hoiem, Jha & Kautz. *Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion.* CVPR, 2020. — arXiv:1912.08795
- **[SOTA]** Chen, Wang, Xu, Yang, Liu, Shi, Xu, Xu & Tian. *Data-Free Learning of Student Networks.* ICCV, 2019. — arXiv:1904.01186
- **[SOTA]** Nayak, Mopuri, Shaj, Babu & Chakraborty. *Zero-Shot Knowledge Distillation in Deep Networks.* ICML, 2019. — arXiv:1905.08114
- **[SOTA]** Truong, Maini, Walls & Papernot. *Data-Free Model Extraction.* CVPR, 2021. — arXiv:2011.14779
- **[SOTA]** Kariyappa, Prakash & Qureshi. *MAZE: Data-Free Model Stealing Attack Using Zeroth-Order Gradient Estimation.* CVPR, 2021.
- **[Theory]** Carlini, Jagielski & Mironov. *Cryptanalytic Extraction of Neural Network Models.* CRYPTO, 2020. — arXiv:2003.04884
- **[Theory]** Rolnick & Kording. *Reverse-engineering deep ReLU networks.* ICML, 2020.
- **[Theory]** Ben-David, Blitzer, Crammer, Kulesza, Pereira & Vaughan. *A theory of learning from different domains.* Machine Learning 79(1–2), 2010.
- **[Measurement]** Jagielski, Carlini, Berthelot, Kurakin & Papernot. *High Accuracy and High Fidelity Extraction of Neural Networks.* USENIX Security, 2020. — arXiv:1909.01838
- **[Measurement]** Stanton, Izmailov, Kirichenko, Alemi & Wilson. *Does Knowledge Distillation Really Work?* NeurIPS, 2021. — arXiv:2106.05945
- **[Applied]** Nagel, van Baalen, Blankevoort & Welling. *Data-Free Quantization Through Weight Equalization and Bias Correction.* ICCV, 2019. — arXiv:1906.04721
- **[Applied]** Cai, Yao, Dong, Gholami, Mahoney & Keutzer. *ZeroQ: A Novel Zero Shot Quantization Framework.* CVPR, 2020. — arXiv:2001.00281
- **[Survey]** Liu, Zhang, Wang, Wang, Oyang & Zhao. *Data-Free Knowledge Transfer: A Survey.* 2021. — arXiv:2112.15278

## 10. Worked Example

Take CIFAR-10, teacher ResNet-34 at $95.6\%$, student ResNet-18. Two transfer sets, both labelled only by the teacher, both trained to convergence.

| Transfer set | Student accuracy $\mathcal{A}_P$ | Agreement $\mathcal{F}_P$ | Agreement on teacher-*wrong* points |
|---|---|---|---|
| Real CIFAR-10 train (50k) | ~94.5% | ~96% | ~50% |
| DeepInversion synthetic (50k) | ~92–93% (reported) | *not reported anywhere* | *not reported anywhere* |

The accuracy column is the one the literature prints, and the gap looks like 2–3 points — modest. Now do the arithmetic on the third column. CIFAR-10 test has 10,000 points; the teacher errs on about 440 of them. Suppose the data-free student reaches $92.5\%$ accuracy. Its agreement with the teacher on the 9,560 correctly-classified points is at most $\approx 92.5\%/95.6\% \cdot 100\% \approx 96.8\%$ of them if every student-correct point is teacher-correct — but on the 440 teacher-errors, agreement is unconstrained by accuracy and can be anywhere from 0% to 100%. Two students with identical $92.5\%$ accuracy can differ by 440 points ($4.4\%$ of the test set) in $\mathcal{F}_P$.

**Where the obstruction becomes visible.** Teacher errors are precisely the points where the decision boundary is idiosyncratic — determined by the training data $P$, not by the task. A synthetic $Q$ built from BatchNorm moments has no reason to place samples near those idiosyncratic regions: the moment constraints are satisfied by texture-like images far from the ambiguous cat/dog boundary. So the metric the field reports (accuracy) is systematically blind to the metric the problem is about (fidelity), and the one number that would expose the gap — agreement on teacher-misclassified points — appears in no data-free distillation paper. That is why this problem is simultaneously empirically open and methodologically blocked: the experiment is cheap, and the measurement is simply not being taken.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*