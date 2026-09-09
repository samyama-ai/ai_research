---
id: 30-synthetic-data/synthetic-image-privacy-utility-frontier
title: "Privacy-Utility Pareto Frontier for Synthetic Images"
topic: 30-synthetic-data
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Privacy-Utility Pareto Frontier for Synthetic Images

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/synthetic-image-privacy-utility-frontier` · **Status:** open

## 1. Problem Statement

Given a sensitive image dataset $D$, a synthesizer $M$ emits a synthetic set $\tilde{D}$ intended to be released. The question is the shape and location of the achievable trade-off curve between a privacy cost and a utility gain.

Three variants, commonly conflated:

- **Measurement variant.** Given a released $\tilde{D}$ and no access to $M$'s internals, estimate where it sits in the privacy-utility plane. Open because per-record empirical privacy loss for image synthesizers has no agreed estimator, and the dominant utility metric (FID) is known not to measure what downstream users need.
- **Method variant.** Build a synthesizer that dominates all current ones: at fixed $(\varepsilon,\delta)$, strictly higher downstream accuracy. Runnable, competitive, and currently held by public-pretraining pipelines whose accounting is contested.
- **Theory variant.** Characterise the *information-theoretically optimal* frontier $U^\star(\varepsilon)$ for a natural image class, and prove a separation between "generate synthetic images, then train" and "train under DP directly". No proof either way for any realistic image distribution.

Solving it means: an estimator that places an arbitrary released set on the plane with calibrated error bars, plus matching upper and lower bounds on $U^\star(\varepsilon)$ within a constant factor.

## 2. Formal Setting

$D = \{x_i\}_{i=1}^n \subset \mathcal{X}$, $\mathcal{X}=[0,1]^{3\times H\times W}$, drawn i.i.d. from $P$. A randomized synthesizer $M:\mathcal{X}^n \to \mathcal{X}^m$ produces $\tilde{D}\sim M(D)$.

**Privacy (declared).** $M$ is $(\varepsilon,\delta)$-DP if for all neighboring $D \simeq D'$ (one record replaced) and all measurable $S$,
$$\Pr[M(D)\in S] \le e^{\varepsilon}\Pr[M(D')\in S] + \delta.$$
As measured: $\varepsilon$ is *not* measured — it is computed analytically from the DP-SGD noise multiplier $\sigma$, sampling rate $q=B/n$, step count $T$, and clipping norm $C$ via a Rényi or PLD accountant. It is a property of the declared mechanism, not of $\tilde{D}$.

**Privacy (empirical lower bound).** For an attack $A$ with true-positive rate $\alpha$ and false-positive rate $\beta$ on the membership game,
$$\varepsilon_{\text{emp}} \ge \log\max\!\left(\frac{1-\delta-\beta}{\alpha},\ \frac{1-\delta-\alpha}{\beta}\right),$$
measured by training $K$ shadow synthesizers, running $A$ on held-in vs held-out canaries, and taking a Clopper–Pearson lower confidence bound at low FPR. Cost scales as $K$ full synthesizer trainings; one-run auditing (Steinke et al., 2023) replaces $K$ trainings with $K$ independent canaries inside one run.

**Utility.** Two non-interchangeable measurements.
1. *Downstream*: $U_{\text{task}} = \mathbb{E}\big[\mathrm{acc}_{P}(f_{\tilde{D}})\big]$, where $f_{\tilde{D}}$ is a fixed-architecture classifier trained only on $\tilde{D}$ and evaluated on a real held-out test split. Measured with fixed seeds, fixed augmentation, fixed $m$.
2. *Distributional*: $\mathrm{FID}(\tilde{D},D_{\text{test}}) = \|\mu_1-\mu_2\|_2^2 + \mathrm{tr}(\Sigma_1+\Sigma_2-2(\Sigma_1\Sigma_2)^{1/2})$ on Inception-V3 pool3 features.

**Frontier.** $U^\star(\varepsilon) = \sup\{U_{\text{task}}(M) : M \text{ is } (\varepsilon,\delta)\text{-DP}\}$, for fixed $n$, $m$, $\delta$, and downstream learner.

**Assumptions, and which fail.**
- *Record-level neighboring relation captures the unit of privacy.* Violated: medical and face corpora contain many images per subject, so record-DP gives a per-subject guarantee inflated by the group-privacy factor $k$.
- *Public pretraining data is independent of $D$.* Violated routinely — ImageNet/LAION pretraining overlaps the support of most benchmark "private" sets (Tramèr et al., ICML 2024).
- *Inception features are a sufficient statistic for image quality.* Violated: FID is dominated by ImageNet class structure (Kynkäänniemi et al., ICLR 2023).
- *i.i.d. sampling in the accountant.* Violated: implementations shuffle rather than Poisson-sample, so the reported $\varepsilon$ is not the $\varepsilon$ of the code that ran.

## 3. State of the Art

**Theory SOTA.** DP-SGD composition bounds (Abadi et al., CCS 2016) plus tight PLD accounting are established. No non-trivial lower bound on $U^\star(\varepsilon)$ exists for image synthesis; the only formal results are generic DP estimation lower bounds that do not bind at pixel dimensionality.

**Empirical SOTA (established).**
- **DPDM** (Dockhorn et al., TMLR 2023): DP diffusion trained from scratch. Strong on MNIST/Fashion-MNIST, collapses on CIFAR-10 — the honest from-scratch baseline.
- **DP fine-tuned diffusion** (Ghalebikesabi et al., 2023, arXiv:2302.13861): ImageNet pretraining, DP fine-tuning on CIFAR-10; first demonstration that synthetic images at single-digit $\varepsilon$ train a usable classifier.
- **Private Evolution / DPSDA** (Lin et al., ICLR 2024): no gradients at all — DP nearest-neighbour voting over API samples. Established that the frontier can be reached without training access.

**Claimed but unablated.**
- That synthetic-then-train beats DP-training-the-classifier-directly. The control arm (De et al., 2022, arXiv:2204.13650: 81.4% CIFAR-10 at $\varepsilon=8$ with JFT pretraining) is usually absent from synthetic-data papers.
- That headline FID numbers at small $\varepsilon$ reflect privacy-preserving generalisation rather than pretrained-prior recall. Unablated: nobody reports the same pipeline with $\varepsilon=0$ (pretrained prior, zero private access) as a floor.

**Benchmark-number-only.** Nearly all CIFAR-10/CelebA-64 DP-synthesis FIDs are single-run, single-seed, single-feature-extractor. No reported variance across synthesizer seeds.

## 4. What Is Known

- **DP-SGD works at ImageNet scale with public pretraining.** 81.4% top-1 on CIFAR-10 at $(\varepsilon{=}8,\delta{=}10^{-5})$ from JFT-300M pretraining; 4,096-image batches, ~2M-example effective batch regimes (De et al., 2022).
- **Undefended diffusion models memorise.** Carlini et al. (USENIX Security 2023) extracted 50 near-verbatim training images from Stable Diffusion out of ~350k generations targeting the 1,000 most-duplicated captions; ~2.5% extraction on a small DDPM trained on CIFAR-10 subsets. Somepalli et al. (CVPR 2023) found ~2% of LAION-trained generations contain significant copies.
- **Synthetic data is not anonymisation by construction.** Stadler et al. (USENIX Security 2022) showed non-DP synthesizers leak linkage and attribute information at rates comparable to naive de-identification, on tabular data at $n\approx10^3$–$10^4$.
- **Auditing gap is large.** Empirical $\varepsilon_{\text{emp}}$ for DP synthetic-data generators sits far below analytic $\varepsilon$ unless the adversary is given worst-case canaries and full gradient access (Annamalai et al., USENIX Security 2024; Nasr et al., S&P 2021).
- **FID and downstream accuracy disagree.** Stein et al. (NeurIPS 2023) showed metric rankings of generative models reverse between FID and human/downstream evaluation at ImageNet scale.
- **Private Evolution reports strong CIFAR-10 FID at sub-1 $\varepsilon$** using an ImageNet-pretrained diffusion API — a number whose interpretation depends entirely on the pretraining-overlap assumption above.

## 5. What Is Not Known

- **Theoretically open.** Any lower bound on $U^\star(\varepsilon)$ for natural images. Whether the two-stage pipeline (DP synthesis → non-private training) can ever dominate one-stage DP training; post-processing invariance says it cannot *exceed* the best DP learner at the same $\varepsilon$ for a single task, but the multi-task / unknown-downstream-task case has no theorem.
- **Empirically open.** The full frontier $\{(\varepsilon, U_{\text{task}})\}$ for $\varepsilon \in \{0, 0.5, 1, 2, 4, 8, \infty\}$ with an $\varepsilon=0$ pretrained-prior floor and a one-stage DP-training control, at ImageNet-1k resolution $\ge 128$. Runnable today; roughly $10^3$–$10^4$ A100-hours. Nobody has published it.
- **Methodologically blocked.** Per-release empirical privacy loss. Given only $\tilde{D}$, there is no accepted estimator of leakage with calibrated error bars — shadow-model MIA on synthetic outputs is a lower bound whose looseness is unquantified, and nearest-neighbour distance to $D$ is known to be a poor proxy. Also blocked: the privacy unit for multi-image-per-subject corpora.

## 6. Why It Is Hard

**The primary obstruction is confounded measurement on both axes simultaneously.**

On the privacy axis, the reported $\varepsilon$ measures a *declared mechanism*, while the attack-based $\varepsilon_{\text{emp}}$ measures a *specific adversary*. The gap between them is typically an order of magnitude and is not an error bar — it is the union of (a) loose accounting, (b) weak attacks, and (c) implementation/accounting mismatch (shuffling vs Poisson sampling). No experiment separates these three without $K\gtrsim10^3$ shadow trainings.

On the utility axis, FID is a biased functional of an ImageNet classifier's features, so "better synthetic images" and "better FID" come apart precisely in the low-$\varepsilon$ regime where samples degrade non-naturally (noise, blur) rather than semantically.

Compounding: public pretraining makes the $x$-axis non-identifiable. Two pipelines reporting $\varepsilon=1$ may have had wildly different *total* information about $P$, because the pretraining corpus is charged at zero. Moving an arbitrary amount of the task into the free prior moves the frontier arbitrarily far up without any change in the accounted $\varepsilon$. Until the $\varepsilon=0$ floor is reported, the curve is not a curve of anything.

## 7. Current Research (as of 2026)

- **Training-free / API-based synthesis.** Private Evolution line (Microsoft Research and collaborators) extended beyond images to text; active work on better DP voting and on foundation-model APIs as the only scalable access mode.
- **Tight auditing.** One-run auditing (Steinke, Nasr, Jagielski) being pushed from classifiers toward generative pipelines; the target is an audit whose cost is one training run and whose $\varepsilon_{\text{emp}}$ is within 2× of analytic $\varepsilon$ *(frontier — verify)*.
- **Pretraining accountability.** Following Tramèr, Kamath and Carlini (ICML 2024), proposals to report a public-data-only baseline alongside every DP number; not yet a norm.
- **Subject-level and concept-level guarantees** for medical imaging — per-patient DP rather than per-image *(frontier — verify)*.
- **Replacing FID** with precision/recall-style decompositions (Alaa et al., ICML 2022: $\alpha$-precision, $\beta$-recall, authenticity) in privacy settings, where *authenticity* doubles as a copying detector.

## 8. Concrete Next Experiment

**Question.** Does DP synthetic image data ever beat one-stage DP training at equal $\varepsilon$, once the free-prior floor is charged?

**Scale.** $D$ = ImageNet-100 (100 classes, $n{=}130{,}000$) at $64\times64$. Synthesizer: latent diffusion pretrained on a *disjoint, deduplicated* corpus (Places365 or a LAION subset with ImageNet-100 near-duplicates removed by CLIP-embedding threshold), DP fine-tuned on $D$. $m = 1{,}000{,}000$ synthetic images. Downstream: fixed ResNet-50, fixed recipe, 3 seeds. Sweep $\varepsilon \in \{0, 0.5, 1, 2, 4, 8\}$, $\delta = 10^{-6}$. Budget ≈ 2,000 A100-hours.

**Control arms (both required).**
1. **$\varepsilon=0$ floor** — sample $m$ images from the pretrained prior with *zero* access to $D$, train the same ResNet-50.
2. **One-stage** — DP-SGD fine-tune the same ResNet-50 backbone directly on $D$ at each $\varepsilon$, identical accountant.

**Deciding number.** $\Delta(\varepsilon) = U_{\text{synth}}(\varepsilon) - \max\{U_{\varepsilon=0}, U_{\text{one-stage}}(\varepsilon)\}$ in top-1 accuracy points on the real ImageNet-100 validation split. If $\Delta(\varepsilon) \le 0$ for all $\varepsilon \le 8$ with 95% CI over seeds, the synthetic-data route buys nothing for single-task utility and the field's headline numbers are prior recall. If $\Delta(\varepsilon) \ge +3$ points at any $\varepsilon \le 4$, the two-stage separation is real and the theory variant becomes urgent.

**Secondary.** Inject 1,000 canary images; report $\varepsilon_{\text{emp}}$ by one-run auditing at each $\varepsilon$. The ratio $\varepsilon/\varepsilon_{\text{emp}}$ is the first published audit-tightness curve for image synthesis.

## 9. Key References

- **[Foundational]** M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, L. Zhang. *Deep Learning with Differential Privacy.* ACM CCS, 2016. — arXiv:1607.00133
- **[Foundational]** C. Dwork, A. Roth. *The Algorithmic Foundations of Differential Privacy.* Foundations and Trends in TCS, 2014.
- **[SOTA]** T. Dockhorn, T. Cao, A. Vahdat, K. Kreis. *Differentially Private Diffusion Models.* TMLR, 2023. — arXiv:2210.09929
- **[SOTA]** S. Ghalebikesabi, L. Berrada, S. Hayes, S. De, S. Ghodsi, J. Hayes, et al. *Differentially Private Diffusion Models Generate Useful Synthetic Images.* 2023. — arXiv:2302.13861
- **[SOTA]** Z. Lin, S. Gopi, J. Kulkarni, H. Nori, S. Yekhanin. *Differentially Private Synthetic Data via Foundation Model APIs 1: Images.* ICLR, 2024. — arXiv:2305.15560
- **[SOTA]** S. De, L. Berrada, J. Hayes, S. L. Smith, B. Balle. *Unlocking High-Accuracy Differentially Private Image Classification through Scale.* 2022. — arXiv:2204.13650
- **[Attack]** N. Carlini, J. Hayes, M. Nasr, M. Jagielski, V. Sehwag, F. Tramèr, B. Balle, D. Ippolito, E. Wallace. *Extracting Training Data from Diffusion Models.* USENIX Security, 2023. — arXiv:2301.13188
- **[Attack]** G. Somepalli, V. Singla, M. Goldblum, J. Geiping, T. Goldstein. *Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models.* CVPR, 2023. — arXiv:2212.03860
- **[Attack]** T. Stadler, B. Oprisanu, C. Troncoso. *Synthetic Data — Anonymisation Groundhog Day.* USENIX Security, 2022. — arXiv:2011.07018
- **[Auditing]** T. Steinke, M. Nasr, M. Jagielski. *Privacy Auditing with One (1) Training Run.* NeurIPS, 2023. — arXiv:2305.08846
- **[Auditing]** M. S. M. S. Annamalai, G. Ganev, E. De Cristofaro. *"What do you want from theory alone?" Experimenting with Tight Auditing of Differentially Private Synthetic Data Generation.* USENIX Security, 2024.
- **[Attack]** N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, F. Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P, 2022. — arXiv:2112.03570
- **[Metrics]** T. Kynkäänniemi, T. Karras, M. Aittala, T. Aila, J. Lehtinen. *The Role of ImageNet Classes in Fréchet Inception Distance.* ICLR, 2023. — arXiv:2203.06026
- **[Metrics]** G. Stein, J. C. Cresswell, R. Hosseinzadeh, Y. Sui, B. L. Ross, et al. *Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models.* NeurIPS, 2023.
- **[Metrics]** A. Alaa, B. van Breugel, E. Saveliev, M. van der Schaar. *How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models.* ICML, 2022.
- **[Position]** F. Tramèr, G. Kamath, N. Carlini. *Position: Considerations for Differentially Private Learning with Large-Scale Public Pretraining.* ICML, 2024. — arXiv:2212.06470
- **[Survey]** Y. Hu, F. Wu, Q. Li, Y. Long, G. Garrido, C. Ge, B. Ding, D. Forsyth, B. Li, D. Song. *SoK: Privacy-Preserving Data Synthesis.* IEEE S&P, 2024.

## 10. Worked Example

Take the standard reported setup: CIFAR-10, $n = 50{,}000$, ImageNet-pretrained diffusion, DP fine-tuned to $(\varepsilon{=}10,\delta{=}10^{-5})$, $m = 50{,}000$ synthetic images, WRN-40-4 downstream. Reported: FID ≈ 10, downstream accuracy ≈ 75%.

Now run the two arithmetic checks the papers skip.

**Check 1 — the free-prior floor.** CIFAR-10's 10 classes are all ImageNet superclasses. Sampling 50,000 class-conditioned images from the *pretrained* model with zero private access ($\varepsilon = 0$) and training the same WRN gives a non-trivial accuracy — plausibly 50–65%, since the same backbone zero-shot-transfers in that band. If the floor is 60%, the entire private signal purchased at $\varepsilon = 10$ is $\approx 15$ points, not 75. The reported point is mostly prior.

**Check 2 — group privacy.** CIFAR-10 is one image per record, so record-DP is fine. Swap in a chest-X-ray corpus with $k = 8$ images per patient. Per-patient $\varepsilon_{\text{subject}} \le k\varepsilon = 80$, and $\delta_{\text{subject}} \le k e^{(k-1)\varepsilon}\delta$. With $\varepsilon=10$, $k=8$, $\delta=10^{-5}$:
$$\delta_{\text{subject}} \le 8 \cdot e^{70} \cdot 10^{-5} \approx 2 \times 10^{26},$$
which is vacuous — the bound carries no information at all. The headline "$\varepsilon = 10$" is a guarantee about one radiograph, and the thing a patient cares about has no guarantee.

**The obstruction made visible.** The same released $\tilde{D}$ sits at "$\varepsilon=10$, utility 75" under the reported convention, at "effective utility 15" once the free prior is charged, and at "no guarantee" under the privacy unit that matters clinically. Three different points in the plane from one artifact, with no measurement distinguishing them. Until the $\varepsilon=0$ arm and the privacy unit are both fixed and reported, the frontier is not being measured — only a convention is.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*