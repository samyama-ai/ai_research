---
id: 22-safety-robustness/certified-robustness-semantic-transformations
title: "Certified Robustness Under Semantic Transformations"
topic: 22-safety-robustness
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Certified Robustness Under Semantic Transformations

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/certified-robustness-semantic-transformations` · **Status:** partially-solved

## 1. Problem Statement

Given a classifier $f$ and an input $x$, produce a machine-checkable guarantee that $f$'s prediction is constant over a set of *semantic* variants of $x$ — rotations, translations, scaling, brightness/contrast shifts, blur, colour-space moves, elastic deformations, and compositions of these — rather than over an $\ell_p$ ball. The set is parameterised by a low-dimensional vector $\alpha \in \mathcal{A}$ (e.g. $\alpha$ = rotation angle in $[-30°, 30°]$), but the induced set in pixel space is a curved, non-convex manifold of far higher $\ell_2$ diameter than any interesting norm ball.

Three variants that are routinely conflated:

- **Theory.** For which transformation families does a sound certificate with non-vacuous radius exist at all, and what is the achievable certified-accuracy/robustness trade-off?
- **Method.** Build a certifier that is *sound* (never certifies a point that a transformation can flip) and *tight enough* to be non-vacuous at ImageNet scale.
- **Measurement.** Decide which $\mathcal{A}$ is worth certifying. Unlike $\ell_p$, the semantic threat model is chosen by the researcher, so a certificate can be simultaneously sound and irrelevant.

Solving it means: a sound, scalable certifier for a *specified* family, plus a defensible argument that the family covers the deployment-relevant perturbations.

## 2. Formal Setting

Let $x \in [0,1]^{d}$ ($d = H \cdot W \cdot C$; for ImageNet $d = 150{,}528$), label set $\mathcal{Y}$, and a transformation $\phi: [0,1]^d \times \mathcal{A} \to [0,1]^d$ with parameter space $\mathcal{A} \subseteq \mathbb{R}^m$, $m$ small ($m=1$ rotation, $m=2$ translation, $m=6$ affine).

**Certified predicate.** $f$ is *certifiably robust* at $x$ for $\mathcal{A}$ if
$$\forall \alpha \in \mathcal{A}: \quad f(\phi(x,\alpha)) = f(x).$$

**Certified accuracy** (the reported quantity): over a test set $D$,
$$\mathrm{CA}(\mathcal{A}) = \frac{1}{|D|}\sum_{(x,y)\in D} \mathbf{1}\big[\text{certifier returns } y\big] \cdot \mathbf{1}[y = f(x)].$$
Measured by running the certifier on each of typically 500 subsampled test images and counting; abstentions count as failures.

**Randomized smoothing base.** The smoothed classifier is $g(x) = \arg\max_c \Pr_{\varepsilon \sim \mathcal{N}(0,\sigma^2 I)}[f(x+\varepsilon)=c]$. Cohen et al.'s bound: with $\underline{p_A}$ a lower confidence bound on the top-class probability, $g$ is constant on the $\ell_2$ ball of radius
$$R = \sigma\,\Phi^{-1}(\underline{p_A}).$$
$\underline{p_A}$ is measured by $n$ Monte-Carlo samples (commonly $n = 10^5$) with a Clopper–Pearson bound at $\alpha_{\text{err}} = 0.001$; the certificate is therefore probabilistic, holding with probability $\ge 1-\alpha_{\text{err}}$ *per input*.

**Resolvability.** $\phi$ is *resolvable* if $\phi(\phi(x,\beta),\alpha) = \phi(x, \gamma(\alpha,\beta))$ exactly — true for brightness/contrast, false for rotation on a pixel grid. For *differentially resolvable* transformations (rotation, scaling), one bounds the interpolation residual
$$M_{\mathcal{A}} = \max_{\alpha \in \mathcal{A}} \ \big\| \phi(x,\alpha) - \phi(\phi(x,\alpha_i),\, \alpha - \alpha_i) \big\|_2$$
over a finite grid $\{\alpha_i\}$, and certifies only if $R > M_{\mathcal{A}}$.

**Assumptions, and which break.**
1. *$M_{\mathcal{A}}$ is computable.* In practice it is estimated by sampling $\alpha$ densely and taking a max, sometimes with a Lipschitz correction — a sampled max is not a bound. **Violated in practice** unless an explicit Lipschitz constant over $\alpha$ is proved.
2. *Pixel space is continuous.* Real images are 8-bit and rotations resample; $\phi$ is not a group action on the grid. **Violated.**
3. *Per-input error rate composes.* Reporting certified accuracy over 500 images with per-input $\alpha_{\text{err}} = 0.001$ gives a family-wise error of up to $0.5$ images, usually unstated. **Ignored, not violated.**
4. *$\mathcal{A}$ is the threat model.* Deployment perturbations (weather, sensor, compression) are not in any parameterised $\mathcal{A}$ used in the literature. **Violated.**

## 3. State of the Art

**Deterministic / relaxation-based (theory SOTA for small nets).** DeepPoly (Singh et al., POPL 2019) and its geometric extension DeepG (Balunović et al., *Certifying Geometric Robustness of Neural Networks*, NeurIPS 2019) compute sound linear relaxations of the interpolation operator, giving *deterministic* certificates for rotation/translation. Established, but limited to CIFAR-scale networks and small parameter intervals (rotations of a few degrees). Ruoss et al. (*Efficient Certification of Spatial Robustness*, AAAI 2021) improved throughput on the same regime. Mohapatra et al. (CVPR 2020) extended CROWN-style bounds to brightness/contrast/rotation on CIFAR-10.

**Smoothing-based (empirical SOTA at scale).** Fischer, Baader, Vechev (*Certified Defense to Image Transformations via Randomized Smoothing*, NeurIPS 2020) gave the first smoothing certificate for rotation with an explicit interpolation-error term. TSS (Li et al., *TSS: Transformation-Specific Smoothing for Robustness Certification*, CCS 2021) is the reference point: the resolvable/differentially-resolvable split above is theirs, and it is the first method with non-trivial ImageNet certificates for rotation. GSmooth (Hao et al., ICML 2022) extends to non-resolvable transformations via a learned surrogate; DeformRS (Alfarra et al., AAAI 2022) certifies parametric and free-form deformations.

**Established vs. claimed.** Established: the resolvable-transformation certificates (brightness, contrast, Gaussian blur, translation-with-black-padding) are exact reductions to $\ell_2$ or $\ell_\infty$ smoothing and have been reproduced. Claimed but under-ablated: (i) GSmooth's surrogate-model soundness — the certificate inherits any error in the learned surrogate, and the paper's guarantee is conditional on a surrogate error bound that is estimated, not proved; (ii) composition results (rotation ∘ brightness ∘ blur) exist mainly as single benchmark table rows with no independent reproduction; (iii) DeformRS's free-form-deformation numbers are benchmark numbers only.

## 4. What Is Known

- **$\ell_2$ baseline, ImageNet (Cohen et al., ICML 2019, ResNet-50, 500 test images):** 49% certified accuracy at $r = 0.5$, 37% at $r = 1.0$. This is the ceiling any smoothing-based semantic certificate inherits.
- **Rotation, ImageNet (TSS, CCS 2021):** 30.4% certified accuracy under rotation within $\pm 30°$ — the first non-vacuous ImageNet certificate for a differentially resolvable transformation. Prior methods certified 0% at that scale.
- **Resolvable transformations, CIFAR-10 (TSS):** certified accuracy above 80% for Gaussian blur and brightness/contrast shifts, close to clean accuracy, because these reduce exactly to $\ell_2$ smoothing with no interpolation loss.
- **Deterministic certification does not scale:** DeepG-class methods handle CIFAR-10 convnets and rotation intervals of roughly $\pm 2°$–$\pm 10°$ with per-image runtimes in the tens of seconds to minutes; no deterministic ImageNet result for rotation exists.
- **Empirical spatial attacks are cheap:** Engstrom et al. (*Exploring the Landscape of Spatial Robustness*, ICML 2019) showed grid search over rotation+translation drops undefended CIFAR-10 accuracy from ~93% to under 10%, and that the loss surface over $\alpha$ is non-concave — so gradient attacks *understate* vulnerability and grid search is the honest empirical baseline.
- **Off-the-shelf diffusion denoisers give strong $\ell_2$ smoothing for free** (Carlini et al., ICLR 2023): 71% certified accuracy at $r=0.5$ on ImageNet, raising the base radius available to semantic certifiers.

## 5. What Is Not Known

- **Theoretically open.** Whether any sound certificate with non-vacuous radius exists for *non-resolvable, non-differentially-resolvable* families (e.g. arbitrary elastic deformation, weather simulation) without assuming a learned surrogate. No impossibility proof either; the question is untouched.
- **Theoretically open.** The correct composition rule. Certifying $\phi_2 \circ \phi_1$ by intersecting individual certificates is sound but exponentially lossy in the number of components; whether a tight joint certificate exists for even two smooth families is unproven.
- **Empirically open.** The certified-vs-attacked gap at ImageNet scale for rotation: nobody has published, on the same model and same 500 images, both the 30.4% certificate and the accuracy under exhaustive grid search over the same $\pm 30°$. The experiment is a day of GPU time.
- **Methodologically blocked.** *Which* $\mathcal{A}$ matters. There is no ground truth mapping from deployment-time distribution shift to a parametric transformation family, so "certified robust to rotation $\pm 30°$" cannot be converted into a statement about field failure rates. This is the blocker that keeps the area academic.
- **Methodologically blocked.** Soundness auditing of interpolation-error constants (see §6).

## 6. Why It Is Hard

The named obstruction is **an unverified constant inside a sound-looking proof**. Every differentially-resolvable certificate has the shape "$R > M_{\mathcal{A}}$, therefore certified", and $M_{\mathcal{A}}$ is a maximum over a continuum of $\alpha$ that is estimated from a finite grid. If the true max exceeds the sampled max on any input, the certificate is *unsound* — it makes a guarantee that is false — and nothing in the reported certified-accuracy number reveals this. The failure is silent: it looks exactly like a valid certificate.

Second obstruction: **the radius budget is arithmetically tiny relative to the perturbation**. A $30°$ rotation moves an ImageNet image by an $\ell_2$ distance on the order of $10$–$50$ in $[0,1]^{150528}$, while the best available smoothing radius is $\approx 1$. Certification is only possible because the grid decomposition reduces the *residual* to sub-radius size — which forces the grid to be dense, and the cost of evaluating $M_{\mathcal{A}}$ to grow accordingly.

## 7. Current Research (as of 2026)

- **Diffusion-denoised smoothing as the base classifier for semantic certifiers** — plugging the Carlini-style denoiser into TSS-style pipelines to spend the larger radius on interpolation error. *(frontier — verify)*
- **Bound-propagation tooling** (`auto_LiRPA` / α,β-CROWN lineage, Zhang, Xu, Hsieh and collaborators) extended to parameterised input transformations, aiming at deterministic certificates beyond CIFAR. *(frontier — verify)*
- **ETH Zurich SRI (Vechev group)** remains the centre of the deterministic/geometric line (DeepG, spatial robustness, smoothing with interpolation bounds); UIUC/CMU (Li, Li) the smoothing line.
- **Certification for generative and VLM pipelines** — semantic invariance of captions/answers rather than labels; largely position papers so far, with no accepted certified metric. *(frontier — verify)*
- **Survey anchor:** Li et al., *SoK: Certified Robustness for Deep Neural Networks*, IEEE S&P 2023.

## 8. Concrete Next Experiment

**Audit the interpolation-error constant.**

- **Scale.** ImageNet, ResNet-50 smoothed classifier, $\sigma = 0.5$, $n = 10^5$ noise samples, 500 test images — i.e. exactly the TSS rotation setting.
- **Procedure.** For each image the certifier certifies under rotation $\pm 30°$, recompute $M_{\mathcal{A}}$ two ways: (a) the paper's sampling grid; (b) a $100\times$ denser grid plus a 1D Lipschitz bound on $\|\partial_\alpha \phi(x,\alpha)\|_2$ obtained by finite differences with a proved remainder. Count images where (b) exceeds the certified radius $R$ while (a) did not.
- **Control arm.** The same 500 images under exhaustive grid attack at $0.1°$ spacing (601 evaluations/image) — the empirical upper bound on achievable robust accuracy.
- **Deciding number.** The **silent-unsoundness rate**: fraction of certified images for which the dense-grid $M_{\mathcal{A}}$ exceeds $R$. If it is $0/500$, sampled $M_{\mathcal{A}}$ is empirically defensible and the field's headline numbers stand. If it exceeds $1\%$, every differentially-resolvable certified-accuracy number in the literature is an upper bound on a quantity that was never certified, and the area needs proved Lipschitz constants before any further benchmark rows.

## 9. Key References

- **[Foundational]** Jeremy Cohen, Elan Rosenfeld, J. Zico Kolter. *Certified Adversarial Robustness via Randomized Smoothing.* ICML, 2019. — arXiv:1902.02918
- **[Foundational]** Mislav Balunović, Maximilian Baader, Gagandeep Singh, Timon Gehr, Martin Vechev. *Certifying Geometric Robustness of Neural Networks.* NeurIPS, 2019.
- **[Foundational]** Gagandeep Singh, Timon Gehr, Markus Püschel, Martin Vechev. *An Abstract Domain for Certifying Neural Networks.* POPL, 2019.
- **[SOTA]** Linyi Li, Maurice Weber, Xiaojun Xu, Luka Rimanic, Bhavya Kailkhura, Tao Xie, Ce Zhang, Bo Li. *TSS: Transformation-Specific Smoothing for Robustness Certification.* ACM CCS, 2021. — arXiv:2002.12398
- **[SOTA]** Marc Fischer, Maximilian Baader, Martin Vechev. *Certified Defense to Image Transformations via Randomized Smoothing.* NeurIPS, 2020.
- **[SOTA]** Zhongkai Hao, Chengyang Ying, Yinpeng Dong, Hang Su, Jian Song, Jun Zhu. *GSmooth: Certified Robustness against Semantic Transformations via Generalized Randomized Smoothing.* ICML, 2022.
- **[SOTA]** Motasem Alfarra, Adel Bibi, Naeemullah Khan, Philip H. S. Torr, Bernard Ghanem. *DeformRS: Certifying Input Deformations with Randomized Smoothing.* AAAI, 2022.
- **[SOTA]** Nicholas Carlini, Florian Tramèr, Krishnamurthy Dvijotham, Leslie Rice, Mingjie Sun, J. Zico Kolter. *(Certified!!) Adversarial Robustness for Free!* ICLR, 2023.
- **[Context]** Logan Engstrom, Brandon Tran, Dimitris Tsipras, Ludwig Schmidt, Aleksander Mądry. *Exploring the Landscape of Spatial Robustness.* ICML, 2019.
- **[Context]** Jeet Mohapatra, Tsui-Wei Weng, Pin-Yu Chen, Sijia Liu, Luca Daniel. *Towards Verifying Robustness of Neural Networks Against A Family of Semantic Perturbations.* CVPR, 2020.
- **[Survey]** Linyi Li, Tao Xie, Bo Li. *SoK: Certified Robustness for Deep Neural Networks.* IEEE Symposium on Security and Privacy, 2023.

## 10. Worked Example

One ImageNet image, $224 \times 224 \times 3$, $d = 150{,}528$, $\sqrt{d} = 388.0$. Certify rotation over $\mathcal{A} = [-30°, 30°]$.

**Step 1 — radius available.** Smoothing with $\sigma = 0.5$; the base classifier's top-class probability lower bound is $\underline{p_A} = 0.99$ ($n = 10^5$, Clopper–Pearson). Then
$$R = 0.5 \cdot \Phi^{-1}(0.99) = 0.5 \times 2.326 = 1.163.$$

**Step 2 — perturbation size.** Rotating a natural image by $30°$ changes a large fraction of pixels substantially; a per-pixel RMS change of $0.15$ (on the $[0,1]$ scale) is typical, giving
$$\|\phi(x,30°) - x\|_2 \approx 0.15 \times 388.0 = 58.2 \gg 1.163.$$
Direct certification is off by a factor of 50. This is why a grid is mandatory.

**Step 3 — grid budget.** The certificate holds only if the *interpolation residual* satisfies $M_{\mathcal{A}} < 1.163$, i.e. per-pixel RMS residual
$$\epsilon_{\text{px}} < \frac{1.163}{388.0} = 0.0030,$$
about $0.76$ of one 8-bit level. Bilinear resampling residual between grid points grows roughly linearly in grid spacing $\Delta\alpha$; hitting $\epsilon_{\text{px}} = 0.003$ on a textured image needs $\Delta\alpha$ on the order of $0.1°$, so $\approx 600$ grid points across $\pm 30°$.

**Step 4 — where the obstruction shows.** The certificate now rests on $\max_{\alpha} \|\text{residual}(\alpha)\|_2 < 1.163$, a max over a continuum, estimated from those $600$ anchors plus some samples between them. The margin is $0.0030$ per pixel. A single high-frequency image region — a picket fence, text, a chain-link mesh — where the true residual peaks between two anchors at $0.005$ per pixel produces $M_{\mathcal{A}} = 1.94 > R$, and the input should not have been certified. Nothing in the output distinguishes that image from a correctly certified one: both return "certified". The reported $30.4\%$ is a count of returned certificates, not of verified guarantees, and §8's audit is the only way to tell the two apart.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*