---
id: 34-diffusion-generative/fid-feature-extractor-dependence
title: "FID Sensitivity to Feature Extractor Choice"
topic: 34-diffusion-generative
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# FID Sensitivity to Feature Extractor Choice

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/fid-feature-extractor-dependence` · **Status:** partially-solved

## 1. Problem Statement

Fréchet Inception Distance (FID) is not a distance between image distributions. It is a distance between Gaussian fits to the pushforward of those distributions under one fixed encoder — Inception-V3 `pool3`, trained on ImageNet-1k classification in 2015. The problem: **how much of a reported FID number, and of the model ranking it induces, is a property of the generative models rather than of that encoder choice?**

Three variants, with different difficulty:

- **Measurement.** Given a set of generative models $\mathcal{M}$ and a family of encoders $\Phi$, quantify the rank instability of FID across $\Phi$, and decide which encoder's ranking best predicts human preference. Runnable today; partly done.
- **Method.** Construct a generative-model evaluation metric whose ranking is provably stable under a stated class of encoder perturbations, or that avoids a learned encoder entirely. Open.
- **Theory.** Characterize which pairs $(\phi, \phi')$ of encoders induce the same ordering on a distribution class, i.e. give conditions under which $\mathrm{FID}_\phi(P,Q) < \mathrm{FID}_\phi(P,R) \Rightarrow \mathrm{FID}_{\phi'}(P,Q) < \mathrm{FID}_{\phi'}(P,R)$. Essentially untouched.

Solving it means: a reported FID comes with a stated encoder, a stated sample size, a stated resize path, and a known bound on how much the ranking would move under a different reasonable encoder.

## 2. Formal Setting

Let $P$ be the reference image distribution (a finite dataset $\mathcal{D}_r$ of $n$ images in practice) and $Q_\theta$ the model distribution, sampled by $m$ generated images $\mathcal{D}_g$. Let $\phi: \mathcal{X} \to \mathbb{R}^d$ be an encoder.

**Measured quantities.** For a set $S$, the empirical moments are
$$\hat{\mu}_S = \frac{1}{|S|}\sum_{x \in S} \phi(x), \qquad \hat{\Sigma}_S = \frac{1}{|S|-1}\sum_{x\in S}(\phi(x)-\hat\mu_S)(\phi(x)-\hat\mu_S)^\top,$$
and
$$\widehat{\mathrm{FD}}_\phi = \lVert \hat\mu_r - \hat\mu_g \rVert_2^2 + \operatorname{tr}\!\left(\hat\Sigma_r + \hat\Sigma_g - 2\big(\hat\Sigma_r \hat\Sigma_g\big)^{1/2}\right).$$
FID is $\widehat{\mathrm{FD}}_\phi$ with $\phi = $ Inception-V3 pool3 ($d = 2048$), $m = 50{,}000$, images bilinearly resized to $299\times299$.

$\phi$ is not the mathematical function alone: it is the composite $\phi = f \circ \rho \circ \kappa$, where $\kappa$ is the JPEG/PNG decode, $\rho$ the resize kernel and library, and $f$ the network weights. All three are measured, and all three are usually unreported.

**Objective.** For a model set $\mathcal{M} = \{\theta_1,\dots,\theta_K\}$ let $r_\phi \in \mathfrak{S}_K$ be the ranking induced by $\widehat{\mathrm{FD}}_\phi$ and $r_H$ the ranking from human 2AFC preference. The two numbers of interest are the cross-encoder agreement $\tau(r_\phi, r_{\phi'})$ and the human agreement $\tau(r_\phi, r_H)$, both Kendall's $\tau$.

**Assumptions, and their status.**

| Assumption | Status |
|---|---|
| $\phi_\\# P$, $\phi_\\# Q$ are Gaussian | Violated. Inception features are non-negative post-ReLU and heavy-tailed; the Fréchet distance ignores all moments beyond the second (Jayasumana et al., CVPR 2024). |
| $\widehat{\mathrm{FD}}$ is an unbiased estimator of $\mathrm{FD}$ | Violated. Bias is $O(1/m)$ and negative in ranking-relevant ways (Chong & Forsyth, CVPR 2020). $\hat\Sigma$ with $d=2048$ needs $m \gg d$; at $m=50$k the covariance estimate is still noisy. |
| $\phi$ preserves the perceptual geometry that matters | Assumed, not established. Inception's features are optimized to discard within-class variation on 1000 ImageNet classes. |
| The resize path is a detail | Violated. Aliased resizing shifts FID by amounts comparable to the gap between competing published methods (Parmar et al., CVPR 2022). |

## 3. State of the Art

**Established (independently reproduced or directly ablated):**

- *clean-FID* (Parmar, Zhang, Zhu, CVPR 2022) isolated resize and compression as a confound and gave a canonical pipeline. Established: the same model, same weights, same samples yields materially different FID under PIL vs. OpenCV vs. TensorFlow resizing.
- *$\mathrm{FID}_\infty$* (Chong & Forsyth, CVPR 2020) established the sample-size bias and gave an extrapolation estimator: fit $\widehat{\mathrm{FD}}(m)$ over several $m$ and extrapolate to $1/m \to 0$.
- *ImageNet-class dependence* (Kynkäänniemi, Karras, Aittala, Aila, Lehtinen, ICLR 2023). Established by construction: FID can be reduced by editing the generated set so its Inception top-class histogram matches the reference, with no perceptual improvement. FID is partly a class-histogram-matching score.
- *Encoder swap changes rankings* (Stein et al., NeurIPS 2023). Their benchmark over diffusion, GAN and autoregressive models on multiple datasets showed FD computed on DINOv2-ViT-L/14 features tracks human ratings better than Inception FID, and that Inception FID systematically ranks diffusion models worse than human raters do.

**Claimed but unablated:**

- That DINOv2 is *the* right encoder. It is the best of the encoders tried in a handful of studies; nobody has ablated DINOv2's own dataset (LVD-142M) biases the way Kynkäänniemi et al. ablated Inception's.
- That CMMD (Jayasumana et al., CVPR 2024 — CLIP features plus an unbiased MMD with a Gaussian RBF kernel) fixes the problem. It removes the Gaussian assumption and the small-sample bias, which is established; that it removes *encoder* dependence is not — it substitutes CLIP for Inception.

**Benchmark-number-only results:** most headline FIDs (ImageNet-256 FID $\approx 1.3$–$2.0$ for recent latent diffusion and masked-transformer models) exist only as single numbers with no seed variance, no encoder ablation, and often no stated resize path.

## 4. What Is Known

- **Rank flips are real, not marginal.** Across the model zoo in Stein et al. (NeurIPS 2023) — order 10 models per dataset, human study with tens of thousands of pairwise responses — the Inception-FD and DINOv2-FD orderings disagree on a substantial minority of pairs, and the disagreements concentrate on GAN-vs-diffusion comparisons.
- **Sample-size bias.** At $m = 50{,}000$ vs. $m = 10{,}000$, FID differs by more than the gap between many published state-of-the-art pairs; FID at small $m$ is biased upward and the bias is model-dependent (Chong & Forsyth, CVPR 2020, on CIFAR-10 and CelebA at $m \in [2\text{k}, 50\text{k}]$).
- **Seed variance is small relative to encoder variance.** Re-drawing 50k samples changes FID by a fraction of a point at ImageNet-256 scale; changing the encoder changes the *ranking*.
- **Self-supervised features are competitive or better.** Morozov, Voynov, Babenko (ICLR 2021) showed SwAV-based FD tracks human judgment at least as well as Inception FD on GAN comparisons — the first clear evidence, pre-dating DINOv2, that ImageNet supervision is not required.
- **Precision/recall decomposition inherits the same dependence.** Improved precision/recall (Kynkäänniemi et al., NeurIPS 2019) and density/coverage (Naeem et al., ICML 2020) are computed in the same feature space and move with it.

## 5. What Is Not Known

- **Theoretically open.** No characterization of when two encoders induce the same ordering. There is no theorem of the form "if $\phi'$ is bi-Lipschitz-equivalent to $\phi$ on the support of $P \cup Q$, then rankings agree", and no lower bound on how badly rankings can disagree for encoders both achieving a given accuracy on some proxy task.
- **Empirically open.** No study has jointly varied encoder ($\geq 6$), sample size, resize path, and model family on a single fixed model set with a preregistered human protocol at ImageNet-256 scale. Each existing study varies one or two axes. The experiment is runnable on 8 GPUs in under a week; nobody has run the full cross.
- **Methodologically blocked.** "Which encoder is correct" presupposes a ground-truth ordering. Human 2AFC preference is the usual stand-in, but it measures per-image appeal and only weakly measures *distribution match* — diversity collapse is nearly invisible in pairwise image comparisons. There is no accepted operationalization of "$Q$ is closer to $P$ as a distribution" that is independent of a learned encoder.

## 6. Why It Is Hard

**Absent ground truth compounded by an evaluation that does not measure what it names.** FID names a distributional distance; what it computes is a second-moment gap in a 2048-dimensional space engineered to be invariant to exactly the within-class variation that generative diversity consists of. To adjudicate between encoders you need a target the metrics are approximating — and the only available target, human preference, is itself blind to the diversity half of the objective. So the two candidate arbiters (encoder-based FD, human raters) each fail on a different term of the quantity in dispute, and there is no third.

Secondary obstruction: **non-identifiability**. Any $\phi$ composed with an invertible linear map leaves FD unchanged, but composed with a *rank-reducing* map — which is what every real encoder is — changes it. The space of "reasonable" encoders is not parameterized, so "the FID is stable under encoder choice" has no well-posed quantifier.

## 7. Current Research (as of 2026)

- **DINOv2-FD as de facto second metric.** Papers increasingly report FD-DINOv2 alongside FID; the NVIDIA and Google DeepMind generative groups both do so. Not yet a replacement.
- **Kernel and distribution-free alternatives.** CMMD (Google Research) and MMD variants; work on unbiased estimators avoiding the Gaussian fit.
- **Encoder-ensemble metrics** — averaging or rank-aggregating FD over several encoders to bound the arbitrariness *(frontier — verify)*.
- **Text-conditional evaluation** displacing FID entirely for text-to-image, with human preference models (PickScore, HPSv2, ImageReward) and VQA-based faithfulness scores. This changes the problem rather than solving it: the preference model is now the encoder, with its own training-set bias.
- **Aalto/NVIDIA line** (Kynkäänniemi, Aila and colleagues) continues on what FID's feature space actually encodes.

## 8. Concrete Next Experiment

**Question:** does the encoder change the *ranking that would be published*, and which encoder's ranking best predicts human preference under a protocol that includes diversity?

- **Scale.** $K = 20$ checkpoints on ImageNet-256: 6 diffusion (varying guidance scale $w \in \{1.0, 1.5, 3.0\}$ — guidance trades diversity for fidelity and is the sharpest available diversity lever), 6 GAN, 4 masked/autoregressive, 4 deliberately diversity-truncated variants. $m = 50{,}000$ samples per checkpoint, 3 seeds.
- **Encoder arm.** $|\Phi| = 6$: Inception-V3 pool3, DINOv2 ViT-L/14, CLIP ViT-L/14, MAE ViT-L, SwAV ResNet-50, and a randomly initialized Inception-V3 (the null encoder). Fixed clean-FID resize path for all.
- **Control arm.** The random-weight Inception. If its ranking has $\tau(r_{\phi_\text{rand}}, r_H)$ close to that of trained encoders, feature *learning* is not what FID is buying, and the whole comparison is measuring image statistics, not perception.
- **Human arm.** Preregistered, two tasks: (a) standard 2AFC image quality, (b) a *set*-level task — given a 3×3 grid from model A and one from model B, which grid better matches a reference grid in variety and content mix. $\geq 30$ raters, $\geq 20{,}000$ comparisons, Bradley–Terry aggregation to $r_H$.
- **The deciding number.** $\Delta\tau = \tau(r_{\phi_\text{DINOv2}}, r_H^{(a+b)}) - \tau(r_{\phi_\text{Inception}}, r_H^{(a+b)})$, with a bootstrap CI over raters and seeds. If $\Delta\tau > 0.15$ with the CI excluding 0, Inception FID should be retired as a primary metric. If the CI contains 0 while $\tau(r_{\phi_\text{DINOv2}}, r_{\phi_\text{Inception}}) < 0.8$, the two metrics disagree with each other but neither tracks humans better — which reclassifies the problem from empirically open to methodologically blocked.

Cost estimate: sampling dominates at roughly 1M images; about 4–6 GPU-days on 8×H100 for the diffusion arm, plus about $6k for the human study.

## 9. Key References

- **[Foundational]** Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter. *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* NeurIPS, 2017. — arXiv:1706.08500
- **[Foundational]** Salimans, Goodfellow, Zaremba, Cheung, Radford, Chen. *Improved Techniques for Training GANs.* NeurIPS, 2016. — arXiv:1606.03498
- **[SOTA]** Stein, Cresswell, Hosseinzadeh, Sui, Ross, Villecroze, Liu, Caterini, Taylor, Loaiza-Ganem. *Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models.* NeurIPS, 2023. — arXiv:2306.04675
- **[SOTA]** Jayasumana, Ramalingam, Veit, Glasner, Chakrabarti, Kumar. *Rethinking FID: Towards a Better Evaluation Metric for Image Generation.* CVPR, 2024. — arXiv:2401.09603
- **[SOTA]** Kynkäänniemi, Karras, Aittala, Aila, Lehtinen. *The Role of ImageNet Classes in Fréchet Inception Distance.* ICLR, 2023. — arXiv:2203.06026
- **[SOTA]** Parmar, Zhang, Zhu. *On Aliased Resizing and Surprising Subtleties in GAN Evaluation.* CVPR, 2022. — arXiv:2104.11222
- **[SOTA]** Chong, Forsyth. *Effectively Unbiased FID and Inception Score and Where to Find Them.* CVPR, 2020. — arXiv:1911.07023
- **[SOTA]** Morozov, Voynov, Babenko. *On Self-Supervised Image Representations for GAN Evaluation.* ICLR, 2021.
- **[Related]** Kynkäänniemi, Karras, Laine, Lehtinen, Aila. *Improved Precision and Recall Metric for Assessing Generative Models.* NeurIPS, 2019. — arXiv:1904.06991
- **[Related]** Naeem, Oh, Uh, Choi, Yoo. *Reliable Fidelity and Diversity Metrics for Generative Models.* ICML, 2020. — arXiv:2002.09797
- **[Survey]** Betzalel, Penso, Navon, Fetaya. *A Study on the Evaluation of Generative Models.* 2022. — arXiv:2206.10935
- **[Related]** Barratt, Sharma. *A Note on the Inception Score.* ICML Workshop, 2018. — arXiv:1801.01973

## 10. Worked Example

Take one classifier-free-guidance sweep on a single ImageNet-256 diffusion checkpoint — the cleanest instance, because only one scalar $w$ changes and its perceptual effect is known.

Sample $m = 50{,}000$ at $w = 1.0$ (no guidance), $w = 1.5$, $w = 3.0$. The reproducible pattern across the ADM/DiT/latent-diffusion literature:

| $w$ | FID (Inception) | Per-image human preference | Diversity |
|---|---|---|---|
| 1.0 | ~10 | lowest | highest |
| 1.5 | ~2 | high | reduced |
| 3.0 | ~5–8 | **highest** | visibly collapsed |

FID is U-shaped in $w$ with a minimum near $1.5$; human per-image preference is monotone increasing in $w$ over this range. So at the two endpoints of the comparison $w=1.5$ vs. $w=3.0$, FID and human raters give **opposite orderings** — and this is with the encoder held fixed.

Now the obstruction. A researcher observing this disagreement has three candidate diagnoses:

1. FID is right and humans are ignoring the diversity loss at $w = 3.0$;
2. Inception's feature space over-weights class-histogram match, so guidance's sharpening of class evidence is being scored as distribution match rather than as mode drop (the Kynkäänniemi et al. mechanism);
3. the Gaussian fit is discarding the higher moments where the collapse actually shows.

Swapping in DINOv2 does not adjudicate. Suppose FD-DINOv2 also bottoms out near $w = 1.5$ — consistent with (1) and (3), and with (2) only if DINOv2 shares Inception's bias, which is unmeasured. Suppose instead it bottoms out at $w = 2.2$ — you have learned that the metrics disagree, and nothing about which is right, because the human data you would use to break the tie is exactly the signal that cannot see diversity.

That is the problem in one figure: three numbers ($\mathrm{FID}$, $\mathrm{FD}_{\text{DINOv2}}$, human preference), a disagreement among them, and no fourth measurement that would tell you which one is wrong.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*