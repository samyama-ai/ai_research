---
id: 25-speech-and-audio/frechet-audio-distance-backbone-sensitivity
title: "Frechet Audio Distance Sensitivity to Backbone Choice"
topic: 25-speech-and-audio
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Fréchet Audio Distance Sensitivity to Backbone Choice

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/frechet-audio-distance-backbone-sensitivity` · **Status:** empirically-open

## 1. Problem Statement

Fréchet Audio Distance (FAD) scores a generative audio system by embedding reference and generated audio with a frozen pretrained network, fitting a Gaussian to each embedding set, and taking the Fréchet distance between them. The score is reported as if it were a property of the system. It is a property of the pair (system, backbone).

Three variants, of different difficulty:

- **Measurement.** Given a set of systems $\{G_1,\dots,G_m\}$ and a set of backbones $\{f_1,\dots,f_K\}$, how often does the FAD-induced ranking change when $f$ changes, and how much of that change is estimator noise rather than a real disagreement about quality? Nobody has run this at the scale of "all published audio-generation backbones × a listening-test-annotated system set".
- **Method.** Produce a scalar audio-generation metric whose system ranking is stable across reasonable backbone choices *and* correlates with human preference — either by fixing a backbone with an argument for it, by aggregating across backbones, or by replacing the Fréchet form (kernel/MMD estimators, per-item scores).
- **Theory.** Characterise which distributional differences FAD can detect as a function of the embedding map $f$. FAD compares only first two moments in $f$-space; the question is what $f$ must satisfy for $\mathrm{FAD}_f(P,Q)=0 \Rightarrow$ perceptual equivalence, and whether any practical audio encoder satisfies it.

Solved would mean: a documented backbone (or ensemble) plus sample-size protocol such that, on a held-out set of systems with human MOS/preference labels, rank correlation with humans exceeds a stated threshold and the ranking is invariant to backbone swaps within a declared family.

## 2. Formal Setting

Let $P$ be the reference audio distribution and $Q$ the distribution induced by a generator. Let $f:\mathcal{X}\to\mathbb{R}^d$ map an audio clip to an embedding. In practice $f$ is frame-wise: a clip $x$ of length $T$ yields frames $f(x)_1,\dots,f(x)_{T'}$, and implementations differ in whether frames are pooled per clip (mean over $T'$) or pooled across the whole corpus. **This choice alone changes $d$-dimensional second moments and is rarely reported.**

Push-forward moments:
$$\mu_P = \mathbb{E}_{x\sim P}[f(x)], \qquad \Sigma_P = \mathrm{Cov}_{x\sim P}[f(x)],$$
and the population score
$$\mathrm{FAD}_f(P,Q) = \|\mu_P-\mu_Q\|_2^2 + \mathrm{Tr}\!\left(\Sigma_P + \Sigma_Q - 2\left(\Sigma_P\Sigma_Q\right)^{1/2}\right).$$

As measured, with $n$ reference and $m$ generated clips, $\hat\mu$ and $\hat\Sigma$ are the sample mean and sample covariance, and the reported number is $\widehat{\mathrm{FAD}}_f = \mathrm{FAD}_f(\hat P,\hat Q)$ — a plug-in estimator, **not** unbiased.

Ranking stability, the quantity of interest:
$$\tau(f_i,f_j) = \text{Kendall-}\tau\!\left(\big(\widehat{\mathrm{FAD}}_{f_i}(G_k)\big)_{k=1}^m,\ \big(\widehat{\mathrm{FAD}}_{f_j}(G_k)\big)_{k=1}^m\right),$$
and human agreement $\rho(f) = \text{Spearman}\big(\widehat{\mathrm{FAD}}_f(G_k),\ -\mathrm{MOS}(G_k)\big)$.

Assumptions, and their status:

| Assumption | Status |
|---|---|
| $f_\\#P$, $f_\\#Q$ are Gaussian | Violated. Audio embeddings are multimodal by genre/source; deep-feature activations are non-Gaussian. |
| $n,m \gg d$ so $\hat\Sigma$ is well conditioned | Violated for $d=2048$ (PANNs) with $m\approx 10^3$ generated clips: $\mathrm{rank}(\hat\Sigma)\le m-1 < d$. |
| Clips are i.i.d. draws | Violated. Frame pooling induces within-clip correlation; MusicCaps-style sets have artist/scene duplication. |
| $f$ preserves perceptually relevant distinctions | Unverified; this is the open problem. |
| Scores are comparable across $f$ | False. FAD is not scale-free; only within-$f$ rankings are meaningful. |

## 3. State of the Art

**Established.**
- FAD was introduced with a VGGish backbone ($d=128$, PCA-whitened, 0.96 s frames) and validated against distortions and a small listening test (Kilgour et al., Interspeech 2019). The original validation is against *artificial degradations*, not against generative systems.
- Gui, Gamper, Braun, Emmanouilidou (ICASSP 2024) showed that swapping VGGish for CLAP/EnCodec-family embeddings changes both the absolute score and the correlation with human judgement on music generation, and introduced two protocol fixes: extrapolation to infinite sample size ($\mathrm{FAD}_\infty$) and per-song FAD. Released as `fadtk`.
- Tailleur, Lee, Lagrange, Choi, Heller, Imoto, Okamoto (EUSIPCO 2024) ran the same question on environmental audio and found the correlation between FAD and human perception is embedding-dependent, with no single embedding best across all their conditions.
- In vision, the analogous defect is proven: FID is sensitive to the classifier's class structure (Kynkäänniemi et al., ICLR 2023) and the plug-in estimator is biased in sample size (Chong & Forsyth, CVPR 2020).

**Claimed but unablated.**
- "CLAP-music is the right FAD backbone for music" — reported on a small number of systems and one annotated set; not reproduced independently across text-to-music, text-to-audio and speech in one study.
- Kernel replacements (MMD-based audio distance, 2025 work following CMMD in vision) claim lower sample-size bias and better human agreement; these exist as benchmark tables on a handful of systems, not as ablations isolating estimator form from backbone.

**Benchmark-number-only.** Nearly all published FAD values in text-to-music/text-to-audio papers (AudioLDM, MusicGen, Stable Audio and successors) are single-backbone, single-sample-size numbers with no confidence interval. They cannot be compared across papers.

## 4. What Is Known

- **Backbone geometry differs by an order of magnitude in dimension.** VGGish $d=128$; OpenL3 $d=512$ or $6144$ depending on configuration; PANNs CNN14 $d=2048$; LAION-CLAP $d=512$; EnCodec latents $d=128$ per frame. FAD's covariance term has $O(d^2)$ free parameters, so estimator variance is not comparable across these.
- **Sample-size dependence is large.** Gui et al. (ICASSP 2024) show FAD decreasing monotonically with the number of evaluated clips, motivating $\mathrm{FAD}_\infty$ by linear extrapolation in $1/n$. Measured at the scale of $10^2$–$10^3$ music clips.
- **Ranking disagreement is real, not hypothetical.** Both Gui et al. (music, MusicCaps-scale) and Tailleur et al. (environmental audio) report that the best-correlating embedding depends on the audio domain and the perceptual attribute being judged.
- **VGGish is a weak backbone.** It is trained for coarse AudioSet tagging at 16 kHz mono; it discards content above 8 kHz entirely, so it is blind by construction to bandwidth and high-frequency artefacts — a common failure mode of neural vocoders and codecs.
- **The vision analogue transfers.** FID's plug-in bias scales with $d$ and $1/n$ (Chong & Forsyth, CVPR 2020); the same algebra applies verbatim to FAD.

## 5. What Is Not Known

- **Empirically open (main gap).** No study has computed $\tau(f_i,f_j)$ over $K\ge 8$ backbones × $m\ge 20$ generative systems × 3 domains (music, general audio, speech) with human labels on the same systems and with bootstrap confidence intervals. The experiment is runnable today on a few GPU-days. It has not been run.
- **Empirically open.** Whether an ensemble (rank-average across backbones) beats the best single backbone on held-out systems, or merely averages away the one backbone that was sensitive to the failure mode present.
- **Methodologically blocked.** "Correlates with human perception" is underspecified: overall quality, text adherence, and artefact audibility give different orderings, and MOS scales are not comparable across listening tests. Without a fixed human protocol, $\rho(f)$ is not a well-defined target.
- **Theoretically open.** No characterisation of the class of distribution pairs $(P,Q)$ that any given audio encoder $f$ makes indistinguishable in first two moments. FAD is a pseudo-metric: $\mathrm{FAD}_f(P,Q)=0$ whenever $f_\\#P$ and $f_\\#Q$ share moments, and the size of that equivalence class is unknown for every deployed $f$.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus absent ground truth**, in that order.

1. *Confounding.* Changing the backbone changes $d$, the frame rate, the pooling, the training corpus and the estimator's conditioning simultaneously. A ranking flip between VGGish ($d=128$) and PANNs ($d=2048$) at $n=1000$ could be a genuine perceptual disagreement or purely the $O(d^2/n)$ bias term. No published comparison separates them.
2. *Absent ground truth.* There is no reference ordering of generative audio systems. Human listening tests are the substitute, but they are expensive ($\sim$\$1–3k per system set), noisy, and attribute-dependent; a metric can only be validated against a target that is itself contested.
3. *Non-identifiability.* Even a perfect experiment identifies which $f$ correlates best on *the systems tested*. Because generators overfit to whatever metric is reported, a backbone selected this way has no guarantee on the next generation of systems — the selection is on a moving distribution.

## 7. Current Research (as of 2026)

- **Protocol fixes over the Fréchet form.** $\mathrm{FAD}_\infty$, per-song FAD, and the `fadtk` toolkit (Microsoft Research + academic collaborators) are the practical baseline for backbone comparison.
- **Kernel/MMD replacements.** Following CMMD in vision (Jayasumana et al., CVPR 2024), audio work in 2025 proposes kernel audio distances that are unbiased and drop the Gaussian assumption. *(frontier — verify)* Adoption in text-to-audio papers is still partial.
- **Better backbones as backbones.** Self-supervised audio and music encoders (MERT, BEATs, audio-MAE variants, CLAP successors) are being tested as FAD feature extractors rather than as classifiers. *(frontier — verify)*
- **Environmental / non-music domains.** Groups at Nantes, Sony CSL and NII continue the embedding-dependence line from the EUSIPCO 2024 result.
- **Reference-free and instruction-following metrics.** Audio-LLM judges and CLAP-score variants sidestep the reference-set problem, and inherit a different one: judge bias. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Is FAD's system ranking backbone-invariant, once estimator bias is removed?

**Scale.** $m=24$ generative systems (8 text-to-music, 8 text-to-audio, 8 neural-vocoder/codec configurations, the last giving cheap controlled quality gradations). $K=8$ backbones: VGGish-128, PANNs-CNN14-2048, OpenL3-512, LAION-CLAP-music-512, LAION-CLAP-general-512, MS-CLAP-1024, EnCodec-128, MERT-1024. $n=m=5{,}000$ clips of 10 s per system per reference set — enough that $n>2d$ for every backbone. Human labels: one MOS-style listening test per domain, 20 listeners × 24 systems × 15 clips, with per-system 95 % CIs. Cost estimate: $\sim$4 GPU-days of embedding extraction plus $\sim$\$6k of listening tests.

**Control arms.** (a) *Bias control*: report both $\widehat{\mathrm{FAD}}$ at $n=5000$ and $\mathrm{FAD}_\infty$ by $1/n$ extrapolation over $n\in\{500,1000,2000,5000\}$; (b) *estimator control*: the same 8 backbones scored with an unbiased kernel/MMD estimator, isolating "Gaussian form" from "backbone"; (c) *null control*: split the reference corpus in half and score one half against the other — the resulting non-zero FAD is the floor for that $(f,n)$.

**Deciding number.** The mean pairwise Kendall $\tau$ across the $\binom{8}{2}=28$ backbone pairs on the 24-system ranking, with bootstrap CI over clips.

- $\bar\tau \ge 0.9$: backbone choice is a reporting nuisance, not a validity threat; standardise on the best-correlating one and move on.
- $\bar\tau \le 0.6$: FAD as reported in the literature does not identify a system ordering, and single-backbone FAD numbers in papers are not comparable — the field needs the ensemble or the kernel form.

Secondary number: $\rho(f)$ per backbone against MOS, to say whether any single $f$ clears $\rho \ge 0.8$ in all three domains.

## 9. Key References

- **[Foundational]** K. Kilgour, M. Zuluaga, D. Roblek, M. Sharifi. *Fréchet Audio Distance: A Reference-Free Metric for Evaluating Music Enhancement Algorithms.* Interspeech, 2019.
- **[Foundational]** S. Hershey et al. *CNN Architectures for Large-Scale Audio Classification.* ICASSP, 2017. — VGGish.
- **[SOTA]** A. Gui, H. Gamper, S. Braun, D. Emmanouilidou. *Adapting Frechet Audio Distance for Generative Music Evaluation.* ICASSP, 2024. — $\mathrm{FAD}_\infty$, per-song FAD, `fadtk`.
- **[SOTA]** M. Tailleur, J. Lee, M. Lagrange, K. Choi, L. M. Heller, K. Imoto, Y. Okamoto. *Correlation of Fréchet Audio Distance With Human Perception of Environmental Audio Is Embedding Dependent.* EUSIPCO, 2024.
- **[Related]** M. F. Chong, D. Forsyth. *Effectively Unbiased FID and Inception Score and Where to Find Them.* CVPR, 2020.
- **[Related]** T. Kynkäänniemi, T. Karras, M. Aittala, T. Aila, J. Lehtinen. *The Role of ImageNet Classes in Fréchet Inception Distance.* ICLR, 2023.
- **[Related]** S. Jayasumana, S. Ramalingam, A. Veit, D. Glasner, A. Chakrabarti, S. Kumar. *Rethinking FID: Towards a Better Evaluation Metric for Image Generation.* CVPR, 2024.
- **[Related]** M. Bińkowski, D. J. Sutherland, M. Arbel, A. Gretton. *Demystifying MMD GANs.* ICLR, 2018.
- **[Backbones]** Q. Kong et al. *PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition.* IEEE/ACM TASLP, 2020. · Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, S. Dubnov. *Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation.* ICASSP, 2023. · A. Défossez, J. Copet, G. Synnaeve, Y. Adi. *High Fidelity Neural Audio Compression.* TMLR, 2023.

## 10. Worked Example

Take the *null control* arm and make the obstruction arithmetic.

Let $P=Q$ exactly — score a reference corpus against an independent draw from the same corpus. Assume $f_\\#P = \mathcal{N}(0, I_d)$ after whitening. The true FAD is $0$. The mean term of the plug-in estimator is not:
$$\mathbb{E}\left[\|\hat\mu_P - \hat\mu_Q\|_2^2\right] = \sum_{j=1}^{d}\mathrm{Var}\big(\hat\mu_{P,j}-\hat\mu_{Q,j}\big) = d\left(\tfrac{1}{n}+\tfrac{1}{m}\right) = \frac{2d}{n}\ \ (n=m).$$

At the sample size most papers use, $n=m=1000$:

| Backbone | $d$ | Mean-term floor $2d/n$ | $\mathrm{rank}(\hat\Sigma)$ vs $d$ |
|---|---|---|---|
| VGGish | 128 | 0.256 | 999 ≥ 128 — full rank |
| LAION-CLAP | 512 | 1.024 | 999 ≥ 512 — marginal |
| MERT | 1024 | 2.048 | 999 < 1024 — **singular** |
| PANNs CNN14 | 2048 | 4.096 | 999 < 2048 — **singular** |

Published VGGish FAD values for text-to-music systems sit roughly in the $1$–$10$ range, and the gaps between competing systems are often $<1$. So:

- Under VGGish, the identical-distribution floor is $0.256$ — small relative to a typical between-system gap.
- Under PANNs, the floor from the mean term alone is $4.096$, plus a covariance term computed from a rank-deficient $\hat\Sigma$ whose matrix square root $(\hat\Sigma_P\hat\Sigma_Q)^{1/2}$ is evaluated on a singular product and systematically under-estimates $\mathrm{Tr}$, inflating the score further.

Now suppose systems $A$ and $B$ score $\mathrm{FAD}_{\text{VGGish}} = 3.1$ vs $3.4$ (A wins) and $\mathrm{FAD}_{\text{PANNs}} = 9.8$ vs $9.1$ (B wins). This is exactly the reported pattern that motivates "backbone sensitivity". But the PANNs gap of $0.7$ sits inside a bias floor of $\ge 4.1$ that depends on the per-system clip count — and generated sets are frequently smaller than reference sets, so the two arms do not even share $n$. **The flip is not evidence that PANNs hears something VGGish misses. It is not evidence of anything until the null-control floor and $\mathrm{FAD}_\infty$ extrapolation are reported alongside it.** That is why the problem is empirically open rather than settled: the experiment separating the two explanations is cheap, and has not been run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*