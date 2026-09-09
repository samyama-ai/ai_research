---
id: 25-speech-and-audio/one-step-vocoder-quality-ceiling
title: "One-Step Vocoder Quality Ceiling"
topic: 25-speech-and-audio
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# One-Step Vocoder Quality Ceiling

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/one-step-vocoder-quality-ceiling` · **Status:** empirically-open

## 1. Problem Statement

A vocoder maps an acoustic feature sequence (mel-spectrogram, or a neural codec's latent) to a waveform. **One-step** means the decoder runs a single forward pass — one network function evaluation (NFE $=1$) — as in HiFi-GAN, BigVGAN, Vocos, or a distilled diffusion model. **Multi-step** means an iterative sampler with NFE $\gg 1$ (WaveGrad, DiffWave, WaveFit).

The question: **is there a quality ceiling intrinsic to NFE $=1$, or is the observed gap an artifact of training objective and budget?**

Three variants, routinely conflated:

- **Theory.** Does a one-step pushforward map have a nonzero lower bound on divergence from the true conditional waveform distribution that a multi-step sampler of the same architecture avoids?
- **Method.** At matched parameters and matched training compute, does any one-step training recipe (GAN, distillation, consistency) reach the multi-step teacher's quality on hard content — full-band 44.1 kHz singing, unseen speakers, non-speech audio?
- **Measurement.** Do current metrics resolve the residual gap at all? Above roughly MOS 4.3 on clean 24 kHz speech, MOS confidence intervals overlap ground truth, and the gap only appears in specific artifact classes.

Solving it means: exhibit a one-step vocoder that is statistically indistinguishable from a strong multi-step vocoder in a MUSHRA test with hidden reference and anchor at matched compute — *or* prove a lower bound showing it cannot be.

## 2. Formal Setting

Let $x \in \mathbb{R}^{L}$ be a waveform at sample rate $f_s$, and $c = \mathcal{M}(x) \in \mathbb{R}^{F \times T}$ its conditioning feature — a mel filterbank of $F$ bands over $T$ frames, computed with hop $H$, so $T = L/H$. The compression ratio is $r = H/F$ (for BigVGAN-style 24 kHz: $H=256$, $F=100$, $r=2.56$).

The target is the conditional law $p(x \mid c)$. Because $\mathcal{M}$ discards phase and band-internal structure, $p(\cdot \mid c)$ is **not** a point mass: its support is the fiber $\mathcal{F}_c = \{x : \mathcal{M}(x) = c\}$, a set of dimension roughly $L(1 - 1/r)$.

A one-step generator is a pushforward
$$q_\theta(\cdot \mid c) = (G_\theta(\cdot, c))_{\\#} \mathcal{N}(0, I_d),$$
with $G_\theta$ a single deterministic pass. Deterministic vocoders are the case $d = 0$: $q_\theta$ is a Dirac at $G_\theta(c)$.

**Quantities as measured.**

- *Perceptual quality* $Q$: MUSHRA score (ITU-R BS.1534-3), $\ge 15$ trained listeners, hidden reference, 3.5 kHz low-pass anchor, per-system mean with bootstrap 95% CI. MOS via ITU-T P.808 crowdsourcing is the cheap substitute and is roughly $3\times$ noisier per listener.
- *Gap*: $\Delta_Q = Q(\text{ref}) - Q(\text{system})$, in MUSHRA points (0–100).
- *Compute*: training FLOPs $C$; inference cost NFE $\times$ params, reported as real-time factor (RTF) on a named device.
- *Distributional distance*: Fréchet Audio Distance $\mathrm{FAD} = \lVert \mu_r - \mu_g \rVert^2 + \mathrm{tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$ on a stated embedding (VGGish, or CLAP — the two disagree in ranking).
- *Ceiling*: the object of interest is
$$\Delta_Q^{\star}(\text{NFE}=1) = \lim_{C \to \infty} \ \inf_{\theta,\ \text{recipe}} \ \Delta_Q(G_\theta),$$
at fixed parameter count. The claim "there is a ceiling" is $\Delta_Q^{\star}(1) > \Delta_Q^{\star}(\infty)$.

**Assumptions, and which are violated.**

1. *$p(\cdot\mid c)$ has a density on a connected manifold.* Violated: the fiber $\mathcal{F}_c$ is a union of components separated by phase-wrap discontinuities; the target is effectively multimodal in phase.
2. *Perceptual equivalence classes are large, so any fiber point suffices.* Violated for transients and for phase coherence across harmonics — listeners hear inter-harmonic phase incoherence as "buzz" or "metallic" texture even when the magnitude spectrum matches.
3. *MUSHRA is monotone in the quantity of interest.* Violated near ceiling: at $\Delta_Q < 5$ points, listener-specific artifact sensitivity dominates and system ranking is unstable across panels.
4. *Training compute is the binding constraint.* Unverified — nobody has run the compute-scaling sweep that would test it.

## 3. State of the Art

**Established (with ablations, reproduced).**

- **HiFi-GAN** (Kong, Kim, Bae, NeurIPS 2020) — one-step GAN, MOS $4.36 \pm 0.05$ vs ground-truth $4.45 \pm 0.04$ on LJSpeech, single speaker, 22.05 kHz. The multi-receptive-field fusion and multi-period discriminator ablations are in the paper and have been reproduced widely.
- **BigVGAN** (Lee, Ping, Ginsburg, Catanzaro, Yoon, ICLR 2023) — 112M params, LibriTTS, 24 kHz. The decisive ablation is the anti-aliased periodic (snake) activation: removing it degrades out-of-distribution and non-speech audio measurably. This is the strongest evidence that the historical one-step gap was an **inductive-bias** defect, not an NFE defect.
- **Vocos** (Siuzdak, ICLR 2024) — predicts STFT magnitude and phase, then one iSTFT. Matches BigVGAN-class quality with roughly an order-of-magnitude lower RTF. Confirms that the decoder need not model raw samples.
- **Pushforward lower bound** (Salmona, de Bortoli, Delon, Desolneux, NeurIPS 2022) — for a target that is a mixture of $K$ well-separated components, any pushforward of a unimodal source with Lipschitz constant $\le \ell$ incurs Wasserstein error bounded below by a quantity growing with component separation and shrinking in $\ell$. This is the only real theoretical handle on the ceiling question.

**Claimed but unablated.**

- That distilled one-step diffusion vocoders (consistency-model style, e.g. CoMoSpeech, ACM MM 2023) match their multi-step teachers. The comparisons are usually against a teacher at reduced NFE, not against the teacher's own asymptotic quality, and the parameter/compute controls are absent.
- That BigVGAN-v2 (2024 release, model card and code, no peer-reviewed ablation paper) closes the 44.1 kHz music gap. This exists as checkpoints and demo pages, not as a controlled study.

**Benchmark-number-only.** Most reported FAD and UTMOS deltas between one-step and multi-step vocoders. UTMOS (Saeki et al., Interspeech 2022) was fit on VoiceMOS-Challenge TTS systems and saturates on vocoded copy-synthesis; a 0.05 UTMOS difference between two good vocoders carries no established perceptual meaning.

## 4. What Is Known

- One-step GAN vocoders are not obviously worse than iterative ones on clean, in-domain 22–24 kHz speech. HiFi-GAN V1 at 4.36 MOS (LJSpeech, ~24 h single speaker) and WaveGrad at 1000 steps sit inside each other's CIs.
- Iterative samplers were, in 2021–2022, better on *out-of-domain* material. WaveFit (Koizumi et al., SLT 2022) reaches WaveRNN-class quality in 5 iterations; SpecGrad (Interspeech 2022) improved on WaveGrad by shaping the diffusion noise with the spectral envelope. Both were motivated by GAN failures on unseen speakers.
- BigVGAN's 2023 result removed most of that generalisation gap at 112M params on ~600 h (LibriTTS), by changing the activation and discriminator, not the step count.
- Codec decoders — DAC (Kumar et al., NeurIPS 2023), 44.1 kHz, ~76M params — are one-step and beat EnCodec at $3\times$ the bitrate on objective metrics, and the multi-band diffusion decoder (Roman et al., NeurIPS 2023) was preferred over EnCodec's one-step decoder at low bitrate. Direction of the gap depends on bitrate, not just NFE.
- No published compute-matched scaling curve exists for one-step vs multi-step vocoders. The largest open vocoder is order $10^8$ params; text and image generators demonstrated NFE-independence only at $10^9$–$10^{10}$.

## 5. What Is Not Known

- **Theoretically open.** Whether $\Delta_Q^\star(1) > 0$. The Salmona et al. bound applies to Lipschitz-constrained pushforwards of a unimodal source; real vocoders are conditioned on $c$, which slices the target into a near-unimodal conditional, and are not Lipschitz-bounded. No adaptation of the bound to the conditional vocoding setting exists. No proof either way.
- **Empirically open (the main gap).** The compute-matched sweep: one-step and multi-step decoders of identical architecture and parameter count, trained over $10^{19}$–$10^{21}$ FLOPs, evaluated by MUSHRA on 44.1 kHz singing and non-speech. Runnable today for under $10^5$ USD. Nobody has published it.
- **Methodologically blocked.** Detecting a residual gap of $\le 3$ MUSHRA points. No accepted metric isolates inter-harmonic phase coherence, and no public artifact-stratified test set exists for vocoder transients. Whatever ceiling exists is currently below the noise floor of the standard evaluation.

## 6. Why It Is Hard

The binding obstruction is **an evaluation that does not measure what it names, compounded by absent ground truth for the failure mode**.

Copy-synthesis MOS saturates: on clean read speech, ground truth itself scores ~4.4–4.5, leaving under 0.15 MOS of headroom, which is the width of a typical CI. So the experiment that matters must run on content where the gap is visible — sustained singing, applause, percussive transients — but there is no standard stratified set for it, so every lab constructs its own and results do not compose.

Second: the comparison is not identified. "One-step vs multi-step" in the literature always co-varies architecture (ConvNeXt vs UNet), objective (adversarial vs score-matching), and training data. No published pair differs *only* in NFE. Any observed gap is therefore attributable to at least three causes, and the ceiling hypothesis is not falsifiable from existing numbers.

## 7. Current Research (as of 2026)

- **Anti-aliased one-step decoders** — NVIDIA's BigVGAN line, continued into codec decoders and 44.1 kHz music. Direction: push one-step quality by fixing inductive bias, not step count.
- **Distillation of audio diffusion to 1–4 steps** — adaptations of progressive distillation (Salimans & Ho, ICLR 2022) and distribution-matching distillation (Yin et al., CVPR 2024) to waveform and latent audio. *(frontier — verify)* Most 2025–2026 reports remain at NFE $\in \{1,2,4\}$ with quality reported against the teacher, not against a compute-matched one-step GAN.
- **Codec-decoder vocoders** — DAC/EnCodec descendants, where the one-step decoder is the default and diffusion decoders are the challenger at very low bitrate.
- **Evaluation** — FAD embedding choice and its instability; artifact-specific listening protocols. This is the least-funded and most load-bearing direction.

## 8. Concrete Next Experiment

**Isolate NFE.** Take one architecture — Vocos-style ConvNeXt backbone, 120M params, 44.1 kHz, iSTFT head — and train four arms on the same 5,000 h corpus (LibriTTS-R + MTG-Jamendo + a singing set), same optimizer, same 400 GPU-hours per arm on H100s ($\approx 3\times10^{20}$ FLOPs):

| Arm | Objective | NFE at inference |
|---|---|---|
| A (control) | multi-step diffusion, DDPM | 50 |
| B | same model as A, deterministic sampler | 4 |
| C | same model as A, consistency-distilled | 1 |
| D | adversarial (HiFi-GAN losses), identical backbone | 1 |

Evaluate on a **stratified** 200-item set: 50 clean speech, 50 unseen-speaker, 50 sustained singing, 50 percussive/applause. MUSHRA, 20 trained listeners, hidden reference plus 3.5 kHz anchor.

**The deciding number:** $\Delta_Q(\text{A}) - \min\{\Delta_Q(\text{C}), \Delta_Q(\text{D})\}$ on the singing + transient strata, with a bootstrap 95% CI. If the interval excludes 0 and the point estimate exceeds 5 MUSHRA points, an NFE ceiling exists at this scale. If the interval contains 0, the ceiling hypothesis is dead at $10^{20}$ FLOPs and the question moves to whether it reappears at $10^{22}$.

Repeat at 30M and 400M params to get a slope: a ceiling that shrinks with scale is an estimation artifact, not a ceiling.

## 9. Key References

- **[Foundational]** Aaron van den Oord et al. *Parallel WaveNet: Fast High-Fidelity Speech Synthesis.* ICML 2018. — arXiv:1711.10433
- **[Foundational]** Jungil Kong, Jaehyeon Kim, Jaekyoung Bae. *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.* NeurIPS 2020. — arXiv:2010.05646
- **[Foundational]** Nanxin Chen et al. *WaveGrad: Estimating Gradients for Waveform Generation.* ICLR 2021. — arXiv:2009.00713
- **[Foundational]** Zhifeng Kong et al. *DiffWave: A Versatile Diffusion Model for Audio Synthesis.* ICLR 2021. — arXiv:2009.09761
- **[SOTA]** Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, Sungroh Yoon. *BigVGAN: A Universal Neural Vocoder with Large-Scale Training.* ICLR 2023. — arXiv:2206.04658
- **[SOTA]** Hubert Siuzdak. *Vocos: Closing the Gap between Time-Domain and Fourier-Based Neural Vocoders for High-Quality Audio Synthesis.* ICLR 2024. — arXiv:2306.00814
- **[SOTA]** Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, Kundan Kumar. *High-Fidelity Audio Compression with Improved RVQGAN.* NeurIPS 2023. — arXiv:2306.06546
- **[Theory]** Antoine Salmona, Valentin de Bortoli, Julie Delon, Agnès Desolneux. *Can Push-forward Generative Models Fit Multimodal Distributions?* NeurIPS 2022. — arXiv:2206.14476
- **[Method]** Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever. *Consistency Models.* ICML 2023. — arXiv:2303.01469
- **[Method]** Tim Salimans, Jonathan Ho. *Progressive Distillation for Fast Sampling of Diffusion Models.* ICLR 2022. — arXiv:2202.00512
- **[Method]** Yuma Koizumi et al. *WaveFit: An Iterative and Non-autoregressive Neural Vocoder based on Fixed-Point Iteration.* IEEE SLT 2022. — arXiv:2210.01029
- **[Evaluation]** Kevin Kilgour, Mauricio Zuluaga, Dominik Roblek, Matthew Sharifi. *Fréchet Audio Distance: A Reference-Free Metric for Evaluating Music Enhancement Algorithms.* Interspeech 2019. — arXiv:1812.08466
- **[Evaluation]** Takaaki Saeki et al. *UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022.* Interspeech 2022. — arXiv:2204.02152
- **[Survey]** Xu Tan, Tao Qin, Frank Soong, Tie-Yan Liu. *A Survey on Neural Speech Synthesis.* 2021. — arXiv:2106.15561

## 10. Worked Example

Take 24 kHz speech, mel with $F = 100$ bands, hop $H = 256$. One second of audio: $L = 24{,}000$ samples, $T \approx 94$ frames, so $c$ carries $9{,}400$ numbers to specify $24{,}000$ samples. The fiber $\mathcal{F}_c$ has roughly $24{,}000 - 9{,}400 = 14{,}600$ free dimensions per second — overwhelmingly phase.

Now make the multimodality concrete. Consider one voiced frame with $f_0 = 200$ Hz and harmonics $k f_0$ for $k = 1..40$. The mel magnitude fixes amplitudes; the relative phases $\phi_k \in [0, 2\pi)$ are free, so the conditional support is a 40-torus. Perceptually, only a lower-dimensional subset sounds natural: phase relations that produce a single glottal pulse per period. Random phase on the same magnitudes produces the classic Griffin-Lim buzz.

So the conditional target is, to first order, a set of well-separated islands on that torus (one per admissible pulse alignment), not a connected blob. That is exactly the regime of the Salmona et al. bound: a one-step pushforward of a unimodal Gaussian must either connect the islands — placing mass on unnatural phase configurations — or use a very large Lipschitz constant.

**Where the obstruction becomes visible.** Suppose the generator places a fraction $\varepsilon = 10^{-3}$ of its mass on the connecting paths. At 24 kHz with 94 frames/s, that is roughly $0.094$ bad frames per second, or one audible glitch every ~11 seconds. In a MUSHRA test with 10-second items, one item in one hears it — a mean shift of perhaps 1–2 points, well inside the $\pm 4$-point CI of a 20-listener panel. FAD on VGGish embeddings, which pool over 0.96 s windows, will not register $10^{-3}$ of frames at all.

The obstruction is not that the ceiling is absent. It is that at the magnitude the theory predicts, **the standard evaluation cannot see it** — which is why the experiment in §8 must be artifact-stratified rather than run on average speech.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*