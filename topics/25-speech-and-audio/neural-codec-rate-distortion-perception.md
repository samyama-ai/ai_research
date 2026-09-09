---
id: 25-speech-and-audio/neural-codec-rate-distortion-perception
title: "Neural Audio Codec Rate-Distortion-Perception Frontier"
topic: 25-speech-and-audio
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Neural Audio Codec Rate-Distortion-Perception Frontier

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/neural-codec-rate-distortion-perception` · **Status:** empirically-open

## 1. Problem Statement

Neural audio codecs are trained with a mixture of reconstruction losses and adversarial losses, then reported at a handful of bitrates. Nobody has measured the surface those losses trade along.

- **Input:** a waveform $x \in \mathbb{R}^{T}$ drawn from a source distribution $p_X$ (speech, music, general audio).
- **Output:** a bitstream of expected length $R$ bits/second and a reconstruction $\hat{x}$.
- **Objective:** characterize $D(R, P)$ — the minimum achievable distortion at rate $R$ subject to a *realism* constraint $d(p_X, p_{\hat X}) \le P$.

Three variants, of very different difficulty:

- **Theory.** Does the Blau–Michaeli bound $D(R, P{=}0) \le 2\,D(R, P{=}\infty)$ (squared error, ICML 2019) apply to perceptually weighted audio distortions, and what is the exact rate penalty when encoder and decoder share no common randomness?
- **Measurement.** Is there any operational estimator of $P$ for audio that behaves like a divergence — zero iff the distributions match, monotone in perturbation strength, insensitive to embedding choice? Fréchet Audio Distance (FAD) and DNSMOS are not known to be.
- **Method.** Given a fixed architecture and rate, can the practitioner *choose* an operating point on the frontier, rather than discovering it by tuning the adversarial loss weight?

Solved would mean: a published, reproducible $D(R,P)$ surface for at least one audio source at three rates, with the three-way tradeoff shown to be a real frontier (moving along it is forced) and not a training artifact.

## 2. Formal Setting

Source $X \sim p_X$ over 1-second frames at sample rate $f_s$. Encoder $f_\theta: \mathbb{R}^T \to \{1,\dots,2^{R}\}$, decoder $g_\phi$, optional common randomness $U$ available to both. Reconstruction $\hat X = g_\phi(f_\theta(X), U)$.

$$D(R,P) \;=\; \inf_{\theta,\phi}\ \mathbb{E}\!\left[\Delta(X,\hat X)\right] \quad \text{s.t.} \quad I(X;\hat X) \le R,\ \ d(p_X, p_{\hat X}) \le P.$$

**As actually measured:**

- $R$: entropy-coded bits per second on a held-out set, *not* $n_q \cdot \log_2 |\mathcal{C}| \cdot f_{\text{tok}}$. Residual vector quantization (RVQ) codebooks are far from uniform; the nominal rate overstates the entropy by 5–20% in published codecs.
- $\Delta$: multi-scale mel-spectrogram L1 over window sizes $\{32,\dots,2048\}$, the standard in DAC/EnCodec. Reported per-frame, mean over the test set.
- $d$: this is the weak link. Candidates: FAD with a fixed embedding (VGGish, or CLAP), or the equilibrium accuracy of a *post-hoc* discriminator $h$ trained from scratch on $(x, \hat x)$ pairs — realism $P \approx 2\,\mathrm{AUC}(h) - 1$, an estimate of total-variation distance. The post-hoc discriminator is the only candidate with a divergence interpretation.
- Subjective anchor: MUSHRA (ITU-R BS.1534-3) with hidden reference and 3.5 kHz anchor, $\ge 20$ trained listeners.

**Assumptions, and which break:**

1. *$p_X$ is stationary and the test set is drawn from it.* Violated: codecs are trained on LibriTTS/Common Voice/MUSDB mixtures and evaluated on the same mixtures; the frontier is source-specific and no cross-source measurement exists.
2. *$I(X;\hat X) \le R$ is tight.* Violated: quantizer collapse means effective rate $\ll$ nominal in the deep RVQ layers.
3. *A single scalar $P$ captures realism.* Violated in the direction that matters: listeners weight artifacts non-uniformly (a metallic transient is worse than a broadband noise floor of equal divergence).
4. *Common randomness is free.* Violated: real codecs are deterministic decoders, and Wagner (2022) shows the RDP frontier without common randomness is strictly worse.

## 3. State of the Art

**Theory SOTA (established).** Blau & Michaeli (ICML 2019) prove $D(R,P)$ is monotone in both arguments and, for squared error, that requiring perfect realism at most doubles the distortion — 3.01 dB. Theis & Wagner (2021) give the coding theorem: the RDP function is operationally achievable with variable-length coding and unlimited common randomness. Wagner (2022) shows common randomness is not merely convenient — without it the achievable region shrinks. Freirich et al. (NeurIPS 2021) give the closed form in Wasserstein space for Gaussian sources. **None of this has been instantiated for an audio source.**

**Systems SOTA (established).** SoundStream (Zeghidour et al., TASLP 2021) at 3 kbps rated above Opus at 12 kbps in MUSHRA. EnCodec (Défossez et al., TMLR 2023) added a multi-scale STFT discriminator and a learned entropy model over RVQ indices. DAC (Kumar et al., NeurIPS 2023) reaches 44.1 kHz at 8 kbps with 9 codebooks — ~90× compression — and identified codebook collapse, fixing it with factorized low-dimensional lookup and L2-normalized codes. Mimi (Défossez et al., 2024) runs at 1.1 kbps, 12.5 Hz frame rate, with a distilled semantic first codebook. WavTokenizer (Ji et al., ICLR 2025) reports single-quantizer operation at 40–75 tokens/s.

**Claimed but unablated.** That adversarial training "improves perceptual quality at fixed rate" is universally asserted and almost never ablated as a *tradeoff* — papers report the GAN arm beating the no-GAN arm on both distortion-like metrics and MOS, which the theory says should be impossible at the frontier. The likely explanation is that neither arm is on the frontier. That is the gap.

**Benchmark-number-only results.** Codec-SUPERB (Wu et al., ACL Findings 2024) tabulates ~10 codecs on downstream tasks. The numbers are real; the comparisons are confounded by differing training data, sample rates, and frame rates, so no tradeoff can be read off them.

## 4. What Is Known

- **The 2× bound.** For MSE, $D(R,0)\le 2D(R,\infty)$ — proven, distribution-free (Blau & Michaeli, ICML 2019). Not proven for mel-domain or perceptually weighted distortions.
- **Universal representations.** A single encoder can serve the entire $P$ axis with only decoder changes, at bounded loss (Zhang, Qian, Yu et al., NeurIPS 2021). Measured on images; untested on audio.
- **Rate scaling, measured.** DAC at 44.1 kHz: 8 kbps, 9 RVQ codebooks, 86 Hz frame rate; ViSQOL ~4.0+ on held-out music. EnCodec 24 kHz: 1.5 → 24 kbps spans roughly the full MUSHRA range from "clearly degraded" to "transparent" on 24 kHz content.
- **Codebook utilization is the binding constraint at low rate.** DAC reports near-100% codebook usage after factorization versus large dead fractions in prior RVQ; this is an independently reproduced regularity across at least three codebases.
- **Semantic distillation trades reconstruction for downstream utility.** SpeechTokenizer (ICLR 2024) and Mimi both show that forcing the first codebook toward HuBERT-like targets *lowers* reconstruction fidelity at fixed rate while raising ASR/LM performance — a measured, sign-consistent tradeoff, at ~1–2 kbps, on LibriSpeech-scale data.
- **ViSQOL and DNSMOS are distortion proxies, not divergences.** Both saturate: a decoder that hallucinates plausible-but-wrong high frequencies can score well. Documented informally across the codec literature; ViSQOL v3 (Chinen et al., QoMEX 2020) is explicitly a similarity metric.

## 5. What Is Not Known

- **Methodologically blocked:** the realism axis. There is no accepted audio $d(p_X,p_{\hat X})$ with divergence properties. FAD depends strongly on embedding choice and sample count; DNSMOS is a per-utterance MOS regressor, i.e. a distortion measure wearing a distribution measure's name. Until $P$ is defined, $D(R,P)$ cannot be plotted. **This is the primary blocker.**
- **Empirically open:** the frontier itself. Given any fixed $P$ estimator, sweeping adversarial weight at fixed rate and fixed architecture is a ~600 GPU-hour experiment. Nobody has published it.
- **Empirically open:** whether the 2× penalty is observed in audio. If measured distortion inflation at perfect realism is $\ll 2\times$, current codecs are far from the frontier and the tradeoff is not yet binding.
- **Theoretically open:** the exact rate cost of zero common randomness for a non-Gaussian, non-stationary audio source. Wagner (2022) establishes the gap is nonzero in general; the audio-relevant magnitude is unproven.
- **Theoretically open:** whether the 2× bound survives replacing MSE with a multi-scale mel distortion (which is not a metric on waveforms — it is invariant to phase).

## 6. Why It Is Hard

**The specific obstruction is confounded measurement compounded by non-identifiability.**

Every reported "perceptual quality" number is a distortion measure. A GAN decoder can lower distributional divergence and raise mel-L1 simultaneously, and the field's metric suite reports only the second — so the tradeoff is invisible in the tables. Worse, the *training* signal for realism is a discriminator co-trained with the generator: its equilibrium value measures neither divergence nor convergence, and it is exactly the quantity papers cite as evidence of realism.

The non-identifiability: at a given adversarial weight, an observed distortion increase can come from (a) genuine movement along the frontier, (b) optimization failure, or (c) reduced effective rate from codebook collapse. Distinguishing these needs the entropy-coded rate, a *post-hoc* (not co-trained) divergence estimate, and a converged baseline — three measurements no published codec paper reports together.

Compute is not the obstruction. A 24 kHz codec trains in ~24 GPU-hours.

## 7. Current Research (as of 2026)

- **Single-quantizer and ultra-low-rate codecs.** WavTokenizer, Mimi, and successors push below 1 kbps for speech LM tokenization. These sit far from the classical frontier by construction — they optimize for LM-modelability, not rate-distortion.
- **Diffusion and flow decoders as realism-first decoders.** Decoding a coarse code with a generative prior is the natural way to hit $P \approx 0$ at high distortion. Multiple groups (Google DeepMind, Meta FAIR, Descript) have shipped variants. *(frontier — verify)* No published version reports the distortion it pays.
- **RDP theory with limited common randomness.** Wagner, Theis, and coauthors; extensions to conditional/per-realization perception measures. Active but image-focused.
- **Codec evaluation.** Codec-SUPERB and successor benchmarks; ITU-T work on objective metrics for generative codecs. *(frontier — verify)* A standardized realism metric is not yet ratified.

## 8. Concrete Next Experiment

**Question:** at fixed rate and architecture, what does perfect realism cost in distortion, and does common randomness reduce that cost?

**Scale.** DAC-24kHz architecture, 6 codebooks, entropy-coded to $\approx 3.0$ kbps (verified, not nominal). Train on LibriTTS + MUSDB18-HQ, 250k steps each. Sweep adversarial weight $\lambda \in \{0, 0.1, 0.3, 1, 3, 10\}$ — 6 runs — crossed with decoder randomness $\in \{$deterministic, private noise, shared-seed noise$\}$ = 18 runs. ~35 GPU-hours each on one A100; **~630 GPU-hours total, under two weeks on 4 GPUs.**

**Control arm.** $\lambda = 0$, deterministic decoder, matched steps and matched entropy-coded rate to within 2%. This is the $P = \infty$ corner: minimum achievable distortion, no realism pressure.

**Measurements per run.** (i) multi-scale mel-L1 on held-out audio; (ii) realism via a *post-hoc* discriminator — ResNet on log-mel, trained from scratch on 50k held-out real/reconstructed pairs after the codec is frozen, reported as $2\,\mathrm{AUC}-1$; (iii) entropy-coded bits/s; (iv) MUSHRA on 12 items for the two endpoint runs only.

**The deciding number.** The distortion inflation ratio
$$\rho \;=\; \frac{\Delta \text{ at the smallest }\lambda\text{ with } 2\,\mathrm{AUC}-1 \le 0.05}{\Delta \text{ at } \lambda = 0}.$$
$\rho \approx 1$ means today's codecs are not on any frontier and adversarial training is a free lunch — the tradeoff is not yet binding, and the field should keep tuning. $1 < \rho \le 2$ confirms the Blau–Michaeli regime holds in the mel domain. $\rho > 2$ means either the mel distortion breaks the bound's assumptions or the runs are not converged — both are publishable and both are currently unknown. Secondary: whether shared-seed randomness lowers $\rho$ at matched rate, the first audio test of Wagner (2022).

## 9. Key References

- **[Foundational]** Y. Blau, T. Michaeli. *The Perception-Distortion Tradeoff.* CVPR, 2018. — arXiv:1711.06077
- **[Foundational]** Y. Blau, T. Michaeli. *Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff.* ICML, 2019. — arXiv:1901.07821
- **[Theory]** L. Theis, A. B. Wagner. *A Coding Theorem for the Rate-Distortion-Perception Function.* ICLR Neural Compression Workshop, 2021. — arXiv:2104.13662
- **[Theory]** A. B. Wagner. *The Rate-Distortion-Perception Tradeoff: The Role of Common Randomness.* 2022. — arXiv:2202.04147
- **[Theory]** D. Freirich, T. Michaeli, R. Meir. *A Theory of the Distortion-Perception Tradeoff in Wasserstein Space.* NeurIPS, 2021.
- **[Theory]** G. Zhang, J. Qian, J. Chen, A. Khisti. *Universal Rate-Distortion-Perception Representations for Lossy Compression.* NeurIPS, 2021. — arXiv:2106.10311
- **[SOTA]** N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, M. Tagliasacchi. *SoundStream: An End-to-End Neural Audio Codec.* IEEE/ACM TASLP, 2021. — arXiv:2107.03312
- **[SOTA]** A. Défossez, J. Copet, G. Synnaeve, Y. Adi. *High Fidelity Neural Audio Compression.* TMLR, 2023. — arXiv:2210.13438
- **[SOTA]** R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, K. Kumar. *High-Fidelity Audio Compression with Improved RVQGAN.* NeurIPS, 2023. — arXiv:2306.06546
- **[SOTA]** A. Défossez, L. Mazaré, M. Orsini, A. Royer, P. Pérez, H. Jégou, E. Grave, N. Zeghidour. *Moshi: a speech-text foundation model for real-time dialogue.* 2024. — arXiv:2410.00037
- **[SOTA]** X. Zhang, D. Zhang, S. Li, Y. Zhou, X. Qiu. *SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models.* ICLR, 2024. — arXiv:2308.16692
- **[Benchmark]** H.-H. Wu et al. *Codec-SUPERB: An In-Depth Analysis of Sound Codec Models.* Findings of ACL, 2024. — arXiv:2402.13071
- **[Metric]** M. Chinen, F. S. C. Lim, J. Skoglund, N. Gureev, F. O'Gorman, A. Hines. *ViSQOL v3: An Open Source Production Ready Objective Speech and Audio Metric.* QoMEX, 2020. — arXiv:2004.09584
- **[Standard]** ITU-R. *BS.1534-3: Method for the subjective assessment of intermediate quality level of audio systems (MUSHRA).* 2015.

## 10. Worked Example

Take DAC at 24 kHz, 6 codebooks, 75 Hz frame rate, 10-bit codebooks. Nominal rate: $6 \times 10 \times 75 = 4500$ bits/s. Entropy-code the indices with a per-codebook static model and the measured index entropies fall to roughly $\{9.8, 9.6, 9.1, 8.4, 7.6, 6.9\}$ bits — the deep layers are the least uniform — giving $\sum \approx 51.4$ bits/frame, or **3855 bits/s, 14% below nominal**. Any comparison run at "4.5 kbps nominal" against a competitor at "4.5 kbps nominal" is comparing two different rates.

Now the tradeoff. Suppose the $\lambda=0$ control reaches mel-L1 $=0.72$ and a post-hoc discriminator separates real from reconstructed at AUC $=0.94$ (realism $0.88$ — trivially detectable). The $\lambda=3$ run reaches AUC $=0.52$ (realism $0.04$, near-indistinguishable) at mel-L1 $=0.97$. Naively $\rho = 0.97/0.72 = 1.35$, comfortably inside the 2× bound.

The obstruction appears when you check the rate: the $\lambda=3$ run's codebook usage dropped, entropy fell to 46.1 bits/frame = 3458 bits/s. It is **10% cheaper**, so it is not the same point on the $R$ axis. Some of the 0.25 mel-L1 increase bought realism; some bought nothing and merely reflects lost rate. With one measurement and two moving variables, $\rho$ is not identified. Recovering it requires re-training with a rate constraint enforced on the *entropy-coded* stream — which no public codec training recipe does.

That is why the frontier is unmeasured. Not compute, not data: the field reports nominal rate, a co-trained discriminator, and a distortion metric labeled "perceptual", and those three choices make the surface unobservable.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*