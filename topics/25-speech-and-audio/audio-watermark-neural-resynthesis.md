---
id: 25-speech-and-audio/audio-watermark-neural-resynthesis
title: "Audio Watermark Robustness Under Neural Re-Synthesis"
topic: 25-speech-and-audio
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Audio Watermark Robustness Under Neural Re-Synthesis

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/audio-watermark-neural-resynthesis` · **Status:** open

## 1. Problem Statement

An audio watermark embeds an imperceptible signal into speech so that a detector can later assert "this was machine-generated" or recover a short payload. The standard threat model is *signal-space editing*: noise, MP3, resampling, time-stretch, filtering. Modern watermarks survive all of it.

**Neural re-synthesis** is a different channel. The attacker passes the watermarked audio through an encoder into a learned, heavily quantized representation (neural codec tokens, mel-spectrogram, self-supervised discrete units, speaker embedding plus $F_0$) and regenerates a waveform from that representation. The output is a *new* waveform, not a perturbation of the old one. It carries the same words in the same voice and is judged near-identical by listeners, but almost nothing sub-perceptual survives.

Three variants, of very different difficulty:

- **Measurement.** Is there an evaluation protocol under which "robust to re-synthesis" is a falsifiable claim? Requires an attacker constraint set that neural re-synthesis actually lives inside — the open question is what that set is.
- **Method.** Does there exist an embedder/detector pair achieving $\mathrm{TPR} \ge 0.95$ at $\mathrm{FPR} = 10^{-3}$ on 1-second windows after re-synthesis through a codec the embedder never saw, at $\le 1$ dB perceptual cost?
- **Theory.** Is there an impossibility result for audio matching the image-domain regeneration bounds — a proof that any watermark surviving a bottleneck of rate $R$ bits/s must be detectable in the semantic content itself?

Solving it means the method variant with an out-of-distribution attack channel, not a whitelist of codecs seen in training.

## 2. Formal Setting

Let $x \in \mathbb{R}^{T}$ be a waveform sampled at $f_s$ Hz, duration $\tau = T/f_s$.

**Embedder.** $W_k(x, m) = x + \delta$ with key $k$, message $m \in \{0,1\}^b$, residual $\delta$. Measured perceptual cost: scale-invariant SNR $\mathrm{SI\text{-}SNR}(x, x_w) = 10\log_{10}\frac{\|\alpha x\|^2}{\|\alpha x - x_w\|^2}$ with $\alpha = \langle x_w, x\rangle/\|x\|^2$; plus ViSQOL MOS-LQO and UTMOS (no-reference naturalness, 1–5).

**Detector.** $D_k: \mathbb{R}^{T'} \to [0,1]^{T'}$ produces a per-sample score; window decision by mean score over a $\tau_w$-second window against threshold $\theta$. Report $\mathrm{TPR}@\mathrm{FPR}{=}10^{-3}$, with FPR estimated on $\ge 10^5$ genuine non-watermarked windows drawn from the *same corpus* (cross-corpus FPR estimation inflates the number).

**Attack channel.** $A = G \circ Q \circ E$: encoder $E$, quantizer $Q$, generator $G$. Its bottleneck rate is
$$R_A \;=\; \frac{1}{\tau}\log_2 |\mathcal{Z}_\tau| \quad \text{bits/s},$$
where $\mathcal{Z}_\tau$ is the set of reachable code sequences for $\tau$ seconds. For a residual-VQ codec with $N_q$ books of size $V$ at frame rate $f_r$: $R_A = f_r N_q \log_2 V$.

**Data-processing limit.** Because $m \to x_w \to z \to \hat{x}$ is a Markov chain,
$$I(m; \hat{x}) \;\le\; I(m; z) \;\le\; H(z) \le R_A \tau .$$
This bounds *any* detector, including one trained against $A$.

**Attacker constraint set.** The attack must preserve content. Measured as: WER of a fixed ASR (Whisper large-v3) on $\hat{x}$ vs on $x$; speaker similarity $\cos(e(\hat x), e(x))$ with an ECAPA-TDNN embedder; UTMOS drop. Define
$$\mathcal{A}_\epsilon = \{A : \Delta\mathrm{WER} \le \epsilon_1,\; \cos \ge \epsilon_2,\; \Delta\mathrm{UTMOS} \le \epsilon_3\}.$$
The game is $\max_{W,D}\min_{A \in \mathcal{A}_\epsilon} \mathrm{TPR}@10^{-3}$.

**Assumptions, and which fail.**
1. *Attacker lacks the key.* Violated: AudioSeal, WavMark and SilentCipher ship public weights, so the detector is a white-box gradient oracle.
2. *Attacks are bounded in signal space* ($\|\hat x - x\|$ small). Violated by construction — re-synthesis produces an unaligned waveform with SI-SNR often below 0 dB against the input while sounding identical.
3. *Perceptual metrics track the constraint.* Violated: ViSQOL and PESQ are intrusive and penalize the phase/alignment changes re-synthesis makes, so they *over*-report attack cost; UTMOS is insensitive to timbre drift and *under*-reports it.
4. *The training-time attack distribution covers deployment.* Violated: new codecs and vocoders appear faster than watermark retraining.

## 3. State of the Art

**Systems/empirical SOTA (established).**
- **AudioSeal** (San Roman, Fernandez, Elsahar, Défossez, Furon, Tran; ICML 2024) — generator/detector pair with sample-level localization, 16-bit payload, trained with an augmentation bank. Established: near-ceiling detection under compression, filtering, and time modification at ~1-second granularity; localization to sub-second resolution, which prior methods lacked.
- **WavMark** (Chen, Wu, Liu, Qin, Xia, Wang; 2023) — invertible-network encoder, 32-bit payload in 1-second patches, synchronization by pattern bits.
- **SilentCipher** (Singh et al., Interspeech 2024) — psychoacoustic-threshold embedding, no perceptual-loss training.
- **Timbre Watermarking** (Liu et al., NDSS 2024) — embeds in the frequency-domain timbre representation specifically so it survives voice cloning that copies the speaker's timbre. This is the only major line that targets the re-synthesis channel directly rather than treating it as one more augmentation.

**Established negative results.** *AudioMarkBench* (Liu, Guo, Jiang, Wang, Gong; NeurIPS 2024 Datasets & Benchmarks) is the reference robustness benchmark: it evaluates AudioSeal, WavMark and Timbre under common perturbations plus white-box and black-box adversarial removal, and finds that adversarial removal succeeds with modest query budgets while leaving audio quality largely intact.

**Claimed but unablated.** Vendor claims of robustness "to re-synthesis" (including Google's SynthID audio watermark, which as of this writing has no peer-reviewed paper describing its audio variant — the Nature 2024 SynthID paper by Dathathri et al. covers *text*) rest on internal evaluation with undisclosed attack channels. Treat as unverified. Papers reporting survival through EnCodec typically trained against EnCodec; the held-out-codec ablation is usually absent.

**Theory SOTA (image domain, not yet ported).** Zhao et al., *Invisible Image Watermarks Are Provably Removable Using Generative AI* (NeurIPS 2024), and Saberi et al., *Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks* (ICLR 2024), both give regeneration/purification arguments bounding detector performance in terms of the distance between watermarked and clean distributions. No audio analogue exists.

## 4. What Is Known

- Watermarks survive conventional DSP. AudioSeal reports detection accuracy near 1.0 and localization IoU above 0.9 across its augmentation suite on ~16 kHz speech (AudioSeal, ICML 2024); WavMark reports bit accuracy above 0.99 under most common edits at 16 kHz, 1-second segments.
- Watermarks do **not** survive adaptive adversaries. AudioMarkBench (NeurIPS 2024) shows both white-box and query-limited black-box removal driving detection to near chance across all three tested schemes at scale of thousands of utterances, with quality degradation the authors characterize as small.
- Neural codecs are extremely lossy in the sub-perceptual band by design. EnCodec (Défossez, Copet, Synnaeve, Adi; TMLR 2023) operates at 1.5–24 kbps; DAC (Kumar et al., NeurIPS 2023) at ~8 kbps for 44.1 kHz. Against 16-bit PCM at 16 kHz (256 kbps), a 6 kbps codec is a $\sim$43× bottleneck.
- Discrete-unit re-synthesis is far more aggressive still. Polyak et al. (*Speech Resynthesis from Discrete Disentangled Self-Supervised Representations*, Interspeech 2021) resynthesize intelligible, speaker-preserving speech from HuBERT units at 50 Hz plus quantized $F_0$ and a speaker embedding — order $10^2$–$10^3$ bits/s.
- Timbre-domain embedding survives *some* voice conversion pipelines (NDSS 2024), establishing that the channel is not universally destructive — content-bearing components can carry a mark.

## 5. What Is Not Known

- **Empirically open.** The clean cross-product — {AudioSeal, WavMark, SilentCipher, Timbre} × {EnCodec, DAC, mel+HiFi-GAN, HuBERT-unit resynthesis, a diffusion vocoder} × {seen, held-out} — with a *matched-perceptual-distortion control arm*, at $\ge 10^3$ utterances and FPR estimated at $10^{-3}$, has not been published. The experiment is runnable on one GPU-week.
- **Methodologically blocked.** There is no agreed measure of attack cost for re-synthesis. Intrusive metrics (PESQ, ViSQOL, SI-SNR) are invalid because the output is not aligned to the input; non-reference metrics (UTMOS, DNSMOS) miss identity drift. Without a valid $\epsilon$, "robust at equal quality" is not a testable statement — this is the blocker that keeps the field reporting incomparable numbers.
- **Theoretically open.** No audio analogue of the regeneration impossibility bounds. Specifically: is there a $R^\star$ such that no watermark with perceptual cost $\le \epsilon$ survives any bottleneck below $R^\star$ bits/s? The data-processing inequality gives a limit but not a constructive threshold, because $I(m;z)$ depends on how the quantizer allocates its rate, not just on $R_A$.

## 6. Why It Is Hard

**The obstruction is non-identifiability of the attacker's budget, compounded by an evaluation that does not measure what it names.**

Every robustness number is of the form "TPR after attack $A$, at quality cost $q$". For additive attacks, $q$ is well defined and both sides agree on it. For re-synthesis, the attacker's true constraint is *perceptual and semantic* — same words, same apparent speaker, natural-sounding — while every deployed metric is *signal-referential*. The consequence is concrete: a re-synthesis attack that a listener rates identical to the original can score SI-SNR near 0 dB and ViSQOL near 3.0, which under the usual protocol is reported as a "high-distortion attack" and therefore excused. The watermark looks robust because the attack was scored as out-of-budget when in fact it was free.

This is not fixed by adding attacks to the training augmentation bank. The bottleneck is a *rate* limit, not a noise process: by the data-processing inequality, once the representation $z$ carries fewer bits than the watermark needs to distinguish itself, no amount of adversarial training helps. The only surviving strategy is to put the mark inside content the codec must preserve — which by construction makes it perceptible-in-principle and trades directly against quality.

## 7. Current Research (as of 2026)

- **Meta AI (FAIR)** — AudioSeal line; continued work on localization and generator-integrated watermarking.
- **Duke (Gong group)** — AudioMarkBench; adversarial removal and robustness benchmarking.
- **Zhejiang University / NDSS line** — timbre- and content-domain watermarking aimed explicitly at voice cloning.
- **Sony / Interspeech line** — SilentCipher and psychoacoustic embedding.
- **Google DeepMind** — SynthID audio, deployed in Lyria/Gemini audio outputs; no published attack analysis *(frontier — verify)*.
- **Joint attention/watermarking architectures** that cross-attend between carrier and message to improve the robustness–imperceptibility frontier appeared at ICML 2025 *(frontier — verify title and authorship before citing)*.
- **Semantic watermarking**, where the mark is placed in prosody or token-level generation choices of the TTS model rather than in the waveform, is the main active response to the rate argument.

## 8. Concrete Next Experiment

**Question.** Is re-synthesis failure caused by the *distortion* it introduces, or by the *rate bottleneck*? These have opposite fixes.

**Scale.** LibriSpeech `test-clean` (2620 utterances, 16 kHz) plus 500 expressive utterances from EXPRESSO. Four watermarks (AudioSeal, WavMark, SilentCipher, Timbre) at their default operating points, each tuned to SI-SNR $= 26 \pm 1$ dB on clean input. Five attack channels: EnCodec 6 kbps, DAC 8 kbps, mel + HiFi-GAN, HuBERT-unit resynthesis (50 Hz, 500 units, + $F_0$ + speaker embedding), and one diffusion vocoder. FPR calibrated on $10^5$ non-watermarked 1-second windows from the same corpora. One GPU-week.

**Control arm (the point of the experiment).** For each (watermark, channel) pair, construct a *matched-distortion* additive attack: coloured Gaussian noise plus band-limited filtering, with its parameters searched so its UTMOS drop and its ECAPA cosine to the original match the re-synthesis channel's to within 0.02. This control has effectively unbounded rate — it destroys quality without imposing a bottleneck.

**Deciding number.**
$$\Delta \;=\; \mathrm{TPR}@\mathrm{FPR}{=}10^{-3}\big|_{\text{matched additive}} \;-\; \mathrm{TPR}@\mathrm{FPR}{=}10^{-3}\big|_{\text{re-synthesis}}$$
on 1-second windows.

- $\Delta < 5$ points: re-synthesis is just a distortion channel. Augmentation training is the right fix, and the problem downgrades to engineering.
- $\Delta > 20$ points: the failure is the rate bottleneck, not the distortion. No augmentation schedule closes it, and the field must move the mark into content the codec preserves — accepting the perceptibility trade.

Secondary quantity, needed for the theory variant: the per-frame HuBERT unit flip rate $p = \Pr[Q(E(x_w))_t \ne Q(E(x))_t]$, which upper-bounds $I(m; z)$ and turns the argument from empirical into information-theoretic.

## 9. Key References

- **[Foundational]** Défossez, A., Copet, J., Synnaeve, G., Adi, Y. *High Fidelity Neural Audio Compression.* TMLR, 2023. — arXiv:2210.13438
- **[Foundational]** Polyak, A., Adi, Y., Copet, J., Kharitonov, E., Lakhotia, K., Hsu, W.-N., Mohamed, A., Dupoux, E. *Speech Resynthesis from Discrete Disentangled Self-Supervised Representations.* Interspeech, 2021. — arXiv:2104.00355
- **[SOTA]** San Roman, R., Fernandez, P., Elsahar, H., Défossez, A., Furon, T., Tran, T. *Proactive Detection of Voice Cloning with Localized Watermarking.* ICML, 2024. — arXiv:2401.17264
- **[SOTA]** Chen, G., Wu, Y., Liu, S., Qin, T., Xia, Y., Wang, H. *WavMark: Watermarking for Audio Generation.* 2023. — arXiv:2308.12770
- **[SOTA]** Liu, C., Zhang, J., Zhang, T., Yang, X., Zhang, W., Yu, N. *Detecting Voice Cloning Attacks via Timbre Watermarking.* NDSS, 2024.
- **[SOTA]** Singh, M. K., Takahashi, N., Liao, W.-H., Mitsufuji, Y. *SilentCipher: Deep Audio Watermarking.* Interspeech, 2024.
- **[Benchmark]** Liu, H., Guo, M., Jiang, Z., Wang, L., Gong, N. Z. *AudioMarkBench: Benchmarking Robustness of Audio Watermarking.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.06979
- **[Theory, image domain]** Zhao, X., Zhang, K., Su, Z., Vasan, S., Grishchenko, I., Kruegel, C., Vigna, G., Wang, Y.-X., Li, L. *Invisible Image Watermarks Are Provably Removable Using Generative AI.* NeurIPS, 2024.
- **[Theory, image domain]** Saberi, M., Sadasivan, V. S., Rezaei, K., Kumar, A., Chegini, A., Wang, W., Feizi, S. *Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks.* ICLR, 2024.
- **[Related]** Kumar, R., Seetharaman, P., Luebs, A., Kumar, I., Kumar, K. *High-Fidelity Audio Compression with Improved RVQGAN.* NeurIPS, 2023.
- **[Related]** Kong, J., Kim, J., Bae, J. *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.* NeurIPS, 2020.

## 10. Worked Example

Take one second of 16 kHz speech, AudioSeal-watermarked with a 16-bit payload at SI-SNR $= 26$ dB. The watermark residual holds $10^{-2.6} \approx 0.25\%$ of signal power.

**Channel A — EnCodec at 6 kbps.** $R_A \tau = 6000$ bits. The payload needs 16. Ratio $375:1$. Rate is not the binding constraint; whether the mark survives depends on whether the RVQ codebooks happen to resolve the residual. This channel is winnable, and augmentation training plausibly wins it.

**Channel B — HuBERT-unit re-synthesis.** 50 frames/s, 500-unit codebook, plus quantized $F_0$ at 50 Hz over 16 bins, plus a speaker embedding amortized over the utterance:
$$R_A \approx 50\log_2 500 + 50\log_2 16 \approx 448 + 200 = 648 \text{ bits/s}.$$
Against 256 kbps PCM that is a $395\times$ bottleneck — but the bottleneck is not the problem. The problem is *where* the rate goes.

The HuBERT quantizer is trained to be invariant to exactly the kind of low-energy, speaker-neutral perturbation the watermark is. Suppose the measured per-frame unit flip rate induced by adding the watermark is $p = 0.02$. Over 50 frames, the expected number of flipped units is 1. Each flip distinguishes at most $\log_2 500 \approx 9$ raw bits of state, and only the component correlated with $m$ is usable, so
$$I(m; z) \;\le\; 50 \cdot H_b(0.02) \cdot \log_2 500 \;\approx\; 50 \times 0.141 \times 9 \;\approx\; 64 \text{ bits}$$
as a loose ceiling — and the *realized* mutual information is far lower, because flips are driven by phonetic ambiguity rather than by the message. If the true $I(m;z)$ falls below 1 bit, then by Fano's inequality no detector, however trained, can separate watermarked from clean 1-second windows at $\mathrm{FPR} = 10^{-3}$ with useful TPR.

**Where the obstruction becomes visible.** Score channel B with the standard protocol. SI-SNR of $\hat x$ against $x$ is roughly $-3$ dB, because the vocoder reconstructs phase freely. ViSQOL lands near 2.8. Under the conventional reading those are catastrophic-distortion numbers, so the attack is dismissed as out-of-budget. Now measure the constraints that matter: Whisper WER rises by 0.4 points, ECAPA speaker cosine holds at 0.91, UTMOS *rises* from 3.9 to 4.1 because the vocoder output is cleaner than the recording.

The attack is free by every metric the deployment cares about and prohibitively expensive by every metric the benchmark reports. That mismatch — not the detector's architecture — is what keeps this problem open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*