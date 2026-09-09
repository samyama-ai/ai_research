---
id: 25-speech-and-audio/provable-synthetic-speech-detection
title: "Provable Detection of Synthetic Speech"
topic: 25-speech-and-audio
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Provable Detection of Synthetic Speech

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/provable-synthetic-speech-detection` · **Status:** open

## 1. Problem Statement

Given a waveform $x$, decide whether it was produced by a human vocal tract recorded through some channel, or synthesized by a generative model (TTS, voice conversion, neural codec resynthesis), **with a guarantee attached to the decision** — a bound on the false-positive rate that holds under distribution shift and under an adversary who may post-process the audio.

Three variants, with very different difficulty:

- **Measurement.** Define an operating point that is meaningful outside the corpus it was tuned on. Current practice reports equal error rate (EER) on a fixed evaluation set; deployment needs TPR at a fixed FPR, e.g. $10^{-3}$, on audio from generators and channels unseen at training time.
- **Method.** Build a detector that hits that operating point. Two families: **passive** (classify the audio as-is) and **proactive** (the generator embeds a keyed watermark; the detector tests for it).
- **Theory.** Prove that a guarantee is achievable. For passive detection the question is whether $p_M$ and $p_H$ stay statistically separated as generators improve. For proactive detection it is whether a watermark can survive a quality-preserving adversary.

Solving it means: a detector plus a proof or a validated statistical certificate that its FPR on genuine human speech is $\le \alpha$, and a lower bound on TPR that holds against a stated adversary class — not a leaderboard number.

## 2. Formal Setting

Let $x \in \mathbb{R}^{T}$ be a waveform at sample rate $f_s$. Let $p_H$ be the distribution of human recordings and $p_M$ that of a generator $M$. A detector is a score $s: \mathbb{R}^T \to \mathbb{R}$ with threshold $\tau$; predict "synthetic" if $s(x) > \tau$.

**Measured quantities.**

- $\mathrm{FPR}(\tau) = \Pr_{x\sim p_H}[s(x) > \tau]$, estimated as the fraction of a held-out human corpus of $n$ utterances above $\tau$. With zero observed false positives the 95% upper bound is the rule of three, $3/n$.
- $\mathrm{TPR}(\tau) = \Pr_{x\sim p_M}[s(x) > \tau]$; EER is $\tau$ where $\mathrm{FPR} = 1-\mathrm{TPR}$.
- ASVspoof scores min t-DCF (2019/2021) and a-DCF (2024/ASVspoof 5), a cost-weighted detection function that folds in a speaker-verification back end rather than reporting the spoof decision alone.

**Channel and adversary.** A channel $C$ (codec, resampling, room impulse response, additive noise) acts on $x$. An adversary picks $x' \in \mathcal{B}(x)$ where $\mathcal{B}$ is a perceptual ball — e.g. $\{x' : \mathrm{ViSQOL}(x,x') \ge 4.0\}$ or $\mathrm{SNR} \ge 20$ dB. Worst-case TPR is $\inf_{x' \in \mathcal{B}(x)} \mathbf{1}[s(x') > \tau]$ averaged over $p_M$.

**Passive upper bound.** For any detector, the Neyman–Pearson optimum is the likelihood ratio, and its ROC obeys

$$\mathrm{AUROC} \;\le\; \tfrac{1}{2} + \tfrac{1}{2}\,\mathrm{TV}(p_M, p_H), \qquad \mathrm{TPR}(\alpha) \le \alpha + \mathrm{TV}(p_M,p_H).$$

So passive detection is bounded by how far the generator still is from the data distribution, after the channel.

**Proactive setting.** The generator holds key $k$, embeds via $E_k$, and the detector computes a statistic $Z_k(x)$ whose null distribution under $x \sim p_H$ is known, giving a p-value. The guarantee is exact only if that null is correct.

**Assumptions, and which are violated.**

1. *Test attacks are drawn from the training attack set* — violated: in-the-wild audio uses generators released after the detector.
2. *Human null is well specified* — violated: the "human" class includes codec artefacts, telephony, enhancement, and denoising, which shift $Z_k$ and $s$ in the same direction as synthesis.
3. *The channel is fixed and known* — violated: social-media re-encoding is unobserved.
4. *Adversary is passive* — violated whenever detection has consequences.
5. *$\mathrm{TV}(p_M,p_H)$ is bounded away from 0* — assumed, never measured; it is not estimable from finite samples in high dimension.

## 3. State of the Art

**Empirical SOTA (passive).** Self-supervised front ends with graph-attention back ends: wav2vec 2.0 XLS-R features feeding AASIST (Tak et al., *Odyssey* 2022) remain the standard strong baseline; AASIST itself is Jung et al., *ICASSP* 2022, and RawNet2 is Tak et al., *ICASSP* 2021. **Established:** these reach low single-digit EER on matched ASVspoof evaluation sets, and their advantage over spectrogram baselines has been reproduced across labs. **Claimed but unablated:** most papers reporting sub-1% EER do not ablate whether the gain comes from spoof cues or from corpus-specific channel/silence artefacts — the known confound since Müller et al. showed silence-duration statistics alone separate ASVspoof classes.

**Empirical SOTA (proactive).** AudioSeal (San Roman et al., *ICML* 2024) does localized watermarking with near-perfect detection under common edits and sample-level localization; WavMark (Chen et al., 2023) embeds payload with high robustness to compression. **Benchmark-only:** the robustness numbers come from fixed edit lists. AudioMarkBench (Liu et al., *NeurIPS* Datasets & Benchmarks 2024) evaluates the same schemes under adaptive attacks and reports substantially worse robustness — the standard edit lists overstate it.

**Theory SOTA.** Two negative and one positive result. Sadasivan et al. give the TV bound above (originally for text, distribution-agnostic). Zhang et al., *Watermarks in the Sand* (*ICML* 2024), show that given a quality oracle and a perturbation oracle, a random-walk attack removes any watermark while preserving quality — strong watermarking is impossible in their model. Positively, Christ and Gunn's pseudorandom error-correcting codes (*CRYPTO* 2024) give watermarks provably robust to a bounded-rate substitution channel, with cryptographic undetectability; the audio instantiation does not exist.

## 4. What Is Known

- **Generalization collapse is real and large.** Detectors at roughly 2–5% EER on matched ASVspoof data degrade to about 30–40% EER on *In-the-Wild* (Müller et al., *Interspeech* 2022; 38 hours, 58 politicians/celebrities, ~31.8k clips). Reproduced widely.
- **Cross-condition degradation within the challenge.** ASVspoof 2021 DF track: RawNet2 baseline ≈ 22% EER; the best submitted single systems landed in the mid-teens, versus ≈ 1–2% EER in the 2021 LA track. Scale: ~600k evaluation trials.
- **Silence is a confound.** Trimming leading/trailing silence changes ASVspoof EER by many points for some architectures — the detector partly measures the recording pipeline, not the vocoder.
- **Watermarks survive benign edits well and adaptive attacks badly.** AudioSeal reports near-1.0 detection AUC under MP3, resampling, and filtering at 16 kHz; AudioMarkBench reports large TPR drops under optimization-based and regeneration attacks at comparable perceptual budgets.
- **Rule-of-three ceiling.** No published audio-detection paper validates an FPR below about $10^{-4}$, because no evaluation corpus is large enough (see §10).

## 5. What Is Not Known

- **Theoretically open.** Whether $\mathrm{TV}(p_M, p_H)$ for frontier speech generators, after a lossy channel, is bounded below by a constant. No proof either way; the TV bound says passive detection dies exactly when it is not.
- **Theoretically open.** Whether the Zhang et al. impossibility binds for audio under a *realistic* perturbation oracle. Their argument needs a quality oracle; whether perceptual audio quality metrics (ViSQOL, PESQ, MUSHRA proxies) are good enough oracles to instantiate the attack is unresolved.
- **Empirically open.** Whether an SSL-based detector trained on $\ge 10^4$ hours across $\ge 100$ generators and $\ge 20$ channels closes the in-the-wild gap. Runnable today; the data curation, not the compute, is the barrier.
- **Methodologically blocked.** Calibrated low-FPR evaluation. There is no accepted human-speech null corpus at the $10^6$-utterance scale, so a claimed $10^{-6}$ FPR is not falsifiable. EER, the field's default metric, is defined at an operating point no deployment uses.
- **Methodologically blocked.** "Synthetic" has no agreed boundary: denoised, bandwidth-extended, and codec-resynthesized human speech pass through neural generative models. Ground truth is a labelling convention, not a physical fact.

## 6. Why It Is Hard

The binding obstruction is **an evaluation that does not measure what it names**, compounded by **absent ground truth at the tail**.

- Reported EER measures separation on a corpus whose two classes differ in recording pipeline as well as in origin. The detector can achieve the number by classifying the pipeline. That is why matched EER of 2% and in-the-wild EER of 35% coexist.
- The quantity deployments need — FPR at $10^{-3}$ or below on genuine speech — cannot be estimated from any public corpus with useful confidence. The measurement is not merely noisy; it is unavailable.
- Passive detection is additionally **non-identifiable in the limit**: if a generator matches $p_H$ on the observable channel, no statistic distinguishes them. Progress in synthesis directly and provably erodes passive detection, which is the argument for proactive watermarking — but watermarking moves the problem to adversarial robustness, where the strongest known result is negative.

## 7. Current Research (as of 2026)

- **ASVspoof 5** (Wang, Yamagishi, Todisco, Evans, Delgado, Nautsch, Kinnunen and colleagues, 2024) moved to crowdsourced source data, adversarial attack conditions, and the a-DCF metric — an explicit attempt to remove the channel confound. Follow-on analyses of whether it succeeded are the live question *(frontier — verify)*.
- **Deepfake-Eval-style in-the-wild corpora** built from real circulated media rather than lab TTS; reported detector accuracy on these is far below challenge numbers *(frontier — verify specific figures)*.
- **Watermarking at generation time** (Meta FAIR's AudioSeal line; Microsoft's WavMark line) plus adaptive-attack benchmarking (AudioMarkBench).
- **Cryptographic watermarking ported to continuous signals** — instantiating pseudorandom error-correcting codes over neural-codec token streams. Plausible and largely unattempted *(frontier — verify)*.
- **Conformal and distribution-free calibration** applied to spoof scores, to convert a score into a valid p-value under exchangeability — which the in-the-wild shift violates.

## 8. Concrete Next Experiment

**Question:** does either detection path deliver a *validated* TPR at FPR $=10^{-3}$ under an adaptive, quality-constrained attack?

- **Scale.** Null corpus: $n = 3\times10^{5}$ genuine utterances (≈ 400 hours) spanning ≥ 8 channels (studio, phone, MP3 64 kbps, Opus 24 kbps, denoised, bandwidth-extended, two room IRs). Positive corpus: 50k utterances from 10 generators, 5 held out from detector training. Compute: single 8-GPU node, days, not weeks.
- **Arms.** (a) wav2vec2-XLS-R + AASIST passive detector; (b) AudioSeal watermark detector on watermarked generations; (c) **control arm** — the *same* attack pipeline applied to genuine human speech, to confirm the attack does not itself inflate FPR, plus an unattacked replicate of (a) and (b).
- **Attack.** Random-walk regeneration through an independent neural codec, constrained to $\mathrm{ViSQOL} \ge 4.0$ against the original, 20 steps.
- **Deciding number.** $\mathrm{TPR}$ at the threshold whose empirically validated $\mathrm{FPR} \le 10^{-3}$ (upper 95% bound on the 300k null). If the watermark arm holds $\mathrm{TPR} \ge 0.90$ post-attack while the passive arm falls below $0.30$, the field should route effort to proactive detection; if the watermark arm also collapses below $0.5$, the Zhang et al. impossibility is empirically operative for audio and the honest conclusion is that provable detection requires provenance metadata (C2PA-style signing), not signal analysis.

## 9. Key References

- **[Foundational]** Todisco, Wang, Vestman, Sahidullah, Delgado, Nautsch, Yamagishi, Evans, Kinnunen, Lee. *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection.* Interspeech, 2019. — arXiv:1904.05441
- **[Foundational]** Yamagishi, Wang, Todisco, Sahidullah, Patino, Nautsch, et al. *ASVspoof 2021: Accelerating Progress in Spoofed and Deepfake Speech Detection.* ASVspoof Workshop, 2021. — arXiv:2109.00537
- **[SOTA]** Jung, Heo, Tak, Shim, Chung, Lee, Yu, Evans. *AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks.* ICASSP, 2022. — arXiv:2110.01200
- **[SOTA]** Tak, Todisco, Wang, Jung, Yamagishi, Evans. *Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2vec 2.0 and Data Augmentation.* Odyssey, 2022.
- **[SOTA]** San Roman, Fernandez, Elsahar, Défossez, Furon, Tran. *Proactive Detection of Voice Cloning with Localized Watermarking.* ICML, 2024. — arXiv:2401.17264
- **[SOTA]** Chen, Wu, Liu, Ren, Tan, et al. *WavMark: Watermarking for Audio Generation.* 2023. — arXiv:2308.12770
- **[Theory]** Zhang, Edelman, Francati, Venturi, Ateniese, Barak. *Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models.* ICML, 2024. — arXiv:2311.04378
- **[Theory]** Sadasivan, Kumar, Balasubramanian, Wang, Feizi. *Can AI-Generated Text Be Reliably Detected?* 2023. — arXiv:2303.11156
- **[Theory]** Christ, Gunn. *Pseudorandom Error-Correcting Codes.* CRYPTO, 2024. — arXiv:2402.09370
- **[Theory]** Christ, Gunn, Zamir. *Undetectable Watermarks for Language Models.* COLT, 2024. — arXiv:2306.09194
- **[Empirical]** Müller, Czempin, Diekmann, Froghyar, Böttinger. *Does Audio Deepfake Detection Generalize?* Interspeech, 2022. — arXiv:2203.16263
- **[Benchmark]** Liu, Hu, Zhang, Jia, Gong, et al. *AudioMarkBench: Benchmarking Robustness of Audio Watermarking.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.06979
- **[Survey]** Yi, Tao, Fu, Wang, et al. *Audio Deepfake Detection: A Survey.* 2023. — arXiv:2308.14970

## 10. Worked Example

**The claim.** A watermark detector reports a per-utterance p-value and is deployed with threshold $p < 10^{-6}$, so "one false accusation per million clips".

**The validation.** Run it on a clean human corpus of $n = 10{,}000$ utterances and observe zero detections. By the rule of three, the 95% upper confidence bound on the true FPR is

$$\hat{\alpha}_{95} = \frac{3}{n} = \frac{3}{10^4} = 3\times10^{-4}.$$

That is **300× above** the claimed $10^{-6}$. The corpus is consistent with a detector whose real FPR is $3\times10^{-4}$, which at YouTube-scale ingest (order $10^{6}$ clips/day) is 300 false accusations per day, not one per day.

**What validation would cost.** To bound $\alpha \le 10^{-6}$ at 95% with zero observed positives requires $n = 3/10^{-6} = 3\times10^{6}$ utterances. At 5 s each that is $1.5\times10^{7}$ s $\approx 4{,}170$ hours of genuine human speech — larger than any public labelled speech corpus, and it must span every channel the deployment sees, because the null shifts with the channel. Split across 8 channel conditions, that is 520 hours per condition.

**The obstruction made visible.** The gap is not that the detector is weak. It is that the guarantee is stated at a rate three orders of magnitude below what any existing corpus can test, and the null distribution used to compute the p-value is derived from a model of "natural audio" that codec artefacts and denoisers violate. Add the adaptive attack of §8 and the TPR side is equally unvalidated. The number in the paper and the number the deployment needs are not the same number, and only one of them has been measured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*