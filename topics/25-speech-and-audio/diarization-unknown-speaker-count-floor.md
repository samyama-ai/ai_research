---
id: 25-speech-and-audio/diarization-unknown-speaker-count-floor
title: "Speaker Diarization Error Floor at Unknown Speaker Count"
topic: 25-speech-and-audio
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Speaker Diarization Error Floor at Unknown Speaker Count

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/diarization-unknown-speaker-count-floor` · **Status:** open

## 1. Problem Statement

Speaker diarization answers "who spoke when" from an audio recording with no enrollment and no prior on the number of speakers. Given a single- or multi-channel waveform, output a set of labelled speech segments; labels are arbitrary symbols, so scoring must first solve an optimal label permutation.

The observed regularity: diarization error rate (DER) is low and roughly flat for 2 speakers, then rises steeply and monotonically with the true speaker count $S$, and a large share of that rise is attributable to systems getting $S$ itself wrong. The problem is whether this is an artifact of current estimators or a floor.

Three variants, which are usually conflated:

- **Measurement.** DER is a single scalar mixing missed speech, false alarm, and speaker confusion, after a permutation that is itself chosen to minimise error. When $\hat S \neq S$ the permutation is not even a bijection, and the resulting penalty is an arbitrary convention rather than a statement about the hypothesis quality. Is there a metric under which the $S$-dependence disappears?
- **Method.** Build an estimator whose DER at $S = 6$ with $S$ unknown matches its DER at $S = 6$ with $S$ given as an oracle. The gap between these two arms is the object of interest, not the absolute DER.
- **Theory.** Under a generative model of overlapping speech with per-speaker embeddings, is the number of speakers identifiable from a finite recording, and what is the minimum-detectable-speaker condition (speaking time, SNR, embedding separation) below which no estimator can recover $S$?

Solving it means: an estimator with a bounded oracle-$S$ gap up to $S \approx 10$ on real conversational audio, or a proof that the gap is irreducible given the observation.

## 2. Formal Setting

Let $x \in \mathbb{R}^{T}$ be the waveform, discretised into frames $t = 1 \ldots T$ of 10 ms. Ground truth is a binary activity matrix $Y \in \{0,1\}^{S \times T}$, $y_{s,t} = 1$ iff speaker $s$ is active in frame $t$. Overlap is $\mathcal{O} = \{t : \sum_s y_{s,t} \ge 2\}$; **measured** as the fraction of speech frames in $\mathcal{O}$ in the reference RTTM. A system emits $\hat Y \in \{0,1\}^{\hat S \times T}$ with $\hat S$ estimated.

DER, as computed by NIST `md-eval` / `dscore`:

$$\mathrm{DER} = \frac{\sum_t \big[\max(N_t, \hat N_t) - C_t(\pi^\star)\big]}{\sum_t N_t}, \quad N_t = \sum_s y_{s,t},\ \hat N_t = \sum_s \hat y_{s,t}$$

where $C_t(\pi)$ counts frames correctly attributed under a global one-to-one map $\pi$ from hypothesis to reference speakers, and $\pi^\star$ is the Hungarian-optimal map. When $\hat S < S$, $S - \hat S$ reference speakers are unmapped and all their speech is counted missed; when $\hat S > S$, the surplus is counted false alarm. **This is the crux**: the $S$-error penalty is set by the metric's convention, not derived.

Decompose $\mathrm{DER} = \mathrm{MS} + \mathrm{FA} + \mathrm{SC}$ (missed, false alarm, confusion), all measured over the same denominator. Define the **oracle-count gap**

$$\Delta(S) = \mathbb{E}\big[\mathrm{DER} \mid \hat S \text{ free}\big] - \mathbb{E}\big[\mathrm{DER} \mid \hat S = S \text{ imposed}\big],$$

both expectations over recordings with true count $S$, same model, same checkpoint. $\Delta(S)$ is the quantity this page is about. Speaker-count accuracy is $\mathrm{SCA} = \Pr[\hat S = S]$; count bias is $\mathbb{E}[\hat S - S]$.

Assumptions the standard setting rests on, and their status:

- **Speakers are discrete and exchangeable.** Violated: the same physical speaker across a channel change (phone hand-off, near/far mic) may be embedded as two identities; children and whispered speech collapse toward each other.
- **Reference boundaries are correct to within a forgiveness collar.** Violated: DIHARD scores with a 0 s collar, so annotation jitter of 50–200 ms enters DER directly; CALLHOME conventions use 0.25 s.
- **Every speaker is a target.** Violated: background TV, laughter, and non-lexical vocalisations are annotated inconsistently across corpora.
- **Embeddings are unimodal per speaker.** Violated under vocal-effort and emotion change; this is exactly what inflates $\hat S$.

## 3. State of the Art

**Clustering pipelines (established).** VBx — Bayesian HMM over x-vector sequences with agglomerative initialisation (Landini et al., *Computer Speech & Language*, 2022) — remains a reproduced, strong baseline; it sets $S$ by a threshold on AHC plus the HMM's own speaker-pruning prior. NME-SC picks the spectral-clustering eigengap automatically and thereby $\hat S$ (Park, Han, Kumar, Narayanan, *IEEE Signal Processing Letters*, 2020). `pyannote.audio` 3.x combines a powerset-loss local EEND with clustering (Plaquet & Bredin, Interspeech 2023; Bredin, Interspeech 2023) and is the most-reproduced open pipeline.

**End-to-end (established for small $S$, claimed for large $S$).** EEND with permutation-invariant training fixes $\hat S$ at train time (Fujita et al., ASRU 2019). EEND-EDA adds an LSTM encoder–decoder that emits attractors until a stop probability fires, making $\hat S$ data-dependent (Horiguchi et al., Interspeech 2020). EEND-VC hybridises local EEND with global vector clustering (Kinoshita, Delcroix, Tawara, ICASSP/Interspeech 2021). Sortformer (Park et al., NVIDIA, 2024) replaces PIT with sort-ordered targets.

**Established vs. claimed.** Established and independently reproduced: EEND-family systems beat clustering on overlap-heavy 2-speaker telephone audio; clustering pipelines remain competitive or better as $S$ grows past ~4. Claimed but under-ablated: that attractor stopping generalises to $S$ unseen in training. Published tables show large-$S$ DER, but almost none report the matched oracle-$S$ arm, so $\Delta(S)$ is essentially unmeasured in the literature. DOVER-Lap fusion (Raj, Garcia-Perera, Huang, Watanabe, Povey, Stolcke, Dehak, SLT 2021) improves challenge numbers; whether it improves $\hat S$ or merely averages boundaries is not separately ablated in most uses.

## 4. What Is Known

- **DER grows steeply in $S$.** On CALLHOME (NIST SRE 2000, ~500 telephone conversations, 2–7 speakers), EEND-EDA reports roughly 8–9% DER on the 2-speaker subset and roughly 30–40% on 5–6 speaker subsets — a 4× degradation within one corpus and one checkpoint (Horiguchi et al., Interspeech 2020, and follow-ups). Numbers are approximate; consult the papers for exact cells.
- **Overlap and $S$ are entangled.** Overlap fraction rises with $S$ in every conversational corpus, so the $S$-effect and the overlap-effect are confounded unless controlled.
- **Absolute difficulty of unconstrained audio.** DIHARD III (Ryant et al., Interspeech 2021), 11 domains, ~34 h eval: winning full-set DER was roughly 16% with oracle SAD and roughly 20% without — an order of magnitude above ASR-style saturation.
- **Meetings remain hard.** `pyannote` 3.1's published benchmark reports roughly 22% DER on AMI single-distant-mic and roughly 25% on MSDWild, versus roughly 11% on VoxConverse — with speaker count a main axis of difference.
- **Count errors are asymmetric.** Clustering systems under-count in overlap-heavy audio (short-turn speakers absorbed); EEND-EDA under-counts beyond its training range, because the stop probability was never trained on those counts.
- **Simulated-conversation training helps.** Training on simulated conversations with realistic turn-taking rather than naive mixtures reduces DER on real data (Landini et al., ICASSP 2022), and part of the gain is better count estimation.

## 5. What Is Not Known

- **Empirically open.** $\Delta(S)$ has not been measured on a shared benchmark across the main system families with a matched oracle-$S$ control. The experiment needs one GPU-week, not a frontier budget. Nobody has run it because the oracle-$S$ arm is not a track in any challenge.
- **Methodologically blocked.** DER's treatment of $\hat S \neq S$ is a convention. There is no agreed metric that separates "found the right speakers, mislabelled frames" from "invented a speaker". Jaccard error rate (DIHARD) and DOVER-based scores change the numbers but not the underlying arbitrariness. Until this is fixed, "the error floor at unknown $S$" is partly a statement about `md-eval`.
- **Theoretically open.** No identifiability result gives conditions on speaking time $\tau_s$, embedding separation $\delta$, and SNR under which $S$ is recoverable in principle. The nearest analogues are mixture-order selection results, which assume i.i.d. samples — false here, since frames within a turn are strongly dependent, so effective sample size is turns, not frames.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability of rare speakers**.

1. A speaker with 4 s of speech in a 10-minute recording contributes about 0.7% of frames. Missing them entirely costs under 1 DER point — below the noise of annotation jitter. So the metric barely rewards finding them, while a false speaker of the same size costs the same tiny amount. **DER is nearly blind to exactly the decision that defines $\hat S$.**
2. The effective sample size for a rare speaker is the number of turns, often 1–3. No estimator distinguishes "one new speaker" from "one speaker in an unusual vocal state" from that evidence, and the ground truth for which of these is happening does not exist in the annotation.
3. $S$, overlap fraction, turn duration, and channel diversity co-vary in every real corpus, so a measured $S$-effect is never a clean $S$-effect without synthetic control.

## 7. Current Research (as of 2026)

- **Attractor and target-speaker methods** extending EEND to larger $S$: EEND-VC and TS-VAD variants, and the Sortformer/streaming line at NVIDIA *(frontier — verify current results)*.
- **LLM post-processing.** DiarizationLM (Wang et al., Google, 2024) rewrites diarized transcripts with an LLM, correcting label assignment using lexical cues; whether it corrects $\hat S$ or only reassigns within a fixed $\hat S$ is the open ablation.
- **Challenge tracks** — CHiME-7/8 DASR (Cornell et al.) push multi-channel, unknown-$S$ meeting audio and score joint diarization+ASR (tcpWER), which sidesteps some DER conventions.
- **Simulated-conversation and count-balanced training data** to give attractor stopping supervision across the full $S$ range *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Is the DER rise with $S$ caused by count errors, or would it persist with $S$ given?

**Scale.** Two corpora: (a) CALLHOME full (~500 conversations, $S = 2\ldots7$); (b) a synthetic set of 600 conversations built from VoxCeleb speakers, 10 min each, with $S \in \{2,3,4,6,8,10\}$ (100 each) and **overlap fraction pinned at 12% for every $S$** — this is the control that breaks the $S$/overlap confound. One GPU-week on a single A100 for inference across three systems.

**Arms.** Three systems: VBx, `pyannote` 3.1, EEND-EDA (or Sortformer). Each run twice:
- **Free arm** — $\hat S$ estimated by the system.
- **Oracle arm (control)** — $\hat S = S$ imposed: fix AHC/spectral cluster count for the clustering systems; force the attractor decoder to emit exactly $S$ attractors.

Report MS/FA/SC decomposition and SCA per $S$ cell.

**Deciding number.** $\Delta(8) = \mathrm{DER}_{\text{free}}(S{=}8) - \mathrm{DER}_{\text{oracle}}(S{=}8)$ on the overlap-pinned synthetic set.

- $\Delta(8) < 3$ DER points → the floor is **not** a counting problem; representation and overlap handling own it, and count estimation is a solved-enough subroutine.
- $\Delta(8) > 8$ DER points → counting is the dominant term and deserves a dedicated loss and a dedicated metric.

Either outcome redirects the field; the ambiguity today is that nobody publishes both arms.

## 9. Key References

- **[Foundational]** Fujita, Kanda, Horiguchi, Xue, Nagamatsu, Watanabe. *End-to-End Neural Speaker Diarization with Self-Attention.* ASRU, 2019.
- **[Foundational]** Horiguchi, Fujita, Watanabe, Xue, Nagamatsu. *End-to-End Speaker Diarization for an Unknown Number of Speakers with Encoder-Decoder Based Attractors.* Interspeech, 2020.
- **[SOTA]** Landini, Profant, Diez, Burget. *Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks.* Computer Speech & Language, 2022.
- **[SOTA]** Plaquet, Bredin. *Powerset multi-class cross entropy loss for neural speaker diarization.* Interspeech, 2023.
- **[SOTA]** Bredin. *pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe.* Interspeech, 2023.
- **[SOTA]** Kinoshita, Delcroix, Tawara. *Integrating end-to-end neural and clustering-based diarization: getting the best of both worlds.* ICASSP, 2021.
- **[Benchmark]** Ryant, Singh, Bhagat, Church, Cieri, Du, Ganapathy, Liberman. *The Third DIHARD Diarization Challenge.* Interspeech, 2021.
- **[Method]** Park, Han, Kumar, Narayanan. *Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap.* IEEE Signal Processing Letters, 2020.
- **[Method]** Raj, Garcia-Perera, Huang, Watanabe, Povey, Stolcke, Dehak. *DOVER-Lap: A Method for Combining Overlap-Aware Diarization Outputs.* IEEE SLT, 2021.
- **[Method]** Landini, Lozano-Diez, Diez, Burget. *From Simulated Mixtures to Simulated Conversations as Training Data for End-to-End Neural Diarization.* Interspeech, 2022.
- **[Frontier]** Wang, Zhang, Cui, Han, et al. *DiarizationLM: Speaker Diarization Post-Processing with Large Language Models.* Google, 2024.
- **[Survey]** Park, Kanda, Dimitriadis, Han, Watanabe, Narayanan. *A Review of Speaker Diarization: Recent Advances with Deep Learning.* Computer Speech & Language, 2022.

## 10. Worked Example

A 10-minute meeting, $T = 60{,}000$ frames. Reference: $S = 6$. Speaking times: A 240 s, B 180 s, C 90 s, D 45 s, E 20 s, F **6 s** (two short interjections). Total speech 581 s; F is **1.03%** of speech.

The system returns $\hat S = 5$, merging F into A (nearest embedding — F interjects over A's turns, so F's segments are partly overlap-corrupted). Everything else is perfect.

- Unmapped reference speaker F → 6 s missed.
- $\mathrm{DER} = 6 / 581 = \mathbf{1.03\%}$.

Now the same system, on the same file, instead splits A into A1/A2 across a cough-induced vocal change, giving $\hat S = 7$, and F is found correctly. Suppose 40 s of A is assigned to the surplus cluster A2.

- $\mathrm{DER} = 40 / 581 = \mathbf{6.88\%}$.

**The obstruction, visible.** The second system is arguably *better* — it found all 6 real speakers — and scores 6.7× worse. The first system deleted a speaker entirely and paid 1 DER point, less than the annotation jitter at a 0 s collar. Any gradient signal, hyperparameter sweep, or challenge leaderboard driven by DER pushes systems toward **under-counting**: dropping a rare speaker is nearly free, inventing one is expensive. That is why $\hat S$ is systematically biased low in overlap-heavy audio, and why "the error floor at unknown speaker count" cannot be attacked without first fixing the metric — the number does not measure the thing its name implies.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*