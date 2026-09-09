---
id: 25-speech-and-audio/polyphonic-sound-event-detection-limit
title: "Environmental Sound Event Detection Under Polyphony"
topic: 25-speech-and-audio
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Environmental Sound Event Detection Under Polyphony

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/polyphonic-sound-event-detection-limit` · **Status:** empirically-open

## 1. Problem Statement

Sound event detection (SED) maps an audio recording to a set of labelled temporal intervals: *(class, onset, offset)*. **Polyphony** is the number of events sounding at the same instant. Monophonic detection is close to solved on constrained vocabularies; detection degrades as polyphony rises, and the shape of that degradation curve is the problem.

Three variants, routinely conflated:

- **Measurement.** Does a metric exist that scores a polyphonic detection without confounding the polyphony effect with class prior, event duration, and annotation collar? Current metrics penalise short overlapping events far more than long isolated ones, so the reported "polyphony penalty" is partly an artifact of the scoring rule.
- **Method.** Given fixed data and a fixed backbone, does explicit source separation, permutation-invariant multi-track decoding, or a larger self-supervised encoder recover the accuracy lost to overlap? Reported gains exist but are not cleanly ablated against the polyphony variable.
- **Theory.** Is there an information-theoretic floor — a polyphony degree $P^\star$ beyond which the mixture no longer identifies the constituent events from a single channel, for a given class vocabulary and SNR distribution? No such bound is known.

Solving it means: a degradation curve $\mathrm{score}(P)$ measured on data where $P$ is the *only* varying factor, a decomposition of that curve into label noise, metric artifact, and model error, and a method whose curve is flat where the human curve is flat.

## 2. Formal Setting

Let $x \in \mathbb{R}^{T_s}$ be a waveform at sampling rate $f_s$ (16 kHz throughout DCASE Task 4). Let $\mathcal{C}$ be the class vocabulary, $C = |\mathcal{C}|$ (DESED: $C=10$; AudioSet-Strong: $C \approx 456$).

**Ground truth** is a set $Y = \{(c_i, t_i^{\mathrm{on}}, t_i^{\mathrm{off}})\}_{i=1}^{N}$. Its frame indicator on a grid of $T$ frames of hop $h$ (typically $h=$ 16 ms after a 128-bin log-mel front end) is $y_{c,t}\in\{0,1\}$.

**Instantaneous polyphony** and its summaries, measured directly from the annotation:

$$P(t) = \sum_{c\in\mathcal{C}} y_{c,t}, \qquad \bar{P} = \frac{1}{T}\sum_{t} P(t), \qquad P_{\max} = \max_t P(t).$$

Note $P(t)$ counts *labelled* events. Unlabelled background sources — traffic, wind, an untargeted voice — are not counted, so $P$ under-reports acoustic overlap. This is the first assumption violated in practice.

**System output** is $\hat{y}_{c,t}\in[0,1]$, thresholded at $\theta$ and median-filtered with a per-class window $w_c$ to produce intervals. The post-processing $(\theta, w_c)$ is itself tuned, so a "model" comparison is really a model-plus-decoder comparison.

**Collar-based F1** (`sed_eval`, Mesaros et al. 2016) counts a detection as true positive if onset falls within $\pm 200$ ms and offset within $\max(200\ \mathrm{ms}, 0.2\,\mathrm{dur})$. **Intersection-based** scoring (Bilen et al. 2020) instead requires
$$\frac{|\hat{e}\cap e|}{|e|}\ge \rho_{\mathrm{GTC}}, \qquad \frac{|\hat{e}\cap e|}{|\hat{e}|}\ge \rho_{\mathrm{DTC}},$$
with $\rho_{\mathrm{GTC}},\rho_{\mathrm{DTC}}$ typically $0.5$ (PSDS1) or $0.1$ (PSDS2). PSDS integrates the per-class true-positive rate against effective false-positive rate $e$ and penalises cross-class variance:
$$\mathrm{PSDS} = \frac{1}{e^{\max}_{\mathrm{FP}}}\int_0^{e^{\max}_{\mathrm{FP}}}\Big(\mu_{\mathrm{TP}}(e) - \alpha_{\mathrm{ST}}\,\sigma_{\mathrm{TP}}(e)\Big)\, de,$$
with $e^{\max}_{\mathrm{FP}} = 100$ FP/hour. The quantity the problem needs is the **polyphony-conditioned** score, e.g. $F_1 \mid P(t_i^{\mathrm{on}}) = p$ or $\mathrm{PSDS}\mid \bar{P}\in[p, p{+}1)$ — computed by partitioning the evaluation set on $P$ and rescoring.

Assumptions and their status:

| Assumption | Status |
|---|---|
| Onsets/offsets are well defined points | Violated — annotator onset std is hundreds of ms for diffuse classes (vacuum, running water) |
| Labels are complete (no missed events) | Violated — weak/crowdsourced labels are incomplete; $P$ is a lower bound |
| Classes are mutually exclusive at a frame | Violated by design — SED is multi-label |
| Synthetic soundscapes match real overlap statistics | Violated — Scaper-generated DESED mixtures have engineered, not natural, co-occurrence priors |
| Evaluation-set polyphony distribution matches deployment | Untested for nearly every deployment |

## 3. State of the Art

**Empirical SOTA (DESED, DCASE Task 4).** Fine-tuning a self-supervised audio transformer into a CRNN with a mean-teacher semi-supervised loss is the dominant recipe. ATST-SED (Shao, Li, Li; ICASSP 2024) reports PSDS1 $\approx 0.583$ and PSDS2 $\approx 0.810$ on the DESED real validation set — the largest single jump on this benchmark in five years, from a two-stage fine-tuning schedule for the pretrained encoder. BEATs (Chen et al., ICML 2023) embeddings became the official DCASE 2023 baseline component, moving baseline PSDS1 from roughly $0.35$ to roughly $0.49$. These are **benchmark numbers**; neither is accompanied by a polyphony-stratified breakdown in the primary paper.

**Established.** Intersection-based PSDS is more stable than collar F1 under post-processing changes (Bilen et al., ICASSP 2020) — ablated and reproduced across DCASE editions. Mean-teacher semi-supervision beats supervised-only training on DESED — reproduced by every top entry since 2019. Temporally-strong labels beat weak labels for localisation: Hershey et al. (ICASSP 2021) show strong AudioSet labels improve both tagging and localisation on the same audio.

**Claimed but unablated.** That sound separation helps polyphonic SED. Turpault et al. (DCASE 2020) report gains from separating with a TDCN++ before detection, and the FUSS dataset (Wisdom et al., ICASSP 2021) exists to support this — but the gain is reported as an aggregate PSDS delta, not as a flattening of $\mathrm{score}(P)$, and separation-based front ends have not held the top of the DCASE leaderboard since. That larger pretrained encoders help *because* they resolve overlap, rather than because they classify better in the clean case, is asserted, not shown.

## 4. What Is Known

- **Degradation with polyphony is real and monotone in the measured range.** Ronchini and Serizel (ICASSP 2022) rescored DCASE 2021 Task 4 submissions on synthetic soundscapes stratified by overlap and found systematic F1 loss as the number of concurrent events grows, over $C=10$ classes and 10-second clips. The effect is present in every system tested, at the scale of tens of F1 points between isolated and heavily overlapped conditions for the worst classes.
- **Non-target (unlabelled) events hurt.** Ronchini et al. (DCASE 2021 Workshop) show adding non-target sources to synthetic soundscapes lowers detection of target events — i.e. acoustic polyphony matters even when labelled polyphony $P$ is held fixed.
- **The annotation is not a gold standard.** Martín-Morató and Mesaros (EUSIPCO 2021; TASLP 2023) show wide inter-annotator disagreement on both presence and boundaries, and build MAESTRO from crowdsourced weak labels with explicit annotator-competence estimation. Human-vs-human agreement on boundaries is comparable to the differences separating leaderboard systems.
- **The scoring rule interacts with overlap.** Ebbers, Haeb-Umbach and Serizel (Interspeech 2024, "Sound Event Bounding Boxes") show that collar and threshold conventions systematically mis-score events whose boundaries are ambiguous — the same class of events that dominate high-polyphony segments.
- **Scale.** DESED real: ~10 s clips, $C=10$, roughly 1.6 k weakly labelled, 14 k unlabelled, 1.6 k synthetic-strong. AudioSet-Strong: ~67 k segments, $C\approx 456$. Neither was designed to vary polyphony as a controlled factor.

## 5. What Is Not Known

- **Theoretically open.** No identifiability result for polyphonic SED. Nobody has stated conditions on class spectro-temporal supports, SNR, and channel count under which $Y$ is recoverable from $x$ — and no converse showing when it is not. The analogous separation literature has permutation and over-determination results, but nothing transferred to *detection* of a fixed vocabulary.
- **Empirically open.** The decomposition of $\mathrm{score}(P)$ into (i) model error, (ii) label incompleteness, (iii) metric artifact. The experiment is runnable today with existing tools (Scaper, PSDS, released checkpoints) and has not been run at a scale that separates the three. Also open: whether the ATST/BEATs-class gains flatten the polyphony curve or shift it uniformly.
- **Methodologically blocked.** The *human* reference curve. There is no accepted protocol for measuring human polyphonic detection accuracy under the same interface, vocabulary, and time budget as the machine, so "the model is below the ceiling" cannot currently be stated as a number.

## 6. Why It Is Hard

**Confounded measurement, compounded by absent ground truth.** Polyphony is not manipulable in real recordings without changing everything else: high-$P$ segments in DESED are also the segments with short events, transient classes, and lower per-source SNR. Isolating $P$ requires synthesis; synthesis introduces a distribution shift whose size is unmeasured, so any polyphony curve is either confounded (real audio) or off-distribution (synthetic audio).

Second obstruction: **the label floor moves with $P$**. Annotators miss more events in denser mixtures, so the reference itself degrades with polyphony. A system that correctly detects a true but unlabelled event at $P=4$ is scored as a false positive. Model error and annotation error are therefore *aliased* along the exact axis being studied — the measurement cannot separate them without a reference of known completeness.

## 7. Current Research (as of 2026)

- **Pretrained-encoder fine-tuning.** ATST/BEATs-into-CRNN pipelines, with schedule engineering to avoid destroying the pretrained representation (Li's group at Westlake; DCASE Task 4 entrants broadly).
- **Metric reform.** Ebbers and Haeb-Umbach (Paderborn) with Serizel (Inria Nancy) on threshold-independent and bounding-box scoring, aimed at removing collar arbitrariness.
- **Label-quality modelling.** Mesaros and Martín-Morató (Tampere) on annotator competence and soft labels; DCASE 2024–25 Task 4 combined DESED and MAESTRO with heterogeneous label quality, forcing systems to handle two annotation regimes at once.
- **Separation-informed detection.** Google/Inria line from FUSS and the 2020 separation baseline; currently a minority approach. *(frontier — verify: whether any 2025–26 entrant re-established separation front ends as competitive.)*
- **Audio-language and open-vocabulary SED.** Text-queried detection over unbounded vocabularies, which raises $C$ and therefore raises attainable $P$ sharply. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question decided:** how much of the polyphony penalty is model error versus label/metric artifact?

**Scale.** Build 3 000 ten-second soundscapes with Scaper from the DESED foreground/background banks, 500 per polyphony level $P_{\max}\in\{1,2,3,4,5,6\}$. Hold constant across levels: per-class event count marginal, event duration distribution, per-event SNR (fixed at 6 dB above background), and background type. Ground truth is exact by construction — no annotator, no missing events. Evaluate three released checkpoints: the DCASE 2023 CRNN baseline, the BEATs-augmented baseline, and ATST-SED.

**Control arms.** (1) **Oracle-mixture control**: score each system on the *isolated stems* of the same soundscapes, one event per clip, same SNR against the same background — this gives the $P=1$ upper bound with identical acoustic content. (2) **Metric control**: rescore every condition with both PSDS1 ($\rho=0.5$) and frame-level macro-F1, which has no collar; any difference in the two degradation slopes is metric artifact, not model error.

**The deciding number.** The slope
$$\Delta = \frac{\mathrm{PSDS1}(P_{\max}{=}1) - \mathrm{PSDS1}(P_{\max}{=}6)}{5}$$
per additional concurrent event, reported for each system, alongside the same slope computed on frame-level F1. If $\Delta_{\mathrm{ATST}} < \Delta_{\mathrm{CRNN}}$ by more than 0.01 PSDS/event, pretrained encoders genuinely resolve overlap. If $\Delta$ is equal across systems while absolute scores differ, the encoders shift the curve without flattening it, and the polyphony penalty is untouched by the last five years of progress. If the frame-F1 slope is under half the PSDS1 slope, the penalty is substantially a boundary-scoring artifact.

## 9. Key References

- **[Foundational]** A. Mesaros, T. Heittola, T. Virtanen. *Metrics for Polyphonic Sound Event Detection.* Applied Sciences, 6(6):162, 2016.
- **[Foundational]** J. F. Gemmeke, D. P. W. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, M. Ritter. *Audio Set: An Ontology and Human-Labeled Dataset for Audio Events.* ICASSP, 2017.
- **[Metric]** Ç. Bilen, G. Ferroni, F. Tuveri, J. Azcarreta, S. Krstulović. *A Framework for the Robust Evaluation of Sound Event Detection.* ICASSP, 2020. — arXiv:1910.08440
- **[Dataset]** N. Turpault, R. Serizel, A. Parag Shah, J. Salamon. *Sound Event Detection in Domestic Environments with Weak Labeling and Soundscape Synthesis.* DCASE Workshop, 2019.
- **[Polyphony]** I. Martín-Morató, A. Mesaros. *What Is the Ground Truth? Reliability of Multi-Annotator Data for Audio Tagging.* EUSIPCO, 2021.
- **[Polyphony]** F. Ronchini, R. Serizel. *A Benchmark of State-of-the-Art Sound Event Detection Systems Evaluated on Synthetic Soundscapes.* ICASSP, 2022.
- **[Polyphony]** F. Ronchini, R. Serizel, N. Turpault, S. Cornell. *The Impact of Non-Target Events in Synthetic Soundscapes for Sound Event Detection.* DCASE Workshop, 2021.
- **[Separation]** N. Turpault, S. Wisdom, H. Erdogan, J. R. Hershey, R. Serizel, E. Fonseca, P. Seetharaman, J. Salamon. *Improving Sound Event Detection in Domestic Environments Using Sound Separation.* DCASE Workshop, 2020.
- **[Separation]** S. Wisdom, H. Erdogan, D. P. W. Ellis, R. Serizel, N. Turpault, E. Fonseca, J. Salamon, J. R. Hershey. *What's All the FUSS About Free Universal Sound Separation Data?* ICASSP, 2021.
- **[SOTA]** N. Shao, X. Li, X. Li. *Fine-Tune the Pretrained ATST Model for Sound Event Detection.* ICASSP, 2024.
- **[SOTA]** S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, F. Wei. *BEATs: Audio Pre-Training with Acoustic Tokenizers.* ICML, 2023.
- **[Labels]** S. Hershey, D. P. W. Ellis, E. Fonseca, A. Jansen, C. Liu, R. C. Moore, M. Plakal. *The Benefit of Temporally-Strong Labels in Audio Event Classification.* ICASSP, 2021.
- **[Metric]** J. Ebbers, R. Haeb-Umbach, R. Serizel. *Sound Event Bounding Boxes.* Interspeech, 2024.
- **[Survey]** T. Virtanen, M. D. Plumbley, D. P. W. Ellis (eds.). *Computational Analysis of Sound Scenes and Events.* Springer, 2018.

## 10. Worked Example

Take a single 10-second DESED-style soundscape, $C=10$, background "kitchen", with four foreground events:

| Event | Class | Onset (s) | Offset (s) | Dur (s) |
|---|---|---|---|---|
| $e_1$ | Running water | 0.0 | 8.0 | 8.0 |
| $e_2$ | Dishes | 3.1 | 3.5 | 0.4 |
| $e_3$ | Dishes | 3.6 | 4.0 | 0.4 |
| $e_4$ | Speech | 3.0 | 5.5 | 2.5 |

Between 3.1 s and 4.0 s, $P(t)=3$. Mean polyphony $\bar{P} = (8.0+0.4+0.4+2.5)/10 = 1.13$ — the clip looks nearly monophonic by its summary statistic while containing a genuinely 3-fold-overlapped second.

Now score a plausible system output: it detects running water (0.2–8.3 s), speech (3.0–5.6 s), and **one** dishes event (3.1–4.0 s) — it merged $e_2$ and $e_3$ across the 100 ms gap, because its median filter for *Dishes* has $w_c = 250$ ms.

- **Collar F1:** water TP, speech TP. The merged dishes detection matches $e_2$'s onset (within 200 ms) but its offset is 4.0 s against $e_2$'s 3.5 s, tolerance $\max(200\ \mathrm{ms}, 0.2\times0.4) = 200$ ms → offset error 500 ms → **false positive**, and $e_2$, $e_3$ are both **false negatives**. Score: 2 TP, 1 FP, 2 FN → $F_1 = 2\cdot2/(2\cdot2+1+2) = 0.571$.
- **PSDS1 ($\rho_{\mathrm{DTC}}=\rho_{\mathrm{GTC}}=0.5$):** the detection covers $e_2$ fully, so GTC $=1.0\ge0.5$; DTC $= 0.4/0.9 = 0.44 < 0.5$ → still fails, by 0.06.
- **PSDS2 ($\rho=0.1$):** DTC $=0.44\ge0.1$, GTC $=1.0$ → the same detection is now a true positive against both $e_2$ and $e_3$. Score: 4 TP, 0 FP, 0 FN → $F_1 = 1.0$.

The obstruction is visible: the *same* output on the *same* audio scores 0.571 or 1.0 depending only on the intersection thresholds, and the swing comes entirely from the 100 ms gap between two overlapping short events — i.e. from the high-polyphony instant. A human annotator listening once would very likely have written a single *Dishes* interval 3.1–4.0 s, making the system's output the correct one and the reference wrong. Any claim of the form "accuracy falls by $x$ points per additional concurrent event" is, at this granularity, a statement about $\rho_{\mathrm{DTC}}$ and $w_c$ as much as about the model. That is why the experiment in §8 must report the slope under two scoring rules, and why synthetic ground truth — where $e_2$ and $e_3$ are separate by construction — is the only way to know which reading is right.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*