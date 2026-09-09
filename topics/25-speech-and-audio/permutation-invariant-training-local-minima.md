---
id: 25-speech-and-audio/permutation-invariant-training-local-minima
title: "Permutation-Invariant Training Local Minima"
topic: 25-speech-and-audio
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Permutation-Invariant Training Local Minima

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/permutation-invariant-training-local-minima` · **Status:** open

## 1. Problem Statement

Permutation-invariant training (PIT) trains a separator by taking, for each training example, the minimum reconstruction loss over all $N!$ assignments of output channels to reference sources. The min makes the objective non-smooth and couples a continuous parameter search to a discrete assignment search. The open problem: **does PIT training converge to poor solutions caused by the assignment search rather than by model capacity, and if so how often, and can the failure be detected or removed?**

Three variants, different difficulty:

- **Measurement.** Given a trained separator and a training run, decide whether the run is in an assignment-induced basin. There is no accepted estimator. The reported loss is already a min over permutations, so a model stuck in a bad assignment regime and a model limited by capacity produce the same scalar.
- **Method.** Replace the hard $\min$ with something that provably or reliably avoids these basins (soft/entropic relaxation, curriculum, auxiliary anchors, order-fixing) without losing the invariance the task requires.
- **Theory.** Characterize the critical points of the lower-envelope objective $\min_{\pi}\ell_\pi$. No convergence guarantee to a global minimum exists for PIT, and no proof exists that spurious minima are *not* the binding constraint.

Solving it means: an estimator that predicts, from a training run, the gap between the achieved SI-SDR and the SI-SDR reachable with the oracle assignment; plus a method that closes that gap at fixed capacity and compute.

## 2. Formal Setting

Mixture $x=\sum_{i=1}^{N}s_i\in\mathbb{R}^T$, references $S=[s_1,\dots,s_N]$. Separator $f_\theta(x)=\hat S=[\hat s_1,\dots,\hat s_N]$. Let $\mathcal{P}_N$ be the permutations of $\{1..N\}$.

**Per-pair loss.** Scale-invariant signal-to-distortion ratio, as actually computed (zero-mean signals):
$$\alpha=\frac{\langle \hat s,s\rangle}{\|s\|^2},\qquad \mathrm{SI\text{-}SDR}(\hat s,s)=10\log_{10}\frac{\|\alpha s\|^2}{\|\hat s-\alpha s\|^2}.$$
Loss $\ell(\hat s,s)=-\mathrm{SI\text{-}SDR}$. Reported metric is *improvement*, $\mathrm{SI\text{-}SDRi}=\mathrm{SI\text{-}SDR}(\hat s,s)-\mathrm{SI\text{-}SDR}(x,s)$.

**PIT objective.**
$$\mathcal{L}_{\mathrm{PIT}}(\theta;x,S)=\min_{\pi\in\mathcal{P}_N}\frac{1}{N}\sum_{i=1}^{N}\ell\!\left(\hat s_{\pi(i)},s_i\right),\qquad \pi^\star(\theta,x)=\arg\min_\pi(\cdot).$$
Utterance-level PIT (uPIT) takes one $\pi$ per utterance; frame-level PIT takes one per frame.

**Measurable quantities.**

- *Permutation flip rate* $\Phi=\Pr_{x\sim\mathcal{D}}[\pi^\star_{t}(x)\neq\pi^\star_{t+\Delta}(x)]$ — fraction of training examples whose argmin assignment changes between checkpoints $\Delta$ steps apart. Logged by caching $\pi^\star$ per example ID.
- *Tie mass* $\tau=\Pr[\,\ell_{\pi^{(1)}}-\ell_{\pi^{(2)}}<\epsilon\,]$ with $\epsilon=0.5$ dB — fraction of examples where the two best permutations are within noise. At a tie the objective is non-differentiable; the subgradient is the convex hull of the two branches.
- *Oracle-assignment gap* $G$ — retrain with $\pi$ frozen to the assignment a reference model would choose (or to an externally supplied speaker-identity assignment), and take the SI-SDRi difference. $G>0$ is direct evidence that the assignment search, not capacity, binds.

**Assumptions and their violations.**

1. *Sources are separable and $N$ is known.* Violated in continuous meeting audio; motivates Graph-PIT (von Neumann et al., Interspeech 2021).
2. *One assignment is correct for the whole utterance.* Violated whenever a speaker is silent for a stretch: the argmin is then genuinely arbitrary on that stretch, and $\tau\to 1$.
3. *$\ell$ is bounded.* SI-SDR is unbounded below as $\|\hat s-\alpha s\|\to\|s\|$ at $\alpha\to 0$ and unbounded above at exact reconstruction; thresholded variants clip at 30 dB.
4. *$\mathcal{L}$ is differentiable a.e.* True, but the a.e. exception set is exactly where the interesting dynamics are, and its measure $\tau$ is not small early in training.

## 3. State of the Art

**Established.**

- PIT (Yu, Kolbæk, Tan, Jensen, ICASSP 2017) and uPIT (Kolbæk et al., TASLP 2017) made speaker-independent separation trainable at all. uPIT-BLSTM reached ~10.0 dB SDRi on WSJ0-2mix; the paper's own ablation shows utterance-level assignment beats frame-level assignment, which is the first direct evidence that the assignment mechanism — not the network — sets the ceiling.
- Deep clustering (Hershey et al., ICASSP 2016) sidesteps the min entirely with a permutation-invariant embedding loss; chimera++ (Wang, Le Roux, Hershey, ICASSP 2018) combines it with mask inference and beats either, again pointing at the objective rather than capacity.
- Sinkhorn-PIT (Tachibana, ICASSP 2021) replaces the $N!$ enumeration with an entropically regularized soft assignment, making $N=10$ tractable ($10!=3{,}628{,}800$ enumerations otherwise). Optimal-permutation training via Hungarian matching, $O(N^3)$ (Dovrat, Nachmani, Wolf, Interspeech 2021), scales to many speakers.
- Architecture SOTA on WSJ0-2mix (8 kHz, min): Conv-TasNet 15.3 dB SI-SDRi (Luo & Mesgarani, TASLP 2019), DPRNN 18.8 dB (Luo et al., ICASSP 2020), SepFormer 20.4 dB / 22.3 dB with dynamic mixing (Subakan et al., ICASSP 2021), TF-GridNet 23.5 dB (Wang et al., TASLP 2023). All use PIT unchanged.

**Claimed but unablated.**

- Probabilistic PIT (Yousefi, Khorram, Hansen, Interspeech 2019) softens the min into a Bayesian-risk-style sum over permutations and reports sub-1 dB gains, attributed to escaping local minima. The attribution is not isolated: no seed sweep, no flip-rate measurement, no oracle-assignment control.
- Curriculum from a 2-speaker checkpoint is standard practice for 3+ speakers and is widely said to be necessary because of PIT local minima. This is folklore-grade: it appears as a recipe detail, not as a controlled ablation.
- The 3-speaker degradation (Conv-TasNet 12.7 dB, SepFormer 17.6 dB on WSJ0-3mix) is a **benchmark number only**. Nobody has decomposed it into task difficulty versus $N!$ search difficulty.

## 4. What Is Known

- $|\mathcal{P}_N|$ grows factorially: 2, 6, 24, 120 for $N=2..5$; enumeration is the default implementation up to $N=4$ in every major toolkit. Hungarian matching reduces this to $O(N^3)$ exactly (Dovrat et al., 2021); Sinkhorn reduces it approximately (Tachibana, 2021). Both are *cost* results, not *quality* results — they do not change which minimum you land in for small $N$.
- Utterance-level beats frame-level assignment by roughly 1 dB SDRi at BLSTM scale on WSJ0-2mix (Kolbæk et al., 2017), and removes the channel-swap artifact frame-PIT produces. Reproduced across toolkits.
- Adding a permutation-free auxiliary objective (deep clustering embeddings) improves over mask-inference-with-PIT alone: 11.5 dB versus ~10 dB SDRi on WSJ0-2mix at BLSTM scale (chimera++, ICASSP 2018).
- Dynamic mixing — resampling mixtures every epoch — adds ~1.9 dB to SepFormer on WSJ0-2mix (20.4 → 22.3). It changes the data distribution, so it also changes the assignment landscape; the two effects have not been separated.
- Upper-bound estimates for single-channel 2-speaker separation put the ceiling near 23 dB SI-SDRi (Lutati, Nachmani, Wolf, Interspeech 2022). TF-GridNet at 23.5 dB is at or past that estimate, which means for $N=2$ the assignment problem, whatever its cost, is not currently binding.

## 5. What Is Not Known

- **Empirically open.** The oracle-assignment gap $G$ at modern scale. Nobody has trained a SepFormer- or TF-GridNet-class model on WSJ0-3mix / WHAMR! with the assignment frozen to a speaker-identity-derived permutation and compared to PIT under matched compute. The experiment is a few hundred GPU-hours; it has not been run.
- **Empirically open.** Seed variance of PIT runs. Published tables report one number per system. The distribution of final SI-SDRi over $\geq 10$ seeds — and whether it is bimodal, the signature of distinct basins — is unmeasured for every headline system.
- **Methodologically blocked.** "PIT local minimum" has no operational definition. $\Phi$ and $\tau$ are computable but nobody has shown they predict final performance, so there is no accepted diagnostic. Without one, every attribution in §3 is post hoc.
- **Theoretically open.** No characterization of the critical points of $\min_\pi \ell_\pi$ for any nontrivial separator class. Even for a linear separator with orthogonal sources it is not proven whether spurious local minima exist. PIT is a hard-EM / $k$-means-style alternating scheme, and hard EM is known to admit spurious fixed points — but that analogy is not a theorem about PIT.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability**. The training loss is itself a minimum over the nuisance variable, so it reports the best-case assignment and hides the assignment error by construction. There is no ground-truth $\pi$ to compare against: the correct assignment is defined only up to the model's own output ordering, which is an arbitrary internal convention the network invents during the first few thousand steps. Two runs that end at different SI-SDRi may differ in initialization, in optimizer noise, or in which channel-labeling convention they settled on — and the standard logs cannot tell these apart. Making it worse, on silent stretches the correct assignment genuinely does not exist (assumption 2 above), so a diagnostic based on flip rate has an irreducible floor that varies with the dataset's silence statistics. The compute cost is secondary: $N!$ is trivial at $N=2,3$. The problem is that the quantity to be measured is not observable in the standard training loop.

## 7. Current Research (as of 2026)

- **Permutation-free objectives.** SA-SDR (von Neumann et al., ICASSP 2022) aggregates error energy across sources before the ratio, flattening the per-source assignment sensitivity; Graph-PIT (Interspeech 2021) generalizes assignment to utterance graphs for continuous meeting audio. Paderborn and NTT continue this line.
- **Order-fixing.** Serialized output training (Kanda et al., Interspeech 2020) fixes source order by first-speech time for multi-talker ASR, removing the permutation search entirely. Whether the same trick transfers to waveform separation without loss of quality is being tested *(frontier — verify)*.
- **Generative separation.** Diffusion and flow-matching separators condition on a source index and may avoid the min altogether; several 2024–2026 systems report competitive WSJ0-2mix numbers this way *(frontier — verify)*.
- **Discrete-optimization view.** Sinkhorn / optimal-transport relaxations of PIT with annealed temperature, motivated explicitly by landscape smoothing. Tachibana's line; scattered follow-ups.
- Groups: Mitsubishi Electric Research Labs (Le Roux, Wang), Paderborn (Haeb-Umbach), NTT (Delcroix, Nakatani), CMU (Watanabe), Tel Aviv (Wolf, Nachmani), SpeechBrain (Subakan, Ravanelli).

## 8. Concrete Next Experiment

**Question.** Is the WSJ0-3mix deficit an assignment-search failure or a capacity/task failure?

**Scale.** SepFormer (~26 M parameters), WSJ0-3mix 8 kHz min, 200 epochs, no dynamic mixing, 10 seeds per arm. About 500 A100-hours total.

**Arms.**

1. *Control* — standard uPIT over all $3!=6$ permutations.
2. *Oracle-assignment* — $\pi$ frozen for the whole run to a fixed rule derived from the reference sources (sort by speaker ID hash, a rule the network cannot infer from the mixture), so the model must learn an arbitrary but consistent labeling. This is the *harder* task; if it matches or beats PIT, PIT's freedom is not buying anything.
3. *Warm-oracle* — 20 epochs with $\pi$ frozen to the sort-by-onset-time rule (learnable from the mixture), then release to standard PIT.

**Logged throughout.** $\Phi$ and $\tau$ ($\epsilon=0.5$ dB) per epoch, per arm.

**Deciding number.** Median final SI-SDRi across seeds, arm 3 minus arm 1. **If the difference exceeds 1.0 dB with non-overlapping interquartile ranges, the 3-speaker deficit is assignment-induced and PIT local minima are real and binding.** If it is under 0.3 dB, the deficit is capacity/task and the folklore is wrong. The secondary readout: whether epoch-20 $\Phi$ in arm 1 correlates with final SI-SDRi at $|r|>0.6$ across seeds — that would give the field its first validated diagnostic.

## 9. Key References

- **[Foundational]** D. Yu, M. Kolbæk, Z.-H. Tan, J. Jensen. *Permutation Invariant Training of Deep Models for Speaker-Independent Multi-talker Speech Separation.* ICASSP, 2017. — arXiv:1607.00325
- **[Foundational]** M. Kolbæk, D. Yu, Z.-H. Tan, J. Jensen. *Multitalker Speech Separation with Utterance-level Permutation Invariant Training of Deep Recurrent Neural Networks.* IEEE/ACM TASLP, 2017. — arXiv:1703.06284
- **[Foundational]** J. R. Hershey, Z. Chen, J. Le Roux, S. Watanabe. *Deep Clustering: Discriminative Embeddings for Segmentation and Separation.* ICASSP, 2016. — arXiv:1508.04306
- **[SOTA]** Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, S. Watanabe. *TF-GridNet: Integrating Full- and Sub-band Modeling for Speech Separation.* IEEE/ACM TASLP, 2023.
- **[SOTA]** C. Subakan, M. Ravanelli, S. Cornell, M. Bronzi, J. Zhong. *Attention Is All You Need in Speech Separation.* ICASSP, 2021. — arXiv:2010.13154
- **[Method]** Y. Luo, N. Mesgarani. *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation.* IEEE/ACM TASLP, 2019. — arXiv:1809.07454
- **[Method]** H. Tachibana. *Towards Listening to 10 People Simultaneously: An Efficient Permutation Invariant Training of Audio Source Separation Using Sinkhorn's Algorithm.* ICASSP, 2021.
- **[Method]** S. Dovrat, E. Nachmani, L. Wolf. *Many-Speakers Single Channel Speech Separation with Optimal Permutation Training.* Interspeech, 2021.
- **[Method]** T. von Neumann, K. Kinoshita, C. Boeddeker, M. Delcroix, R. Haeb-Umbach. *Graph-PIT: Generalized Permutation Invariant Training for Continuous Separation of Arbitrary Numbers of Speakers.* Interspeech, 2021.
- **[Method]** M. Yousefi, S. Khorram, J. H. L. Hansen. *Probabilistic Permutation Invariant Training for Speech Separation.* Interspeech, 2019.
- **[Method]** Z.-Q. Wang, J. Le Roux, J. R. Hershey. *Alternative Objective Functions for Deep Clustering.* ICASSP, 2018.
- **[Bound]** S. Lutati, E. Nachmani, L. Wolf. *SepIt: Approaching a Single Channel Speech Separation Bound.* Interspeech, 2022.

## 10. Worked Example

Two speakers, 4 s at 8 kHz, split into two 2 s halves. Assume the halves are equal-energy, $\|s_i^{(h)}\|^2=E$ for $i\in\{1,2\}$, $h\in\{1,2\}$, and the sources are mutually orthogonal. Compare three parameter configurations under utterance-level PIT with SI-SDR.

**(a) Correct.** $\hat s_1=s_1$, $\hat s_2=s_2$. Loss $\to-\infty$; in practice a trained SepFormer lands near $-20$ dB (i.e. 20 dB SI-SDR).

**(b) Mixture collapse.** $\hat s_1=\hat s_2=x=s_1+s_2$. Projection of $x$ on $s_1$ is $s_1$; residual is $s_2$ with energy $2E$ against signal energy $2E$:
$$\mathrm{SI\text{-}SDR}=10\log_{10}(2E/2E)=0\text{ dB}.$$
Both permutations tie exactly, so $\tau=1$ here.

**(c) Half-swap.** $\hat s_1=[s_1^{(1)},\,s_2^{(2)}]$, $\hat s_2=[s_2^{(1)},\,s_1^{(2)}]$. Against $s_1$: error $=[0,\;s_2^{(2)}-s_1^{(2)}]$, energy $2E$; signal energy $2E$; again $0$ dB. Identity and swap permutations both give $0$ dB.

**The obstruction, visible.** Under *frame-level* PIT, (c) is a **global** minimum — every frame is perfectly reconstructed by some permutation, loss $\to-\infty$ — while under uPIT it scores 0 dB, identical to the collapse (b). Three qualitatively different models produce two identical scalars, 0 dB and 0 dB, and the loss curve alone cannot distinguish "the network has not learned to separate" from "the network separates perfectly but cannot maintain a consistent channel assignment across 2 seconds." The two require opposite interventions: more capacity/data for (b), longer-range context or an order-fixing rule for (c). Only the flip-rate diagnostic separates them — a model at (c) shows $\Phi\approx 0.5$ when the assignment is recomputed on 2 s windows, a model at (b) shows $\tau=1$ with near-zero gradient signal. Neither statistic is logged by any standard separation recipe, which is precisely why the question in §1 is still open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*