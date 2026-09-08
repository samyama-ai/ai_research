---
id: 16-state-space-models/discretization-step-extrapolation
title: "Extrapolation of Continuous-Time Discretization Step Size"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Extrapolation of Continuous-Time Discretization Step Size

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/discretization-step-extrapolation` · **Status:** open

## 1. Problem Statement

A deep SSM layer (S4, S4D, S5, Mamba) is defined as a continuous-time linear system that is *discretized* with a step size $\Delta$ before being run as a recurrence. The advertised benefit is **resolution invariance**: the learned object is the continuous operator, so changing the sampling rate at test time should require only rescaling $\Delta$, with no retraining and no loss of accuracy.

The problem: **state precisely when that transfer holds, and how far $\Delta$ can be moved before the model breaks.**

Three variants, of very different difficulty:

- **Measurement.** Given a trained checkpoint, an input resampled by factor $r$, and the rescaled step $\Delta/r$, what is the task-metric gap versus the same model at training resolution? Runnable today; almost never reported outside audio.
- **Method.** Find a parameterization/discretization rule under which the gap is $O(\epsilon)$ for $r$ over some range — including for **selective** SSMs where $\Delta_t$ is an input-dependent function, not a constant to divide.
- **Theory.** Prove a bound on the gap in terms of $r$, the eigenvalue spectrum $\{\lambda_i\}$, the discretization rule, and the input's smoothness. Establish whether *discrete-token* models (language) have any well-posed continuous limit at all, i.e. whether $\Delta$ is a physical timescale or merely a reparameterization of the decay radius.

Solving it means: a stated $r$-range with a proved or measured gap bound, and a rule that tells a practitioner which channels to trust when $r \neq 1$.

## 2. Formal Setting

Per channel, the continuous system is

$$x'(t) = A x(t) + B u(t), \qquad y(t) = C x(t), \qquad A \in \mathbb{C}^{N \times N}.$$

**Zero-order hold (ZOH)** discretization with step $\Delta$:

$$\bar A = e^{\Delta A}, \qquad \bar B = A^{-1}(e^{\Delta A} - I)B, \qquad x_k = \bar A x_{k-1} + \bar B u_k .$$

**Euler-$B$** (used in Mamba's kernel): $\bar B = \Delta B$, with $\bar A = e^{\Delta A}$ unchanged.

*Measured quantities.*

- $\Delta$: for S4/S4D, $\exp$ of a learned scalar per channel, initialized log-uniform on $[10^{-3},10^{-1}]$. For Mamba, $\Delta_t = \mathrm{softplus}(W_\Delta u_t + b)$ — a per-token, per-channel tensor; measure it as the empirical distribution over a held-out corpus, not as a hyperparameter.
- $\lambda_i = \mathrm{eig}(A)$; the quantity that actually controls behaviour is the **dimensionless pole** $z_i = \Delta \lambda_i$. Measure $|\mathrm{Re}\, z_i|$ per channel; the memory horizon in steps is $\tau_i = 1/|\mathrm{Re}\, z_i|$.
- **Resolution-transfer gap.** For a task metric $M$ (accuracy, or bits/byte),
  $$G(r) = M\big(\theta;\ \mathcal{R}_r(u),\ \Delta/r\big) - M\big(\theta;\ u,\ \Delta\big),$$
  where $\mathcal{R}_r$ is a resampling operator (decimation, interpolation, or for text: nothing — the operator is undefined).
- **DC-gain drift**, the cheapest diagnostic: $H(0) = \bar C \bar B/(1-e^{z})$. Invariant in $\Delta$ under ZOH; not under Euler-$B$.

*Assumptions, and which are violated.*

1. **The input is piecewise constant on the grid.** Exactly what ZOH assumes; exactly what makes $\Delta$-rescaling exact. Violated for any bandlimited signal (audio, EEG) — resampling changes the within-interval shape, not just the grid.
2. **A continuous signal exists.** Violated outright for text and code. Token index is not time; $\mathcal{R}_r$ has no definition, so $G(r)$ is not measurable without inventing a resampling convention.
3. **$\Delta$ is a scalar that can be divided.** Violated for selective SSMs: $\mathrm{softplus}$ is not homogeneous, so $\Delta_t/r$ is not reachable by any input-independent edit to $W_\Delta, b$.
4. **Nonlinearity commutes with resampling.** Violated: the layer's GELU/gating and normalization act pointwise on a grid whose density just changed by $r$.

## 3. State of the Art

**Established.**

- Zero-shot sample-rate transfer works for audio with the S4 parameterization. Gu, Goel & Ré (ICLR 2022) train on 16 kHz Speech Commands and test at 8 kHz with $\Delta$ doubled: reported **96.3% → 91.3%**, while a CNN baseline (WaveGAN-D) collapses to near-chance. Reproduced in follow-ups; this is the load-bearing evidence for the whole "continuous-time" claim.
- *How to Train Your HiPPO* (Gu et al., ICLR 2023) gives the basis-projection account of why $\Delta$ has a timescale meaning: HiPPO-LegS/LagT parameterizations correspond to projecting onto orthogonal polynomial bases with an explicit window length $\propto \Delta^{-1}$.
- S4D (Gu, Gupta, Goel & Ré, NeurIPS 2022) establishes that a diagonal $A$ with log-uniform $\Delta$ initialization suffices; the $\Delta$-range, not the HiPPO structure, does most of the work on long-range tasks.

**Claimed but unablated.**

- That Mamba/Mamba-2 inherit resolution invariance. Mamba (Gu & Dao, 2023/COLM 2024) presents $\Delta$ as a selective timescale and shows a synthetic-task story, but publishes **no $\Delta$-rescaling ablation** on language; Euler-$B$ breaks the ZOH invariance argument (§10).
- That the continuous framing is necessary. Orvieto et al. (ICML 2023, LRU) match S4-class LRA numbers with a directly-parameterized complex diagonal recurrence and **no discretization at all**, which is evidence that $\Delta$ is a reparameterization of the pole radius rather than a physical step. This is a strong counter-claim and has not been decisively answered.

**Benchmark-number-only.** DeciMamba (Ben-Kish et al., 2024) reports large context extension for Mamba by pooling/manipulating the selective scan; the mechanism is described in terms of an "effective receptive field", and the connection to a formal $\Delta$-rescaling law is asserted, not derived.

## 4. What Is Known

- **ZOH is exactly $\Delta$-invariant for piecewise-constant inputs.** DC gain $-CB/\lambda$ is independent of $\Delta$; the composed transition over $r$ sub-steps equals $e^{\Delta\lambda}$ exactly.
- **Euler-$B$ is not.** DC gain ratio between step $\Delta$ and $r$ steps of $\Delta/r$ is $\frac{1-e^{z}}{r(1-e^{z/r})}$, which is $1 - \frac{z}{2}\frac{r-1}{r} + O(z^2)$. At $|z| = \Delta|\lambda| = 1$ and $r = 2$: **20% gain error** (§10). At $|z| = 0.05$: 1.2%.
- **Measured $\Delta$-ranges.** S4/S4D: $\Delta \in [10^{-3}, 10^{-1}]$ at init, $N = 64$, sequence length up to $16{,}384$ (Path-X). Mamba-370M/1.4B: $\Delta$ initialized so $\mathrm{softplus}$ output spans roughly $[10^{-3}, 10^{-1}]$, with $d_{\text{state}} = 16$.
- **Audio transfer degrades asymmetrically**: the 16→8 kHz direction (coarsening, $r<1$ in samples) is the one reported; the fine-grained direction is not.
- **Length extrapolation is separately known to fail.** Mamba trained at 2k context degrades sharply past ~2–4k tokens (DeciMamba, 2024) — a failure in the same $\Delta$-timescale currency, on data where $\Delta$ has no physical referent.
- **Memory theory bounds what can be extrapolated at all.** Wang & Xue (NeurIPS 2023) prove layer-wise-nonlinear SSMs are universal approximators but with *exponentially decaying memory*; Wang & Li (ICLR 2024) give the inverse ("curse of memory") direction. Any $\Delta$ rescale that pushes $\tau_i$ beyond the trained horizon leaves the regime these results cover.

## 5. What Is Not Known

- **Theoretically open.** No bound of the form $|G(r)| \le f(r, \{\lambda_i\}, \|u\|_{H^s})$ for a *deep, gated, normalized* stack. Single-layer LTI error analysis is classical numerical ODE theory; the composition with pointwise nonlinearity and layer norm across a grid change is not analyzed. Also open: whether a selective SSM has any continuous-time limit that is well-posed as $\Delta_t \to 0$ when $\Delta_t$ depends on $u_t$.
- **Empirically open.** Nobody has published a $\Delta$-sweep on a language-scale Mamba: take a 1.4B checkpoint, scale post-softplus $\Delta$ by $r \in \{0.25,\dots,4\}$, report bits/byte. The run costs a few GPU-hours. It has not been done at that scale in public.
- **Methodologically blocked.** For text there is no resampling operator, so $G(r)$ has no denominator. Any "$\Delta$ extrapolation" claim on language is currently a claim about an *unnamed* quantity — which is why the literature slides between "step size", "effective receptive field", and "context length" without a conversion.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability plus a missing referent**.

$(\Delta, A, B, C)$ and $(\Delta/c,\ cA,\ cB,\ C)$ give the identical discrete recurrence. Only the product $z = \Delta\lambda$ is identifiable from data. So training cannot pin $\Delta$; it pins $z$. The claim "rescale $\Delta$ and the model transfers" therefore has content only if something *outside* the discrete model fixes $\Delta$'s meaning — for audio, the sample rate does. For text, nothing does. That is not a hard measurement, it is an undefined one.

Compounding it: the two published discretizations disagree. ZOH is invariant, Euler-$B$ is invariant only to $O(z)$, and the fastest channels — the ones carrying local structure and the largest $|z|$ — are exactly where the $O(z)$ term is worst. So an aggregate accuracy number confounds "the continuous story failed" with "the fast channels were discretized sloppily".

## 7. Current Research (as of 2026)

- **Selective-scan timescale editing** for context extension: DeciMamba (Ben-Kish, Bar-Shalom, Wolf et al.), and follow-on test-time $\Delta$-scaling methods for Mamba long context *(frontier — verify the 2025 variants; several exist as preprints with inconsistent baselines)*.
- **Discretization-free recurrences**: LRU-style direct parameterization (Orvieto, De, Smith, Gu, et al., DeepMind), Mamba-2's SSD view (Dao & Gu, ICML 2024) which reads the layer as structured attention and makes $\Delta$ a scalar decay per token — deliberately weakening the continuous-time reading.
- **Genuinely continuous-time modeling** where $\Delta$ is data (irregular sampling): neural ODE/CDE lineage (Chen et al. 2018; Kidger 2022), Liquid-S4 (Hasani et al., ICLR 2023). Here the extrapolation question is well posed and least studied at scale.
- **Memory/approximation theory** for SSM length extension (S. Wang and collaborators, NUS).

## 8. Concrete Next Experiment

**Question decided:** is Mamba's $\Delta$ a transferable timescale, or a fitted decay radius?

- **Scale.** Two checkpoints: (a) Mamba-370M trained on 30B tokens of the Pile; (b) an S4D-audio model, $N=64$, trained on 16 kHz Speech Commands. Total cost: one 8×A100-day for training (a) if not reusing a public checkpoint, plus <10 GPU-hours of evaluation.
- **Intervention.** At inference, multiply the post-$\mathrm{softplus}$ $\Delta_t$ by $r \in \{0.25, 0.5, 1, 2, 4\}$, patched *after* the nonlinearity (so the rescale is exact, not approximated by a bias shift). For audio, resample the input by the matching factor. For text, hold the token sequence fixed — the null resampling — and report that this is the convention.
- **Control arms.** (i) Same sweep with $\bar B$ replaced by exact ZOH $A^{-1}(e^{\Delta A}-I)B$, no retraining. (ii) A discretization-free LRU of matched parameter count, where $r$ maps to $\lambda \mapsto \lambda^{1/r}$ — the reparameterization-only hypothesis.
- **Deciding number.** $\Delta(\text{bits/byte})$ at $r = 2$ on held-out Pile, ZOH arm minus Euler-$B$ arm. **If the ZOH arm's degradation is smaller by $\ge 0.05$ bits/byte, the continuous-time framing has real, exploitable content and the current kernel is throwing it away. If the two arms degrade within $0.01$ bits/byte of each other and both degrade like the LRU control, $\Delta$ is a reparameterization and resolution invariance does not survive outside audio.**

## 9. Key References

- **[Foundational]** Albert Gu, Karan Goel, Christopher Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022. — arXiv:2111.00396
- **[Foundational]** Albert Gu, Isys Johnson, Aman Timalsina, Atri Rudra, Christopher Ré. *How to Train Your HiPPO: State Space Models with Generalized Orthogonal Basis Projections.* ICLR 2023. — arXiv:2206.12037
- **[SOTA]** Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré. *On the Parameterization and Initialization of Diagonal State Space Models.* NeurIPS 2022. — arXiv:2206.11893
- **[SOTA]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024. — arXiv:2312.00752
- **[SOTA]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[SOTA]** Jimmy T.H. Smith, Andrew Warrington, Scott W. Linderman. *Simplified State Space Layers for Sequence Modeling.* ICLR 2023. — arXiv:2208.04933
- **[Counter-claim]** Antonio Orvieto, Samuel L. Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, Soham De. *Resurrecting Recurrent Neural Networks for Long Sequences.* ICML 2023. — arXiv:2303.06349
- **[Theory]** Shida Wang, Beichen Xue. *State-space models with layer-wise nonlinearity are universal approximators with exponentially decaying memory.* NeurIPS 2023.
- **[Theory]** Shida Wang, Qianxiao Li. *Inverse Approximation Theory for Nonlinear Recurrent Neural Networks.* ICLR 2024.
- **[Empirical]** Assaf Ben-Kish, Itamar Zimerman, Shady Abu-Hussein, Nadav Cohen, Amir Globerson, Lior Wolf, Raja Giryes. *DeciMamba: Exploring the Length Extrapolation Potential of Mamba.* ICLR 2025.
- **[Continuous-time]** Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. *Neural Ordinary Differential Equations.* NeurIPS 2018.
- **[Continuous-time]** Ramin Hasani, Mathias Lechner, Tsun-Hsuan Wang, Makram Chahine, Alexander Amini, Daniela Rus. *Liquid Structural State-Space Models.* ICLR 2023.
- **[Survey]** Matteo Tiezzi, Michele Casoni, Alessandro Betti, Marco Gori, Stefano Melacci. *State-Space Modeling in Long Sequence Processing: A Survey on Recurrence in the Transformer Era.* 2024.

## 10. Worked Example

One channel, $\lambda = -10$, $B = C = 1$, $\Delta = 0.1$, so $z = \Delta\lambda = -1$ — a fast channel, well inside S4D's initialization range. Halve the step: $r = 2$, $\Delta' = 0.05$, sequence length doubled, input held constant (the best case for ZOH — assumption 1 satisfied exactly).

**ZOH.** DC gain $= \bar B/(1-e^{z}) = \frac{(e^{z}-1)/\lambda}{1-e^{z}} = -1/\lambda = 0.1$ at both steps. **Drift: 0.0%.** Two half-steps compose to $e^{-0.5}e^{-0.5} = e^{-1}$ exactly.

**Euler-$B$** ($\bar B = \Delta B$, Mamba's kernel):

$$H_\Delta(0) = \frac{0.1}{1-e^{-1}} = \frac{0.1}{0.63212} = 0.15820, \qquad H_{\Delta/2}(0) = \frac{0.05}{1-e^{-0.5}} = \frac{0.05}{0.39347} = 0.12708 .$$

Ratio $0.12708/0.15820 = 0.8033$. **The channel's steady-state response drops 19.7% from a step-size change that the continuous-time story says should be a no-op.**

Now the same channel with $\lambda = -0.5$ ($z = -0.05$, a slow channel): ratio $= 0.048771/(2\times 0.024690) = 0.9877$, a **1.2%** drift.

**What this makes visible.** Inside a single layer with $\Delta\lambda$ spanning $[10^{-3}, 1]$ — the standard init — a factor-2 step change silently reweights channels against each other by up to 20 percentage points, monotonically in $|z|$. The fast channels, which carry local phonetic or token-adjacent structure, are exactly the ones corrupted. A downstream accuracy drop is therefore *not* evidence that continuous-time transfer fails; it is consistent with a $O(z)$ discretization artifact that the ZOH control arm in §8 removes for free. Until that arm is run, every published $\Delta$-extrapolation result on a Mamba-family model confounds the two.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*