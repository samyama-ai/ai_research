---
id: 34-diffusion-generative/hybrid-continuous-discrete-generative-modeling
title: "Continuous-Discrete Hybrid Modeling of Mixed-Type Data"
topic: 34-diffusion-generative
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Continuous-Discrete Hybrid Modeling of Mixed-Type Data

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/hybrid-continuous-discrete-generative-modeling` · **Status:** open

## 1. Problem Statement

A single datum carries both continuous and categorical coordinates: a molecule is 3D positions plus atom types and bond orders; a table row is age and income plus occupation and ZIP code; a protein backbone is $SE(3)$ frames plus amino-acid identity. Generative models built on stochastic interpolation need a corruption process, and the two coordinate families admit different natural ones — Gaussian noise on $\mathbb{R}^{d_c}$, a continuous-time Markov chain (CTMC) on a finite set.

The problem: **build a joint generative model of $p(x^c, x^d)$ whose two corruption processes are coupled in a principled way, and say what "principled" means numerically.**

Three variants, with different difficulty:

- **Method.** Construct a sampler that beats both single-modality reductions — everything-continuous (relax categories into $\mathbb{R}$) and everything-discrete (quantize the reals) — on joint fidelity, not on marginals. Largely solved in engineering terms; unsolved in the sense that no reduction is known to be dominated.
- **Measurement.** Define a scalar that says the two schedules are *aligned* at every $t$, and that is invariant to reparametrizations of the continuous block. No accepted definition exists.
- **Theory.** Characterize when a factorized forward process $q(x_t \mid x_0) = q^c(x^c_t \mid x^c_0)\, q^d(x^d_t \mid x^d_0)$ admits a reverse process whose denoiser is learnable at a rate not worse than the harder marginal. Open.

Solving it means: an alignment criterion that is invariant to smooth reparametrization, a schedule derived from it, and a demonstration that the derived schedule beats a tuned scalar sweep at fixed compute.

## 2. Formal Setting

Data $x_0 = (x_0^c, x_0^d)$, with $x_0^c \in \mathbb{R}^{d_c}$ and $x_0^d \in \prod_{j=1}^{d_d} [K_j]$, drawn from $p_{\text{data}}$.

**Continuous channel.** Variance-preserving forward kernel
$$q^c(x_t^c \mid x_0^c) = \mathcal{N}\!\left(\alpha_t x_0^c,\ \sigma_t^2 I\right), \qquad \mathrm{SNR}(t) = \alpha_t^2/\sigma_t^2 .$$
Measured as: the empirical $\sigma_t$ used by the sampler, times the per-dimension data standard deviation after whatever normalization the codebase applies.

**Discrete channel.** CTMC with rate matrix $R_t \in \mathbb{R}^{K\times K}$. For the masking (absorbing) case, $q^d(x_t^d = \texttt{[M]} \mid x_0^d) = \gamma_t$ with $\gamma_0=0$, $\gamma_1=1$. Measured as: the realized mask fraction at time $t$ over a sample batch.

**Alignment functional.** The natural candidate is per-modality retained information,
$$\mathcal{I}^c(t) = I(x_0^c; x_t^c), \qquad \mathcal{I}^d(t) = I(x_0^d; x_t^d) = (1-\gamma_t)\, H(x_0^d) \ \ \text{(masking, exactly)},$$
with an *alignment residual* $\Delta(t) = \mathcal{I}^c(t)/\mathcal{I}^c(0) - \mathcal{I}^d(t)/\mathcal{I}^d(0)$. For a Gaussian channel, $\mathcal{I}^c$ is measured through the I-MMSE identity $\frac{d}{d\,\mathrm{SNR}} I = \tfrac12\,\mathrm{mmse}(\mathrm{SNR})$, so $\mathcal{I}^c(t)$ is estimable from the trained denoiser's own MSE curve — no extra estimator needed.

**Objective.** The joint ELBO decomposes as
$$-\log p_\theta(x_0) \le \underbrace{\mathbb{E}\!\int_0^1 w(t)\,\|\hat{x}^c_\theta(x_t,t)-x_0^c\|^2 dt}_{\text{continuous}} + \underbrace{\mathbb{E}\!\int_0^1 \sum_j \tfrac{\gamma_t'}{1-\gamma_t}\,\mathrm{CE}\!\left(\hat{p}_\theta^{(j)}(x_t,t),\, x_{0,j}^d\right) dt}_{\text{discrete}} + \text{const},$$
two terms in incommensurable units (squared data units vs. nats) joined by a free scalar $\lambda$ in every implementation.

**Assumptions, and which fail.**
- *Forward factorization across modalities.* Assumed everywhere; false for the data — atom type and bond length are strongly dependent, so an independently-corrupted $x_t$ is off the data manifold in a correlated way the model must undo.
- *Reparametrization invariance of $\mathcal{I}^c$.* **Violated.** $I(x_0^c; x_t^c)$ is invariant to invertible maps of $x_0^c$ only if the noise is added in the same coordinates. Cartesian vs. internal coordinates vs. a learned VAE latent give different $\mathcal{I}^c(t)$ for the same distribution. This is the crux of §6.
- *$\lambda$ is a nuisance parameter.* Assumed tunable-and-forget; in practice it interacts with the schedule, so a schedule ablation at fixed $\lambda$ is confounded.

## 3. State of the Art

**Established (reproduced, ablated).**
- **EDM** (Hoogeboom, Satorras, Vignac, Welling, ICML 2022): $E(3)$-equivariant joint diffusion over 3D coordinates and one-hot atom types, with the categorical block relaxed into $\mathbb{R}$ and diffused under a *single* Gaussian schedule. Widely reproduced. Establishes that the everything-continuous reduction is a strong baseline.
- **Masked discrete diffusion** (MDLM, Sahoo et al., NeurIPS 2024; MD4, Shi et al., NeurIPS 2024): the discrete ELBO reduces to a time-weighted cross-entropy, independently derived twice. This is what makes a clean hybrid ELBO writable at all.
- **D3PM** (Austin et al., NeurIPS 2021) and **SEDD** (Lou, Meng, Ermon, ICML 2024) fix the discrete side: score-entropy training closed most of the perplexity gap to autoregressive models.

**Claimed but not independently ablated.**
- **MiDi** (Vignac et al., ECML PKDD 2023): joint continuous coordinates + discrete atom types and bond orders, with a per-modality noise schedule. The paper reports large gains over EDM on GEOM-DRUGS. The gain is attributed to the discrete treatment, but the schedule choice is not isolated from the adaptive-noise and architecture changes shipped alongside it.
- **Multiflow** (Campbell, Yim, Barzilay, Rainforth, Jaakkola, ICML 2024): explicitly multimodal flows on mixed state spaces for protein sequence-structure co-design. The multimodal factorization is derived; the *relative* schedule between the $SE(3)$ and sequence components is set by hand.
- **TabSyn** (Zhang et al., ICLR 2024): sidesteps hybridity by encoding mixed-type rows into a single continuous VAE latent and diffusing there. Reports large column-wise density-error reductions over baselines on 6 tables — a single-paper benchmark number.

**Benchmark-number-only.** Nearly all mixed-type tabular results (TabDDPM, Kotelnikov et al., ICML 2023; TabSyn) are reported as machine-learning-efficiency deltas on small public tables, with dataset-selection and hyperparameter budgets not matched across arms.

## 4. What Is Known

- Discrete diffusion is not a handicap per se: SEDD reports 25–75% perplexity reduction over prior discrete diffusion and reaches parity-band with GPT-2 at 355M parameters on standard LM benchmarks (ICML 2024).
- The everything-continuous reduction works at small scale: EDM on QM9 (~130k molecules, ~5M parameters) reports about 98.7% atom stability and 82.0% molecule stability, against 99.0%/95.2% for the data itself. The gap is real and the reduction is the cause candidate, but not the demonstrated cause.
- **Analog Bits** (Chen, Zhang, Hinton, ICLR 2023) shows a purely continuous model over binary-encoded categories is competitive on discrete image and captioning tasks — evidence that the continuous reduction is not intrinsically broken.
- The absorbing-state hybrid ELBO is exact and simulation-free in both blocks; no score-matching approximation is needed on the discrete side (MDLM/MD4, 2024).
- Latent unification (TabSyn) beats explicit hybrid handling on small tables. That is one modality family, at $10^4$–$10^5$ rows, not a general result.

## 5. What Is Not Known

- **Methodologically blocked.** There is no reparametrization-invariant alignment criterion. $\mathcal{I}^c(t)$ depends on the coordinate system in which noise is injected, and no principle selects that system. Until this is fixed, "the schedules are aligned" is not a measurable claim, and every reported schedule is a tuned hyperparameter wearing a derivation.
- **Empirically open.** Whether native hybrid corruption beats the best-tuned everything-continuous reduction *at matched compute and matched hyperparameter search budget*. Every published comparison varies architecture simultaneously. The experiment is a 1–10 GPU-day sweep on QM9; nobody has published it as a controlled arm.
- **Empirically open.** Whether hybrid gains survive scale. All controlled evidence is below $10^8$ parameters and below $10^6$ training examples.
- **Theoretically open.** No sample-complexity or approximation result for the joint denoiser. Existing diffusion convergence bounds (e.g. score-based results under $L^2$-accurate scores) cover the continuous case; the CTMC case has separate results; nothing bounds the joint under a factorized forward with dependent data.
- **Theoretically open.** Whether the factorized forward is ever *optimal*. Coupled forward processes (corrupting types conditionally on geometry) are unexplored.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the alignment criterion under reparametrization**, compounded by **confounded measurement**.

Mutual information between $x_0^c$ and $x_t^c$ is a property of the *channel*, not of the distribution. Add Gaussian noise in Cartesian coordinates and you destroy global position first; add it in bond-length/angle coordinates and you destroy local chemistry first. Both describe the same $p_{\text{data}}$. So "match information destruction across modalities" picks out a different schedule for each choice, with no data-driven tiebreak. §10 makes this numeric.

The second obstruction: the loss-weight $\lambda$, the schedule, and the sampler step allocation are three knobs that trade against each other, and papers move all three at once. A reported hybrid win is therefore not attributable. Combined with small-benchmark saturation — QM9 molecule stability sits above 95% for several methods, inside the range where seed variance and validity-checker conventions dominate — the evaluation does not measure the thing it names.

## 7. Current Research (as of 2026)

- **Unified generator-matching frameworks.** Discrete Flow Matching (Gat et al., NeurIPS 2024) and the multimodal flow construction of Campbell et al. (ICML 2024) supply a common language for mixed state spaces; the open work is deriving, not choosing, the relative schedule. *(frontier — verify: whether any 2026 paper derives it from a variational criterion rather than a sweep.)*
- **Latent unification vs. native hybrid.** TabSyn-style "encode everything continuous, then diffuse" is spreading beyond tables into molecules (GeoLDM lineage). The unresolved question is whether the VAE hides the alignment problem or merely relocates it into the encoder's implicit metric.
- **Protein and small-molecule co-design.** MIT (Jaakkola/Barzilay), EPFL (Frossard), Amsterdam (Welling/Hoogeboom lineage), Stanford (Ermon) are the active groups.
- **Any-modality masked models.** Extending masking-based discrete diffusion to continuous blocks via per-block "mask" tokens, so one schedule governs both. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Does the relative continuous/discrete schedule matter, beyond seed noise, at fixed architecture and fixed $\lambda$?

**Scale.** QM9, 100k train / 18k val molecules, with hydrogens. MiDi architecture, ~5M parameters, 1000 epochs, ~1 GPU-day per run on a single A100. Total 15 runs ≈ 15 GPU-days.

**Design.** One scalar $\tau$ warps the discrete schedule against the continuous one: $\gamma_t = 1-(1-t)^{\tau}$ with the continuous VP schedule held fixed. Sweep $\tau \in \{0.5, 0.71, 1.0, 1.41, 2.0\}$, 3 seeds each. Hold $\lambda$ (the loss balance) fixed at the MiDi default across all arms; sample with 1000 steps everywhere.

**Control arm.** The same backbone with atom types and bond orders relaxed to one-hot vectors diffused under the *identical* continuous schedule (EDM-style single-channel), 3 seeds. This is the everything-continuous reduction, architecture-matched.

**Deciding number.** Molecule stability (fraction of generated molecules with all atoms at correct valency), measured on 10k samples per run. Decide: schedule alignment matters iff
$$\max_\tau \bar{S}(\tau) - \bar{S}(\tau{=}1) > 3\,\hat{\sigma}_{\text{seed}},$$
with $\hat{\sigma}_{\text{seed}}$ estimated from the 3 seeds at $\tau=1$ (expected ≈0.5 percentage points at this scale, so the threshold is ≈1.5 pp). Report the control arm's $\bar{S}$ on the same axis. If the $\tau$-sweep spread is under 1.5 pp *and* the control arm is within 1.5 pp of the best hybrid, the hybrid machinery is not earning its complexity at this scale — a publishable negative result that no current paper rules out.

## 9. Key References

- **[Foundational]** Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS, 2020. — arXiv:2006.11239
- **[Foundational]** Austin, Johnson, Ho, Tarlow, van den Berg. *Structured Denoising Diffusion Models in Discrete State-Spaces.* NeurIPS, 2021. — arXiv:2107.03006
- **[Foundational]** Campbell, Benton, De Bortoli, Rainforth, Deligiannidis, Doucet. *A Continuous Time Framework for Discrete Denoising Models.* NeurIPS, 2022. — arXiv:2205.14987
- **[Foundational]** Hoogeboom, Nielsen, Jaini, Forré, Welling. *Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions.* NeurIPS, 2021. — arXiv:2102.05379
- **[SOTA]** Hoogeboom, Satorras, Vignac, Welling. *Equivariant Diffusion for Molecule Generation in 3D.* ICML, 2022. — arXiv:2203.17003
- **[SOTA]** Vignac, Osman, Toni, Frossard. *MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation.* ECML PKDD, 2023. — arXiv:2302.09048
- **[SOTA]** Campbell, Yim, Barzilay, Rainforth, Jaakkola. *Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design.* ICML, 2024. — arXiv:2402.04997
- **[SOTA]** Lou, Meng, Ermon. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.* ICML, 2024. — arXiv:2310.16834
- **[SOTA]** Sahoo, Arriola, Schiff, Gokaslan, Marroquin, Chiu, Rush, Kuleshov. *Simple and Effective Masked Diffusion Language Models.* NeurIPS, 2024. — arXiv:2406.07524
- **[SOTA]** Shi, Han, Wang, Doucet, Titsias. *Simplified and Generalized Masked Diffusion for Discrete Data.* NeurIPS, 2024. — arXiv:2406.04329
- **[SOTA]** Gat, Remez, Shaul, Kreuk, Chen, Synnaeve, Adi, Lipman. *Discrete Flow Matching.* NeurIPS, 2024. — arXiv:2407.15595
- **[Applied]** Kotelnikov, Baranchuk, Rubachev, Babenko. *TabDDPM: Modelling Tabular Data with Diffusion Models.* ICML, 2023. — arXiv:2209.15421
- **[Applied]** Zhang, Zhou, Yao, Chu, Han. *Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space.* ICLR, 2024. — arXiv:2310.09656
- **[Applied]** Vignac, Krawczuk, Siraudin, Wang, Cevher, Frossard. *DiGress: Discrete Denoising Diffusion for Graph Generation.* ICLR, 2023. — arXiv:2209.14734
- **[Applied]** Chen, Zhang, Hinton. *Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning.* ICLR, 2023. — arXiv:2208.04202
- **[Survey]** Yang, Zhang, Song, Hong, Xu, Zhao, Zhang, Cui, Yang. *Diffusion Models: A Comprehensive Survey of Methods and Applications.* ACM Computing Surveys, 2023. — arXiv:2209.00796

## 10. Worked Example

Take one QM9 molecule with hydrogens. Atom-type marginal over $\{$H, C, N, O, F$\}$ is roughly $(0.51, 0.35, 0.05, 0.08, 0.01)$, giving per-atom entropy
$$H(x_0^d) = -\sum_k p_k \ln p_k \approx 1.11 \text{ nats}.$$
Under masking, retained discrete information is exactly $(1-\gamma_t)\cdot 1.11$ nats.

Now ask where the continuous channel retains the same amount, per atom. For 3 isotropic dimensions with per-axis data scale $s$ and noise $\sigma_t$,
$$\mathcal{I}^c(t) = \tfrac{3}{2}\ln\!\left(1 + s^2/\sigma_t^2\right).$$

**Cartesian coordinates**, $s = 1.5$ Å (QM9 atoms spread over a few Å):
$$\tfrac32\ln(1+2.25/\sigma_t^2) = 1.11 \ \Rightarrow\ 1+2.25/\sigma_t^2 = e^{0.74}=2.10 \ \Rightarrow\ \sigma_t^\star \approx 1.43\ \text{Å}.$$

**Internal coordinates** (bond lengths / angles), where the informative per-dimension scale is the bond-length spread, $s = 0.10$ Å:
$$\tfrac32\ln(1+0.01/\sigma_t^2) = 1.11 \ \Rightarrow\ \sigma_t^\star \approx 0.096\ \text{Å}.$$

Same molecule, same distribution, same criterion — and the "matched" noise level differs by **15×**. Under a standard cosine VP schedule that moves the crossing time from roughly $t\approx0.75$ to $t\approx0.25$: the discrete block would be half-masked either three-quarters of the way through corruption, or one-quarter of the way, depending on a coordinate choice the data does not make for you.

That is the obstruction in one number. The alignment criterion is not a property of $p_{\text{data}}$; it is a property of the parametrization, and nothing in the current theory selects the parametrization. Consequently every published "principled" hybrid schedule is, on inspection, a tuned constant — and the experiment in §8 exists precisely because nobody has measured whether that constant matters.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*