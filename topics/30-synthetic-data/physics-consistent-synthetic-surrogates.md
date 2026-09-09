---
id: 30-synthetic-data/physics-consistent-synthetic-surrogates
title: "Physically Consistent Synthetic Data for Scientific Surrogates"
topic: 30-synthetic-data
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Physically Consistent Synthetic Data for Scientific Surrogates

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/physics-consistent-synthetic-surrogates` · **Status:** open

## 1. Problem Statement

Scientific surrogates (neural PDE solvers, weather emulators, molecular force fields) are trained on simulation output. High-fidelity simulation is the cost bottleneck: one turbulence DNS trajectory can cost more GPU-hours than training the surrogate on it. The proposal is to expand the training set with *generated* samples — from a generative model, a coarse solver, symmetry augmentation, or the surrogate itself — instead of running more of the expensive solver.

The problem: generated samples are cheap but need not obey the physics. A sample can look right and still violate mass conservation, break a symmetry, or carry a wrong energy spectrum. Training on it can degrade the surrogate rather than improve it.

Three variants, of very different difficulty:

- **Measurement.** Given a synthetic sample $\tilde{u}$, define a scalar "physical consistency" that predicts downstream surrogate quality. Currently unsolved: the reported residual metrics correlate weakly with rollout error.
- **Method.** Produce a generator $G$ whose samples raise surrogate accuracy per unit of total compute versus spending the same compute on more ground-truth solver runs. This is the operative claim and it is rarely tested against that control.
- **Theory.** Bound the surrogate's excess risk as a function of the generator's physics violation. No such bound exists for a nonlinear PDE with autoregressive rollout.

A solution: a generator plus a consistency measure such that filtering on the measure provably (or reproducibly) improves rollout error at fixed *total* compute, on at least two PDE families, against a real-data-only arm.

## 2. Formal Setting

Let $u: \Omega \times [0,T] \to \mathbb{R}^d$ solve $\partial_t u = \mathcal{N}(u; \theta_p)$ with parameters $\theta_p \sim \mu$ (viscosity, forcing, boundary data). The reference solver $S_h$ at resolution $h$ produces $u^{(h)}$; the *ground-truth* distribution is $p^\star = \text{law}(S_h(\theta_p))$ at the $h$ where the solver is converged — which is itself an assumption, and one that is checked by grid refinement at best on a few cases.

**Surrogate.** $f_\phi: u^n \mapsto u^{n+1}$, trained on $\mathcal{D} = \mathcal{D}_{\text{real}} \cup \mathcal{D}_{\text{syn}}$, $|\mathcal{D}_{\text{real}}| = N_r$, $|\mathcal{D}_{\text{syn}}| = N_s$.

**Rollout error**, the target quantity, measured as normalized RMSE over $K$ autoregressive steps:
$$\mathcal{E}_K(\phi) = \mathbb{E}_{\theta_p}\left[\frac{\|f_\phi^{\circ K}(u^0) - u^{K}\|_2}{\|u^{K}\|_2}\right].$$

**Constraint residual.** For $m$ conservation/constraint operators $C_j$ (e.g. $C_1 u = \nabla\!\cdot u$ for incompressibility, $C_2 u = \int_\Omega u\,dx$ for mass):
$$R_j(\tilde u) = \frac{\|C_j \tilde u\|_2}{\|\tilde u\|_2}, \qquad R(\tilde u) = \max_j R_j(\tilde u).$$
Measured with the *same* discrete operator the solver used — a spectral divergence and a second-order finite-difference divergence of the same field differ by orders of magnitude, so $R$ is only defined relative to a discretization.

**Spectral fidelity.** With $E(k)$ the shell-averaged energy spectrum,
$$\Delta_{\text{spec}} = \max_k \left|\log_{10} \tilde{E}(k) - \log_{10} E^\star(k)\right|.$$

**Distributional gap.** $W_2(\tilde{p}, p^\star)$ over trajectory statistics. In practice unmeasurable in the ambient dimension ($10^5$–$10^7$ per snapshot); estimated only on low-dimensional summaries (spectra, PDFs of increments, extreme quantiles).

**Compute budget.** $B = c_{\text{sim}} N_r + c_{\text{gen}} N_s + c_{\text{train}}$, in GPU-hours. Every honest comparison holds $B$ fixed. Most published comparisons hold $N_r$ fixed instead, which is not the decision anyone faces.

**Assumptions known to be violated.** (i) $p^\star$ is the true physics — false; $S_h$ has its own truncation and turbulence-model error. (ii) Synthetic samples are i.i.d. from a distribution — false when generated autoregressively by the surrogate, giving correlated errors and the recursive-training pathology. (iii) Constraint satisfaction implies distributional correctness — false, and this is the crux (§10). (iv) Rollout error is the deployment objective — often false; downstream use is a functional (drag, extreme quantile) that a low-RMSE model can still get wrong.

## 3. State of the Art

**Established (independently reproduced).**
- Hard constraint layers work as stated: projecting onto a divergence-free or conserved-quantity subspace drives $R_j$ to machine precision. Négiar, Mahoney & Krishnapriyan (ICLR 2023) and Hansen et al., *ProbConserv* (ICML 2023) both show constraint satisfaction is achievable without hurting accuracy.
- Lie point symmetry augmentation is a generator whose samples are exactly on-manifold. Brandstetter et al. (ICML 2022) report error reductions of tens of percent on KdV/Burgers in the low-data regime; Mialon et al. (NeurIPS 2023) confirm the gain transfers to self-supervised pretraining.
- Solver-in-the-loop training (Um et al., NeurIPS 2020) and the pushforward trick (Brandstetter et al., ICLR 2022) fix distribution shift from self-generated rollouts and are now standard.

**Claimed but unablated.**
- Diffusion models as PDE data generators (Kohl, Um & Thuerey, ACDM, 2023; Shu et al., JCP 2023) produce samples with plausible spectra. Whether training a *separate* surrogate on those samples beats spending the generator's training compute on more DNS is, to our knowledge, not reported at fixed $B$.
- "Physics-informed" losses added to generators are reported to reduce $R$ — which is the loss they optimize — with rollout error of a downstream surrogate rarely the reported endpoint.

**Benchmark-number-only.** PDEBench (Takamoto et al., NeurIPS D&B 2022) and The Well (Ohana et al., NeurIPS D&B 2024) supply the data but no fixed-compute synthetic-augmentation protocol. Scores on them are single numbers with no baseline-tuning audit — precisely the failure McGreivy & Hakim (*Nature Machine Intelligence*, 2024) document, finding weak baselines in 79% of the ML-for-fluid-PDE papers they surveyed.

## 4. What Is Known

- **Scale helps and transfers.** Subramanian et al. (NeurIPS 2023) show FNO pretraining on Poisson/advection/Helmholtz transfers downstream with power-law improvement in dataset size, over $10^1$–$10^4$ training samples at $128^2$ resolution.
- **Constraint residual is cheap to zero out.** Leray projection of a $128^2$ velocity field costs two FFTs, $O(10^{-4})$ s on GPU, versus $O(10)$ s per DNS step. Cost is not the barrier to constraint satisfaction.
- **PINN soft constraints fail non-obviously.** Krishnapriyan et al. (NeurIPS 2021) show convection/reaction PINNs with residual loss driven to $10^{-4}$ still give $O(1)$ relative solution error at high coefficient values — residual smallness does not certify solution correctness. Wang, Yu & Perdikaris (JCP 2022) attribute this to NTK eigenvalue imbalance across loss terms.
- **Recursive synthetic training degrades.** Shumailov et al. (*Nature*, 2024) show tail collapse under fully-replacing recursive training; Gerstgrasser et al. (COLM 2024) show *accumulating* real plus synthetic data bounds the degradation. Both were shown on language/VAE settings, not PDEs — the transfer is assumed, not measured.
- **Rollout stability is a separate axis.** PDE-Refiner (Lippe et al., NeurIPS 2023) extends stable KS rollouts from roughly tens to ~100 time units at equal parameter count, by fixing the low-amplitude spectrum the MSE loss ignores.

## 5. What Is Not Known

- **Methodologically blocked.** No validated scalar consistency measure. $R$, $\Delta_{\text{spec}}$ and low-dim $W_2$ each capture a slice; no published study reports the rank correlation between any of them and $\mathcal{E}_K$ across a sample pool. Until that correlation is measured, "physically consistent" names a property no one can score.
- **Empirically open.** The fixed-total-compute comparison — generator + synthetic data versus more solver runs — is runnable today on PDEBench at $128^2$ for a few thousand GPU-hours. It has not been published for any PDE family.
- **Theoretically open.** No excess-risk bound of the form $\mathcal{E}_K \le a\,\mathcal{E}_K^{\text{real}} + b(K)\,\varepsilon(\tilde p, p^\star)$ for autoregressive surrogates under a nonlinear $\mathcal{N}$. Even the right $\varepsilon$ is unsettled: constraint residual is provably insufficient (§10), and $W_2$ in ambient dimension is unestimable at these sample counts.

## 6. Why It Is Hard

**The specific obstruction: the constraint manifold is enormously larger than the solution manifold, and the measurable violation is nearly orthogonal to the error that matters.**

The set of divergence-free fields on a $128^2$ grid is a linear subspace of dimension ~$1.6\times10^4$; the attractor of forced 2D turbulence at a given $\mathrm{Re}$ is a measure-zero subset of it. Any $R_j$ measures distance to the subspace, not to the attractor. A generator can be pushed to $R = 10^{-14}$ and still place mass in regions the flow never visits.

Secondary: **absent ground truth.** $p^\star$ is defined by the solver, so a synthetic sample cannot be scored against the physics, only against a discretization that has its own error. And **confounded measurement** — augmentation studies hold $N_r$ fixed, which credits the synthetic arm with the generator's training cost for free.

## 7. Current Research (as of 2026)

- **Symmetry- and structure-preserving generation.** Lie-point augmentation (Brandstetter, Welling, and the Amsterdam/JKU line) extended to higher-dimensional systems where the symmetry group is smaller and the augmentation buys less. *(frontier — verify current scope.)*
- **PDE foundation models trained on mixed corpora** — Poseidon (Herde et al., NeurIPS 2024), Universal Physics Transformers (Alkin et al., NeurIPS 2024) — which makes corpus composition, including synthetic fraction, a live design variable.
- **Diffusion emulators for weather ensembles**, GenCast (Price et al., *Nature*, 2025), where calibration against reanalysis is the physics check rather than a residual. Best current template for what a consistency metric should look like.
- **Data-curation-for-science** groups (Polymathic AI, The Well) building the corpora on which the fixed-compute experiment could be run. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** At fixed total GPU-hours, does synthetic augmentation beat more solver runs?

**Scale.** 2D incompressible Navier–Stokes, $128^2$, $\mathrm{Re}\in[10^3,10^4]$, PDEBench-style pseudospectral DNS. Budget $B = 512$ A100-hours per arm. Surrogate: FNO-2D, ~30M parameters, identical for all arms.

**Arms.**
- **Control (real-only):** spend all 512 h on DNS → about 2,000 trajectories; train.
- **Synthetic:** 128 h DNS (~500 trajectories) + 128 h training a conditional diffusion generator + 32 h sampling 4,000 synthetic trajectories + 224 h surrogate training.
- **Ablation A:** synthetic samples Leray-projected ($R = 0$).
- **Ablation B:** synthetic samples filtered to the best 50% by $\Delta_{\text{spec}}$.

**Deciding number.** $\mathcal{E}_{50}$, normalized RMSE at 50 autoregressive steps on a held-out DNS test set of 200 trajectories, 5 seeds. The synthetic arm wins only if its mean $\mathcal{E}_{50}$ is below the control's by more than $2\times$ the seed standard deviation. Secondary and equally informative: Spearman $\rho$ between per-sample $R$, $\Delta_{\text{spec}}$, and the leave-one-out change in $\mathcal{E}_{50}$. If $|\rho| < 0.2$ for $R$, the residual metric is dead as a filter and §5's methodological block is confirmed empirically.

## 9. Key References

- **[Foundational]** Raissi, Perdikaris & Karniadakis. *Physics-Informed Neural Networks.* Journal of Computational Physics, 2019.
- **[Foundational]** Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart & Anandkumar. *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR, 2021. — arXiv:2010.08895
- **[SOTA]** Brandstetter, Welling & Worrall. *Lie Point Symmetry Data Augmentation for Neural PDE Solvers.* ICML, 2022. — arXiv:2202.07643
- **[SOTA]** Lippe, Veeling, Perdikaris, Turner & Brandstetter. *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers.* NeurIPS, 2023. — arXiv:2308.05732
- **[SOTA]** Herde, Raonić, Rohner, Käppeli, Molinaro, de Bézenac & Mishra. *Poseidon: Efficient Foundation Models for PDEs.* NeurIPS, 2024. — arXiv:2405.19101
- **[SOTA]** Price, Sanchez-Gonzalez, Alet, Andersson, El-Kadi, Masters, Ewalds, Stott, Mohamed, Battaglia, Lam & Willson. *Probabilistic weather forecasting with machine learning.* Nature, 2025.
- **[Critical]** Krishnapriyan, Gholami, Zhe, Kirby & Mahoney. *Characterizing Possible Failure Modes in Physics-Informed Neural Networks.* NeurIPS, 2021. — arXiv:2109.01050
- **[Critical]** McGreivy & Hakim. *Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations.* Nature Machine Intelligence, 2024. — arXiv:2407.07218
- **[Critical]** Shumailov, Shumaylov, Zhao, Papernot, Anderson & Gal. *AI models collapse when trained on recursively generated data.* Nature, 2024.
- **[Method]** Um, Brand, Fei, Holl & Thuerey. *Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers.* NeurIPS, 2020. — arXiv:2007.00016
- **[Method]** Hansen, Maddix, Alizadeh, Gupta & Mahoney. *Learning Physical Models that Can Respect Conservation Laws.* ICML, 2023. — arXiv:2302.11002
- **[Benchmark]** Takamoto, Praditia, Leiteritz, MacKinlay, Alesiani, Pflüger & Niepert. *PDEBench: An Extensive Benchmark for Scientific Machine Learning.* NeurIPS Datasets & Benchmarks, 2022. — arXiv:2210.07182
- **[Benchmark]** Ohana et al. *The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2412.00568
- **[Survey]** Karniadakis, Kevrekidis, Lu, Perdikaris, Wang & Yang. *Physics-informed machine learning.* Nature Reviews Physics, 2021.

## 10. Worked Example

**Setup.** Forced 2D turbulence, $\Omega = [0,2\pi]^2$, $128^2$ grid, $\Delta x = 2\pi/128 \approx 0.049$. A pseudospectral DNS field $u^\star$ has relative divergence $R_1 \approx 10^{-13}$ (round-off). A diffusion-model sample $\tilde u$ has $R_1 \approx 2\times10^{-2}$.

**Step 1 — repair the constraint.** Leray projection in Fourier space, $\hat{u}^{\perp}_k = (I - \hat{k}\hat{k}^\top)\hat{u}_k$. Two FFTs. Afterwards $R_1 \approx 10^{-14}$: constraint violation removed exactly.

**Step 2 — how much of the field did that change?** The removed component is the compressive part. Its share of kinetic energy is bounded by $R_1^2$ in the spectral norm, so
$$\frac{\|\tilde u - \tilde u^\perp\|_2^2}{\|\tilde u\|_2^2} \lesssim (2\times10^{-2})^2 = 4\times10^{-4}.$$
Projection touched **0.04% of the energy**.

**Step 3 — the error that remains.** The same sample carries $\Delta_{\text{spec}} = 0.35$ over $k \in [32, 64]$, i.e. about $10^{0.35} \approx 2.2\times$ excess energy in the dissipation-range shells. Those shells hold roughly 3–8% of total energy in a $k^{-3}$-ish 2D spectrum, so the spectral error is $\sim10^{-2}$ of the energy — **two orders of magnitude larger than what projection removed**, and entirely inside the divergence-free subspace.

**Step 4 — downstream.** A surrogate trained on projected samples inherits the small-scale bias. Because the FNO's autoregressive map amplifies high-$k$ error roughly geometrically, a $2.2\times$ small-scale bias compounds over 50 steps into $O(1)$ rollout error, while $R_1$ reads a clean $10^{-14}$ throughout.

**The obstruction made visible.** The physics check that is cheap, exact and universally reported ($R$) certifies 0.04% of the problem. The 1% that decides the surrogate lives in a direction $R$ is blind to by construction, because the projection operator and the error are orthogonal. This is why the field has hard-constraint layers that work and still has no usable definition of "physically consistent synthetic data".

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*