---
id: 09-model-design/equivariance-vs-augmentation-scale
title: "Equivariance Versus Data Augmentation at Scale"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Equivariance Versus Data Augmentation at Scale

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/equivariance-vs-augmentation-scale` · **Status:** empirically-open

## 1. Problem Statement

A symmetry group $G$ acts on the input space (rotations for molecules, translations for images, permutations for sets). Two ways to make a model respect it:

- **Hard equivariance** — constrain the hypothesis class so $f(g\cdot x) = g\cdot f(x)$ holds by construction (steerable kernels, tensor-product message passing).
- **Augmentation** — leave the architecture unconstrained and sample group elements during training, so the symmetry is learned approximately.

**The question.** As training compute $C$ and dataset size $N$ grow, does the advantage of hard equivariance shrink to zero, stay a fixed offset, or invert?

Three variants, different difficulties:

- **Measurement.** Is there a metric that isolates the symmetry contribution from the confounds of parameter count, throughput, and expressivity? Currently no agreed one.
- **Method.** At a fixed *compute* budget (not fixed parameters), which arm has the lower loss, and does the crossover point exist? Runnable, largely unrun outside one or two domains.
- **Theory.** Does the generalization gain of an equivariant class decay as $O(N^{-1})$, $O(|G|^{-1})$, or not at all in the overparameterized regime? Open for anything beyond kernels and linear models.

**Solved** means: a fitted scaling law for both arms in $\geq 3$ domains, with the sign and magnitude of the exponent difference reported and reproduced.

## 2. Formal Setting

Let $G$ be a compact group acting on $\mathcal{X}$ via $\rho_{\mathcal{X}}$ and on $\mathcal{Y}$ via $\rho_{\mathcal{Y}}$. Target $f^\star$ is $G$-equivariant: $f^\star(\rho_{\mathcal{X}}(g)x) = \rho_{\mathcal{Y}}(g) f^\star(x)$.

**Hypothesis classes.** $\mathcal{F}_{\mathrm{eq}} \subset \mathcal{F}_{\mathrm{free}}$, where $\mathcal{F}_{\mathrm{eq}}$ is the equivariant subclass. Augmentation trains over $\mathcal{F}_{\mathrm{free}}$ with the orbit-averaged risk
$$\hat{R}_{\mathrm{aug}}(f) = \tfrac{1}{N}\sum_{i=1}^{N} \mathbb{E}_{g\sim\mu_G}\,\ell\!\left(f(\rho_{\mathcal{X}}(g)x_i),\, \rho_{\mathcal{Y}}(g) y_i\right),$$
$\mu_G$ the Haar measure, estimated in practice with $K$ samples per example per step ($K=1$ typically; AlphaFold 3's diffusion module uses dozens).

**Equivariance error**, as actually measured — the only defensible operational definition:
$$\mathcal{E}(f) = \mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{g\sim\mu_G} \frac{\lVert f(\rho_{\mathcal{X}}(g)x) - \rho_{\mathcal{Y}}(g)f(x)\rVert}{\lVert f(x)\rVert}.$$
For connected Lie groups the infinitesimal version — the Lie derivative $\mathcal{L}_X f$ along generator $X$ (Gruver et al., ICLR 2023) — is cheaper and less variance-bound. $\mathcal{E}(f)=0$ by construction for $\mathcal{F}_{\mathrm{eq}}$ up to float error (in practice $10^{-6}$–$10^{-4}$ relative, from spherical-harmonic evaluation in float32).

**Compute.** $C \approx 6 N_{\mathrm{par}} D$ FLOPs is the standard estimator but is *wrong for equivariant models*: tensor-product layers have low arithmetic intensity, so wall-clock per FLOP differs by 3–10× against dense attention on the same GPU. Measure $C$ as **measured device-seconds × achievable FLOP/s**, not the analytic count.

**Decision predicate.** Fit $L(C) = aC^{-\alpha} + L_\infty$ per arm. The question is the sign of $\alpha_{\mathrm{eq}} - \alpha_{\mathrm{aug}}$ and whether $\log a_{\mathrm{eq}} - \log a_{\mathrm{aug}}$ is a constant offset (a fixed data/compute multiplier) or shrinking.

**Assumptions known to be violated.**
- *Exact symmetry of the target.* Violated almost everywhere: crystals break continuous rotation, images have a gravity prior, real molecular datasets carry pose bias from the sampling protocol.
- *Exact symmetry of the data distribution.* $\mathcal{D}$ is rarely $G$-invariant even when $f^\star$ is; augmentation then changes the input distribution, equivariance does not.
- *Equal expressivity outside the symmetry constraint.* Equivariant and free models differ in depth, nonlinearity, and attention structure, so the comparison is never clean.
- *$6N_{\mathrm{par}}D$ compute accounting.* Violated, as above.

## 3. State of the Art

**Established.**
- Group-equivariant CNNs (Cohen & Welling, ICML 2016) and E(3)-equivariant message passing (Thomas et al. 2018; NequIP, Batzner et al., *Nature Communications* 2022; MACE, Batatia et al., NeurIPS 2022) win decisively in the **small-data** regime for atomistic systems — this is reproduced across many groups.
- Elesedy & Zaidi (ICML 2021) prove a *strictly positive* generalization gain for equivariant models in linear/kernel settings, with the gain governed by the fraction of the function's energy outside the symmetric subspace.
- Bietti, Venturi & Bruna (NeurIPS 2021) and Mei, Misiakiewicz & Montanari (COLT 2021) give sample-complexity gains of order $|G|$ (finite groups) or a dimension-dependent factor (Lie groups) for kernel/random-feature models.

**Claimed but unablated.**
- "Equivariance stops mattering at scale." Widely repeated after AlphaFold 3 (Abramson et al., *Nature* 2024) dropped the equivariant structure module for a non-equivariant diffusion decoder trained with random global rotations. No controlled equivariant-arm ablation at matched compute was published. This is an *existence proof that augmentation suffices*, not a comparison.
- "Equivariance still matters at scale." Brehmer, Bose, de Haan & Cohen, *Does equivariance matter at scale?* (2024) fit compute-optimal scaling laws for an equivariant transformer against augmented baselines on rigid-body dynamics and report the equivariant arm ahead across the whole budget range — but in one domain, one group family, one architecture pair.

**Benchmark-number-only.** OC20 leaderboard positions. EquiformerV2 (Liao et al., ICLR 2024) held the top S2EF slot with equivariance; EScAIP (Qu & Krishnapriyan, NeurIPS 2024) argued a non-equivariant attention model reaches comparable accuracy with far better throughput. These are leaderboard entries under different training recipes, not matched-compute ablations.

## 4. What Is Known

- **Small-data gain is large and real.** NequIP matched or beat then-current potentials on MD-17 while training on $\sim10^3$ configurations, a claimed ~$10^3\times$ data-efficiency gain over non-equivariant baselines of that era. Scale: $10^3$–$10^4$ structures, $<10^7$ parameters.
- **Learned equivariance grows with scale.** Gruver et al. (ICLR 2023) measured Lie derivatives across ~400 ImageNet models and found large ViTs are *more* translation/rotation-equivariant than architecturally-translation-equivariant CNNs — architecture is a weak predictor of measured equivariance once data is plentiful. Scale: ImageNet-1k/21k, $10^7$–$10^9$ parameters.
- **Augmentation is not free.** Chen, Dobriban & Lee (JMLR 2020) show augmentation acts as a variance-reduction/regularization operator whose benefit is bounded by the orbit-averaging projection — it can recover the invariant estimator in expectation but with higher variance at finite $K$.
- **Approximate symmetry beats both extremes when the symmetry is broken.** Wang, Walters & Yu (ICML 2022) and Petrache & Trivedi (NeurIPS 2023) quantify an approximation–generalization trade-off: hard equivariance is a *bias* when $f^\star$ is only approximately equivariant.
- **Throughput gap is the practical decider today.** Tensor-product layers of degree $\ell_{\max}=2$–$3$ cost roughly an order of magnitude more device-time per parameter than dense attention on current accelerators, which is why matched-*parameter* comparisons systematically flatter equivariant models.

## 5. What Is Not Known

- **Empirically open (the main gap).** No matched-compute scaling-law comparison of the two arms exists in more than one or two domains. The specific unrun experiment: both arms swept over $\geq 4$ compute decades in molecules, images, and one non-geometric group (e.g. permutation-symmetric tabular/set data), with device-time-normalized budgets.
- **Theoretically open.** Whether the equivariance gain persists in the overparameterized/feature-learning regime. All existing strict-benefit theorems are linear, kernel, or random-feature. No result predicts $\alpha_{\mathrm{eq}} - \alpha_{\mathrm{aug}}$ for a deep network.
- **Theoretically open.** Whether augmentation with finite $K$ per step converges to the invariant risk minimizer at a rate that depends on $|G|$ or only on the Haar-sampling variance.
- **Methodologically blocked.** "Matched compute" for architectures with different arithmetic intensity is not well defined — analytic FLOPs, device-seconds, and energy give different orderings of the same two models. Until the catalog fixes one, published comparisons are not commensurable.
- **Methodologically blocked.** No agreed measure of *how symmetric a dataset actually is*, so "the symmetry is only approximate here" is asserted rather than measured.

## 6. Why It Is Hard

**The obstruction is confounded measurement compounded by compute cost.**

Every published comparison varies at least three things at once: the symmetry constraint, the parameter count, and the achievable hardware utilization. Equivariant models are typically 3–10× slower per parameter, so a matched-parameter comparison hands them a large compute advantage and a matched-wall-clock comparison hands them a large parameter disadvantage. The two protocols can produce opposite conclusions from the same code.

Second, the crossover — if it exists — is expected in a regime where a single arm costs $10^{20}$–$10^{22}$ FLOPs. Fitting two scaling laws over four decades with enough seeds to separate exponents is a multi-hundred-GPU-day study per domain, which is why it is repeatedly asserted and rarely run.

Third, there is no ground truth for the target's true symmetry group. When augmentation wins, one cannot distinguish "equivariance stopped helping" from "the imposed group was wrong for this data."

## 7. Current Research (as of 2026)

- **Scaling-law comparisons.** Extension of the Brehmer et al. protocol to more domains and to $SE(3)$ vs $E(3)$ vs $SO(3)$ constraint strengths. *(frontier — verify)*
- **Relaxed and learned symmetry.** Canonicalization networks (Kaba et al., ICML 2023) and approximately-equivariant layers, which convert the binary choice into a tunable constraint strength — the most likely resolution shape.
- **Hardware-aware equivariance.** Fused tensor-product kernels (e3nn/cuEquivariance line) aiming to close the throughput gap; if the gap closes, the empirical question changes answer. *(frontier — verify)*
- **Foundation models for atoms.** MACE-MP, and successor universal potentials, trained at $10^7$–$10^8$ structures — the first regime where the augmentation arm is even competitive on chemistry.
- **Biomolecular structure.** Post-AlphaFold-3 work testing whether the non-equivariant-plus-augmentation recipe holds at smaller data budgets.

## 8. Concrete Next Experiment

**Domain.** Interatomic potentials on OC20 S2EF ($\sim1.3\times10^8$ structures), which is large enough that the small-data equivariance advantage should have decayed if it decays at all.

**Arms.** One codebase, one training loop, one optimizer schedule:
- **Arm A (equivariant):** MACE-style or EquiformerV2-style, $\ell_{\max}=2$, exact $E(3)$.
- **Arm B (control, augmented):** the *same* network with tensor products replaced by unconstrained MLP/attention channels of matched hidden width, trained with $K=1$ uniform random $SO(3)$ rotation per example per step.
- **Arm C (null control):** Arm B with no augmentation. Required — it establishes how much of Arm B's performance is augmentation versus architecture.

**Scale.** Six compute budgets, log-spaced over $10^{18}$–$10^{22}$ FLOPs of *measured device-time × achieved FLOP/s* on one GPU type. Three seeds per point: 54 runs. At compute-optimal token allocation per budget, estimated ~600–900 A100/H100-days total.

**Deciding number.**
$$\Delta = \log_{10}\!\left(\frac{C_{\mathrm{aug}}(L_0)}{C_{\mathrm{eq}}(L_0)}\right)$$
— the compute multiplier the augmented arm needs to reach a fixed force-MAE target $L_0 = 20$ meV/Å, evaluated at the two largest budgets. If $\Delta$ falls by more than $0.3$ (a 2× shrink) between the two largest budgets, the equivariance advantage is decaying with scale. If $\Delta$ is flat within seed noise, it is a permanent constant-factor gain. Report $\Delta$ with bootstrap CIs over seeds; anything without CIs does not settle it.

## 9. Key References

- **[Foundational]** Taco Cohen, Max Welling. *Group Equivariant Convolutional Networks.* ICML, 2016. — arXiv:1602.07576
- **[Foundational]** Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhoff, Patrick Riley. *Tensor Field Networks: Rotation- and Translation-Equivariant Neural Networks for 3D Point Clouds.* 2018. — arXiv:1802.08219
- **[Theory]** Bryn Elesedy, Sheheryar Zaidi. *Provably Strict Generalisation Benefit for Equivariant Models.* ICML, 2021. — arXiv:2102.10333
- **[Theory]** Alberto Bietti, Luca Venturi, Joan Bruna. *On the Sample Complexity of Learning under Geometric Stability.* NeurIPS, 2021.
- **[Theory]** Song Mei, Theodor Misiakiewicz, Andrea Montanari. *Learning with Invariances in Random Features and Kernel Models.* COLT, 2021.
- **[Theory]** Shuxiao Chen, Edgar Dobriban, Jane H. Lee. *A Group-Theoretic Framework for Data Augmentation.* JMLR, 2020.
- **[Empirical]** Simon Batzner, Albert Musaelian, Lixin Sun, Mario Geiger, Jonathan P. Mailoa, Mordechai Kornbluth, Nicola Molinari, Tess E. Smidt, Boris Kozinsky. *E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials.* Nature Communications, 2022.
- **[Empirical]** Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner, Gábor Csányi. *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields.* NeurIPS, 2022.
- **[SOTA]** Johann Brehmer, Sönke Behrends, Pim de Haan, Taco Cohen. *Does equivariance matter at scale?* 2024.
- **[SOTA]** Yi-Lun Liao, Brandon Wood, Abhishek Das, Tess Smidt. *EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations.* ICLR, 2024.
- **[SOTA]** Eric Qu, Aditi S. Krishnapriyan. *The Importance of Being Scalable: Improving the Speed and Accuracy of Neural Network Interatomic Potentials Across Chemical Domains.* NeurIPS, 2024.
- **[Measurement]** Nate Gruver, Marc Finzi, Micah Goldblum, Andrew Gordon Wilson. *The Lie Derivative for Measuring Learned Equivariance.* ICLR, 2023.
- **[Relaxation]** Rui Wang, Robin Walters, Rose Yu. *Approximately Equivariant Networks for Imperfectly Symmetric Dynamics.* ICML, 2022.
- **[Relaxation]** Mircea Petrache, Shubhendu Trivedi. *Approximation-Generalization Trade-offs under (Approximate) Group Equivariance.* NeurIPS, 2023.
- **[Application]** Josh Abramson et al. *Accurate structure prediction of biomolecular interactions with AlphaFold 3.* Nature, 2024.
- **[Survey]** Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković. *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.* 2021. — arXiv:2104.13478

## 10. Worked Example

Take one budget point and show why the published comparisons cannot be added together.

Two models on the same GPU:

| | Arm A (equivariant, $\ell_{\max}=2$) | Arm B (unconstrained + aug) |
|---|---|---|
| Parameters | $3.0\times10^7$ | $3.0\times10^7$ |
| Analytic FLOPs/structure | $\approx 1.0\times$ | $\approx 1.0\times$ |
| Achieved FLOP/s (fraction of peak) | ~8% | ~45% |
| Structures/second | 320 | 1,800 |

Fix a wall-clock budget of 24 GPU-hours.

- Arm A sees $320 \times 86{,}400 \approx 2.8\times10^7$ structures.
- Arm B sees $1{,}800 \times 86{,}400 \approx 1.6\times10^8$ structures — **5.6×** more.

Suppose Arm A reaches force MAE 21 meV/Å and Arm B reaches 24 meV/Å.

- **Matched-parameter reading:** equivariance wins by 12% relative error at equal size.
- **Matched-wall-clock reading:** Arm B, given 5.6× more data for the same money, is only 14% behind. Extrapolate its own $L(D)\propto D^{-0.3}$ curve and it passes 21 meV/Å at roughly $1.6\times10^8 \times (24/21)^{1/0.3} \approx 2.7\times10^8$ structures — 1.7× more wall-clock, not 5.6×. So $\Delta \approx \log_{10}1.7 = 0.23$, a modest constant, not a categorical advantage.

Same runs. Opposite headlines. The obstruction is not that the experiment is unrunnable — it is that "equal compute" is undefined across architectures with 8% versus 45% hardware utilization, and *nobody has fixed a convention*. Until $\Delta$ is reported as a device-time-normalized compute multiplier with seed CIs at two or more budgets, the field is comparing numbers that are not on the same axis.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*