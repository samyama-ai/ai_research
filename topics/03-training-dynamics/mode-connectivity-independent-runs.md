---
id: 03-training-dynamics/mode-connectivity-independent-runs
title: "Loss Landscape Connectivity of Independently Trained Models"
topic: 03-training-dynamics
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Loss Landscape Connectivity of Independently Trained Models

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/mode-connectivity-independent-runs` · **Status:** partially-solved

## 1. Problem Statement

Two networks of identical architecture, trained on the same data from different random seeds, reach parameter vectors $\theta_A,\theta_B$ with near-identical test loss. The linear interpolant $\theta(\alpha)=(1-\alpha)\theta_A+\alpha\theta_B$ usually has much *higher* loss in between — a **barrier**. The question: is that barrier intrinsic, or an artifact of the network's discrete symmetries?

Three variants, with different difficulty:

- **Measurement.** Given $\theta_A,\theta_B$, compute the barrier reliably. Blocked in practice by BatchNorm statistics and by activation-variance collapse at the midpoint, which inflate the measured barrier for reasons unrelated to the landscape.
- **Method.** Find a permutation $\pi$ of hidden units per layer such that $\theta_A$ and $\pi(\theta_B)$ are linearly connected with barrier below $\epsilon$. Solving it exactly is combinatorial; the practical question is how close cheap heuristics get.
- **Theory.** Prove or refute the **permutation conjecture** (Entezari et al., 2022): for wide enough networks, SGD solutions form a single basin *modulo permutation*, so almost all seed pairs admit a zero-barrier linear path after alignment.

Solving it means: a per-architecture width threshold above which alignment provably yields barrier $\le\epsilon$, plus an algorithm that attains it at ImageNet scale.

## 2. Formal Setting

Let $f(\cdot;\theta)$ be a network with $L$ layers of widths $h_1,\dots,h_L$, $\theta\in\mathbb{R}^d$, and empirical risk $\hat L(\theta)=\frac1n\sum_i \ell(f(x_i;\theta),y_i)$ measured on a held-out split of size $n$ (report both train and test; they differ).

**Barrier.** For the interpolant $\theta(\alpha)$,
$$B(\theta_A,\theta_B)=\max_{\alpha\in[0,1]}\Big[\hat L(\theta(\alpha))-\big((1-\alpha)\hat L(\theta_A)+\alpha \hat L(\theta_B)\big)\Big].$$
Measured on a grid $\alpha\in\{0,0.05,\dots,1\}$; the max typically sits at $\alpha=0.5$. Report the error-rate barrier too — loss and error barriers can disagree in sign at small $\epsilon$.

**Symmetry group.** Let $\mathcal{S}=\{\pi=(P_1,\dots,P_{L-1})\}$ be per-layer permutation matrices acting as $W_\ell\mapsto P_\ell W_\ell P_{\ell-1}^\top$. Every $\pi$ is a loss-preserving reparameterization: $\hat L(\pi(\theta))=\hat L(\theta)$. Define the **aligned barrier** $B^\star(\theta_A,\theta_B)=\min_{\pi\in\mathcal{S}}B(\theta_A,\pi(\theta_B))$. $|\mathcal S|=\prod_\ell h_\ell!$, so $B^\star$ is never computed exactly above toy width; every reported value is an upper bound from a heuristic $\pi$.

**Alignment objectives.** Weight matching maximizes $\sum_\ell \langle W_\ell^A,\,P_\ell W_\ell^B P_{\ell-1}^\top\rangle$ — a sum-of-bilinear-assignments problem, NP-hard in general; solved by coordinate descent over layers with the Hungarian algorithm per layer. Activation matching instead maximizes cross-correlation of unit activations over a probe batch.

**Normalization confound.** For BatchNorm nets, $\hat L(\theta(\alpha))$ is undefined until running statistics are set. Two conventions exist — inherit interpolated statistics, or recompute by a forward pass over training data ("reset BN"). They give different barriers. REPAIR (Jordan et al., 2023) goes further and rescales each interpolated unit so its pre-activation mean/variance matches the interpolation of the endpoints' statistics.

**Assumptions known to be violated.** (i) That $\mathcal{S}$ is the full symmetry group — it is not: scaling symmetries under ReLU+normalization, and sign/rotation symmetries in some layers, are also present and are ignored by standard matching. (ii) That endpoints are equally trained — barriers are asymmetric when the runs differ in final loss. (iii) That the barrier is architecture-independent given width — depth and residual structure change it materially. (iv) That the loss surface is what matters — the objects being connected are functions, and the correspondence between "one basin" and "one function class" is assumed, not shown.

## 3. State of the Art

**Established.**
- Nonlinear connectivity is settled: independently trained modes are joined by low-loss *curved* paths (Garipov et al., NeurIPS 2018; Draxler et al., ICML 2018). Quadratic-Bezier / polygonal chain paths of near-training loss exist for VGG/ResNet on CIFAR-10/100.
- Linear connectivity within a run's own trajectory is settled: after a short "stability" phase, two SGD runs branched from a shared checkpoint with different data order are linearly connected (Frankle, Dziugaite, Roy & Carbin, ICML 2020).
- Alignment reduces the barrier substantially and monotonically with width for small architectures (Entezari et al., ICLR 2022; Ainsworth, Hayase & Srinivasa, ICLR 2023).

**Claimed but unablated / benchmark-only.**
- "Zero barrier" claims for aligned ResNets are reported at specific width multipliers with a specific BN-reset convention, and are sensitive to that convention. Jordan et al. (ICLR 2023) show the residual barrier after weight matching is largely *variance collapse* — interpolated units have shrunken activation variance — and that correcting it (REPAIR) removes most of what remains. Whether the corrected quantity is still "the loss barrier" is an open definitional point, not an ablation.
- ImageNet-scale ResNet-50 remains the honest failure case: aligned barriers are reduced but not eliminated in every published attempt.
- Model soups (Wortsman et al., ICML 2022) and task arithmetic exploit *fine-tuning* connectivity from a shared pretrained initialization — a strictly easier regime than independent seeds, and often cited as if it were evidence for the harder one.

**Theory SOTA.** Ferbach, Goujaud, Gidel & Dieuleveut (AISTATS 2024) prove linear mode connectivity modulo permutation via an optimal-transport argument, for two-layer and multilayer networks, with width requirements that grow with the target $\epsilon$ and degrade sharply with depth. Earlier: Kuditipudi et al. (NeurIPS 2019) explain connectivity via dropout stability and noise stability; Nguyen (ICML 2019) proves connectedness of sublevel sets when one hidden layer exceeds $n$; Simsek et al. (ICML 2021) characterize the permutation-induced geometry of global minima manifolds.

## 4. What Is Known

- **Curved paths, CIFAR-10, VGG-16 / ResNet-164:** Bezier paths between independent modes with train loss along the path at roughly the endpoint level, versus an unaligned linear interpolant that rises to near-chance error at $\alpha=0.5$ (Garipov et al., 2018; Draxler et al., 2018).
- **Instability analysis, CIFAR-10 ResNet-20:** branching at initialization gives an error barrier of several percent; branching after a small fraction of training (order 1–2 epochs) gives barrier $\approx 0$. On ImageNet ResNet-50 the stability point arrives later in training (Frankle et al., ICML 2020). This is the cleanest reproduced regularity in the area.
- **Width dependence, MNIST/CIFAR-10 MLPs and small CNNs:** aligned barrier falls monotonically with width, approaching but not provably reaching zero at the largest widths tested (Entezari et al., ICLR 2022).
- **Alignment at CIFAR-10 scale:** weight matching drives the aligned barrier of wide ResNet-20 (width multiplier 8–32) to near zero; at standard width, and for VGG-16, a nonzero barrier persists (Ainsworth et al., ICLR 2023).
- **Variance collapse:** the midpoint network's internal activation variance is systematically smaller than either endpoint's; rescaling to fix it removes a large majority of the residual barrier across CIFAR-10 and ImageNet architectures (Jordan et al., ICLR 2023).
- **Feature-level connectivity:** layerwise linear *feature* connectivity co-occurs with LMC (Zhou et al., NeurIPS 2023), suggesting the phenomenon is representational, not merely parametric.

## 5. What Is Not Known

- **Theoretically open.** Whether the permutation conjecture holds at *realistic* widths. Existing proofs need widths far above deployed models, and depth dependence is not tight. No lower bound rules out a small-width counterexample; none has been exhibited either.
- **Theoretically open.** Whether the transition from barrier to no-barrier as width grows is sharp (a threshold) or smooth.
- **Empirically open.** Whether an ImageNet ResNet-50 seed pair admits a zero-barrier aligned linear path. Runnable today; the blocker is alignment search quality, not compute for training.
- **Empirically open.** Whether independently pretrained transformer LMs at $\ge 1$B parameters are permutation-connectable at all. Attention heads and residual streams add symmetry structure that per-layer matching does not capture.
- **Methodologically blocked.** What "the barrier" means for normalized networks. BN-reset and REPAIR are corrections applied to the *measurement*; there is no agreed definition that is simultaneously symmetry-respecting, convention-free, and reported by all papers. Cross-paper barrier numbers are therefore not comparable.

## 6. Why It Is Hard

The obstruction is **non-identifiability under an intractable symmetry group combined with a confounded measurement**. $B^\star$ is a minimum over $\prod_\ell h_\ell!$ elements; every published number is an upper bound from a heuristic, so a positive barrier never refutes the conjecture — it may only mean the search failed. Symmetrically, a near-zero barrier does not establish it, because the measurement was taken after a normalization correction (BN reset, REPAIR) whose effect on the barrier is comparable in size to the barrier itself. The falsifier and the confirmer are both blocked, from opposite directions. Compute is secondary: CIFAR-scale experiments are cheap, and the field has run thousands of them without closing the question.

## 7. Current Research (as of 2026)

- Alignment beyond permutations: matching over the fuller symmetry group (scaling, sign, block-rotation in normalized and attention layers), and differentiable/straight-through alignment objectives that optimize the barrier directly rather than a proxy correlation.
- Connectivity of transformers and LLM merging: whether independently pretrained LMs can be aligned, versus the easier fine-tuning-soup regime that current merging practice relies on. *(frontier — verify)*
- Mechanistic framing: whether models on the same basin implement the same circuits, following Lubana et al. (ICML 2023) on mechanistic mode connectivity — low-loss paths can connect mechanistically distinct solutions, which weakens "one basin ⇒ one function".
- Tightening width requirements in the optimal-transport proof, and searching for small-width counterexamples by direct barrier maximization.
- Groups active in this line include Gidel's at Mila, Srinivasa/Hayase at UW, Dziugaite/Roy on instability, and Zhou/Ma on feature-level connectivity. *(frontier — verify current affiliations)*

## 8. Concrete Next Experiment

**Question:** is the residual ImageNet barrier a search failure or a real one?

**Scale.** ResNet-50 on ImageNet-1k, standard recipe, 3 seeds → 3 pairs. Alignment budget: 200 GPU-hours per pair spent purely on searching $\pi$ (weight matching, activation matching on a 50k probe set, straight-through barrier minimization, and simulated annealing over layer-local swaps, all initialized from each other's best).

**Control arm.** The *same* alignment pipeline applied to a pair that is known-connectable: two runs branched from a shared checkpoint at 20% of training with different data order. The control must reach barrier $\le 0.01$ nats; if it does not, the pipeline, not the landscape, is the finding.

**Deciding number.** Test-error barrier at $\alpha=0.5$, reported under three conventions (BN reset, REPAIR, and no correction), as a function of alignment compute. **Decision rule:** if the barrier under BN-reset-only plateaus above $2$ percentage points while still falling on the control arm, independently trained ResNet-50s are *not* permutation-connected at standard width — a counterexample at deployed scale. If it falls below $0.5$ points, the remaining gap is search, and the conjecture survives at scale.

Cost: order 1,000 GPU-hours total, dominated by search, not training.

## 9. Key References

- **[Foundational]** Garipov, Izmailov, Podoprikhin, Vetrov & Wilson. *Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs.* NeurIPS, 2018. — arXiv:1802.10026
- **[Foundational]** Draxler, Veschgini, Salmhofer & Hamprecht. *Essentially No Barriers in Neural Network Energy Landscape.* ICML, 2018. — arXiv:1803.00885
- **[Foundational]** Frankle, Dziugaite, Roy & Carbin. *Linear Mode Connectivity and the Lottery Ticket Hypothesis.* ICML, 2020. — arXiv:1912.05671
- **[SOTA — conjecture]** Entezari, Sedghi, Saukh & Neyshabur. *The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks.* ICLR, 2022. — arXiv:2110.06296
- **[SOTA — method]** Ainsworth, Hayase & Srinivasa. *Git Re-Basin: Merging Models modulo Permutation Symmetries.* ICLR, 2023. — arXiv:2209.04836
- **[SOTA — measurement]** Jordan, Sedghi, Saukh, Entezari & Neyshabur. *REPAIR: REnormalizing Permuted Activations for Interpolation Repair.* ICLR, 2023. — arXiv:2211.08403
- **[SOTA — theory]** Ferbach, Goujaud, Gidel & Dieuleveut. *Proving Linear Mode Connectivity of Neural Networks via Optimal Transport.* AISTATS, 2024.
- Kuditipudi, Wang, Lee, Zhang, Li, Hu, Ge & Arora. *Explaining Landscape Connectivity of Low-cost Solutions for Multilayer Nets.* NeurIPS, 2019.
- Simsek, Ged, Jacot, Spadaro, Hongler, Gerstner & Brea. *Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances.* ICML, 2021.
- Lubana, Bigelow, Dick, Krueger & Tanaka. *Mechanistic Mode Connectivity.* ICML, 2023.
- Zhou, Yang, Yang, Yan & Wang. *Going Beyond Linear Mode Connectivity: The Layerwise Linear Feature Connectivity.* NeurIPS, 2023.
- **[Applied]** Wortsman et al. *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy Without Increasing Inference Time.* ICML, 2022. — arXiv:2203.05482
- **[Survey]** Nguyen. *On Connected Sublevel Sets in Deep Learning.* ICML, 2019.

## 10. Worked Example

Take a 3-layer MLP on MNIST, hidden widths $h_1=h_2=512$, two seeds, each reaching $\approx1.7\%$ test error.

**Unaligned.** Midpoint test error $\approx 60$–$90\%$ — near chance. Barrier is enormous.

**Aligned.** Weight matching by layerwise Hungarian coordinate descent converges in a few passes. Midpoint error drops to a few percent above the endpoints. The number that matters is what is *left*.

**Where the obstruction becomes visible.** The search space is $512!\times512!\approx 10^{2340}$. Coordinate descent returns a local optimum of a bilinear objective; restarting from 100 random initializations gives 100 different $\pi$ with a spread of midpoint errors. Suppose the best gives a $0.9$-point barrier. Two readings are consistent with this:

1. $B^\star=0$ and the search left $0.9$ points on the table;
2. $B^\star=0.9$ and the search is optimal.

Nothing in the experiment distinguishes them. Now widen to $h=4096$: the barrier falls to $\approx0.1$ points — evidence for reading (1) — but the search space grew to $(4096!)^2$, so search quality also changed. **The width knob and the search-difficulty knob move together and cannot be separated by this measurement.** That coupling, not compute, is why the conjecture has stood since 2022 with strong supporting evidence and no proof at practical width.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*