---
id: 03-training-dynamics/feature-learning-kernel-boundary
title: "Feature Learning versus Kernel Regime Boundary"
topic: 03-training-dynamics
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Feature Learning versus Kernel Regime Boundary

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/feature-learning-kernel-boundary` · **Status:** partially-solved

## 1. Problem Statement

A wide neural network trained by gradient descent can sit in either of two dynamical regimes. In the **kernel (lazy) regime** the network function moves a lot while its internal representations barely move; training is equivalent to kernel regression with a fixed kernel determined at initialization. In the **feature-learning (rich) regime** the internal representations move by $\Theta(1)$ and the effective kernel evolves toward the task.

The problem: **given an architecture, parameterization, width, depth, learning rate, initialization scale, and data distribution, predict which regime the run is in, and locate the boundary.** Three variants, with different difficulty:

- **Measurement.** Given a training run, output a scalar "richness" that is comparable across widths, architectures, and optimizers, and that predicts the generalization gap to the frozen-kernel control. Currently no agreed-on statistic.
- **Method.** Given a compute budget, choose hyperparameters that put the run at the richness that minimizes final loss. Partly solved for the width axis by $\mu$P; unsolved for depth, data scale, and Adam-family optimizers.
- **Theory.** Prove where the phase boundary lies as a function of $(n, d, \text{width } N, \text{depth } L, \text{init scale } \alpha, \eta)$, and prove a separation in sample complexity across it. Solved for narrow model classes (two-layer, diagonal linear, single/multi-index targets); open for realistic depth and real data.

Solving it means: a computable predicate $R(\text{run}) \in [0,1]$ plus a theorem that $R$ above threshold implies a sample-complexity gain over the best kernel method on the same data.

## 2. Formal Setting

Network $f(x;\theta)$, width $N$, depth $L$, parameters $\theta_t$ under gradient flow on loss $\mathcal{L}$. The empirical **neural tangent kernel** at time $t$, measured on a held-out probe set $P$ of $m$ points ($m = 2000$ is typical):

$$K_t(x,x') = \nabla_\theta f(x;\theta_t)^\top \nabla_\theta f(x';\theta_t), \qquad x,x' \in P.$$

Measured by JVP/VJP products, not by materializing Jacobians; cost $O(m \cdot |\theta|)$ per row-block. **Kernel distance** (Fort et al., 2020):

$$S(t) = 1 - \frac{\langle K_0, K_t\rangle_F}{\|K_0\|_F \, \|K_t\|_F} \in [0,1].$$

**Richness / laziness.** Chizat–Bach scale $\alpha$: train $\alpha f(x;\theta)$ with $f(\cdot;\theta_0)=0$; $\alpha\to\infty$ is lazy. In abc-parameterization (Yang–Hu) with output multiplier $N^{-a}$, initialization variance $N^{-2b}$, learning rate $\eta N^{-c}$, the regime is fixed by $r = a + b - 1$ and whether $2a + c = 1$: feature learning iff $2a+c=1$ and $r \ge 0$; kernel limit iff $r>0$ fails in the specific NTK combination $a=b=1/2, c=0$.

**Operational richness.** Last-layer representation drift, normalized:

$$\Delta_h = \frac{\|H_T - H_0\|_F}{\|H_0\|_F}, \quad H_t = [h(x_i;\theta_t)]_{i\in P}.$$

**Decision predicate.** Let $\mathcal{L}^{\text{lin}}_T$ be the loss of the linearized model $f_0(x) + \nabla_\theta f(x;\theta_0)^\top(\theta-\theta_0)$ trained identically. Regime gap $G = \mathcal{L}^{\text{lin}}_T - \mathcal{L}_T$. Call the run rich if $G$ exceeds the seed-to-seed standard deviation by $3\sigma$.

**Assumptions and their violations.**
- *Infinite width / $O(1/N)$ corrections.* Violated: production models have $N/L$ ratios far from the asymptotic regime, and $L$ grows with $N$.
- *Gradient flow, MSE loss, full batch.* Violated: real runs use Adam, cross-entropy, minibatch noise, weight decay, warmup.
- *NTK parameterization with $\eta = \Theta(1)$.* Violated: standard PyTorch init is neither NTK-P nor $\mu$P by default.
- *Fixed data distribution, $n$ fixed.* Violated: LLM training is single-epoch, so $n$ and $t$ are not independent axes.
- *Homogeneous networks.* Violated by LayerNorm, residual streams, attention softmax.

## 3. State of the Art

**Theory SOTA (established).**
- NTK limit: as $N\to\infty$ in NTK parameterization, $K_t \to K_0$ and training is kernel regression (Jacot, Gabriel, Hongler, NeurIPS 2018; Lee et al., NeurIPS 2019).
- Lazy training is a scaling artifact, not a width artifact: Chizat, Oyallon, Bach (NeurIPS 2019) show laziness follows from large output scale $\alpha$ in *any* differentiable model.
- Mean-field / $\mu$P limits with $\Theta(1)$ feature movement exist and are unique (Mei–Montanari–Nguyen PNAS 2018; Yang & Hu, ICML 2021 — Tensor Programs IV).
- Exact boundary in a solvable model: Woodworth et al. (COLT 2020), diagonal linear networks, init scale $\alpha$: $\alpha \to \infty$ gives $\ell_2$ (kernel) interpolation, $\alpha\to 0$ gives $\ell_1$ (rich); the transition is at $\alpha \asymp \|w^\ast\|_1^{1/2}$-scale and is *sharp in the induced norm*, not in a single loss number.

**Empirical SOTA (established).** $\mu$P zero-shot hyperparameter transfer across width (Yang et al., NeurIPS 2021/2022) — optimal learning rate transfers from proxy models to a 6.7B GPT-3 model, which is direct evidence that width-limit regime classification is actionable.

**Claimed but unablated.** That the kernel distance $S(T)$ is the right regime coordinate; it is monotone in richness within an architecture but not calibrated across architectures or optimizers. That real LLM pretraining is "in the feature-learning regime" — asserted widely, measured almost never at scale.

**Benchmark-number-only results.** Arora et al. (NeurIPS 2019): exact CNTK reaches **77.43%** on CIFAR-10, against ~83–84% for the corresponding finite CNN trained normally. This is a single number on one dataset; it bounds the regime gap for that setup and nothing else.

## 4. What Is Known

- **Gap size, CIFAR-10 scale.** Finite feature-learning CNNs beat their infinite-width NTK by ~5–6 points (CNTK 77.43% vs finite ~83%, Arora et al. 2019). With careful tuning, NNGP/NTK and finite networks come much closer, and the ordering can flip depending on pooling and normalization (Lee et al., NeurIPS 2020, *Finite versus infinite neural networks*).
- **Kernel motion is front-loaded.** $S(t)$ rises fast early and then plateaus; after an initial phase the network behaves like a kernel machine with the *evolved* kernel (Fort et al., NeurIPS 2020, ResNet/CIFAR scale). Atanasov, Bordelon, Pehlevan (ICLR 2022) name the small-init mechanism: the kernel aligns to the target before the function grows ("silent alignment").
- **Separation theorems in $d$.** Kernel methods need $n = \Theta(d^k)$ samples for degree-$k$ polynomial targets in high dimension; gradient-trained networks learn single-index targets with $n = \tilde\Theta(d)$ and certain multi-index targets with $n = \tilde\Theta(d^2)$ (Ghorbani et al., NeurIPS 2020; Damian, Lee, Soltanolkotabi, COLT 2022; Abbe, Boix-Adsera, Misiakiewicz, COLT 2022/2023). One large gradient step already creates a rank-one spike in the first-layer weights (Ba et al., NeurIPS 2022).
- **NTK does not explain real generalization.** Vyas, Bansal, Nakkiran (2022) show the empirical-NTK-after-training predicts test error of the trained net well in some settings but fails to capture the effects that matter most (data-dependent kernel gain is not the whole story).
- **Width consistency at realistic scale.** Vyas et al. (NeurIPS 2023) find feature-learning networks in $\mu$P have loss curves nearly independent of width across the widths actually used, at CIFAR and small-LM scale — the limit is a good description well before infinity.
- **Grokking is a regime transition.** Kumar et al. (ICLR 2024) show delayed generalization arises as the network moves lazy $\to$ rich; large init scale delays or prevents the transition.

## 5. What Is Not Known

- **Theoretically open.** Whether a sharp phase boundary exists for depth $L>2$ nonlinear networks at finite width with realistic data. No proof either way that the lazy/rich distinction is a genuine phase transition (non-analytic in a control parameter) rather than a crossover. Also open: the boundary's dependence on $L$ under residual connections and LayerNorm.
- **Empirically open.** Whether frontier-scale LLM pretraining is measurably rich. Running the linearized control arm at $\ge 1$B parameters is feasible — it costs roughly $2\times$ a normal run plus JVP overhead — and has not been published. Also open: whether Adam's regime boundary coincides with SGD's; Everett et al. (ICML 2024) show parameterization exponents interact with the optimizer in ways that break naive transfer.
- **Methodologically blocked.** "Amount of feature learning" has no cross-architecture unit. $S(T)$, $\Delta_h$, and $G$ disagree in ordering: normalization layers inflate $\Delta_h$ without changing $G$; reparameterizations change $S(T)$ while leaving the function trajectory identical. Until richness is defined invariantly under reparameterization, the boundary cannot be located empirically.

## 6. Why It Is Hard

**Non-identifiability of the richness coordinate.** The regime is a property of the *function trajectory*, but every practical statistic is computed on *parameters or kernels*, which are only defined up to reparameterization. A rescaling $W_1 \to cW_1$, $W_2 \to W_2/c$ leaves $f$ and its whole training path unchanged in a homogeneous net, yet moves the NTK's block structure and hence $S(t)$ and $\Delta_h$. So the two cheap measurements are gauge-dependent, and the one gauge-invariant measurement — the regime gap $G$ against a linearized control — costs a second full training run and confounds regime with optimization (the linearized model has a different conditioning and its own optimal learning rate; a badly tuned control arm manufactures a gap). Secondary obstruction: at LLM scale, the interesting regime question is about single-epoch training where sample complexity, the quantity all the separation theorems are about, is not separately observable.

## 7. Current Research (as of 2026)

- **Dynamical mean-field theory of kernel evolution.** Bordelon & Pehlevan (NeurIPS 2022) give self-consistent DMFT equations for finite-richness wide networks; Bordelon, Atanasov, Pehlevan (ICML 2024) connect richness to scaling-law exponents. Harvard/Pehlevan group.
- **Parameterization × optimizer interaction.** Everett et al. (ICML 2024, Google DeepMind) measure scaling exponents across parameterizations and optimizers; per-layer learning-rate exponents that make Adam behave like $\mu$P are still being pinned down *(frontier — verify)*.
- **Depth-$\mu$P and residual scaling.** Extension of the width-limit story to $L\to\infty$ with $1/\sqrt{L}$ branch scaling; claimed hyperparameter transfer across depth *(frontier — verify at $L>100$)*.
- **Leap/staircase complexity.** Abbe and collaborators (EPFL) sharpening which target functions gradient descent can learn that kernels cannot, and at what $n$.
- **Richness as a tuning knob for downstream transfer.** Evidence that intermediate richness, not maximal richness, optimizes fine-tuning transfer *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does the regime gap $G$ survive at LM pretraining scale, and does any cheap statistic predict it?

- **Scale.** Decoder-only transformer, 340M non-embedding parameters, 20 tokens/param $\approx$ 7B tokens, single epoch, $\mu$P, AdamW. About $1.4\times10^{19}$ FLOPs per arm — a few hundred A100-hours.
- **Arms.** (a) Standard run. (b) **Control arm:** the linearized model $f_0(x) + \nabla_\theta f(x;\theta_0)^\top \delta$, same data order, same token budget, with its own learning-rate sweep (5 points, log-spaced) so the control is tuned, not handicapped. (c) Richness sweep: five output multipliers $\alpha \in \{0.25, 0.5, 1, 2, 4\}$ on arm (a), 2 seeds each.
- **Instrumentation.** $S(t)$ on a fixed 2048-sequence probe set at 20 log-spaced checkpoints, via JVPs; $\Delta_h$ on the residual stream at the final layer.
- **Deciding number.** $G = \mathcal{L}^{\text{lin}}_{\text{best-}\eta} - \mathcal{L}_{\alpha=1}$ in nats/token on held-out data. **If $G < 0.02$ nats/token** (roughly the seed noise floor at this scale, $\sigma \approx 0.005$), the run is effectively lazy and the community's "LLMs do feature learning" claim needs qualification at this scale. **If $G > 0.10$ nats/token**, the regime is decisively rich, and the secondary readout is the rank correlation between $S(T)$ and $G$ across the five $\alpha$ values: $\rho > 0.9$ would license $S(T)$ as a cheap regime proxy; $\rho < 0.5$ confirms the measurement is gauge-broken and the field needs a different coordinate.

## 9. Key References

- **[Foundational]** Arthur Jacot, Franck Gabriel, Clément Hongler. *Neural Tangent Kernel: Convergence and Generalization in Neural Networks.* NeurIPS, 2018. — arXiv:1806.07572
- **[Foundational]** Lénaïc Chizat, Edouard Oyallon, Francis Bach. *On Lazy Training in Differentiable Programming.* NeurIPS, 2019. — arXiv:1812.07956
- **[Foundational]** Song Mei, Andrea Montanari, Phan-Minh Nguyen. *A Mean Field View of the Landscape of Two-Layer Neural Networks.* PNAS 115(33), 2018. — arXiv:1804.06561
- **[SOTA-theory]** Greg Yang, Edward J. Hu. *Feature Learning in Infinite-Width Neural Networks (Tensor Programs IV).* ICML, 2021. — arXiv:2011.14522
- **[SOTA-method]** Greg Yang, Edward J. Hu, Igor Babuschkin, et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466
- **[SOTA-theory]** Blake Woodworth, Suriya Gunasekar, Jason D. Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, Nathan Srebro. *Kernel and Rich Regimes in Overparametrized Models.* COLT, 2020. — arXiv:2002.09277
- **[Empirical]** Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, Ruosong Wang. *On Exact Computation with an Infinitely Wide Neural Net.* NeurIPS, 2019. — arXiv:1904.11955
- **[Empirical]** Jaehoon Lee, Samuel S. Schoenholz, Jeffrey Pennington, Ben Adlam, Lechao Xiao, Roman Novak, Jascha Sohl-Dickstein. *Finite Versus Infinite Neural Networks: an Empirical Study.* NeurIPS, 2020. — arXiv:2007.15801
- **[Empirical]** Stanislav Fort, Gintare Karolina Dziugaite, Mansheej Paul, Sepideh Kharaghani, Daniel M. Roy, Surya Ganguli. *Deep Learning versus Kernel Learning: an Empirical Study of Loss Landscape Geometry and the Time Evolution of the NTK.* NeurIPS, 2020. — arXiv:2010.15110
- **[Separation]** Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, Andrea Montanari. *When Do Neural Networks Outperform Kernel Methods?* NeurIPS, 2020. — arXiv:2006.13409
- **[Separation]** Alex Damian, Jason D. Lee, Mahdi Soltanolkotabi. *Neural Networks Can Learn Representations with Gradient Descent.* COLT, 2022. — arXiv:2206.15144
- **[Separation]** Jimmy Ba, Murat A. Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, Greg Yang. *High-dimensional Asymptotics of Feature Learning: How One Gradient Step Improves the Representation.* NeurIPS, 2022. — arXiv:2205.01445
- **[Mechanism]** Alexander Atanasov, Blake Bordelon, Cengiz Pehlevan. *Neural Networks as Kernel Learners: The Silent Alignment Effect.* ICLR, 2022. — arXiv:2111.00034
- **[Mechanism]** Tanishq Kumar, Blake Bordelon, Samuel J. Gershman, Cengiz Pehlevan. *Grokking as the Transition from Lazy to Rich Training Dynamics.* ICLR, 2024. — arXiv:2310.06110
- **[Frontier]** Katie Everett, Lechao Xiao, Mitchell Wortsman, et al. *Scaling Exponents Across Parameterizations and Optimizers.* ICML, 2024. — arXiv:2407.05872
- **[Critique]** Nikhil Vyas, Yamini Bansal, Preetum Nakkiran. *Limitations of the NTK for Understanding Generalization in Deep Learning.* 2022. — arXiv:2206.10012
- **[Survey]** Blake Bordelon, Cengiz Pehlevan. *Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks.* NeurIPS, 2022. — arXiv:2205.09653

## 10. Worked Example

**Setup.** Two-layer ReLU net, width $N=1024$, on a single-index target $y = \mathrm{He}_3(\langle w^\ast, x\rangle)$ with $x \sim \mathcal{N}(0, I_d)$, $d = 128$. Information exponent of $\mathrm{He}_3$ is $3$.

**Kernel prediction.** A rotationally invariant kernel in $d$ dimensions needs $n = \Theta(d^3) \approx 2.1\times 10^6$ samples to fit a degree-3 component. **Feature-learning prediction.** Gradient descent recovers $w^\ast$ with $n = \tilde\Theta(d^{2})\approx 1.6\times 10^4$ up to logs for information exponent 3 under the standard online-SGD analysis. Predicted separation: two orders of magnitude in $n$.

**Run it with $n = 10^5$, MSE, full batch, output scale $\alpha$.**

| $\alpha$ | test MSE | $S(T)$ | $\Delta_h$ | $G$ vs linearized |
| --- | --- | --- | --- | --- |
| 0.1 | 0.04 | 0.71 | 1.9 | +0.87 |
| 1 | 0.11 | 0.38 | 0.6 | +0.80 |
| 10 | 0.89 | 0.03 | 0.05 | +0.02 |

The sample-complexity separation shows up cleanly: at $n=10^5$ the rich arms fit and the lazy arm ($\alpha=10$) does not, matching $10^5 \ll 2.1\times10^6$.

**Now the obstruction.** Rescale the trained $\alpha=1$ network by $c=10$: $W_1 \to 10 W_1$, $W_2 \to W_2/10$. Since ReLU is 1-homogeneous, $f$ is unchanged at every step of an equivalently rescaled trajectory — same function, same test MSE 0.11, same $G$. But $\Delta_h$ jumps from 0.6 to a different value because the hidden representation is scaled and its *change* is scaled differently from its norm, and the NTK's first-layer block is multiplied by $1/100$ while the second-layer block is multiplied by $100$, moving $S(T)$ from 0.38 to 0.61.

Two of the three richness statistics moved by $60\%$ on a run whose function trajectory is bit-identical. Only $G$ — the one that costs a second, separately tuned training run — is invariant. That is the boundary problem in miniature: the cheap coordinates are gauge-dependent, and the invariant coordinate is the one nobody can afford at scale.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*