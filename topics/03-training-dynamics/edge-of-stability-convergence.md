---
id: 03-training-dynamics/edge-of-stability-convergence
title: "Edge of Stability Convergence Theory"
topic: 03-training-dynamics
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Edge of Stability Convergence Theory

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/edge-of-stability-convergence` · **Status:** open

## 1. Problem Statement

Full-batch gradient descent on neural networks does not obey the descent lemma. Classical analysis requires step size $\eta < 2/L$ where $L$ is the smoothness constant. In practice the largest Hessian eigenvalue (the *sharpness*) rises until it reaches $\approx 2/\eta$, then stays pinned there while the loss keeps falling non-monotonically over long horizons. Cohen et al. (ICLR 2021) named this the Edge of Stability (EoS).

Three separable variants:

- **Theory.** Give a convergence theorem for GD with fixed $\eta$ on a function class that contains realistic networks, covering the regime $\eta \lambda_1 > 2$: a rate on $\min_{t\le T} L(\theta_t)$ or on $L(\theta_T) - L^\star$, with constants that do not degrade to vacuity when $\lambda_1$ is unbounded. No such theorem exists for anything beyond low-dimensional or heavily structured models.
- **Method.** Predict, from the loss geometry alone, what step size a given architecture will tolerate, and whether raising $\eta$ improves or destroys generalization — without running the sweep.
- **Measurement.** Define EoS for stochastic, adaptive, batch-normalized, warmup-scheduled training, where $\lambda_1$ is estimated by Lanczos on minibatches and the "stability threshold" is no longer $2/\eta$.

Solving the theory variant means: a non-asymptotic bound proving that self-stabilizing GD converges to a $2/\eta$-stable minimum, with an explicit dependence on the third derivative of the loss along the top eigenvector.

## 2. Formal Setting

Let $\theta \in \mathbb{R}^p$, loss $L: \mathbb{R}^p \to \mathbb{R}$ twice differentiable, GD update $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$.

**Sharpness.** $\lambda_1(t) := \lambda_{\max}(\nabla^2 L(\theta_t))$. Measured by Lanczos or power iteration on Hessian-vector products (double backprop), on the *full* training set. On a 5k-example CIFAR-10 subset with a small CNN this is $\sim$20 HVPs per estimate; on a 7B-parameter model over 100B tokens it is not measurable at all at per-step resolution.

**Stability ratio.** $\rho(t) := \eta \lambda_1(t) / 2$. Classical descent requires $\rho < 1$; EoS is the empirical regime $\rho \approx 1$ (typically $1.0$–$1.1$).

**Progressive sharpening.** $\lambda_1(t)$ increasing while $\rho < 1$. This is a property of the network parameterization, not the loss: for squared loss with model $f_\theta$, $\nabla^2 L = \frac{1}{n}\sum_i \nabla f_i \nabla f_i^\top + \frac{1}{n}\sum_i r_i \nabla^2 f_i$ (Gauss–Newton plus residual term), and sharpening is driven mostly by the Gauss–Newton block growing as the network fits.

**Self-stabilization** (Damian, Nichani, Lee, ICLR 2023). Write $u$ for the top eigenvector, $x := \langle \theta - \theta^\star, u\rangle$ the oscillation coordinate, and
$$\alpha := \langle \nabla L, \nabla \lambda_1 \rangle, \qquad \beta := \|\nabla \lambda_1\|^2 .$$
The claim is that GD tracks a constrained trajectory: gradient flow on $L$ subject to $\lambda_1(\theta)\le 2/\eta$, with the constraint enforced by the cubic term $\tfrac12 x^2 \nabla \lambda_1$ that the oscillation itself generates. Convergence needs $\langle \nabla L, \nabla\lambda_1\rangle$ bounded and $\nabla\lambda_1 \ne 0$.

**Central flow** (Cohen et al., ICLR 2025). Model the oscillating iterate as $\theta_t = \bar\theta_t + \delta_t$ with $\mathbb{E}[\delta\delta^\top] = \Sigma$, and evolve the average $\bar\theta$ by
$$\dot{\bar\theta} = -\nabla\big(L(\bar\theta) + \tfrac12\langle \Sigma, \nabla^2 L(\bar\theta)\rangle\big),$$
with $\Sigma$ solving a complementarity condition that keeps every oscillating eigenvalue exactly at $2/\eta$.

**Assumptions known to be violated in practice.**
- $L$-smoothness with finite global $L$: false; $\lambda_1$ grows unboundedly along the trajectory.
- Isolated top eigenvalue: false late in training — 2 to 10 eigenvalues sit at the $2/\eta$ threshold simultaneously, so the scalar analysis is wrong.
- Full batch: essentially never used. With SGD the threshold is not $2/\eta$ and depends on gradient noise.
- Fixed $\eta$, no momentum, no adaptivity, no normalization layers: violated by every production run.
- Twice differentiability: violated by ReLU, though this appears benign in practice.

## 3. State of the Art

**Theory SOTA (established).**
- Ahn, Zhang, Sra (ICML 2022): unstable convergence provably occurs even for simple functions; identify forward-invariant sets and show a "relative progress" quantity still decreases.
- Arora, Li, Panigrahi (ICML 2022): for normalized/normalizable losses and small $\eta$, GD after entering EoS provably tracks a *sharpness-reduction flow* $\dot\theta \propto -\nabla \lambda_1$ on the manifold of minimizers. Rigorous, but requires a $\sqrt{\eta}$-scale separation and a manifold of minimizers.
- Damian, Nichani, Lee (ICLR 2023): self-stabilization theorem — under a cubic Taylor model and explicit conditions on $\alpha,\beta$, GD's trajectory stays within $O(\eta)$ of the constrained flow for a bounded time window. This is the strongest general statement; it is local and finite-horizon, not a convergence theorem.
- Chen & Bruna (ICML 2023), Zhu et al. (ICLR 2023), Song & Yun (NeurIPS 2023): exact analyses of 2-to-4-parameter models ($\ell(xy)$, single-neuron nets) via period-doubling bifurcation. Complete, but the models are toys.

**Empirical SOTA (established).** Cohen et al. (ICLR 2021) is the reference measurement across MLPs, CNNs, and transformers on CIFAR-10 subsets. Cohen et al. (2022) extends to Adam/RMSProp, where the relevant object is $\lambda_{\max}$ of the *preconditioned* Hessian $P^{-1}\nabla^2 L$ and the threshold is $\approx 38/\eta$ for default $\beta_2$.

**Claimed but unablated.** That EoS is causally responsible for the generalization benefit of large learning rates. The correlation (large $\eta$ → flatter minimum → better test error) is repeatedly observed, but no ablation isolates the oscillation from the effective-step-size and noise confounds. Central flows predict optimizer behavior well on the reported nets; the coverage across architecture families is a set of benchmark curves, not a tested general law.

## 4. What Is Known

- **Sharpness pins at $2/\eta$, not below.** Cohen et al. (ICLR 2021), full-batch GD, 5k-example CIFAR-10 subsets, MLPs and VGG-style CNNs: across $\eta$ spanning roughly two orders of magnitude, $\lambda_1$ equilibrates in the band $\eta\lambda_1 \in [2, 2.2]$ and stays there for the rest of training.
- **Loss falls non-monotonically.** Over 1-to-10-step windows the loss rises; over 100-step windows it falls. Training reaches near-zero training loss anyway.
- **The threshold is optimizer-specific.** Cohen et al. (2022): Adam's preconditioned sharpness equilibrates near $38/\eta$ (from the stability boundary of the momentum + preconditioner linear system), verified on transformers and CNNs.
- **Warmup works by sharpness reduction.** Gilmer et al. (ICLR 2022) show learning-rate warmup and gradient clipping act by keeping $\eta\lambda_1$ below the instability threshold early; on ResNets, the interventions that fix divergence are exactly those that lower $\eta\lambda_1$.
- **Minima that GD can reach are constrained.** Mulayoff, Michaeli, Soudry (NeurIPS 2021) and Wu, Ma, E (NeurIPS 2018): only $2/\eta$-stable minima are reachable, which by itself bounds the flatness of the endpoint.
- **Sharpness dynamics have reproducible phases.** Kalra & Barkeshli (NeurIPS 2023): early sharpness reduction, then progressive sharpening, then EoS, with phase boundaries scaling predictably in width and initialization scale, measured on MLPs and CNNs up to a few million parameters.

## 5. What Is Not Known

- **Theoretically open.** No convergence theorem for GD at EoS on a function class containing multi-layer networks. Self-stabilization is a local, finite-horizon tracking result under a cubic truncation; nobody has proven the truncation error stays controlled indefinitely, nor handled the degenerate case $\nabla\lambda_1 \approx 0$, nor the multi-eigenvalue case where 5+ eigenvalues sit at threshold at once.
- **Theoretically open.** Why progressive sharpening happens. It is universally observed and has no proof outside two-layer and toy settings.
- **Empirically open.** Whether EoS occurs at LLM pretraining scale under standard recipes. Measuring $\lambda_{\max}$ of the preconditioned Hessian per step on a 7B model over 100B tokens is affordable at coarse cadence (every $10^3$ steps) but has not been published at that scale with the step-size sweep needed to test the $\approx 38/\eta$ law.
- **Empirically open.** Whether the oscillation *causes* the generalization gain, separably from effective learning rate. Runnable: hold $\eta$ fixed and damp the top-eigenvector oscillation by projecting it out.
- **Methodologically blocked.** "Sharpness" under SGD with batch norm. BN makes $\lambda_1$ scale-dependent (rescaling weights rescales curvature at fixed function), so $\eta\lambda_1$ is not gauge-invariant. There is no agreed batch-stochastic stability threshold to compare against.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the mechanism under a confounded measurement**, compounded by scale.

Raising $\eta$ changes at least four things at once: the stability threshold $2/\eta$, the effective per-step distance travelled, the SGD noise scale $\eta/B$, and the implicit-regularization strength $\eta \|\nabla L\|^2/4$ in the backward-error (modified-equation) sense. Every EoS/generalization claim varies $\eta$ and therefore varies all four. No published design separates "the iterate oscillates along $u_1$" from "the effective step is larger."

Second, the theory needs third-order information. Self-stabilization turns on $\nabla \lambda_1$ — a third derivative of $L$. Estimating $\nabla\lambda_1$ requires differentiating through a Lanczos solve; it is $\sim$50–100× the cost of a gradient, and its variance under minibatching has not been characterized. Theorems are stated in terms of a quantity nobody routinely measures.

Third, the scalar picture is wrong where it matters. At convergence the Hessian spectrum has a cluster at threshold, so the 1-D bifurcation analyses that give the clean toy answers do not extend; the multi-dimensional version is a complementarity problem, which is why central flows are numerically integrated rather than solved in closed form.

## 7. Current Research (as of 2026)

- **Central flows** — Cohen, Damian, Talwalkar, Kolter, Lee (ICLR 2025). Differential-equation models that predict time-averaged trajectories of GD, RMSProp, and Adam including the oscillation's effect. The most promising route to a predictive theory; convergence guarantees for the flow itself are not established. *(frontier — verify current extensions to distributed/large-batch settings.)*
- **Bifurcation-theoretic exact analyses** — Song & Yun (KAIST), Chen & Bruna (NYU). Push toy models to more parameters, aiming for a trajectory-alignment theorem.
- **Adaptive-optimizer stability thresholds** — extending the $38/\eta$ result to Adam variants, Muon, and Shampoo-family preconditioners. *(frontier — verify.)*
- **Sharpness-aware training** — SAM (Foret et al., ICLR 2021) and successors, read as explicit versions of the implicit sharpness reduction EoS provides.
- **Warmup and instability at LLM scale** — loss-spike diagnosis via curvature. Public evidence is largely anecdotal from training reports rather than controlled sweeps.

## 8. Concrete Next Experiment

**Question.** Does the EoS oscillation itself contribute to generalization, or only the larger effective step?

**Scale.** ResNet-18 on CIFAR-10, full 50k training set, full-batch GD (or batch size 12,500 with gradient accumulation), 4,000 steps, 5 seeds. Roughly 200 A100-hours total including curvature estimation. Deliberately small: the confound, not the scale, is the blocker.

**Arms.**
1. **EoS arm (treatment).** Plain GD at $\eta_0$ chosen so training enters EoS by step ~300.
2. **Control arm (oscillation-damped).** Identical $\eta_0$, but each step project the update off the top-$k$ Hessian eigenvectors ($k=5$, refreshed by Lanczos every 20 steps) and take a small stable step $\eta_{\text{sub}} = 0.5 \cdot 2/\lambda_1$ in that subspace. Same $\eta$, same data order, same seed. This removes the oscillation while holding the nominal step size fixed.
3. **Second control (step-size-matched).** Plain GD at $\eta' < \eta_0$ tuned so that the *average per-step parameter displacement* $\|\theta_{t+1}-\theta_t\|$ matches arm 2's. Separates "smaller effective step" from "no oscillation."

**Deciding number.** Final test accuracy of arm 1 minus arm 2, averaged over 5 seeds, with seed standard deviation reported. A gap $\ge 1.0$ percentage point with $\text{SD} \le 0.3$ pp establishes that the oscillation contributes beyond effective step size. A gap $\le 0.3$ pp says EoS is epiphenomenal for generalization and the theory question is purely about convergence, not implicit bias.

**Secondary readout.** $\eta\lambda_1$ trace for all arms, confirming arm 2 holds $\eta\lambda_1 < 2$ throughout while arm 1 pins at $\approx 2.05$.

## 9. Key References

- **[Foundational]** Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar. *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.* ICLR, 2021. — arXiv:2103.00065
- **[Foundational]** Aitor Lewkowycz, Yasaman Bahri, Ethan Dyer, Jascha Sohl-Dickstein, Guy Gur-Ari. *The large learning rate phase of deep learning: the catapult mechanism.* 2020. — arXiv:2003.02218
- **[SOTA — theory]** Alex Damian, Eshaan Nichani, Jason D. Lee. *Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability.* ICLR, 2023. — arXiv:2209.15594
- **[SOTA — theory]** Sanjeev Arora, Zhiyuan Li, Abhishek Panigrahi. *Understanding Gradient Descent on the Edge of Stability in Deep Learning.* ICML, 2022. — arXiv:2205.09745
- **[SOTA — predictive model]** Jeremy M. Cohen, Alex Damian, Ameet Talwalkar, J. Zico Kolter, Jason D. Lee. *Understanding Optimization in Deep Learning with Central Flows.* ICLR, 2025. — arXiv:2410.24206
- **[Theory]** Kwangjun Ahn, Jingzhao Zhang, Suvrit Sra. *Understanding the unstable convergence of gradient descent.* ICML, 2022.
- **[Adaptive optimizers]** Jeremy M. Cohen, Behrooz Ghorbani, Shankar Krishnan, Naman Agarwal, Sourabh Medapati, Michal Badura, Daniel Suo, David Cardoze, Zachary Nado, George E. Dahl, Justin Gilmer. *Adaptive Gradient Methods at the Edge of Stability.* 2022. — arXiv:2207.14484
- **[Empirical]** Justin Gilmer, Behrooz Ghorbani, Ankush Garg, Sneha Kudugunta, Behnam Neyshabur, David Cardoze, George E. Dahl, Zachary Nado, Orhan Firat. *A Loss Curvature Perspective on Training Instability in Deep Learning.* ICLR, 2022. — arXiv:2110.04369
- **[Toy-model analysis]** Xingyu Zhu, Zixuan Wang, Xiang Wang, Mo Zhou, Rong Ge. *Understanding Edge-of-Stability Training Dynamics with a Minimalist Example.* ICLR, 2023.
- **[Toy-model analysis]** Minhak Song, Chulhee Yun. *Trajectory Alignment: Understanding the Edge of Stability Phenomenon via Bifurcation Theory.* NeurIPS, 2023.
- **[Stability of minima]** Rotem Mulayoff, Tomer Michaeli, Daniel Soudry. *The Implicit Bias of Minima Stability: A View from Function Space.* NeurIPS, 2021.
- **[Phenomenology]** Dayal Singh Kalra, Maissam Barkeshli. *Universal Sharpness Dynamics in Neural Network Training: Fixed Point Analysis, Edge of Stability, and Route to Chaos.* NeurIPS workshop / ICLR, 2023–2024.

## 10. Worked Example

Take the 2-parameter loss $L(a,b) = \tfrac12 (ab - 1)^2$ — the minimal model in Zhu et al. (2023). Minimizers form the hyperbola $ab=1$. At a minimizer $(a, 1/a)$ the Hessian is $\begin{pmatrix} 1/a^2 & 0 \\ 0 & a^2\end{pmatrix}$ in the natural basis, so $\lambda_1 = \max(a^2, a^{-2})$. Sharpness is unbounded along the minimizer manifold: this is progressive sharpening in two parameters.

Run GD with $\eta = 0.4$. The stability threshold is $\lambda_1 = 2/\eta = 5$, i.e. $|a| = \sqrt{5} \approx 2.236$.

Start at $(a_0, b_0) = (3.0, 0.30)$, so $a_0 b_0 = 0.9$, $L = 0.005$, and $\lambda_1 \approx 9.0$ — already unstable, $\eta\lambda_1 = 3.6$.

GD update: $a \leftarrow a - \eta (ab-1) b$, $b \leftarrow b - \eta (ab-1) a$.

- Step 1: $r = -0.1$. $a_1 = 3.0 + 0.4(0.1)(0.30) = 3.012$, $b_1 = 0.30 + 0.4(0.1)(3.0) = 0.420$. Now $a_1b_1 = 1.265$, $L = 0.0351$ — **the loss increased 7×**.
- Step 2: $r = 0.265$. $a_2 = 3.012 - 0.4(0.265)(0.420) = 2.968$, $b_2 = 0.420 - 0.4(0.265)(3.012) = 0.101$. $a_2b_2 = 0.300$, $L = 0.245$ — loss increased again, 50× the start.
- The iterate oscillates across the manifold, and each overshoot moves $a$ down. After a few hundred steps $|a|$ falls through $\sqrt{5}$ and the oscillation dies; the run converges to a point with $\eta\lambda_1 \lesssim 2$.

**What the example makes visible.** The trajectory's *progress* is not in the loss — the loss went up 50× in two steps — it is in $\lambda_1$, which fell from 9.0 toward 5. Any convergence proof must therefore find a Lyapunov function that is not $L$. Self-stabilization supplies a candidate built from $\nabla\lambda_1$, and here $\nabla \lambda_1 \ne 0$ everywhere off $|a|=1$, so it works cleanly.

Now perturb: start at $a_0 = 1.0 + 10^{-3}$. At $|a|=1$ the two eigenvalues coincide at 1 and $\nabla\lambda_1$ is not defined; the self-stabilization argument's key quantity vanishes into a degenerate cluster. In a 2-parameter model this is a measure-zero curiosity. In a ResNet at convergence, 5 to 10 eigenvalues sitting within a few percent of $2/\eta$ is the *typical* case, not the exception. That is the obstruction: the mechanism with the cleanest proof is stated for an isolated top eigenvalue, and real networks at EoS do not have one.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*