---
id: 23-privacy-memorization/hidden-state-privacy-accounting
title: "Hidden State Privacy Accounting for Convex and Nonconvex Training"
topic: 23-privacy-memorization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hidden State Privacy Accounting for Convex and Nonconvex Training

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/hidden-state-privacy-accounting` · **Status:** partially-solved

## 1. Problem Statement

Standard DP-SGD accounting charges privacy for **every** gradient step, because the composition argument assumes the adversary sees every intermediate iterate $\theta_1,\dots,\theta_T$. In deployment the adversary usually sees only the released final model $\theta_T$. The **hidden state** question: how much smaller is the true $(\varepsilon,\delta)$ of the map $D \mapsto \theta_T$ than the composed bound?

Three variants, with very different difficulty:

- **Theory.** Prove an upper bound on the Rényi divergence between last-iterate distributions on adjacent datasets that is $o(T)$ — ideally $O(1)$ as $T \to \infty$. Solved for smooth convex losses; open in general for nonconvex.
- **Method.** Produce an accountant a practitioner can run: takes $(n, B, \eta, \sigma, T, C)$ plus curvature assumptions, returns a valid $\varepsilon$ smaller than the composition $\varepsilon$, for the architectures people actually train.
- **Measurement.** Empirically certify the gap. Given only final-model access, produce a *lower* bound $\hat\varepsilon$ tight enough to distinguish "hidden-state amplification is real at this scale" from "the bound is loose but the algorithm is not actually more private."

Solving it means: a bound that holds for a nonconvex loss without unrealistic assumptions, plus an audit that brackets it.

## 2. Formal Setting

Dataset $D = (x_1,\dots,x_n)$, loss $\ell(\theta; x)$, $\theta \in \mathbb{R}^d$. Noisy SGD:

$$\theta_{t+1} = \Pi_{\mathcal{K}}\Big(\theta_t - \eta\big(\tfrac{1}{B}\textstyle\sum_{i \in B_t} \mathrm{clip}_C(\nabla \ell(\theta_t; x_i)) + \tfrac{\sigma C}{B}\, \xi_t\big)\Big), \quad \xi_t \sim \mathcal{N}(0, I_d).$$

Measured quantities: $C$ is the clipping norm as configured; $\sigma$ the noise multiplier as configured; $B$ the *expected* batch size under Poisson sampling with rate $q = B/n$; $\eta$ the step size actually used at step $t$ (schedules break stationarity arguments); $n$ the dataset size.

Let $P_T^D$ be the law of $\theta_T$. Rényi DP of order $\alpha$:

$$\varepsilon_\alpha(T) = \sup_{D \simeq D'} \; \tfrac{1}{\alpha-1} \log \mathbb{E}_{\theta \sim P_T^{D'}}\big[(dP_T^D/dP_T^{D'})^\alpha\big].$$

Composition gives $\varepsilon_\alpha(T) \le T \cdot \varepsilon_\alpha^{\text{step}}$. Hidden-state accounting asks for a bound on the left directly. The mechanism is contraction: if the gradient step map is $c$-Lipschitz with $c<1$ (guaranteed by $\lambda$-strong convexity, $\beta$-smoothness and $\eta \le 2/(\lambda+\beta)$, with $c = 1-\eta\lambda$), a shifted-divergence argument gives a geometric rather than linear accumulation,

$$\varepsilon_\alpha(\infty) \;\lesssim\; \frac{\alpha S^2}{2\varsigma^2}\cdot\frac{1}{1-c^2}, \qquad S = \text{per-step sensitivity} = \eta C/n,\ \ \varsigma = \eta\sigma C/n,$$

so $T$ effective steps collapse to $1/(1-c^2)$.

**Assumptions, and which are violated.**
- *Convexity / strong convexity* — violated by every neural network of interest.
- *$\beta$-smoothness* — violated by ReLU, attention softmax saturation, and by clipping itself, which is non-smooth.
- *Bounded domain $\mathcal{K}$ with projection* — not used in practice; unconstrained training is standard.
- *Secrecy of the minibatch sequence* — Poisson subsampling amplification assumes the sampled indices are hidden; real code shuffles, and shuffling does not give the same amplification.
- *Only $\theta_T$ released* — violated by checkpointing, early-stopping on a validation curve, federated learning, and any published training-curve artifact.

## 3. State of the Art

**Theory (established).**
- *Privacy amplification by iteration* (Feldman, Mironov, Talwar, Thakurta, FOCS 2018): for convex smooth Lipschitz losses, the privacy cost of an example used at step $i$ decays like $O(1/(T-i))$ in the released last iterate. Requires contractive (non-expansive) updates.
- *Langevin/noisy-GD dynamics* (Chourasia, Ye, Shokri, NeurIPS 2021): for strongly convex smooth losses, full-batch noisy GD has $\varepsilon_\alpha(T)$ converging to a finite constant as $T \to \infty$. Extended to the convex (non-strongly) case by Ye & Shokri (NeurIPS 2022).
- *Saturation for convex SGD* (Altschuler & Talwar, NeurIPS 2022): for smooth convex Lipschitz losses on a bounded domain, the last-iterate privacy loss stops growing after $\tilde O(n^2)$ steps — "more iterations without more privacy loss."
- *Tight constants* (Bok, Su, Altschuler, ICML 2024, "Shifted Interpolation for Differential Privacy"): shifted interpolated processes give the first exact/near-exact hidden-state $f$-DP for strongly convex objectives, removing the slack in earlier Rényi arguments and handling unbounded domains.

**Theory (negative, established).** Annamalai (2024) constructs a *nonconvex* loss on which hidden-state DP-SGD gets **no** amplification: the last-iterate privacy loss grows with $T$ at the composition rate. This closes the naive hope that nonconvexity is a technicality.

**Empirical SOTA.** Auditing under the hidden-state threat model (Cebere, Bellet, Papernot, 2024) obtains materially higher $\hat\varepsilon$ lower bounds than prior final-model audits by crafting adversarial gradient-canary inputs; the reported numbers are benchmark numbers on small convex and small-CNN setups, not a general accountant. Nasr et al. (USENIX Security 2023) established that tight auditing requires the *strong* adversary — all intermediate iterates plus worst-case data; with final-model-only access on natural data, empirical $\hat\varepsilon$ falls far below theoretical $\varepsilon$.

**Claimed but unablated.** That the convex saturation results "morally apply" to deep networks because training is locally convex near convergence. No paper isolates this: no ablation varies measured local curvature and shows the audit lower bound tracks the predicted saturation.

## 4. What Is Known

- Composition is loose for the final iterate whenever contraction holds. Established at the level of theorems, not just experiments.
- For strongly convex objectives, the converged $\varepsilon$ is independent of $T$; the effective number of charged steps is $\approx 1/(1-(1-\eta\lambda)^2) \approx 1/(2\eta\lambda)$. At $\eta\lambda = 10^{-2}$ this is $\approx 50$ steps regardless of whether you run $10^3$ or $10^6$.
- For convex smooth losses on a bounded domain, saturation kicks in at $\tilde O(n^2)$ steps (Altschuler–Talwar) — for $n = 50{,}000$ that is $\sim 10^9$ steps, i.e. the asymptotic regime is far beyond any real training run. The result is qualitatively decisive and quantitatively inert at practical $T$.
- Final-model audits are weak. Steinke, Nasr, Jagielski (NeurIPS 2023, one-run auditing) report $\hat\varepsilon \approx 1.3$ against a theoretical $\varepsilon = 4$ in the white-box/all-iterates setting on CIFAR-10-scale models; black-box final-model-only numbers are substantially smaller still, typically well under 1 at the same theoretical $\varepsilon$.
- Shuffling breaks the accounting people report. Annamalai, Balle, De Cristofaro, Hayes (2024–2025) audit DP-SGD implemented with shuffling but accounted as Poisson-subsampled, and obtain $\hat\varepsilon$ lower bounds exceeding the reported $\varepsilon$ — a *violation*, in the direction opposite to hidden-state amplification.

## 5. What Is Not Known

- **Theoretically open.** Whether any nontrivial structural condition weaker than convexity (e.g. Polyak–Łojasiewicz, one-point convexity near the trajectory, or bounded negative curvature along the path) yields $T$-independent hidden-state bounds. Asoodeh & Diaz argue convergence "might" hold for some nonconvex losses under trajectory conditions; no general theorem. Annamalai's counterexample rules out the unconditional claim, not conditional ones.
- **Theoretically open.** Hidden-state accounting for *shuffled* (random-reshuffling) minibatches, which is what implementations do. All amplification-by-iteration results assume Poisson or worst-case ordering.
- **Empirically open.** Nobody has audited a hidden-state trained model at LLM fine-tuning scale ($\ge 10^9$ parameters) with a canary strong enough to bracket the convex-theory prediction. The experiment is runnable; it costs GPU-months of repeated training runs.
- **Methodologically blocked.** There is no accepted way to *measure* the contraction factor $c$ of a real training run. Without it, the theory has no input, and "does hidden-state amplification apply here?" is not a well-posed empirical question.

## 6. Why It Is Hard

The obstruction is **absent ground truth plus a two-sided gap that neither side can close**. The true $\varepsilon$ of the final-iterate map is a supremum over adjacent dataset pairs of a divergence between two $d$-dimensional distributions that no one can sample from except by retraining. Upper bounds come only from theory whose hypotheses (convexity, smoothness, projection) are false for the models people run. Lower bounds come only from auditing, which requires $\sim 10^3$–$10^6$ training runs for a usable confidence interval and which is *provably* weak in the final-model threat model — Nasr et al. show the strong-adversary gap is intrinsic, not an artifact of weak attacks. So a measured $\hat\varepsilon = 0.2$ against a composed $\varepsilon = 3.2$ is consistent with (a) real amplification of $16\times$ and (b) a bound that is tight but an attack that is weak. The two hypotheses are **non-identifiable** with current instruments.

Second obstruction: even where theory applies, the constants are inert. Saturation at $\tilde O(n^2)$ steps never triggers in a 100-epoch run.

## 7. Current Research (as of 2026)

- **Shifted-composition / interpolation machinery.** Altschuler (Penn), Bok, Su; Niles-Weed (NYU). Sharpening hidden-state bounds to exact $f$-DP and pushing beyond strong convexity. Extension to non-log-concave targets is active *(frontier — verify)*.
- **Nonconvex last-iterate bounds under trajectory assumptions.** Asoodeh & Diaz; Kong & Ribero (cyclically-sampled DP-SGD on nonconvex composite losses). Bounds depend on quantities along the realized trajectory, which are themselves data-dependent — the accounting-validity question is unresolved *(frontier — verify)*.
- **Hidden-state auditing.** Cebere, Bellet, Papernot (Inria / Toronto); Annamalai, Balle, Hayes, De Cristofaro (UCL / Google DeepMind / UC Riverside). Direction: stronger canaries under final-model-only access, and auditing the shuffling gap.
- **Implementation-faithful accounting.** Making the reported $\varepsilon$ match the sampler actually used. This is the direction with the clearest near-term payoff, because it currently produces *violations*, not amplification.

## 8. Concrete Next Experiment

**Question.** Does hidden-state amplification survive at a scale and architecture people actually use, or does it vanish with nonconvexity?

**Setup.** Fine-tune a 125M-parameter GPT-2-small on a 50,000-example text corpus with DP-SGD: $C=1$, $\sigma=1.0$, $q=0.01$, $T=10{,}000$ steps, $\delta=10^{-5}$. Composed $\varepsilon \approx 7.8$.

**Arms** (each requires $\ge 1{,}000$ retrainings with one canary in/out, or one-run auditing with 1,000 independent canaries):
1. **Treatment:** release only $\theta_T$. Audit with gradient-space canaries.
2. **Control A:** release all 10,000 iterates, same seeds, same canaries. This is the composition-tight regime; it calibrates attack strength.
3. **Control B:** replace the network with a linear (convex) model of matched input dimension, same hyperparameters. Theory predicts amplification here.

**Deciding number.** The ratio $R = \hat\varepsilon_{\text{all-iterates}} / \hat\varepsilon_{\text{final-only}}$ at fixed 95% confidence. If $R \le 1.5$ in the nonconvex arm while $R \ge 4$ in the convex arm, hidden-state amplification is architecture-dependent and the convex theory does not transfer. If $R \ge 4$ in both, there is empirical warrant for a nonconvex bound and the theory gap is the bottleneck. Cost: $\sim 10^3$ fine-tunes $\approx$ 2,000 A100-hours per arm.

## 9. Key References

- **[Foundational]** V. Feldman, I. Mironov, K. Talwar, A. Thakurta. *Privacy Amplification by Iteration.* FOCS, 2018. — arXiv:1808.06651
- **[Foundational]** R. Chourasia, J. Ye, R. Shokri. *Differential Privacy Dynamics of Langevin Diffusion and Noisy Gradient Descent.* NeurIPS, 2021.
- **[SOTA]** J. Altschuler, K. Talwar. *Privacy of Noisy Stochastic Gradient Descent: More Iterations without More Privacy Loss.* NeurIPS, 2022. — arXiv:2205.13710
- **[SOTA]** J. Bok, W. Su, J. Altschuler. *Shifted Interpolation for Differential Privacy.* ICML, 2024. — arXiv:2403.00278
- **[Negative result]** M. S. M. S. Annamalai. *It's Our Loss: No Privacy Amplification for Hidden State DP-SGD with Non-Convex Loss.* Preprint, 2024.
- **[SOTA — auditing]** T. Cebere, A. Bellet, N. Papernot. *Tighter Privacy Auditing of DP-SGD in the Hidden State Threat Model.* Preprint, 2024.
- **[SOTA — auditing]** T. Steinke, M. Nasr, M. Jagielski. *Privacy Auditing with One (1) Training Run.* NeurIPS, 2023.
- **[Established]** M. Nasr, J. Hayes, T. Steinke, B. Balle, F. Tramèr, M. Jagielski, N. Carlini, A. Terzis. *Tight Auditing of Differentially Private Machine Learning.* USENIX Security, 2023.
- **[Related]** J. Ye, R. Shokri. *Differentially Private Learning Needs Hidden State (Or Much Faster Convergence).* NeurIPS, 2022.
- **[Related]** M. S. M. S. Annamalai, B. Balle, E. De Cristofaro, J. Hayes. *To Shuffle or Not to Shuffle: Auditing DP-SGD with Shuffling.* Preprint, 2024.

## 10. Worked Example

Full-batch noisy GD, $n = 50{,}000$, $C = 1$, $\eta = 1.0$, noise multiplier $\sigma = 50$, $T = 1{,}000$ steps, $\delta = 10^{-5}$.

Per-step RDP (add/remove adjacency, sensitivity $C/n$, noise std $\sigma C/n$): $\varepsilon_\alpha^{\text{step}} = \alpha/(2\sigma^2) = \alpha/5000$.

**Composition arm.** $\varepsilon_\alpha = 1000 \cdot \alpha/5000 = 0.2\alpha$. Convert: minimize $0.2\alpha + \log(1/\delta)/(\alpha-1)$ with $\log(1/\delta)=11.5$ at $\alpha = 8.6$, giving

$$\varepsilon \approx 1.72 + 1.52 = \mathbf{3.24}.$$

**Hidden-state arm, assuming $\lambda$-strong convexity with $\eta\lambda = 10^{-2}$.** Contraction $c = 0.99$, geometric sum $1/(1-c^2) = 50.3$. Effective charged steps drop from 1,000 to 50:

$$\varepsilon_\alpha \approx 0.01\alpha \;\Rightarrow\; \alpha = 34.9, \quad \varepsilon \approx 0.35 + 0.34 = \mathbf{0.69}.$$

A $4.7\times$ improvement, from an assumption of strong convexity.

**Where the obstruction becomes visible.** Swap the linear model for a 2-layer ReLU network on the same data. Now $\lambda \le 0$: the geometric sum $1/(1-c^2)$ diverges, the bound falls back to $3.24$, and Annamalai's construction shows this is not merely a proof artifact — there exist nonconvex losses where $3.24$ is the truth. Meanwhile, a one-run audit of the released final model on this setup returns $\hat\varepsilon \approx 0.2$ at 95% confidence.

The three numbers are $0.2 \le \varepsilon_{\text{true}} \le 3.24$, with the convex prediction $0.69$ sitting inside. The audit cannot exclude $0.69$, cannot exclude $3.24$, and cannot confirm either. Nothing about the measurement gets better by running longer — a $16\times$ bracket is what final-model auditing delivers at this scale. That gap, not the theorem, is the state of the problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*