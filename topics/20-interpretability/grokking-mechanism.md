---
id: 20-interpretability/grokking-mechanism
title: "Grokking Phase Transitions Mechanism"
topic: 20-interpretability
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Grokking Phase Transitions Mechanism

> **Topic:** Interpretability · **ID:** `20-interpretability/grokking-mechanism` · **Status:** partially-solved

## 1. Problem Statement

Grokking is the observation that a network can fit its training set to near-zero loss, sit at chance test accuracy for orders of magnitude more optimization steps, and then rise sharply to near-perfect test accuracy without any change to data, objective, or hyperparameters. The problem is to explain *why the delay exists and what sets its length*.

Three variants, different difficulty:

- **Measurement.** Given a training run, produce a scalar progress measure computable from weights and training data alone (no test set) that moves monotonically during the plateau and predicts the transition step. Solved for one task family, not in general.
- **Method.** Given an architecture, task, and optimizer, predict *a priori* whether grokking occurs and at which step; and produce an intervention that removes the delay without harming final test accuracy.
- **Theory.** Prove, for a stated model class, that the test-loss curve has the delayed-transition shape, and give the delay as a function of dataset size $n$, weight decay $\lambda$, initialization scale, and learning rate.

Solving it means: the theory variant proved for at least one non-trivial nonlinear architecture on a task where the generalizing solution is not hand-planted, plus a progress measure that transfers to a task it was not designed on.

## 2. Formal Setting

Data. A finite group task is standard: $\mathcal{X} = \mathbb{Z}_p \times \mathbb{Z}_p$, $y = (a+b) \bmod p$, $p = 113$ in most replications. The full set has $p^2 = 12{,}769$ examples; a fraction $\alpha_{\text{data}}$ is the train set $S$, $|S| = n$, the rest is test.

Model. A one-layer transformer $f_\theta$ (Nanda et al.: $d_{\text{model}}=128$, 4 heads, no LayerNorm, ~$4\times10^5$ parameters), full-batch AdamW, weight decay $\lambda$.

Objective:
$$\mathcal{L}_S(\theta) = \frac{1}{n}\sum_{(x,y)\in S} \ell(f_\theta(x), y) + \frac{\lambda}{2}\|\theta\|_2^2 .$$

Measured quantities, as actually computed:

- **Memorization time** $t_{\text{mem}} = \min\{t : \text{acc}_S(\theta_t) \ge 0.99\}$; **generalization time** $t_{\text{gen}} = \min\{t: \text{acc}_{\text{test}}(\theta_t)\ge 0.99\}$; **delay ratio** $\tau = t_{\text{gen}}/t_{\text{mem}}$. Grokking is conventionally declared at $\tau \gtrsim 10$.
- **Fourier concentration.** Embed logits over the $p$-dim DFT basis; let $\hat{L}_k$ be the amplitude at frequency $k$. The **key frequencies** are the top-$m$ $k$ by $\|\hat L_k\|$. Nanda et al.'s *restricted loss* recomputes training loss after zeroing all non-key frequencies; *excluded loss* recomputes it after zeroing only the key ones. Both are test-set-free.
- **Circuit efficiency** (Varma et al.): $E_C = \text{logit-margin}(C)/\|\theta_C\|$, the logits a circuit $C$ produces per unit parameter norm.
- **Richness** (Kumar et al.): output scale $\alpha$ in $f_\theta = \alpha \tilde f_\theta$; $\alpha \to \infty$ is lazy (NTK), small $\alpha$ is rich (feature-learning).

Assumptions and their violations. (i) The generalizing circuit is unique and identifiable — violated: at $p=59$ the same setup produces two mechanisms, the "clock" and the "pizza" (Zhong et al. 2023). (ii) Regularization is the driver — violated: grokking occurs at $\lambda=0$ under Adam via slingshots/numerical effects (Thilak et al. 2022; Prieto et al. 2025). (iii) The transition is a property of the data–architecture pair — violated: it is partly an artifact of large initialization norm, removable by rescaling weights (Liu et al. 2023). (iv) Train accuracy $\to 1$ implies gradient signal is exhausted — violated: cross-entropy keeps a logit-growth gradient after 100% train accuracy.

## 3. State of the Art

**Established (mechanistically ablated).** Nanda, Chan, Lieberum, Smith & Steinhardt, *Progress Measures for Grokking via Mechanistic Interpretability* (ICLR 2023) fully reverse-engineered the $p=113$ modular addition transformer: the network computes $\cos/\sin$ of $\omega_k(a+b)$ over ~5 key frequencies and reads off the answer by trigonometric identity. They split training into **memorization → circuit formation → cleanup**, and show the Fourier circuit forms *during* the plateau, before test loss moves. The claim is causally supported by ablation (restricted/excluded loss), not just correlation. This is the strongest result in the area.

**Established (theory, restricted classes).** Mohamadi, Li, Wu & Sutherland, *Why Do You Grok? A Theoretical Analysis of Grokking Modular Addition* (ICML 2024) prove for two-layer quadratic-activation networks on modular addition that $O(p)$ samples suffice for memorization by kernel-regime dynamics while $\Omega(p^2 \cdot \text{polylog})$-free feature learning takes provably longer — a separation that produces the delay. Lyu et al. (ICLR 2024) prove grokking for a homogeneous network where early training is kernel-like and late training is driven by margin maximization under weight decay. Xu, Wang, Frei, Vardi & Hu (ICLR 2024) prove grokking for ReLU nets on XOR-cluster data.

**Claimed but not fully ablated.** Varma, Shah, Kenton, Kramár & Kumar, *Explaining Grokking through Circuit Efficiency* (2023) argue memorizing and generalizing circuits compete and the generalizing one wins because it is more parameter-efficient at large $n$; they predict and observe two new phenomena — **ungrokking** (regression to memorization when $n$ is cut below a critical $D_{\text{crit}}$) and **semi-grokking** (transition to intermediate test accuracy near $D_{\text{crit}}$). The predictions replicate on algorithmic tasks; the efficiency measure $E_C$ has not been shown to predict $t_{\text{gen}}$ on non-algorithmic data.

**Benchmark-number-only.** Reports of grokking in LLM-scale settings — hierarchical syntax (Murty et al., ACL 2023), implicit multi-hop reasoning in transformers (Wang, Yue, Su & Sun, NeurIPS 2024) — establish the *curve shape* at that scale but carry no circuit-level ablation.

## 4. What Is Known

- Original observation: modular division mod 97, 50% train fraction, small decoder transformer — train accuracy saturates near $10^3$ steps, test accuracy near $10^5$; $\tau \approx 10^2$ (Power et al. 2022).
- Weight decay is the strongest single accelerant: increasing $\lambda$ can reduce $t_{\text{gen}}$ by roughly an order of magnitude on modular arithmetic; at $\lambda = 0$ with sufficient data the transition can be pushed past $10^6$ steps or not observed (Power et al. 2022; Nanda et al. 2023).
- **Omnigrok** (Liu, Michaud & Tegmark, ICLR 2023): grokking on MNIST is induced by scaling initialization weights up by $\sim\!8\times$, and *removed* by constraining $\|\theta\|$ to a "Goldilocks" radius during training. This shows a large part of the delay is a weight-norm effect, not a data effect. Measured on MLPs, 1k MNIST examples.
- Effective theory of representation learning (Liu, Kitouni, Nolte, Michaud, Tegmark & Williams, NeurIPS 2022) maps a four-regime phase diagram — comprehension, grokking, memorization, confusion — in (learning rate, weight decay) on toy addition.
- Lazy-to-rich: Kumar, Bordelon, Gershman & Pehlevan (ICLR 2024) show on polynomial regression and MNIST that grokking appears when initialization puts the net in the lazy regime and the transition coincides with the onset of feature learning; the delay scales with output scale $\alpha$.
- Numerical: Prieto, Barsbey, Mediano & Birdal (2025) attribute much late-phase behavior to **Softmax Collapse** — floating-point round-off in the cross-entropy softmax after train accuracy saturates — and remove grokking on modular arithmetic by a gradient orthogonalization ($\perp$Grad) that avoids the naive-loss-minimization direction.
- Delayed generalization is not confined to plateaus: Humayun, Balestriero & Baraniuk (ICML 2024) report delayed robustness ("grokking") after standard training on CIFAR-scale vision nets, measured through decision-boundary partition geometry.

## 5. What Is Not Known

- **Theoretically open.** No proof of grokking for a transformer with softmax attention on any task. All proofs cover two-layer nets with quadratic/ReLU activations or linear estimators. Also open: a general lower bound on $t_{\text{gen}}$ in terms of $(n, \lambda, \alpha)$ — current results give existence of a delay, not its length.
- **Theoretically open.** Whether the several proposed causes (large init norm, lazy→rich, circuit efficiency, softmax collapse, slingshots) are the *same* mechanism written in different coordinates, or genuinely distinct sufficient causes. No unification theorem exists.
- **Empirically open.** Does anything grok at $\ge 10^9$ parameters on natural language in a way that survives a circuit-level ablation? The runs are affordable at 1B scale; the ablation methodology exists; nobody has published the combination.
- **Methodologically blocked.** "Grokking" has no agreed operational definition. $\tau \ge 10$ vs. "sharp transition in test loss" vs. "delayed circuit cleanup" select different run sets. Progress measures (restricted loss, key-frequency count) are defined only where the target circuit is already known — circular on any task whose solution is not pre-reverse-engineered.

## 6. Why It Is Hard

The central obstruction is **non-identifiability of the cause under confounded interventions**. Every known knob that removes grokking — raising $\lambda$, shrinking init norm, lowering $\alpha$, orthogonalizing gradients — changes several candidate causes at once. Rescaling weights changes norm *and* effective richness *and* softmax logit scale simultaneously; weight decay changes efficiency pressure *and* norm *and* the late-phase implicit bias. So an experiment showing "intervention X removes grokking" cannot distinguish which of the five theories is right, and all five explain the same curve.

Second: **absent ground truth off algorithmic data.** The Nanda result is trustworthy because the correct circuit for $(a+b) \bmod p$ is known in closed form, so restricted loss is checkable. For natural language no such reference circuit exists, so a progress measure cannot be validated — only asserted.

Third, mild: compute is *not* the binding constraint. A $p=113$ grokking run is minutes on one GPU. That is precisely why the field has many competing theories and no discriminating experiment: cheap runs generate hypotheses faster than they falsify them.

## 7. Current Research (as of 2026)

- **Mechanistic-interpretability groups** (Nanda and collaborators; DeepMind's alignment team, following Varma et al.) continue circuit-competition accounts and the $D_{\text{crit}}$ / semi-grokking phenomenology.
- **Physics-of-learning groups** (Tegmark's group at MIT; Gromov at Maryland; Ringel and collaborators on grokking as a first-order phase transition, ICLR 2024) pursue analytic solutions and phase diagrams; Gromov's closed-form two-layer solution for modular arithmetic is the cleanest exact construction.
- **Learning-theory groups** (Pehlevan/Bordelon at Harvard; Hu at Michigan; Sutherland's group) push the lazy-to-rich and sample-complexity-separation program toward deeper models.
- **Optimizer-artifact line** (Prieto et al.; earlier Thilak et al. at Apple) argues part of the phenomenon is numerical rather than statistical. *(frontier — verify)* Reports that grokking-like delays appear during LLM pretraining on memorized-then-generalized factual associations remain circuit-unablated.

## 8. Concrete Next Experiment

**Question.** Is the delay caused by weight norm, or by lazy-regime richness — the two confounded in every published intervention?

**Design: a $2\times2$ factorial that decouples them.** Modular addition $p=113$, one-layer transformer, $d_{\text{model}}=128$, $\alpha_{\text{data}}=0.4$, full-batch AdamW, $\lambda=1$, 5 seeds per cell — 20 runs, ~4 GPU-hours total on one A100.

- Factor A: initialization norm $\|\theta_0\|$ at $1\times$ vs $8\times$ the standard value.
- Factor B: output scale $\alpha$ (a multiplier on the unembedding, compensated at init so $\theta_0$'s norm is held fixed by construction) at $1\times$ vs $0.1\times$.
- **Control arm:** the $(1\times, 1\times)$ standard run, plus a norm-projection arm reproducing Omnigrok's constrained-sphere training at each $\alpha$.

**Deciding number.** The delay ratio $\tau = t_{\text{gen}}/t_{\text{mem}}$ at the 99% thresholds. Fit $\log_{10}\tau \sim \beta_A \cdot \mathbb{1}[\text{norm}=8\times] + \beta_B\cdot\mathbb{1}[\alpha=0.1] + \beta_{AB}$. If $\beta_B$ explains $\tau$ with $|\beta_A| < 0.3$ (i.e. norm changes $\tau$ by less than $2\times$ at fixed richness), the norm account is a proxy and lazy-to-rich is the mechanism. If $\beta_A \ge 1.0$ at fixed $\alpha$, norm is causal independently. A large $\beta_{AB}$ falsifies both as single causes.

**Second readout, free:** log restricted and excluded loss every 100 steps in all 20 cells. If circuit-formation onset (first step where excluded loss rises above its memorization value by $2\sigma$) tracks $\alpha$ but not $\|\theta_0\|$, that corroborates the same conclusion mechanistically rather than only by curve shape.

## 9. Key References

- **[Foundational]** Power, Burda, Edwards, Babuschkin, Misra. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* ICLR 2022 MATH-AI Workshop. — arXiv:2201.02177
- **[SOTA — mechanism]** Nanda, Chan, Lieberum, Smith, Steinhardt. *Progress Measures for Grokking via Mechanistic Interpretability.* ICLR 2023. — arXiv:2301.05217
- **[SOTA — theory]** Mohamadi, Li, Wu, Sutherland. *Why Do You Grok? A Theoretical Analysis of Grokking Modular Addition.* ICML 2024.
- Liu, Kitouni, Nolte, Michaud, Tegmark, Williams. *Towards Understanding Grokking: An Effective Theory of Representation Learning.* NeurIPS 2022. — arXiv:2205.10343
- Liu, Michaud, Tegmark. *Omnigrok: Grokking Beyond Algorithmic Data.* ICLR 2023. — arXiv:2210.01117
- Varma, Shah, Kenton, Kramár, Kumar. *Explaining Grokking through Circuit Efficiency.* 2023. — arXiv:2309.02390
- Kumar, Bordelon, Gershman, Pehlevan. *Grokking as the Transition from Lazy to Rich Training Dynamics.* ICLR 2024. — arXiv:2310.06110
- Lyu, Jin, Li, Du, Lee, Hu. *Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking.* ICLR 2024.
- Zhong, Liu, Tegmark, Andreas. *The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks.* NeurIPS 2023. — arXiv:2306.17844
- Barak, Edelman, Goel, Kakade, Malach, Zhang. *Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit.* NeurIPS 2022. — arXiv:2207.08799
- Thilak, Littwin, Zhai, Saremi, Susskind. *The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon.* 2022. — arXiv:2206.04817
- Prieto, Barsbey, Mediano, Birdal. *Grokking at the Edge of Numerical Stability.* 2025. — arXiv:2501.04697
- **[Survey-adjacent]** Gromov. *Grokking Modular Arithmetic.* 2023. — arXiv:2301.02679

## 10. Worked Example

Take $p=113$, $n = 0.3 \cdot 12{,}769 \approx 3{,}831$ training pairs, one-layer transformer, AdamW, $\lambda = 1$. A typical run: train accuracy crosses 99% near step $t_{\text{mem}} \approx 1.0\times10^3$; test accuracy crosses 99% near $t_{\text{gen}} \approx 1.2\times10^4$. So $\tau \approx 12$.

Now count parameters against data to see why the "efficiency" story is *plausible* but not *decided*. Memorizing 3,831 input–output pairs over 113 classes needs at least $3831 \cdot \log_2 113 \approx 2.6\times10^4$ bits. The generalizing circuit needs only the 5 key frequencies: for each, a $\cos$ and $\sin$ direction in the $113$-dim embedding plus a readout, roughly $5 \cdot (2\cdot 113 + 113) \approx 1.7\times10^3$ parameters. The model has $\sim\!4\times10^5$ parameters, so both fit comfortably; the generalizing circuit is roughly $15\times$ cheaper in parameter norm. Under weight decay at $\lambda=1$, that norm gap gives the Fourier circuit a per-step advantage — the efficiency account.

Here is the obstruction, made concrete. Run the identical configuration with $\lambda=0$ but initialization weights scaled to $8\times$: grokking still appears, $\tau$ of the same order. Weight decay was not applied, so no efficiency pressure existed, yet the delay persisted — the norm account explains this run and the efficiency account does not. Run it again at $\lambda=0$, standard init, but with $\perp$Grad: the delay largely vanishes, which neither the norm nor the efficiency account predicts, and which the softmax-collapse account does. Three interventions, three winners, one curve. Every one of these runs costs under 20 minutes on a single GPU. The bottleneck is not compute — it is that no published experiment holds norm, richness, and logit scale fixed independently, so the observed $\tau \approx 12$ remains attributable to at least three causes at once.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*