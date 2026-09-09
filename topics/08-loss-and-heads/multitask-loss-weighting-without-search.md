---
id: 08-loss-and-heads/multitask-loss-weighting-without-search
title: "Multi-Task Loss Weighting Without Validation Search"
topic: 08-loss-and-heads
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Multi-Task Loss Weighting Without Validation Search

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/multitask-loss-weighting-without-search` · **Status:** open

## 1. Problem Statement

A network with one trunk and $T$ output heads is trained on $\mathcal{L}(\theta) = \sum_t w_t \ell_t(\theta)$. The weights $w$ are usually found by grid or random search over validation runs, costing $O(k^{T-1})$ full trainings. The problem: **produce $w$ from a single training run, at a cost comparable to one run, with end-task quality no worse than the tuned grid.**

Three variants, routinely conflated:

- **Measurement.** Given a fixed architecture and budget, is there a scalar summarizing "multi-task quality" that a weighting rule can be said to optimize? The field's default, $\Delta_m\%$ (mean per-task relative change against single-task baselines, Maninis et al., CVPR 2019), is itself an implicit uniform weighting over tasks and inherits the arbitrariness it was meant to remove.
- **Method.** Find an online rule $w_t^{(s)} = f(\text{state at step } s)$ that matches tuned fixed weights. This is where GradNorm, uncertainty weighting, MGDA-UB, PCGrad, CAGrad, Nash-MTL and FAMO live.
- **Theory.** Characterize when a single-run rule *can* recover a tuned-search optimum. Includes the classical fact that scalarization only reaches the convex hull of the Pareto front, and the newer question of whether the reachable set matters at deep-network scale.

Solved would mean: on $\ge 3$ task suites and $\ge 2$ scales, a search-free rule attains within noise (defined below) of an $O(k^{T-1})$ grid search, with the comparison ablated against a *tuned-scalarization* control, not an equal-weight one.

## 2. Formal Setting

Tasks $t \in \{1,\dots,T\}$; shared parameters $\theta_{\mathrm{sh}}$, head parameters $\theta_t$. Per-task empirical risk on the training split:
$$\ell_t(\theta) = \tfrac{1}{n_t}\sum_{i=1}^{n_t} L_t\!\big(h_t(f(x_i;\theta_{\mathrm{sh}});\theta_t),\, y_{t,i}\big).$$
Measured as: the mean minibatch loss for head $t$, logged every step, EMA-smoothed over $\sim 100$ steps (raw per-step values have relative standard deviation 0.1–0.5 and are unusable as a control signal).

Weighting $w \in \Delta^{T-1}$ (or $\mathbb{R}_{>0}^T$; scale is absorbed by the learning rate). Objective $\mathcal{L}_w = \sum_t w_t \ell_t$.

**Evaluation.** Held-out per-task metric $m_t$ (mIoU, accuracy, negative RMSE), and
$$\Delta_m\% = \frac{1}{T}\sum_{t=1}^T (-1)^{\delta_t}\,\frac{m_t - m_t^{\mathrm{STL}}}{m_t^{\mathrm{STL}}}\times 100,$$
$\delta_t=1$ if lower is better, $m_t^{\mathrm{STL}}$ the single-task baseline. **Measured as:** requires $T$ extra single-task runs, each tuned; in most papers those baselines are tuned less than the MTL arm.

**Gradient conflict.** $c_{ts} = \cos\!\big(\nabla_{\theta_{\mathrm{sh}}}\ell_t,\, \nabla_{\theta_{\mathrm{sh}}}\ell_s\big)$, measured on a fixed probe batch (whole-dataset gradients are unaffordable; minibatch estimates of $c_{ts}$ at batch size 64 have absolute bias toward 0 of order $1/\sqrt{B}$).

**Pareto set.** $\theta$ is Pareto-optimal if no $\theta'$ has $\ell_t(\theta')\le \ell_t(\theta)$ for all $t$ with one strict. Scalarization with $w>0$ yields, at best, points on the convex hull of the attainable loss set.

**Assumptions, and which are violated.**
1. *Losses are commensurable up to scale* — violated: cross-entropy in nats and depth RMSE in metres have different curvature, so a single $w_t$ cannot fix both the gradient magnitude and the Hessian scale.
2. *A fixed $w$ is optimal for the whole run* — violated: loss ratios move by 1–2 orders of magnitude between step 0 and convergence.
3. *Validation metric is a monotone function of validation loss* — violated for mIoU, BLEU, pass@1.
4. *The optimizer reaches a stationary point of $\mathcal{L}_w$* — violated; with Adam, per-parameter normalization makes the update largely invariant to a *global* loss rescale and only partly sensitive to *relative* $w_t$, which is the source of the identifiability problem in §6.

## 3. State of the Art

**Established (reproduced, ablated).**
- *Tuned scalarization is a strong baseline.* Kurin et al., "In Defense of the Unitary Scalarization for Deep Multi-Task Learning" (NeurIPS 2022) and Xin et al., "Do Current Multi-Task Optimization Methods in Deep Learning Even Help?" (NeurIPS 2022) independently found that once regularization and learning rate are tuned equally on both arms, specialized multi-task optimizers (PCGrad, CAGrad, MGDA, IMTL, GradDrop) give no reliable gain over $w_t = 1$ on Cityscapes, NYUv2, CelebA and Meta-World MT10/MT50.
- *Scalarization traces essentially the same front.* Hu et al., "Revisiting Scalarization in Multi-Task Learning: A Theoretical Perspective" (NeurIPS 2023) show scalarization is not Pareto-complete in general, yet with random weight sampling explores fronts comparable to specialized MTO in deep nets.
- *Mixture weights are predictable in language pretraining.* Fernandes et al., "Scaling Laws for Multilingual Neural Machine Translation" (ICML 2023) fit per-task loss as a function of mixing weight and scale; Ye et al., "Data Mixing Laws" (ICLR 2024) predict held-out loss under unseen mixtures from small proxy runs.

**Claimed but unablated.** GradNorm (Chen et al., ICML 2018), uncertainty weighting (Kendall et al., CVPR 2018), DWA (Liu et al., CVPR 2019), Nash-MTL (Navon et al., ICML 2022), FAMO (Liu et al., NeurIPS 2023) each report gains over equal weighting; almost none report against a *tuned* fixed-$w$ grid at matched compute. The Nash-MTL and FAMO numbers on NYUv2/CityScapes exist mainly as leaderboard $\Delta_m\%$ entries under the MTAN protocol.

**Benchmark-number-only.** Most reported $\Delta_m\%$ improvements of 1–3 points on NYUv2 come from single seeds under a shared protocol; seed variance on that benchmark is of similar size.

## 4. What Is Known

- Uncertainty weighting (Kendall et al., 2018) on CityScapes-style semantics + instance + depth: learned weights beat equal weighting and approach the best point of a coarse grid over $w$ swept in their Fig. 4 — a 3-task, single-architecture, single-scale result (VGG-class encoder, $\sim$10 M–20 M params).
- MGDA-UB (Sener & Koltun, NeurIPS 2018) gives a single-backward upper bound on the min-norm direction; on MultiMNIST and CelebA (40 binary tasks, ResNet-18) it reduces per-task degradation versus uniform, at $\approx$1.2–2$\times$ step cost.
- Kurin et al. (2022): on CelebA-40 and MT10, unitary scalarization plus tuned weight decay/early stopping matched or beat every MTO method tested; MT10 success rates differ by less than the $\pm$5–10 point seed spread ($n\!=\!10$ seeds).
- Xin et al. (2022): with equal hyperparameter budget, MTO gains on Cityscapes/CelebA fell inside seed noise; scalarization with searched $w$ was on or above the MTO front.
- Royer et al., "Scalarization for Multi-Task and Multi-Domain Learning at Scale" (NeurIPS 2023): at ImageNet-scale multi-domain training, population-based tuning of a *fixed* $w$ beat adaptive schemes.
- DoReMi (Xie et al., NeurIPS 2023): domain weights from a 280 M proxy transfer to an 8 B model, reaching baseline downstream accuracy with 2.6$\times$ fewer steps on The Pile — an existence proof that cheap-proxy weight transfer works in one regime.
- Task grouping matters as much as weighting: Standley et al. (ICML 2020) and Fifty et al. (TAG, NeurIPS 2021) show which-tasks-together dominates how-weighted on Taskonomy.

## 5. What Is Not Known

- **Theoretically open.** No characterization of when a *causal, online* rule (using only past loss/gradient state) can match the best fixed $w$ chosen with oracle validation access. Regret bounds for online scalarization against the best hindsight $w$ in non-convex deep training do not exist.
- **Theoretically open.** Whether the non-convex portion of the Pareto front is ever reached by deep multi-task training in practice, i.e. whether scalarization's incompleteness is binding at all.
- **Empirically open.** Nobody has run the decisive comparison: adaptive rules vs. a *dense, tuned* $w$-grid at matched total compute, multi-seed, at both $10^7$ and $10^9$ parameters. Runnable today; roughly $10^3$ GPU-hours.
- **Empirically open.** Whether data-mixture laws (Ye et al. 2024) extend from data-domain weights to *loss-head* weights (e.g. LM + reward + auxiliary contrastive head).
- **Methodologically blocked.** "Multi-task quality" has no agreed scalar. $\Delta_m\%$ presupposes a uniform preference over tasks — exactly the preference the weighting was supposed to encode. Until the target is fixed by a stated user preference, "optimal $w$" is not defined.

## 6. Why It Is Hard

**Non-identifiability.** For head-local parameters $\theta_t$ under SGD, scaling $w_t \to \alpha w_t$ is exactly equivalent to scaling the head's learning rate by $\alpha$; only the *ratio* of $w_t$ across tasks acts on the shared trunk. Under Adam, per-coordinate normalization removes even part of that: the head update is nearly invariant to $w_t$, so the same measured $\Delta_m\%$ change can be produced by weights, head learning rates, weight decay, or early stopping. This is the concrete mechanism behind the Kurin/Xin results — MTO gains were absorbed into the regularization budget of the baseline.

**Confounded measurement.** The reported gain is a difference of held-out metrics whose seed spread on NYUv2/Cityscapes is comparable to the claimed 1–3 $\Delta_m\%$ improvement, and single-task baselines are usually undertuned, inflating $\Delta_m\%$ for every MTL arm equally.

**Absent ground truth.** The oracle — best fixed $w$ under full search — is $O(k^{T-1})$ trainings and is essentially never computed for $T>3$, so "matches the tuned grid" has almost never been checked; papers compare to equal weights instead.

## 7. Current Research (as of 2026)

- **Mixture-law extrapolation.** Fitting parametric per-task loss surfaces on small proxies and solving for $w$ analytically: DoReMi, DoGE (Fan et al., ICML 2024), Data Mixing Laws. Active at Google DeepMind, Stanford, Tsinghua. Extension to loss-head weights is *(frontier — verify)*.
- **Post-training head balance.** Weighting SFT / preference / auxiliary-KL terms in RLHF-style objectives from a single run; largely unpublished internal practice *(frontier — verify)*.
- **Cheap online rules.** FAMO's $O(1)$-per-step loss-balancing lineage; the open question is whether they beat tuned constants, not equal ones.
- **Preference-conditioned single models.** Training one network conditioned on $w$ so the front is traversable at inference (Pareto hypernetworks, controllable-MTL line), sidestepping search rather than solving it.

## 8. Concrete Next Experiment

**Question.** Does any search-free rule match a tuned fixed $w$ at matched compute?

**Scale.** $T=3$ (semantic segmentation, depth, surface normals), NYUv2, SegNet/MTAN encoder, $\approx$ 20 M params, 200 epochs, 5 seeds per arm. Second scale: ViT-B/16, $\approx$ 86 M, same tasks.

**Arms.**
1. **Control (the one that matters):** fixed $w$ on a simplex grid, $k=7$ per axis, 28 grid points $\times$ 5 seeds; pick the grid point with best mean validation $\Delta_m\%$, report its test $\Delta_m\%$.
2. Equal weights, 5 seeds (the weak baseline everyone uses).
3. Uncertainty weighting, GradNorm, Nash-MTL, FAMO — 5 seeds each, learning rate and weight decay tuned with the *same* per-arm budget as one grid axis.

Every arm gets an identical hyperparameter-search budget in GPU-hours; report each adaptive arm's cost including its tuning.

**Deciding number.** $D = \Delta_m\%(\text{best adaptive}) - \Delta_m\%(\text{tuned grid})$, with a paired 5-seed 95% bootstrap CI. If the CI for $D$ lies above $-0.5$ points at both scales, single-run weighting is settled affirmatively for this regime. If the CI excludes $0$ from below at either scale, adaptive rules are confirmed to be a cheaper-but-worse approximation, and the field's equal-weight baselines are confirmed as the source of reported gains. Estimated cost: $\approx$ 900 GPU-hours on A100s.

## 9. Key References

- **[Foundational]** Alex Kendall, Yarin Gal, Roberto Cipolla. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR, 2018. — arXiv:1705.07115
- **[Foundational]** Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, Andrew Rabinovich. *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.* ICML, 2018. — arXiv:1711.02257
- **[Foundational]** Ozan Sener, Vladlen Koltun. *Multi-Task Learning as Multi-Objective Optimization.* NeurIPS, 2018. — arXiv:1810.04650
- **[Foundational]** Jean-Antoine Désidéri. *Multiple-gradient descent algorithm (MGDA) for multiobjective optimization.* Comptes Rendus Mathematique, 2012.
- **[SOTA]** Vitaly Kurin, Alessandro De Palma, Ilya Kostrikov, Shimon Whiteson, M. Pawan Kumar. *In Defense of the Unitary Scalarization for Deep Multi-Task Learning.* NeurIPS, 2022. — arXiv:2201.04122
- **[SOTA]** Derrick Xin, Behrooz Ghorbani, Ankush Garg, Orhan Firat, Justin Gilmer. *Do Current Multi-Task Optimization Methods in Deep Learning Even Help?* NeurIPS, 2022. — arXiv:2209.11379
- **[SOTA]** Yuzheng Hu, Ruicheng Xian, Qilong Wu, Qiuling Fan, Lang Yin, Han Zhao. *Revisiting Scalarization in Multi-Task Learning: A Theoretical Perspective.* NeurIPS, 2023. — arXiv:2308.13985
- **[SOTA]** Amelie Royer, Tijmen Blankevoort, Babak Ehteshami Bejnordi. *Scalarization for Multi-Task and Multi-Domain Learning at Scale.* NeurIPS, 2023. — arXiv:2310.08910
- **[Method]** Bo Liu, Yihao Feng, Peter Stone, Qiang Liu. *FAMO: Fast Adaptive Multitask Optimization.* NeurIPS, 2023. — arXiv:2306.03792
- **[Method]** Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, Ethan Fetaya. *Multi-Task Learning as a Bargaining Game.* ICML, 2022. — arXiv:2202.01017
- **[Method]** Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn. *Gradient Surgery for Multi-Task Learning.* NeurIPS, 2020. — arXiv:2001.06782
- **[Method]** Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Barret Zoph, Adams Wei Yu. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS, 2023. — arXiv:2305.10429
- **[Method]** Jiasheng Ye, Peiju Liu, Tianxiang Sun, Yunhua Zhou, Jun Zhan, Xipeng Qiu. *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance.* ICLR, 2024. — arXiv:2403.16952
- **[Related]** Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, Silvio Savarese. *Which Tasks Should Be Learned Together in Multi-Task Learning?* ICML, 2020. — arXiv:1905.07553
- **[Survey]** Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, Luc Van Gool. *Multi-Task Learning for Dense Prediction Tasks: A Survey.* IEEE TPAMI, 2021. — arXiv:2004.13379

## 10. Worked Example

Two tasks on a shared trunk: token cross-entropy $\ell_1$ (nats, starts $\approx 10.4$, ends $\approx 2.6$) and an auxiliary depth head with MSE $\ell_2$ in metres$^2$ (starts $\approx 4.0$, ends $\approx 0.35$). Objective $\ell_1 + w\,\ell_2$.

*Gradient-magnitude balancing* (the GradNorm/uncertainty family) sets $w$ so the two gradient norms on the trunk match. Suppose at step 1k, $\|\nabla \ell_1\| = 0.80$ and $\|\nabla \ell_2\| = 0.05$, giving $w = 16$. At step 50k, $\|\nabla \ell_1\| = 0.11$, $\|\nabla \ell_2\| = 0.02$, giving $w = 5.5$. The rule produces a schedule, for free, in one run.

Now the obstruction. Change the depth head's output units from metres to centimetres. Then $\ell_2 \to 10^4 \ell_2$, $\|\nabla\ell_2\| \to 10^2\|\nabla\ell_2\|$, and the balancing rule returns $w = 0.16$ and $0.055$ — the *product* $w\,\ell_2$ is $10^4\times$ larger at every step. A units change that alters nothing about the task changes the effective task weight by four orders of magnitude. Uncertainty weighting, which learns $\sigma_t$ and uses $\ell_t/2\sigma_t^2 + \log\sigma_t$, is scale-equivariant in the first term but not the second, so its fixed point also moves under the same relabeling.

Then the identifiability half: run the metres version with $w=5.5$ and the centimetres version with $w=0.055$ under Adam. Because Adam normalizes per coordinate, the depth head's own updates are near-identical in both; only the trunk sees the ratio. Measured test mIoU differs by 0.3 points — inside the 5-seed spread of $\pm 0.6$. So the experiment cannot distinguish "the rule found a good weight" from "the optimizer was insensitive to the weight". Deciding this needs the tuned-grid control of §8, not a comparison to $w=1$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*