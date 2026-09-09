---
id: 31-distributed-training/elastic-batch-size-without-retuning
title: "Elastic Batch Size Without Hyperparameter Re-Tuning"
topic: 31-distributed-training
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Elastic Batch Size Without Hyperparameter Re-Tuning

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/elastic-batch-size-without-retuning` · **Status:** empirically-open

## 1. Problem Statement

Elastic training systems add and remove workers mid-run: spot-instance preemption, job co-scheduling, hardware failure, or deliberate resource rebalancing. The cheapest response is to change the global batch size $B$ to match the worker count. The alternative — hold $B$ fixed and vary gradient-accumulation steps — preserves optimization semantics exactly but wastes the elasticity, since throughput per step then falls with the worker count instead of step time falling.

**The problem.** Find a mapping from a batch-size trajectory $B(t)$ to a hyperparameter trajectory $\theta(t)$ (learning rate, optimizer moment decay rates, weight decay, $\epsilon$, warmup state) such that a run with time-varying $B(t)$ reaches the same loss at the same token budget as a fixed-$B$ run whose hyperparameters were tuned by sweep — with no sweep for the elastic run.

Three variants, of different difficulty:

- **Measurement.** Define "same result" for two runs that consumed identical tokens under different $B(t)$. Loss at equal tokens? At equal wall-clock? Downstream task scores? These do not agree.
- **Method.** Produce a rule $\theta = f(B, \text{state})$ that empirically holds across a switch. This is where the field is.
- **Theory.** Prove that some $f$ makes the parameter law of the elastic run converge, as the discretization is refined, to that of the fixed-$B$ run. Established only for SGD and for adaptive methods in an SDE limit; not for realistic $B$ near the critical batch size.

Solved means: for a preemption trace with $B$ varying over an $8\times$ range, final validation loss is within seed noise of a tuned fixed-$B$ control at equal tokens.

## 2. Formal Setting

Loss $L(w) = \mathbb{E}_{x\sim\mathcal{D}}[\ell(w,x)]$, parameters $w\in\mathbb{R}^d$. At step $k$, batch $\mathcal{B}_k$ of size $B_k$ gives $g_k = \frac{1}{B_k}\sum_{x\in\mathcal{B}_k}\nabla\ell(w_k,x)$, with per-example covariance $\Sigma(w) = \operatorname{Cov}_x[\nabla\ell(w,x)]$, so $\operatorname{Cov}[g_k] = \Sigma(w_k)/B_k$.

**Gradient noise scale** (McCandlish et al., 2018), measured — not assumed — by computing gradients at two batch sizes $B_{\text{small}}, B_{\text{big}}$ and solving the unbiased estimator for $\operatorname{tr}\Sigma$ and $|\nabla L|^2$:

$$B_{\text{noise}} = \frac{\operatorname{tr}\Sigma(w)}{\nabla L(w)^{\top} H(w)\, \nabla L(w)/|\nabla L(w)|^2}\ \approx\ \frac{\operatorname{tr}\Sigma(w)}{|\nabla L(w)|^2}\quad(\text{simple noise scale}).$$

In practice one reports the simple noise scale in *tokens*, averaged over a window, because the Hessian-weighted form needs Hessian-vector products at every step.

**Critical batch size** $B_{\text{crit}}(t)$: the empirical knee of the steps-to-target-loss curve $S(B)$. Measured by the Pareto fit $\left(\frac{S}{S_{\min}}-1\right)\left(\frac{E}{E_{\min}}-1\right)=1$, where $E=BS$ is examples processed; $B_{\text{crit}}=E_{\min}/S_{\min}$. This requires a sweep of full runs at several $B$, each with its own tuned learning rate.

**The elastic budget.** Token budget $T$, trajectory $B:[0,T]\to\{B_1,\dots,B_m\}$ from a preemption trace. Rule $f$ gives $\eta_k=f_\eta(B_k,\cdot)$, $\beta_{1,k},\beta_{2,k}=f_\beta(B_k,\cdot)$. Success predicate, at equal tokens:

$$\Delta = L_{\text{val}}\big(w_T^{\text{elastic}}\big) - L_{\text{val}}\big(w_T^{\text{fixed},\star}\big) \le \tau,$$

with $\tau$ set to the seed-to-seed standard deviation of $L_{\text{val}}$ measured over $\ge 3$ seeds of the control.

**Candidate rules.** Linear (Krizhevsky 2014; Goyal et al. 2017): $\eta\propto B$. Square-root (Hoffer et al. 2017): $\eta\propto\sqrt B$. Adam SDE rule (Malladi et al. 2022): $\eta\propto\sqrt B$ *and* $1-\beta_1,1-\beta_2 \propto B$, $\epsilon\propto\sqrt B$ — i.e. hold the optimizer's EMA timescale fixed in *examples*, not steps. Noise-scale rule: $\eta(B)=\eta_{\max}/(1+B_{\text{noise}}/B)$.

**Assumptions, and which are violated.**
- *Constant $\Sigma$, $H$ over the switch window* — violated; $B_{\text{noise}}$ grows by an order of magnitude over a pretraining run.
- *SDE limit valid* — requires $\eta$ small relative to curvature; violated at the large-LR edge-of-stability regime where LLM pretraining actually runs.
- *$B\ll B_{\text{crit}}$* — violated by design; elasticity is only interesting when large $B$ is on the table.
- *Optimizer state is $B$-invariant* — false. $v_k$ (second moment) contains a $\operatorname{tr}\Sigma/B$ term, so it changes discontinuously in expectation at a switch even if $w$ does not.
- *Loss surface is scheduler-agnostic* — violated; warmup and cosine-decay position interact with the switch.

## 3. State of the Art

**Established (reproduced, ablated).**
- Linear scaling with gradual warmup: ResNet-50/ImageNet at $B=8192$ matches the $B=256$ baseline to 76.3% top-1, 1 hour on 256 P100s (Goyal et al., 2017). Degrades above $B\approx 8\text{k}$.
- LARS (You et al., 2017) and LAMB (You et al., 2020) extend the usable range by layerwise trust-ratio normalization: ResNet-50 at $B=32\text{k}$, BERT pretraining at $B=32\text{k}$ in 76 minutes on 1024 TPUv3 chips.
- No single scaling heuristic works across workloads: Shallue et al. (JMLR 2019) swept $B$ from 2 to $2^{17}$ over 6 model families and found neither linear nor square-root predicted the optimal LR; the exponent varied by workload.
- Increasing $B$ is a partial substitute for decaying $\eta$ (Smith et al., ICLR 2018) — an existence proof that $B(t)$ trajectories can be benign.

**Claimed but unablated for elasticity.**
- Pollux (Qiao et al., OSDI 2021) co-adapts $B$ and $\eta$ online using a gradient-noise-scale-derived statistical-efficiency term, reporting $1.3$–$3.7\times$ reduction in average job completion time on a 64-GPU cluster. The LR rule is a component of a scheduler benchmark, not an isolated ablation against a tuned fixed-$B$ control at matched tokens.
- The Adam square-root + EMA-timescale rule (Malladi et al., NeurIPS 2022) is verified on fixed-$B$ pairs, not on mid-run switches.
- Batch-size invariance for PPO via $\epsilon$ and Adam-timescale correction (Hilton et al., 2022) is demonstrated in RL, and is the closest thing to a *switching* result; it has not been reproduced in supervised or self-supervised pretraining.

**Benchmark-number-only.** DeepSeek LLM (2024) fits $B\propto C^{0.3271}$, $\eta\propto C^{-0.125}$ against compute $C$. This is a fit over a family of *fixed-$B$* runs, not a within-run rule.

## 4. What Is Known

- Perfect-scaling region exists and ends. Shallue et al. (2019): steps-to-target falls as $1/B$ up to a workload-dependent knee, then flattens; the knee ranged over more than two orders of magnitude across their 6 workloads at ImageNet/LM1B scale.
- $B_{\text{crit}}$ scales with *data* seen, not model size: Zhang et al. (ICLR 2025) measured critical batch size in Transformer pretraining up to 1.2B parameters on C4 and found the dominant dependence on token budget, with model size nearly irrelevant at fixed data.
- Learning-rate sensitivity is sharp but not knife-edge. In Transformer pretraining, being $2\times$ off the optimal LR typically costs order $10^{-2}$ nats of final validation loss; being $4\times$ off can cost $10^{-1}$ nats or diverge (Wortsman et al., 2023, at 1.2B scale; Everett et al., ICML 2024, across parameterizations).
- Seed-to-seed noise band is small: repeated pretraining seeds at the $10^8$–$10^9$ parameter scale differ in final validation loss by roughly $10^{-3}$–$10^{-2}$ nats. This sets $\tau$ and is why LR errors are detectable.
- $\mu$P (Yang et al., 2022) transfers LR across *width* at fixed $B$. It explicitly does not solve batch-size transfer; $\mu$Transfer papers tune $B$ separately.

## 5. What Is Not Known

- **Empirically open (dominant).** No published experiment runs a realistic preemption trace ($B$ varying $\ge 4\times$, multiple switches) against a token-matched, LR-swept fixed-$B$ control at $\ge 10^9$ parameters and reports $\Delta$ against a seed-noise band. The experiment is runnable today for well under $10^5$ GPU-hours. Nobody has published it.
- **Empirically open.** Whether optimizer moment state should be rescaled, reset, or left alone at a switch. Three defensible choices; no head-to-head.
- **Theoretically open.** No theorem covers switches at $B\sim B_{\text{crit}}$ with $\eta$ at the edge of stability. The SDE results (Malladi et al., 2022) assume a small-LR limit that production runs violate.
- **Theoretically open.** Whether path dependence is real: does a run that spent its first 20% of tokens at $B=0.5$M reach a different basin than one at $B=2$M, given identical total tokens and correctly scaled $\eta$?
- **Methodologically blocked.** The success criterion itself. Equal-token and equal-wall-clock comparisons rank rules differently, and downstream benchmark scores at these scales have variance larger than the loss gap being measured. There is no agreed metric for "the elastic run was not harmed."

## 6. Why It Is Hard

**Confounded measurement, on top of an $O(n^2)$ control cost.** A mid-run batch-size switch cannot be A/B tested cheaply: the only honest control is a *complete* fixed-$B$ run at the same token budget, with its own LR sweep. So testing one rule against one trace at one scale costs (sweep width $\times$ full runs) $+$ (seeds $\times$ full runs). At 1B parameters and 100B tokens that is roughly 10–20 full pretraining runs.

**Non-identifiability compounds it.** At a switch, $\eta$, $\beta_2$, $\epsilon$, weight decay, and the schedule position all change the trajectory. A single end-of-run loss number cannot attribute the gap to any one of them. The rules also *disagree by less than the sweep resolution in some regimes and by more than the tolerance in others* — so a null result at one $B$ range does not generalize.

**The interesting regime is where theory is silent.** All clean scaling theorems live at $B \ll B_{\text{crit}}$ and small $\eta$. Elasticity is only worth doing near or above $B_{\text{crit}}$, at production LRs.

## 7. Current Research (as of 2026)

- **Critical-batch-size scaling laws.** Following Zhang et al. (ICLR 2025), several groups fit $B_{\text{crit}}(D)$ against token budget to derive within-run schedules. Extending these fits to *time-varying* $B$ is active and unresolved *(frontier — verify)*.
- **Optimizer-side invariance.** Work on Adam variants and normalized/sign-based updates (Muon, Shampoo-family, Lion) claims reduced LR sensitivity; whether that translates into batch-size-switch robustness is untested *(frontier — verify)*.
- **Elastic systems.** TorchElastic/Ray Train, Pollux-descended schedulers, and spot-instance LLM training stacks default to fixed-$B$ with variable accumulation — an admission that the scaling rule is not trusted.
- **Batch-size warmup in production.** Several open LLM recipes ramp $B$ over early training (a monotone, hand-tuned $B(t)$). Recipes report the schedule; almost none report the fixed-$B$ control.

## 8. Concrete Next Experiment

**Scale.** Decoder-only Transformer, 1.3B parameters, 100B tokens of a public corpus (FineWeb-Edu or C4), AdamW, cosine schedule with 2B-token warmup. Roughly 3–4k H100-hours per run.

**Arms.**
1. *Control (must exist).* Fixed $B=2$M tokens, LR swept over $\{1,2,4\}\times$ around the $\mu$P-predicted optimum, best arm run at 3 seeds. Report $\hat\sigma$ of final validation loss.
2. *Trace, no correction.* $B(t)$ from a real spot-preemption trace over $\{0.5, 2, 8\}$M tokens, switching every $\approx 5$B tokens, $\eta$ held at the control value.
3. *Trace + square-root.* $\eta\propto\sqrt B$, moments untouched.
4. *Trace + Adam SDE rule.* $\eta\propto\sqrt B$, $1-\beta_1,1-\beta_2,\epsilon^2\propto B$ (EMA timescale fixed in tokens).
5. *Trace + noise-scale rule.* $\eta=\eta_{\max}/(1+\hat B_{\text{noise}}(t)/B)$ with $\hat B_{\text{noise}}$ estimated online from the two-batch estimator.

Nine runs total. Measure and log $\hat B_{\text{noise}}(t)$ in every arm.

**The deciding number.** $\Delta = L_{\text{val}}^{\text{arm}} - L_{\text{val}}^{\text{control}}$ at 100B tokens, in nats, compared against $\hat\sigma$. A rule *passes* if $\Delta \le 2\hat\sigma$ — concretely, $\Delta \le 0.01$ nats if $\hat\sigma\approx 0.005$. If arm 2 already passes, the problem is smaller than believed and the finding is that Adam is near-invariant at these switch magnitudes. If no arm passes, the finding is that no published rule survives a switch, which is currently the assumed-but-unverified state of the field.

## 9. Key References

- **[Foundational]** McCandlish, Kaplan, Amodei, OpenAI Dota Team. *An Empirical Model of Large-Batch Training.* arXiv, 2018. — arXiv:1812.06162
- **[Foundational]** Goyal, Dollár, Girshick, Noordhuis, Wesolowski, Kyrola, Tulloch, Jia, He. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv, 2017. — arXiv:1706.02677
- **[Foundational]** Shallue, Lee, Antognini, Sohl-Dickstein, Frostig, Dahl. *Measuring the Effects of Data Parallelism on Neural Network Training.* JMLR 20(112), 2019. — arXiv:1811.03600
- **[SOTA]** Zhang, Morwani, Vyas, Wu, Zou, Ghosh, Ghosh, Kakade. *How Does Critical Batch Size Scale in Pre-training?* ICLR, 2025.
- **[SOTA]** Malladi, Lyu, Panigrahi, Arora. *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms.* NeurIPS, 2022. — arXiv:2205.10287
- **[SOTA]** Qiao, Choe, Subramanya, Neiswanger, Ho, Zhang, Ganger, Xing. *Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning.* OSDI, 2021.
- **[SOTA]** You, Li, Reddi, Hseu, Kumar, Bhojanapalli, Song, Demmel, Keutzer, Hsieh. *Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes.* ICLR, 2020. — arXiv:1904.00962
- **[Related]** Smith, Kindermans, Ying, Le. *Don't Decay the Learning Rate, Increase the Batch Size.* ICLR, 2018. — arXiv:1711.00489
- **[Related]** Zhang, Li, Nado, Martens, Sachdeva, Dahl, Shallue, Grosse. *Which Algorithmic Choices Matter at Which Batch Sizes? Insights from a Noisy Quadratic Model.* NeurIPS, 2019.
- **[Related]** Yang, Hu, Babuschkin, Sidor, Liu, Farhi, Ryder, Pachocki, Chen, Gao. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021 / arXiv, 2022. — arXiv:2203.03466
- **[Related]** Hilton, Cobbe, Schulman. *Batch Size Invariance for Policy Optimization.* NeurIPS, 2022. — arXiv:2110.00641
- **[Related]** Everett, Xiao, Wortsman, Alemi, Novak, Liu, Gur, Sohl-Dickstein, Kaelbling, Lee, Pennington. *Scaling Exponents Across Parameterizations and Optimizers.* ICML, 2024. — arXiv:2407.05872

## 10. Worked Example

A 1.3B-parameter run loses half its workers at token 30B. $B$ drops $2\text{M}\to 0.5\text{M}$ tokens; later it recovers to 2M. What multiplier should $\eta$ take on the way back up?

Measured input: the simple noise scale at token 30B is $\hat B_{\text{noise}} \approx 2$M tokens — the same order as $B$ itself, which is exactly the elastic-interesting regime.

Three rules, all defensible, all published:

| Rule | $\eta(2\text{M})/\eta(0.5\text{M})$ |
|---|---|
| Linear ($\eta\propto B$) | $4.00$ |
| Square-root ($\eta\propto\sqrt B$) | $2.00$ |
| Noise-scale, $\eta=\eta_{\max}/(1+\hat B_{\text{noise}}/B)$ | $\dfrac{1+2/0.5}{1+2/2}=\dfrac{5}{2}=2.50$ |

The rules span $2.00\times$ to $4.00\times$ — a factor of 2 disagreement about the learning rate. From Section 4, a $2\times$ LR error costs order $10^{-2}$ nats at this scale. The tolerance $\tau\approx 2\hat\sigma \approx 0.01$ nats. So **the spread between published rules is at least as large as the entire error budget**: picking the wrong one is guaranteed to fail the success predicate, and there is no published measurement telling you which is right at a switch.

It gets worse on the state. Adam's second moment carries $\mathbb{E}[v]\approx |\nabla L|^2 + \operatorname{tr}\Sigma/B$. With $\hat B_{\text{noise}}=\operatorname{tr}\Sigma/|\nabla L|^2 = 2$M, at $B=0.5$M we have $\mathbb{E}[v]\approx|\nabla L|^2(1+4)=5|\nabla L|^2$; at $B=2$M, $\mathbb{E}[v]\approx 2|\nabla L|^2$. The preconditioner $1/\sqrt{v}$ therefore shifts by $\sqrt{5/2}\approx 1.58\times$ *on its own*, over the $\beta_2$ relaxation window — an uncontrolled transient that partially cancels whatever $\eta$ multiplier you chose, with a timescale of $1/(1-\beta_2)\approx 1000$ steps.

That is the obstruction in one instance: the intended intervention ($\eta$) and an unintended one (the preconditioner transient) act in the same direction, over overlapping timescales, and only their product is observable in the final loss. Distinguishing them requires the full nine-run design of Section 8, not a single elastic run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*