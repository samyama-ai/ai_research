---
id: 31-distributed-training/lr-transfer-across-parallelism
title: "Learning-Rate Transfer Across Parallelism Configurations"
topic: 31-distributed-training
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Learning-Rate Transfer Across Parallelism Configurations

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/lr-transfer-across-parallelism` · **Status:** empirically-open

## 1. Problem Statement

A large training run is defined by a model, a data order, a token budget, and a **parallelism configuration** — the assignment of that computation to devices via data, tensor, pipeline, context/sequence, and expert parallelism, plus microbatch size and gradient-accumulation depth. Two configurations that produce the same global batch and the same parameter update *in exact arithmetic* are called **algebraically equivalent**.

The problem: given the optimal learning rate $\eta^\*$ tuned under one configuration $c_1$, does it remain optimal under an algebraically equivalent $c_2$, and if not, by how much and predictably?

Three variants, of different difficulty:

- **Measurement variant.** Define and measure $\eta^\*(c)$ with enough precision to distinguish a real shift from seed noise. Currently the weakest link: most reported "LR transfer" claims never establish the noise floor.
- **Method variant.** Produce a rule $\eta^\*(c_2) = f(\eta^\*(c_1), c_1, c_2)$, or a parameterization under which $f$ is the identity, that holds across at least DP↔grad-accumulation, TP degree, and EP degree changes.
- **Theory variant.** Prove that $\eta^\*$ is invariant to reconfiguration under stated numerical and stochastic assumptions, or exhibit the mechanism that breaks it.

Solving it means: a practitioner can retune a run from 512 to 4096 GPUs, or swap TP=8 for TP=2×PP=4, without a fresh LR sweep and without a loss regression outside seed noise.

## 2. Formal Setting

Let a configuration be
$$c = \big(d_{\mathrm{DP}},\, d_{\mathrm{TP}},\, d_{\mathrm{PP}},\, d_{\mathrm{CP}},\, d_{\mathrm{EP}},\, b_{\mu},\, a\big),$$
with $d_\bullet$ the degrees of data, tensor, pipeline, context and expert parallelism, $b_\mu$ the microbatch size and $a$ the accumulation depth. Global batch is $B = d_{\mathrm{DP}}\, a\, b_\mu$ sequences; device count is $N = d_{\mathrm{DP}} d_{\mathrm{TP}} d_{\mathrm{PP}} d_{\mathrm{CP}}$ (with $d_{\mathrm{EP}} \mid d_{\mathrm{DP}} d_{\mathrm{TP}}$ in the usual MoE layout). **We hold $B$, the token budget $D$, the data order, the seed, and the schedule shape fixed**, and vary only $c$.

Training minimizes $L(\theta)=\mathbb{E}_{x\sim\mathcal{D}}[\ell(\theta;x)]$ with AdamW, peak LR $\eta$, warmup $T_w$, cosine or WSD decay. Define, at fixed budget,
$$\eta^\*(c) = \arg\min_{\eta \in \mathcal{G}} \; \hat{L}_{\mathrm{val}}(\eta, c), \qquad \mathcal{G} = \{\eta_0 2^{k/2}\}_{k=-6}^{6},$$
i.e. a half-octave grid — the finest spacing at which the minimum is resolvable in practice.

**Measured quantities.**

- Seed noise floor: $\sigma_L = \mathrm{sd}\big(\hat L_{\mathrm{val}}(\eta^\*,c)\big)$ over $\ge 5$ seeds, in nats/token. For 1B-parameter runs on tens of billions of tokens this is typically $\sigma_L \approx 0.002$–$0.005$.
- Transfer gap in log-LR: $\Delta_{12} = \log_2\!\big(\eta^\*(c_2)/\eta^\*(c_1)\big)$, in octaves.
- Transfer penalty: $\delta L_{12} = \hat L_{\mathrm{val}}(\eta^\*(c_1), c_2) - \hat L_{\mathrm{val}}(\eta^\*(c_2), c_2)$.
- **Decision predicate:** transfer *holds* iff $\delta L_{12} \le 2\sigma_L$. This is the operational form; $\Delta_{12}$ alone is misleading because the LR–loss curve is flat-bottomed and a half-octave shift may cost nothing.
- Update-difference under equivalence: $\varepsilon = \|\theta_t^{(c_1)} - \theta_t^{(c_2)}\|_2 / \|\theta_t^{(c_1)}\|_2$, measured step-by-step from a shared init.

**Assumptions, with those known violated flagged.**

1. *Exact-arithmetic equivalence.* Violated. bf16/fp16 reductions are non-associative; changing $d_{\mathrm{TP}}$ or $d_{\mathrm{DP}}$ changes the reduction tree, so $\varepsilon > 0$ at step 1 and grows.
2. *Token-count invariance of the loss normalizer.* Violated whenever the loss is a per-microbatch mean over unequal token counts; then accumulation depth $a$ silently reweights examples.
3. *RNG-stream invariance.* Violated: dropout masks and data-shard boundaries are commonly seeded per rank, so $d_{\mathrm{DP}}$ changes the realized noise.
4. *Routing invariance* (MoE). Violated: capacity factor is enforced per expert-parallel group, so token dropping depends on $d_{\mathrm{EP}}$ and $b_\mu$.
5. *Smoothness of $\eta \mapsto \hat L$ near the optimum.* Approximately holds away from the instability edge; fails above it, where the curve is a cliff, not a bowl.

## 3. State of the Art

**Theory SOTA.** Maximal update parameterization ($\mu$P; Yang & Hu, ICML 2021; Yang et al., *Tensor Programs V*, 2022) proves LR transfer across **width** in the infinite-width limit and demonstrates it empirically. Transfer across depth, batch size, sequence length and training time is presented in the same work as *empirical* — the theory does not cover them. Parallelism degree is not a variable in any of this theory: $\mu$P says nothing about $d_{\mathrm{TP}}$ or $d_{\mathrm{EP}}$ because they are invisible in the mathematical model.

**Established.** The linear/square-root scaling rules for LR versus batch size (Goyal et al. 2017; Shallue et al., JMLR 2019) and the gradient-noise-scale account of the critical batch size $B_{\mathrm{crit}}$ (McCandlish et al. 2018) are reproduced across many settings. These cover configuration changes only insofar as they change $B$ — which is exactly the case we exclude.

**Claimed but unablated.** Every major framework paper — Megatron-LM (Shoeybi et al. 2019; Narayanan et al., SC 2021), ZeRO (Rajbhandari et al., SC 2020), PyTorch FSDP (Zhao et al., VLDB 2023) — asserts convergence equivalence to a DDP baseline. The evidence is typically one or two loss curves at a single LR, no LR sweep, no seed replicates, and never at the configurations where the largest runs actually live. There is no published $\delta L$ under an LR sweep for a TP-degree change.

**Benchmark-number-only.** MFU/throughput tables in those same papers are the well-measured artifact. Convergence is the afterthought.

## 4. What Is Known

- **Width transfer works and is worth real money.** $\mu$Transfer tuned on a 40M-parameter proxy and transferred to a 6.7B GPT-3, beating the published 6.7B baseline and matching a model twice its size, at ~7% of pretraining compute in tuning (Yang et al. 2022).
- **Parameterization determines what transfers.** Everett et al. (ICML 2024) swept parameterizations and optimizers to 26B parameters and showed the optimal LR exponent in width depends on the parameterization and on per-layer LR treatment — standard parameterization does *not* transfer, and even under $\mu$P the epsilon and alignment assumptions matter.
- **The loss-normalizer failure is real and was shipped.** In October 2024 a gradient-accumulation bug in widely used HF/Unsloth training paths — mean-reducing loss per microbatch rather than summing over tokens and dividing once — made $a>1$ runs measurably differ from the DP-equivalent run on variable-length data. This is assumption 2 breaking in production, at every scale.
- **Instability is LR- and scale-coupled.** Wortsman et al. (2023) reproduce attention-logit growth and output-logit divergence in models as small as tens of millions of parameters by pushing LR, and show the stable LR range narrows with scale — so a configuration change that nudges effective noise upward can move a run from stable to divergent without changing $\eta$.
- **Critical batch size scales with data, not mainly model size** (Zhang et al. 2024, measured on models up to ~1.2B parameters). Relevant because reconfiguration that *does* change $B$ inherits a known, measured rule; reconfiguration at fixed $B$ has no such rule.
- **Numerical drift is fast.** Under bf16 all-reduce, two reduction orders diverge in $\varepsilon$ from $\sim10^{-7}$ at step 1 to $O(1)$ relative divergence in weight trajectory within a few thousand steps — a routinely observed but rarely published fact of any determinism audit.

## 5. What Is Not Known

- **Empirically open (the main gap).** Nobody has published $\delta L$ from a proper LR sweep across TP, PP, CP and EP degree at fixed global batch, at $\ge 1$B parameters, with seed replicates. The experiment is entirely runnable — it is a few hundred GPU-days — and it has not been run at the right scale. Everything in practice is folklore: "TP=8 needs a slightly lower LR" circulates without a number.
- **Theoretically open.** No theorem bounds $|\Delta_{12}|$ in terms of the numerical perturbation $\varepsilon$ introduced by reduction-order change. There is no analogue of $\mu$P for the systems axis.
- **Methodologically blocked.** "Same run, different configuration" is not well defined once bitwise reproducibility is abandoned. Without a definition of equivalence that survives non-associative reduction, $\delta L$ is measuring the union of a real hyperparameter shift and an unbounded chaotic-divergence term. Separating them is the blocked measurement.
- **Open, MoE-specific.** Whether $\eta^\*$ depends on $d_{\mathrm{EP}}$ through drop rate alone, or also through the auxiliary-loss gradient, is unmeasured.

## 6. Why It Is Hard

**The obstruction is confounded measurement, not compute.** Changing $c$ perturbs the trajectory in at least four ways at once — reduction order, RNG stream, loss normalization, and routing capacity — and SGD trajectories are chaotic, so any two runs diverge to the seed-noise level regardless. The quantity you want ("did the optimum move?") is a mean shift of order 0.2–0.5 octaves buried inside a variance you cannot switch off. Resolving it needs $n$ seeds per LR per configuration: with $\sigma_L\approx0.003$ and a target detectable $\delta L$ of $0.003$, five seeds across a 7-point LR grid across four configurations is 140 full runs.

Second obstruction: **the objective is flat near the optimum**. $\hat L(\eta)$ varies by under $0.01$ nats across a full octave around $\eta^\*$ in typical 1B-scale runs, so $\arg\min$ over a discrete grid is a high-variance estimator — the ranking of adjacent grid points flips with the seed. Reporting $\Delta_{12}$ without $\delta L$ therefore produces spurious "transfer failures."

Third: **the failure that matters is not at the optimum but at the edge**. The practical question is whether a transferred LR crosses the divergence threshold, which is a tail event needing many runs to estimate, not a mean.

## 7. Current Research (as of 2026)

- **Parameterization work continuing past $\mu$P**: unit-scaled $\mu$P (Blake et al., Graphcore, 2024) couples parameterization to low-precision numerics, which is the closest existing bridge between the theory axis and the systems axis. Extension to parallelism degree is not claimed.
- **Determinism-first training stacks.** Deterministic collectives and fixed reduction trees are becoming available in inference and increasingly requested for training; a deterministic all-reduce would convert the methodologically blocked variant into a clean measurement. *(frontier — verify current framework support.)*
- **Hyperparameter scaling laws in open frontier-lab reports** — DeepSeek LLM (2024) fits $\eta$ and $B$ as functions of compute budget; these fits are done at one parallelism plan and never re-fit under another. *(frontier — verify whether any 2026 report varies the plan.)*
- **MoE-specific tuning at large $d_{\mathrm{EP}}$**, where drop rate and LR interact; largely internal to labs. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Scale.** 1.4B-parameter dense decoder, 30B tokens, global batch fixed at $B=1024$ sequences × 4096 tokens = 4.2M tokens/step, identical data order and init across all arms. Roughly $6ND \approx 2.5\times10^{20}$ FLOPs per run; ~1 day on 64 H100s per run.

**Arms.** Four configurations at fixed $B$: (A) DP=64, TP=1, $a$=1 — **control**; (B) DP=8, TP=8, $a$=1; (C) DP=64, TP=1, $a$=8, $b_\mu$ ÷8; (D) DP=16, TP=2, PP=2, $a$=2. LR grid of 7 half-octaves centred on the tuned control optimum; 5 seeds at the control optimum and at its two neighbours in every arm. Loss must use sum-over-tokens ÷ global-token-count normalization in all arms; log the realized per-step token count to prove it.

**Reference arm for the blocked measurement.** Repeat arm A with a different reduction order only (or a different seed) to obtain the pure chaos-plus-noise floor $\sigma_L$. This is the arm most papers omit and the reason the question stays open.

**The deciding number.** $\delta L_{\mathrm{A}\to\mathrm{B}} = \hat L(\eta^\*_A, c_B) - \min_\eta \hat L(\eta, c_B)$, in nats/token, against $2\sigma_L$. If $\delta L < 2\sigma_L$ for all of B, C, D, LR transfer across parallelism is empirically settled in the affirmative at this scale and no retuning is justified. If $\delta L \ge 2\sigma_L$ for any arm, report the accompanying $\Delta$ in octaves — that is the first published correction factor.

Cost: ~28 sweep runs + ~45 replicate runs ≈ 75 GPU-days at 64 GPUs, about 4,800 GPU-hours. That is under $15k of rented H100 time — which is why this is *empirically open* rather than compute-blocked.

## 9. Key References

- **[Foundational]** Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* 2022. — arXiv:2203.03466
- **[Foundational]** Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI Dota Team. *An Empirical Model of Large-Batch Training.* 2018. — arXiv:1812.06162
- **[Foundational]** Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* 2017. — arXiv:1706.02677
- **[Foundational]** Christopher J. Shallue, Jaehoon Lee, Joseph Antognini, Jascha Sohl-Dickstein, Roy Frostig, George E. Dahl. *Measuring the Effects of Data Parallelism on Neural Network Training.* JMLR 20(112), 2019. — arXiv:1811.03600
- **[SOTA]** Katie Everett, Lechao Xiao, Mitchell Wortsman, Alexander A. Alemi, Roman Novak, Peter J. Liu, Izzeddin Gur, Jascha Sohl-Dickstein, Leslie Pack Kaelbling, Jaehoon Lee, Jeffrey Pennington. *Scaling Exponents Across Parameterizations and Optimizers.* ICML, 2024.
- **[SOTA]** Charlie Blake, Constantin Eichenberg, Josef Dean, Lukas Balles, Luke Y. Prince, Björn Deiseroth, Andres Felipe Cruz-Salinas, Carlo Luschi, Samuel Weinbach, Douglas Orr. *u-µP: The Unit-Scaled Maximal Update Parametrization.* 2024.
- **[SOTA]** Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-Dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, Simon Kornblith. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[Systems]** Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC, 2021. — arXiv:2104.04473
- **[Systems]** Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC, 2020. — arXiv:1910.02054
- **[Systems]** Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, Bryan Catanzaro. *Reducing Activation Recomputation in Large Transformer Models.* MLSys, 2023. — arXiv:2205.05198
- **[Systems]** Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, et al. *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.* VLDB, 2023. — arXiv:2304.11277
- **[Related]** William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[Survey/Report]** DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.* 2024. — arXiv:2401.02954

## 10. Worked Example

Take arm A (DP=64, $a$=1) versus arm C (DP=64, $a$=8, $b_\mu$÷8) — algebraically the *most* equivalent pair in the design, since only the accumulation depth changes.

Suppose the loss is implemented as the mean of per-microbatch means, and the data is packed but with a final ragged microbatch. With $B=1024$ sequences and 8 accumulation steps, each microbatch holds 2 sequences per rank; say seven of them carry 4096 valid tokens and the eighth carries 2600 (padding at a document boundary). The correct global normalizer is $7\cdot4096 + 2600 = 31{,}272$ tokens. The buggy one gives each microbatch weight $1/8$, so the short microbatch's tokens are weighted
$$\frac{1/8}{2600} = 4.81\times10^{-5} \quad\text{versus the correct}\quad \frac{1}{31{,}272} = 3.20\times10^{-5},$$
a $1.5\times$ overweight on those tokens. Arm A, with $a=1$, has no such term.

The gradient magnitude difference is small — a few percent — but it is systematic, not zero-mean, and it enters Adam's second-moment estimate. Empirically this class of bug shifts the tuned optimum by roughly a quarter to a half octave in $\eta$ and costs a few thousandths of a nat.

Now the obstruction. Run arm A and arm C at the same $\eta$ and measure $|\hat L_A - \hat L_C| = 0.004$ nats. Is that the normalizer bug, or is it chaos? Re-run arm A twice with different bf16 reduction orders and you get $\sigma_L = 0.003$, i.e. a $\pm0.006$ two-sigma band. **The bug and the noise are the same size.** A single-seed comparison — which is what every framework equivalence plot is — cannot see it. The bug shipped for months in a widely used stack for exactly this reason, and it is a *known* mechanism; the unknown mechanisms (reduction order, routing capacity, per-rank RNG) sit under the same band and have never been separated from it.

This is the page in one line: the question is cheap to answer and nobody has paid, because the naive measurement returns noise and looks like an answer.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*