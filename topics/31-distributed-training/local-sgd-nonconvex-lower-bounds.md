---
id: 31-distributed-training/local-sgd-nonconvex-lower-bounds
title: "Local SGD Communication-Round Lower Bounds for Nonconvex Objectives"
topic: 31-distributed-training
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Local SGD Communication-Round Lower Bounds for Nonconvex Objectives

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/local-sgd-nonconvex-lower-bounds` · **Status:** open

## 1. Problem Statement

Local SGD (FedAvg, and in LLM practice DiLoCo) runs $K$ stochastic gradient steps independently on each of $M$ machines, then averages. The question: **how many communication rounds $R$ are unavoidably required to reach an $\epsilon$-stationary point of a smooth nonconvex objective, given $M$ machines and $K$ local steps per round?**

Three variants, which are routinely conflated:

- **Theory variant.** Prove a lower bound on $R$ for *any* algorithm in the intermittent-communication (IC) graph — the oracle model where each machine may take $K$ sequential stochastic-gradient queries between synchronizations — that matches a known upper bound, for nonconvex $F$. Equivalently: is the minibatch-SGD round complexity $\Theta(H\Delta/\epsilon^2)$ optimal in the IC model, or can local steps provably buy rounds?
- **Method variant.** Exhibit an algorithm in the IC graph whose round complexity is strictly below minibatch SGD's for nonconvex $F$, under assumptions that hold for neural training.
- **Measurement variant.** Define the empirical quantity — rounds to reach a target loss at fixed total token budget — so that it is comparable across $(M, K, \eta_{\text{inner}}, \eta_{\text{outer}})$ and not silently a statement about tuned learning-rate schedules.

Solved means: a lower bound $R = \Omega(g(H,\Delta,\sigma,\zeta,\epsilon,M,K))$ and an algorithm matching it to constants/logs, in the nonconvex setting, for both homogeneous and heterogeneous data.

## 2. Formal Setting

$M$ machines, machine $m$ holds distribution $\mathcal{D}_m$. Objective
$$F(x) = \frac{1}{M}\sum_{m=1}^{M} F_m(x), \qquad F_m(x) = \mathbb{E}_{z\sim\mathcal{D}_m}\,f(x;z), \quad x \in \mathbb{R}^d .$$

Measured quantities:

- $\Delta = F(x_0) - \inf_x F$. Measured as initial loss minus best loss reached by any run in the sweep (a lower bound on the true $\Delta$).
- $H$: smoothness, $\|\nabla F(x) - \nabla F(y)\| \le H\|x-y\|$. Measured as the top Hessian eigenvalue via power iteration on Hessian-vector products, at a checkpoint.
- $\sigma^2$: per-machine gradient-noise variance, $\mathbb{E}\|\nabla f(x;z) - \nabla F_m(x)\|^2 \le \sigma^2$. Measured as the trace of the empirical covariance of per-microbatch gradients at a fixed checkpoint.
- Heterogeneity. First-order: $\frac{1}{M}\sum_m \|\nabla F_m(x) - \nabla F(x)\|^2 \le \zeta^2$. Second-order (mean smoothness of the differences): $\|\nabla F_m(x)-\nabla F_m(y) - (\nabla F(x)-\nabla F(y))\| \le \tau\|x-y\|$. Measured by computing per-shard full gradients on held-out shards at the same checkpoint and taking the empirical dispersion.
- Third-order smoothness $Q$: $\|\nabla^2 F(x) - \nabla^2 F(y)\| \le Q\|x-y\|$. Not directly measurable at scale; estimated only via finite differences of Hessian-vector products.

Protocol: for $r = 1..R$, each machine starts from $x^{r-1}$, takes $K$ sequential steps using one fresh stochastic gradient each, then the server aggregates. Total oracle calls $N = MKR$. Success predicate: output $\hat{x}$ with $\mathbb{E}\|\nabla F(\hat{x})\| \le \epsilon$.

**Assumptions known to be violated in practice.** (i) Uniform bounded variance — gradient noise in transformer training is strongly anisotropic and grows with $\|\nabla F\|$; (ii) global $H$-smoothness — loss landscapes show $H$ varying by 1–2 orders of magnitude along a trajectory and edge-of-stability behavior where the effective $H$ is *set by* the step size; (iii) bounded $\zeta$ uniformly in $x$ — heterogeneity is measured only at checkpoints, not along the whole path; (iv) the IC oracle assumes one gradient per step, whereas real workers use large local batches, so the mapping $K \leftrightarrow$ wall-clock is not the model's.

## 3. State of the Art

**Theory SOTA.**
- *Established.* Arjevani, Carmon, Duchi, Foster, Srebro, Woodworth (*Math. Programming*, 2023) give the tight sample lower bound $\Omega(\sigma^2 H\Delta \epsilon^{-4})$ for nonconvex stochastic first-order optimization, plus the deterministic $\Omega(H\Delta\epsilon^{-2})$ term (Carmon, Duchi, Hinder, Sidford, *Math. Programming*, 2020). These constrain $N = MKR$ but say nothing about the split between $K$ and $R$.
- *Established.* Glasgow, Wu, Duchi (AISTATS 2022) prove a tight lower bound for local SGD in the **convex** homogeneous IC setting, showing the known upper bound is unimprovable — the first time local SGD's rate was pinned rather than bounded above.
- *Established.* Patel, Glasgow, Zindari, Wang, Stich, Cheng, Joshi, Srebro (COLT 2024) show local SGD is **not** better than minibatch SGD for heterogeneous convex problems under first- and second-order heterogeneity assumptions, and *is* provably better under third-order smoothness. This is the sharpest statement of when local steps buy rounds — and it is convex.
- *Open in nonconvex.* No matching algorithm-independent round lower bound for the nonconvex IC graph exists. Upper bounds for local SGD in the nonconvex case (Stich ICLR 2019; Yu, Yang, Zhu AAAI 2019; Khaled, Mishchenko, Richtárik AISTATS 2020; Koloskova et al. ICML 2020) all carry an extra term of the form $\left(\frac{\sigma\zeta H\Delta}{R}\right)^{2/3}$-type or $\frac{H^2K\sigma^2}{\cdot}$ that no lower bound is known to require.

**Systems/empirical SOTA.** DiLoCo (Douillard et al., 2023) trains language models with $H\!\approx\!500$ local AdamW steps per round and Nesterov outer momentum, reporting near-parity with fully synchronous training at ~500× less communication. Streaming DiLoCo (Douillard et al., 2025) reports further bandwidth reduction by partial-parameter synchronization and overlap. **These are benchmark numbers, not ablations of the round-complexity question:** they are reported at fixed token budgets with separately tuned inner/outer optimizers, so they do not isolate whether $R$ could be reduced further, nor at what $\epsilon$ the parity breaks.

## 4. What Is Known

- **Sample complexity is settled.** $\Theta(\sigma^2 H \Delta \epsilon^{-4})$ oracle calls, tight up to constants (Arjevani et al. 2023). Minibatch SGD with batch $MK$ per round attains $\epsilon^2 \lesssim \frac{H\Delta}{R} + \sqrt{\frac{H\Delta\sigma^2}{MKR}}$ — so $R = O\!\left(\frac{H\Delta}{\epsilon^2}\right)$ rounds always suffice once $MKR$ is large enough.
- **Local SGD cannot beat this in the worst heterogeneous convex case** (Patel et al., COLT 2024); the separation requires third-order smoothness.
- **The convex homogeneous local-SGD rate is tight** (Glasgow et al., AISTATS 2022), including the $K$-dependence.
- **Empirically, $R$ is far below the theory's worst case.** DiLoCo: 8 workers × 500 local steps on C4 with ~150M-parameter decoders, reported to match or beat synchronous data-parallel at equal token count. Reported LLM-scale runs extend the pattern to the 1B-parameter range. No published run at any scale has found the round budget that theory's $\zeta$-dependent terms would predict to be necessary.
- **The heterogeneity term is real when heterogeneity is real.** SCAFFOLD (Karimireddy et al., ICML 2020) removes the client-drift term with control variates and improves round counts on pathologically split federated benchmarks (e.g. label-partitioned EMNIST/CIFAR at $M = 100$s of clients) — evidence the $\zeta$ term is not an artifact in the federated regime, and evidence it does *not* bind in the LLM data-center regime where shards are IID.

## 5. What Is Not Known

- **Theoretically open.** Whether any IC-graph algorithm achieves $R = o(H\Delta/\epsilon^2)$ for nonconvex $F$ with $\sigma>0$, and what the matching lower bound is. Also open: the tight $K$-dependence of local SGD's nonconvex rate — even in the homogeneous case, existing upper bounds have a $K$-dependent drift term with no lower-bound counterpart.
- **Theoretically open.** Whether the third-order-smoothness separation of Patel et al. (2024) extends from convex to nonconvex objectives, and whether outer momentum (the DiLoCo ingredient) changes the round complexity class at all or only constants.
- **Empirically open.** Whether the DiLoCo-style parity survives at fixed *final loss* (not fixed tokens) as $K$ grows past $10^3$ and $M$ past $10^2$. Runnable today at ~1B parameters; nobody has published the $K \times M$ grid.
- **Methodologically blocked.** The empirical "round complexity" of a training run is not well defined: $R$ depends on the outer optimizer, on $\eta_{\text{inner}}$ tuned per $K$, and on the loss target chosen. Comparing $R$ across $K$ without a protocol that fixes what is tuned is comparing tuning budgets.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the hard instance under the assumptions that actually hold**. Lower bounds in the IC graph work by constructing a "zero-chain" function where progress requires one coordinate per sequential oracle call. Local steps on a zero-chain are not useless — they advance the chain — so the construction must instead make local progress *misleading*, which requires heterogeneity ($\zeta$ or $\tau$) or high curvature variation. But the constructions that are hard for local SGD are exactly the ones that violate third-order smoothness, and Patel et al. showed the separation flips sign across that assumption. So the answer depends on a constant ($Q$) that **cannot be measured at scale**: estimating third-order smoothness of a 1B-parameter loss requires finite differences of Hessian-vector products with a step size that is itself the thing being probed. Secondary obstruction: compute. A clean $K \times M$ grid at 1B parameters with three seeds is $\sim\!10^2$ full pretraining runs.

## 7. Current Research (as of 2026)

- **Lower bounds in the IC graph.** Srebro's group (TTIC) and collaborators (Patel, Joshi, Glasgow, Stich) continue the convex→nonconvex program; the natural next theorem is a nonconvex analogue of the COLT 2024 separation. *(frontier — verify)*
- **Local-update LLM training.** Google DeepMind's DiLoCo line (Douillard et al.), plus open replications (Prime Intellect's INTELLECT runs, Nous Research DisTrO/DeMo). These are systems results; their scaling-law claims for local-update training are *(frontier — verify)*.
- **Bridging.** Analyses of outer momentum as an inexact-proximal or lookahead method, aimed at explaining why DiLoCo tolerates $K \sim 500$ where theory predicts drift. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does increasing $K$ at fixed total tokens cost rounds beyond the minibatch-SGD baseline, and does the cost track measured $\tau$?

- **Scale.** 400M-parameter decoder, C4, 40B tokens fixed for every arm. $M = 16$ workers. $K \in \{1, 16, 64, 256, 1024\}$ (so $R$ = tokens/($MK$·batch)).
- **Control arm.** $K=1$ is exactly minibatch SGD with batch $M \cdot b$ — the same total batch and the same token budget, so the only difference is the number of syncs. Second control: shuffled-IID shards vs. domain-partitioned shards (C4 by source domain), giving a low-$\tau$ and a high-$\tau$ condition with everything else held.
- **Protocol fix for the measurement block.** Tune $\eta_{\text{inner}}$ on a 3-point log grid *within each arm* and report the best; hold outer optimizer fixed (Nesterov, $\eta_{\text{outer}}=0.7$). Report tuning budget per arm.
- **Deciding number.** $R^\star(K)$ = rounds to reach validation loss 2.90 nats, normalized: $\rho(K) = R^\star(K)\cdot K / R^\star(1)$. If $\rho(K) \le 1.1$ out to $K=1024$ in the high-$\tau$ arm, local steps are free at this scale and the $\zeta/\tau$ terms in every existing nonconvex upper bound are loose. If $\rho$ breaks above $1.5$ at some $K_c$, report $K_c$ against measured $\tau$ — that pair is the first empirical constraint on the true bound.

Cost estimate: 10 arms × 2 shard conditions × 40B tokens ≈ 8×$10^{20}$ FLOPs, ~2k A100-days.

## 9. Key References

- **[Foundational]** Yossi Arjevani, Yair Carmon, John C. Duchi, Dylan J. Foster, Nathan Srebro, Blake Woodworth. *Lower Bounds for Non-Convex Stochastic Optimization.* Mathematical Programming, 2023. — arXiv:1912.02365
- **[Foundational]** Yair Carmon, John C. Duchi, Oliver Hinder, Aaron Sidford. *Lower Bounds for Finding Stationary Points I.* Mathematical Programming, 2020.
- **[Foundational]** Sebastian U. Stich. *Local SGD Converges Fast and Communicates Little.* ICLR, 2019. — arXiv:1805.09767
- **[SOTA-theory]** Margalit Glasgow, Honglin Yuan, Tengyu Ma. *Sharp Bounds for Federated Averaging (Local SGD) and Continuous Perspective.* AISTATS, 2022.
- **[SOTA-theory]** Kumar Kshitij Patel, Margalit Glasgow, Ali Zindari, Lingxiao Wang, Sebastian U. Stich, Ziheng Cheng, Nirmit Joshi, Nathan Srebro. *The Limits and Potentials of Local SGD for Distributed Heterogeneous Learning with Intermittent Communication.* COLT, 2024.
- **[SOTA-theory]** Blake Woodworth, Kumar Kshitij Patel, Nathan Srebro. *Minibatch vs Local SGD for Heterogeneous Distributed Learning.* NeurIPS, 2020.
- **[SOTA-theory]** Blake Woodworth, Kumar Kshitij Patel, Sebastian U. Stich, Zhen Dai, Brian Bullins, H. Brendan McMahan, Ohad Shamir, Nathan Srebro. *Is Local SGD Better than Minibatch SGD?* ICML, 2020.
- **[SOTA-method]** Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, Ananda Theertha Suresh. *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.* ICML, 2020. — arXiv:1910.06378
- **[SOTA-systems]** Arthur Douillard, Qixuan Feng, Andrei A. Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc'Aurelio Ranzato, Arthur Szlam, Jiajun Shen. *DiLoCo: Distributed Low-Communication Training of Language Models.* 2023. — arXiv:2311.08105
- **[Survey]** Anastasia Koloskova, Nicolas Loizou, Sadra Boreiri, Martin Jaggi, Sebastian U. Stich. *A Unified Theory of Decentralized SGD with Changing Topology and Local Updates.* ICML, 2020.

## 10. Worked Example

Take $M=16$, $K=256$, $R=1000$, so $N = 4.1\times10^6$ oracle calls. Suppose measured at a checkpoint: $H = 40$, $\Delta = 3.0$ nats, $\sigma^2 = 2.5$ (per-microbatch trace), target $\epsilon = 0.05$.

**Minibatch SGD bound.** $\epsilon^2 \lesssim \frac{H\Delta}{R} + \sqrt{\frac{H\Delta\sigma^2}{N}} = \frac{120}{1000} + \sqrt{\frac{300}{4.1\times10^6}} = 0.120 + 0.0086 = 0.129$. So $\epsilon \approx 0.36$ — the **round-limited** term dominates by 14×. To hit $\epsilon = 0.05$ this bound needs $R \ge H\Delta/\epsilon^2 = 48{,}000$ rounds.

**Local SGD bound.** Add the standard drift term. Even in the homogeneous case, Koloskova et al.-style analyses contribute roughly $\left(\frac{H\Delta\sigma}{R\sqrt{K}}\right)^{2/3}\cdot K^{1/3}$-order corrections; with $K=256$ this is *larger* than the minibatch term, so **the theory says local SGD is strictly worse here**.

**What is actually observed.** DiLoCo-style runs at this $K$ and $M$ reach the same validation loss as the $K=1$ control at the same token count. So the empirical $R$ is ~1000 where both bounds demand ~$5\times10^4$, and the *ordering* the theory predicts ($K=1$ better than $K=256$) is not observed.

**The obstruction made visible.** Two explanations fit the data equally well: (a) the true nonconvex round lower bound is $\Theta(H\Delta/(\epsilon^2 K^\alpha))$ for some $\alpha>0$ and all current upper bounds are loose; (b) the bounds are tight but the measured $H=40$ is the wrong constant — the trajectory's *effective* smoothness along the low-curvature manifold the optimizer occupies is $10\times$ smaller, and $Q$ (unmeasured) is small enough to place this instance in the regime where Patel et al.'s separation favors local steps. Nothing measurable at 400M parameters distinguishes (a) from (b): both predict identical loss curves. That is why the problem is open rather than merely unrun — the deciding constant, $Q$, has no scalable estimator.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*