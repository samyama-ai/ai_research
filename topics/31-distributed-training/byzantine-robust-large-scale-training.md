---
id: 31-distributed-training/byzantine-robust-large-scale-training
title: "Provable Byzantine Robustness for Large-Scale Nonconvex Training"
topic: 31-distributed-training
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Provable Byzantine Robustness for Large-Scale Nonconvex Training

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/byzantine-robust-large-scale-training` · **Status:** solved-but-impractical

## 1. Problem Statement

$n$ workers compute stochastic gradients of a shared nonconvex loss. An unknown subset of size $\le \delta n$ is *Byzantine*: it may send arbitrary vectors, collude, and know the honest workers' gradients and the aggregation rule. Aggregate the $n$ messages each step so that training converges to a stationary point of the honest objective at a rate degraded only by a bounded function of $\delta$.

Three variants, with very different difficulty:

- **Theory.** Prove a rate $\mathbb{E}\|\nabla F(x_{\text{out}})\|^2 \le \epsilon + c(\delta)$ against a worst-case adaptive adversary, and prove $c(\delta)$ is unimprovable. **Largely settled** for homogeneous and bounded-heterogeneity data.
- **Method.** Build an aggregator whose per-step cost, memory, and communication pattern survive at LLM scale ($d \sim 10^{10}$–$10^{12}$ parameters, sharded optimizer state, ring/tree all-reduce that never materializes a per-worker gradient). **Open.**
- **Measurement.** Define what "robust" means for a 100B-parameter pretraining run where no attack-free rerun exists and loss curves differ by seed. **Blocked.**

A solution is a system that trains a $\ge$10B-parameter model to within a stated loss gap of an attack-free control, under a named adaptive attack with $\delta = 0.2$, at $\le 1.1\times$ the step time of plain all-reduce.

## 2. Formal Setting

Honest worker $i$ holds distribution $\mathcal{D}_i$; the objective is $F(x) = \frac{1}{|\mathcal{H}|}\sum_{i \in \mathcal{H}} f_i(x)$, $f_i(x) = \mathbb{E}_{\xi \sim \mathcal{D}_i}[\ell(x;\xi)]$, $x \in \mathbb{R}^d$. $\mathcal{H}$ is the honest set, $|\mathcal{H}| \ge (1-\delta)n$.

Measured quantities:

- **Gradient noise** $\sigma^2$: $\mathbb{E}\|g_i - \nabla f_i\|^2 \le \sigma^2$. Measured as the trace of the empirical covariance of per-microbatch gradients on one worker, in $\text{loss-units}^2$; it scales as $\sigma_1^2/B$ for batch size $B$.
- **Heterogeneity** $\zeta^2$: $\frac{1}{|\mathcal{H}|}\sum_{i\in\mathcal{H}}\|\nabla f_i(x) - \nabla F(x)\|^2 \le \zeta^2$. Measured by holding $x$ fixed, computing each shard's full-batch gradient, and taking the empirical variance across shards. Rarely reported; it is *not* constant along the trajectory.
- **Aggregator quality** $(c,\delta)$-**robustness** (Karimireddy et al., ICML 2021): for inputs $\{y_i\}$ with honest variance $\rho^2$, the output $\hat y$ satisfies
$$\|\hat y - \bar y_{\mathcal{H}}\|^2 \le c\,\delta\,\rho^2 .$$
Measured directly: feed synthetic inputs with known $\bar y_\mathcal{H}$ and $\rho^2$, sweep attacks, take the max ratio.
- **Cost**: aggregator FLOPs/step, peak bytes the aggregation point must hold, and wall-clock step time — all measured on the same hardware as the control.

Target guarantee for smooth $F$ ($L$-smooth, $F(x_0)-F^\star \le \Delta$), $T$ steps:
$$\min_t \mathbb{E}\|\nabla F(x_t)\|^2 \;=\; O\!\left(\sqrt{\tfrac{L\Delta\sigma^2}{nT}}\right) \;+\; O\big(\delta\,(\sigma^2 + \zeta^2)\big).$$
The second term is a **non-vanishing floor**: no amount of compute removes it.

Assumptions known to be violated in practice: (i) bounded $\zeta^2$ — data shards in real pretraining are domain-partitioned and $\zeta$ drifts by orders of magnitude across phases; (ii) uniform $\sigma^2$ across workers — microbatch composition and sequence packing differ; (iii) $L$-smoothness — transformer loss surfaces show loss spikes inconsistent with a single global $L$; (iv) static $\mathcal{H}$ — real failures (bit flips, silent data corruption, a bad NIC) are intermittent, so $\delta$ fluctuates step to step.

## 3. State of the Art

**Theory SOTA (established).**
- Coordinate-wise median and trimmed mean give order-optimal statistical rates for strongly convex and smooth losses (Yin, Chen, Ramchandran, Bartlett, ICML 2018).
- Byzantine-resilient nonconvex SGD with $\tilde O(1/\epsilon^4)$ complexity and *no* bounded-gradient assumption (Allen-Zhu, Ebrahimian, Li, Alistarh, ICLR 2021).
- **Momentum is the key primitive**: worker-side momentum shrinks the variance the aggregator sees, restoring robustness against time-coupled attacks (Karimireddy, He, Jaggi, ICML 2021 — centered clipping; El-Mhamdi, Guerraoui, Rouault, ICLR 2021).
- **Heterogeneity floor.** Under $\zeta$-heterogeneity, no algorithm can drive error below $\Omega(\delta\zeta^2)$; bucketing plus a robust aggregator matches it (Karimireddy, He, Jaggi, ICLR 2022). This is a genuine lower bound, not a proof artifact.
- Variance reduction removes the $\sigma^2$ part of the floor for finite sums (Gorbunov, Horváth, Richtárik, Gidel, ICLR 2023, Byz-VR-MARINA).

**Claimed but unablated.** Most robust-aggregator papers report accuracy under a fixed attack list (bit-flip, label-flip, ALIE, IPM) and omit: adaptive attacks tuned to the *deployed* aggregator, $\zeta$ measurement, and wall-clock cost. Reported "robustness" is often a benchmark number on CIFAR-10/MNIST with $n \le 50$, not an ablation.

**Systems SOTA.** Production frameworks (Megatron-LM, DeepSpeed, FSDP, PyTorch DDP) ship **no** Byzantine defense. Redundancy-based schemes — DRACO (Chen, Wang, Charles, Papailiopoulos, ICML 2018) and DETOX (Rajput et al., NeurIPS 2019) — are the only designs with all-reduce-compatible cost, and they buy it with $2$–$3\times$ gradient replication. Secure aggregation (Bonawitz et al., CCS 2017) protects privacy, not correctness.

## 4. What Is Known

- **Naive averaging fails at $\delta = 1/n$.** One worker sending $-n\cdot\bar g + v$ moves the update to any $v$. Established analytically; no scale caveat.
- **Distance-based aggregators break under small colluding perturbations.** "A Little Is Enough" (Baruch, Baruch, Goldberg, NeurIPS 2019) drives Krum/median/trimmed-mean models to near-random accuracy on CIFAR-10 with perturbations of $\approx 0.3\sigma$, at $n = 51$, $\delta \approx 0.24$. Inner-product manipulation (Xie, Koyejo, Gupta, UAI 2019) reproduces this against Krum and Bulyan.
- **Bulyan** (El Mhamdi, Guerraoui, Rouault, ICML 2018) fixes the ALIE-style leakage but requires $n \ge 4\delta n + 3$ and costs $O(n^2 d)$.
- **Momentum + bucketing works empirically where plain robust means do not**: ICLR 2022 bucketing results on heterogeneous MNIST/CIFAR-10 splits with $n = 25$, $\delta$ up to $0.2$, recovering most of the attack-free accuracy where unbucketed median collapses.
- **Cost scaling is measured and bad.** Krum/Bulyan are $\Theta(n^2 d)$ FLOPs and require the aggregation point to hold all $n$ gradients: $O(nd)$ bytes, versus $O(d)$ for ring all-reduce.
- **Largest published Byzantine-robust training runs are small.** The literature's scale is ResNet-20/CIFAR-10 and small LSTMs, $n \le 100$, $d \le 10^8$. No peer-reviewed run at $d \ge 10^9$ with an adaptive adversary is known to this catalog as of 2026.

## 5. What Is Not Known

- **Theoretically open.** Whether the $\Omega(\delta\zeta^2)$ floor is avoidable when honest workers can *communicate with each other* rather than only with a server, and whether any aggregator achieves $(c,\delta)$-robustness with $O(nd)$ FLOPs *and* $O(d)$ peak memory simultaneously. Also open: rates under intermittent (per-step-resampled) corruption, which is the realistic hardware-fault model.
- **Empirically open.** Does momentum + bucketing + centered clipping preserve loss at $d \approx 10^{10}$? The experiment is runnable today on ~256 GPUs; nobody has published it. Also open: the actual value of $\zeta^2$ in domain-sharded LLM pretraining — a single measurement run would settle whether the theory's floor is $10^{-2}$ or $10^{2}$ in loss units.
- **Methodologically blocked.** "Robustness" at pretraining scale has no control arm: a single 10B run costs $\sim$$10^5$ GPU-hours, seed variance in final loss is comparable to the effect being measured, and there is no ground truth for which worker was faulty in a real cluster incident. Until an accepted attack-free-control protocol exists, negative results are unfalsifiable.

## 6. Why It Is Hard

The specific obstruction is **structural incompatibility with collective communication**, not proof difficulty.

Every $(c,\delta)$-robust aggregator with a proof is a *nonlinear* function of the $n$ individual gradients. Ring and tree all-reduce achieve their bandwidth optimality precisely because summation is associative and can be done in fragments: no node ever holds worker $i$'s full gradient. A robust aggregator forces materialization of $n$ full gradient vectors at one point. For $n = 256$, $d = 7\times10^9$, bf16, that is $256 \times 14\,\text{GB} = 3.6$ TB of transient state at the aggregation point — three orders of magnitude past a single accelerator's HBM. Sharding the aggregator across the $d$ axis (coordinate-wise median, trimmed mean) restores memory feasibility but is exactly the class ALIE defeats; sharding the geometric/distance-based rules is not possible without recomputing $O(n^2)$ distances over full vectors.

A second obstruction is **confounded measurement**: the quantity that decides whether a defense works, $\zeta^2$, is trajectory-dependent and essentially never reported, so a defense's failure cannot be attributed to the aggregator versus a violated assumption.

## 7. Current Research (as of 2026)

- **Resilient-averaging-of-momentums** line (Farhadkhani, Guerraoui, Gupta, Pinot, Stephan, ICML 2022) — reducing all robust aggregators to one analysis, plus tight $\delta$ thresholds. EPFL/DCL continues here.
- **Variance-reduced Byzantine methods** (Gorbunov, Richtárik and collaborators, KAUST/MBZUAI) — removing the $\sigma^2$ term from the floor.
- **Robustness under communication compression and asynchrony**, the regime that matters for geo-distributed training (DiLoCo-style low-communication training, Douillard et al., 2023, has no Byzantine analysis) *(frontier — verify)*.
- **Verified/attested execution** as an alternative to statistical robustness: TEEs on accelerators shift the threat model from "arbitrary vector" to "crash fault", which linear all-reduce already tolerates *(frontier — verify)*.
- **Silent-data-corruption studies** from hyperscalers (Meta, Google) motivating the intermittent-fault model rather than the static-adversary model.

## 8. Concrete Next Experiment

**Scale.** Pretrain a 1.4B-parameter decoder-only transformer, 30B tokens, $n = 64$ data-parallel workers, domain-sharded data (not IID). Roughly 3,000 A100-hours per arm; five arms is affordable at a university cluster.

**Arms.**
1. **Control:** plain all-reduce SGD/AdamW, no attack. Establishes the reference loss and seed variance (run 2 seeds).
2. Plain all-reduce, $\delta = 0.15$ (10 of 64 workers), ALIE attack.
3. Centered clipping + worker momentum ($\beta = 0.9$) + bucketing ($s = 2$), same attack.
4. Same defense, **adaptive** attack: the adversary knows the clipping radius and momentum state and solves for the maximal-damage perturbation inside the clip ball.
5. Defense, no attack — the *cost of the defense itself*.

**Instrumentation.** Report measured $\zeta^2$ every 500 steps (fixed $x$, per-shard full-batch gradients) and per-step wall-clock.

**Deciding number.** Final validation loss gap $\Delta\mathcal{L} = \mathcal{L}_{\text{arm 4}} - \mathcal{L}_{\text{arm 1}}$, in nats/token, against the arm-1 seed spread $s$. If $\Delta\mathcal{L} \le 0.01$ nats and step time $\le 1.1\times$ control, the method variant is solved at 1.4B and the question moves to $10^{10}$. If $\Delta\mathcal{L} > 3s$, adaptive attacks defeat the current SOTA defense at realistic heterogeneity — which no published experiment currently establishes or refutes.

## 9. Key References

- **[Foundational]** P. Blanchard, E. M. El Mhamdi, R. Guerraoui, J. Stainer. *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent.* NeurIPS, 2017.
- **[Foundational]** D. Yin, Y. Chen, K. Ramchandran, P. Bartlett. *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.* ICML, 2018. — arXiv:1803.01498
- **[Foundational]** Y. Chen, L. Su, J. Xu. *Distributed Statistical Machine Learning in Adversarial Settings: Byzantine Gradient Descent.* POMACS, 2017.
- **[Attack]** G. Baruch, M. Baruch, Y. Goldberg. *A Little Is Enough: Circumventing Defenses For Distributed Learning.* NeurIPS, 2019.
- **[Attack]** C. Xie, O. Koyejo, I. Gupta. *Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation.* UAI, 2019.
- **[Attack/Defense]** E. M. El Mhamdi, R. Guerraoui, S. Rouault. *The Hidden Vulnerability of Distributed Learning in Byzantium.* ICML, 2018.
- **[SOTA]** S. P. Karimireddy, L. He, M. Jaggi. *Learning from History for Byzantine Robust Optimization.* ICML, 2021.
- **[SOTA]** S. P. Karimireddy, L. He, M. Jaggi. *Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing.* ICLR, 2022.
- **[SOTA]** Z. Allen-Zhu, F. Ebrahimian, J. Li, D. Alistarh. *Byzantine-Resilient Non-Convex Stochastic Gradient Descent.* ICLR, 2021.
- **[SOTA]** S. Farhadkhani, R. Guerraoui, N. Gupta, R. Pinot, J. Stephan. *Byzantine Machine Learning Made Easy by Resilient Averaging of Momentums.* ICML, 2022.
- **[SOTA]** E. Gorbunov, S. Horváth, P. Richtárik, G. Gidel. *Variance Reduction Is an Antidote to Byzantines: Better Rates, Weaker Assumptions and Communication Compression as a Cherry on the Top.* ICLR, 2023.
- **[Systems]** L. Chen, H. Wang, Z. Charles, D. Papailiopoulos. *DRACO: Byzantine-resilient Distributed Training via Redundant Gradients.* ICML, 2018.
- **[Systems]** S. Rajput, H. Wang, Z. Charles, D. Papailiopoulos. *DETOX: A Redundancy-based Framework for Faster and More Robust Gradient Aggregation.* NeurIPS, 2019.
- **[Systems]** K. Bonawitz et al. *Practical Secure Aggregation for Privacy-Preserving Machine Learning.* ACM CCS, 2017.
- **[Survey]** P. Kairouz, H. B. McMahan et al. *Advances and Open Problems in Federated Learning.* Foundations and Trends in Machine Learning, 2021. — arXiv:1912.04977

## 10. Worked Example

Take a 7B-parameter model, $d = 7\times10^9$, bf16 gradients, $n = 128$ data-parallel workers, $\delta = 0.1$ (12 Byzantine), one A100-80GB per worker.

**Baseline.** Ring all-reduce moves $2d(n-1)/n \approx 2d$ elements per worker $= 28$ GB, at 200 GB/s effective interconnect $\approx 140$ ms. Peak extra memory per node: two $O(d/n)$ chunks, about 220 MB.

**Krum.** Needs all pairwise distances: $n^2 d = 128^2 \times 7\times10^9 = 1.15\times10^{14}$ FLOPs. An A100 at 150 TFLOP/s bf16 sustained needs $\approx 0.8$ s — already $5.5\times$ the all-reduce. Worse, the aggregation point must hold $n$ full gradients: $128 \times 14\,\text{GB} = 1.79$ TB. That is $22\times$ a single A100's HBM. The rule cannot run at all without a distributed rewrite that does not exist.

**Coordinate-wise trimmed mean.** Shardable across $d$: each of 128 nodes owns $d/n = 5.5\times10^7$ coordinates, gathers 128 values per coordinate ($128 \times 5.5\times10^7 \times 2$ B $= 14$ GB in, fits), sorts, trims 10% each end. FLOPs $\approx O(nd\log n) \approx 6\times10^{12}$ — 40 ms, acceptable. Communication becomes all-to-all rather than ring: same $O(d)$ bytes per worker. So the *cheap* rule is feasible.

**Now the attack.** With $\sigma \approx 1$ (per-coordinate, normalized) and $n = 128$, ALIE sets each Byzantine gradient to $\bar g - z\sigma$ with $z$ chosen so the perturbation sits inside the honest range. With 12 attackers and trimming 12 per side, the attackers survive trimming by construction whenever their values fall between the honest quantiles; the resulting bias per coordinate is $O(\delta\sigma) \approx 0.1$, and over $d = 7\times10^9$ coordinates the update is displaced by $\|b\| \approx 0.1\sqrt{d} \approx 8.4\times10^3$ in $\ell_2$ — larger than a typical gradient norm.

**The obstruction, visible.** The only aggregator that fits the memory budget (coordinate-wise trimming) is the one whose bias grows as $\sqrt{d}$ under a small colluding perturbation; the only aggregators that resist that perturbation (Krum, Bulyan, geometric median variants) require $O(nd)$ materialized state and $O(n^2d)$ work. Momentum shrinks the effective $\sigma$ that ALIE can hide behind by $\sqrt{1-\beta}$ — a factor of $\approx 3.2$ at $\beta = 0.9$ — which reduces the bias but does not change the $\sqrt{d}$ scaling. Whether that constant-factor reduction is enough at $d = 7\times10^9$ is exactly the unrun experiment of Section 8.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*