---
id: 15-mixture-of-experts/auxiliary-loss-free-balancing-guarantees
title: "Auxiliary-Loss-Free Balancing Guarantees"
topic: 15-mixture-of-experts
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Auxiliary-Loss-Free Balancing Guarantees

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/auxiliary-loss-free-balancing-guarantees` · **Status:** open

## 1. Problem Statement

Sparse MoE layers route each token to $k$ of $N$ experts. Left alone, routing collapses onto a few experts, wasting capacity and stalling distributed training. The standard fix adds an auxiliary balance loss to the training objective, which injects gradients that are not aligned with the language-modelling objective. **Auxiliary-loss-free (ALF) balancing** removes that loss and instead adds a per-expert bias $b_i$ to the routing scores, updating $b_i$ by a control rule from observed load — no gradient, no interference term.

The open problem: **ALF balancing has no guarantee.** Three variants, different difficulty:

- **Theory.** Given a bias-update rule with step size $u$, prove a bound on the steady-state load imbalance as a function of $u$, batch size $T$, expert count $N$, and the drift rate of the router's score distribution. No such bound exists. Prove or refute that the bias, at convergence, is a dual variable for the capacity-constrained assignment problem.
- **Method.** Design a rule with a *certificate*: a computable quantity, available during training, that upper-bounds imbalance for the next $m$ steps, or triggers when the guarantee is void.
- **Measurement.** Decide what balancing should be *worth*. Imbalance metrics (MaxVio) and quality metrics (validation loss) are both reported; the causal link between them, at fixed hardware, is not isolated.

Solving it means: a bound that holds under the non-stationarity of real training, not just for a frozen router.

## 2. Formal Setting

One MoE layer, $N$ experts, top-$k$ routing, batch of $T$ tokens with hidden states $h_t \in \mathbb{R}^d$.

**Affinity score** (measured as the router head's output after its nonlinearity — sigmoid in DeepSeek-V3, softmax in Switch):
$$s_{i,t} = \sigma\!\left(w_i^\top h_t\right), \quad i \in [N].$$

**Biased selection.** ALF selects the top-$k$ of $s_{i,t} + b_i$ but *gates* with the unbiased score:
$$\mathcal{T}_t = \operatorname*{arg\,top-}k_i \, (s_{i,t} + b_i), \qquad y_t = \sum_{i \in \mathcal{T}_t} g_{i,t}\, E_i(h_t), \quad g_{i,t} \propto s_{i,t}.$$
The bias moves *who is chosen*, not *how much they count*. This split is the crux of the method and the source of its unquantified distortion.

**Load** (measured by counting, per micro-batch, before dispatch):
$$c_i = \sum_{t=1}^{T} \mathbf{1}[i \in \mathcal{T}_t], \qquad \bar{c} = \frac{kT}{N}.$$

**Maximal violation**, the standard imbalance metric:
$$\mathrm{MaxVio} = \frac{\max_i c_i - \bar{c}}{\bar{c}} \in [0, N/k - 1].$$
Two variants are reported: $\mathrm{MaxVio}_{\text{global}}$ (counts over the whole validation set — measures expert specialization) and $\mathrm{MaxVio}_{\text{batch}}$ (per micro-batch — measures the actual hardware bottleneck). They differ by an order of magnitude and are not interchangeable.

**Update rule** (Loss-Free Balancing; DeepSeek-V3):
$$b_i \leftarrow b_i + u \cdot \operatorname{sign}(\bar{c} - c_i), \quad u > 0 .$$

**Sensitivity**, the quantity that controls everything and is essentially never logged:
$$\rho_i = \frac{\partial \mathbb{E}[c_i]}{\partial b_i} = T \cdot p_i(0),$$
where $p_i$ is the density of the margin $m_{i,t} = (s_{i,t}+b_i) - (k\text{-th largest competitor score})$ at zero — the number of tokens sitting on expert $i$'s decision boundary.

**Assumptions, and which are violated:**

| Assumption | Status in practice |
|---|---|
| Score distribution stationary across steps | **Violated.** Router weights and $h_t$ co-evolve; specialization shifts fastest early in training. |
| Tokens i.i.d. within a batch | **Violated.** Sequence packing correlates tokens; domain-homogeneous batches skew loads. |
| $\rho_i > 0$ and bounded | **Violated at both ends.** Near-tied scores give huge $\rho$ (overshoot); confidently specialized experts give $\rho \approx 0$ (bias must grow without bound). |
| Balancing $\Rightarrow$ better loss | Unestablished; the reported correlation is confounded with dropped-token rate. |

## 3. State of the Art

**Empirical SOTA.** *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts* (Wang, Chen, Xu, Zhu, Dai, Guo, Luo et al., 2024, arXiv:2408.15664) introduced the sign-update bias. Deployed at scale in **DeepSeek-V3** (DeepSeek-AI, 2024, arXiv:2412.19437), 671B total / 37B active, 256 routed experts per layer, with $u$ annealed to $0$ late in training and a small sequence-wise auxiliary loss retained as a backstop. Subsequent large open MoEs adopted the same recipe.

**Established:** on the paper's own 1B and 3B ablations, ALF matches or beats the auxiliary-loss arm on validation perplexity *and* reduces MaxVio by an order of magnitude. The comparison is like-for-like (same data, same tokens).

**Claimed but unablated:** that ALF is why DeepSeek-V3 trains stably. V3 changed router nonlinearity (sigmoid), added node-limited routing, shared experts, and a residual auxiliary loss in the same release. No single-variable ablation at that scale is public. The V3 result is a benchmark number, not an isolated cause.

**Theory SOTA — for a different problem.** Balancing-by-assignment has guarantees: **BASE Layers** (Lewis et al., ICML 2021) solves a linear assignment per batch, so balance is exact by construction; **Sinkhorn/entropic OT** routing converges linearly for a fixed cost matrix; **Expert Choice** (Zhou et al., NeurIPS 2022) makes loads exactly equal by inverting the selection direction. None of these bound the *online, single-scalar-per-expert, sign-update* regime that ALF actually uses. The nearest analytic frame is Bertsekas' **auction algorithm** (1988), where per-item prices under $\epsilon$-complementary slackness yield an assignment within $N\epsilon$ of optimal — ALF's $b_i$ is structurally a price, but with one price per *expert* rather than per slot, and no termination test.

## 4. What Is Known

- **1B params, 100B tokens** (Loss-Free Balancing paper): validation perplexity 9.56 (aux-loss control) → 9.50 (ALF); $\mathrm{MaxVio}_{\text{global}}$ 0.72 → 0.04.
- **3B params, 200B tokens**, same paper: perplexity 7.97 → 7.92; $\mathrm{MaxVio}_{\text{global}}$ 0.52 → 0.04. Effect size on perplexity is ~0.05, roughly one seed's worth of noise in many setups; the imbalance effect is ~10×, far outside noise.
- **Step size trades off.** Larger $u$ balances faster and oscillates more; the paper's sweep found a broad low-sensitivity basin around $u \approx 10^{-3}$ at these scales. No scaling law for $u$ as a function of $N$, $T$, or model size.
- **Update-before-vs-after matters little; expected-load vs counted-load matters.** Using a per-batch count rather than a running average was reported as the better estimator.
- **Auxiliary losses cost quality.** Switch Transformer (Fedus, Zoph, Shazeer, JMLR 2022) and ST-MoE (Zoph et al., 2022) both document a balance-coefficient sweep in which too much balancing degrades loss — this is the motivating regularity ALF exploits, and it reproduces.
- **Dropless training is possible independently** (MegaBlocks, Gale et al., MLSys 2023), which decouples "imbalance" from "token loss" and therefore changes what balancing buys.

## 5. What Is Not Known

- **Theoretically open.** No bound of the form $\mathrm{MaxVio} \le f(u, T, N, \delta)$ under any non-trivial drift model. No proof that the sign update converges, even for a *frozen* router — the discrete top-$k$ makes $c_i(b)$ piecewise constant, so standard subgradient arguments do not apply off the shelf. No proof or refutation that $b^\star$ is a dual optimum of the capacity-constrained assignment.
- **Empirically open.** Whether ALF's advantage survives at 100B+ active parameters *without* the residual auxiliary loss. Runnable — it is a single-variable ablation on a frontier pretraining run — and unrun in public.
- **Empirically open.** Whether the bias term degrades routing quality: how often is the selected expert not the argmax of the unbiased affinity, and does per-token loss rise for those tokens? Cheap to measure, not reported.
- **Methodologically blocked.** "Balanced" has no agreed operational definition. $\mathrm{MaxVio}_{\text{global}}$, $\mathrm{MaxVio}_{\text{batch}}$, entropy of the load distribution, and realized step time rank methods differently, and no paper reports all four.

## 6. Why It Is Hard

**Non-identifiability of the sensitivity $\rho_i$, compounded by confounded measurement.** The bias update is a feedback controller whose plant gain is $\rho_i = T \cdot p_i(0)$ — the boundary-token density. That gain varies by orders of magnitude across experts and across training, is never logged, and cannot be inferred from load alone: an expert at $c_i = \bar{c}$ may be there because $\rho_i$ is large and the controller is holding it, or because $\rho_i \approx 0$ and nothing is happening. A guarantee needs a bound on $\rho_i$; getting one requires assumptions about the score distribution that the training dynamics actively falsify.

Second obstruction: **the evaluation does not measure what it names.** MaxVio is reported as the balancing outcome, but the thing that matters is wall-clock per token at fixed hardware, which depends on the all-to-all critical path and the parallelism layout, not on $\max_i c_i$ alone. So the quantity the theory would bound is not the quantity practitioners need bounded.

Third: the decisive ablation is at frontier scale. A single-variable ALF-vs-aux-loss arm at 100B active parameters costs a full pretraining run, so it is run once inside labs and never as a controlled pair.

## 7. Current Research (as of 2026)

- **DeepSeek** continues the bias-control line; V3's annealed $u$ plus residual sequence-wise loss is the current production compromise. Whether the backstop is load-bearing is untested publicly.
- **Differentiable-routing alternatives** that sidestep the discrete top-$k$: ReMoE (ReLU routing, Wang et al., ICLR 2025) makes the routing count a continuous function of parameters, which restores the possibility of a gradient-based balance argument.
- **Open-model reporting.** OLMoE (Muennighoff et al., 2024) publishes router-saturation and expert-usage diagnostics, making cross-lab imbalance comparisons possible for the first time.
- *(frontier — verify)* Control-theoretic framings of the bias update — treating it as integral control with dead-zone and deriving stability from a Lipschitz bound on the score drift — are circulating in workshop form; no peer-reviewed bound as of this writing.
- *(frontier — verify)* Post-training and inference-time expert balancing (expert replication / placement rebalancing under serving load) is where the practical pressure has moved; it is a different, easier problem because the router is frozen.

## 8. Concrete Next Experiment

**Question:** is ALF's benefit *balance*, or is it *the absence of the auxiliary gradient*?

- **Scale.** 3B total / ~0.5B active, 64 experts, top-8, 200B tokens. Three seeds. Roughly the smallest scale at which the published effect is visible.
- **Arms.**
  1. *Control A* — auxiliary loss, coefficient swept over $\{10^{-3}, 10^{-2}, 10^{-1}\}$.
  2. *Control B (the decisive one)* — **frozen-bias replay**: take the $b_i$ trajectory from the ALF run and replay it as a fixed schedule, with no feedback. Same balance profile, no controller.
  3. *Treatment* — ALF, $u = 10^{-3}$.
  4. *Null arm* — no balancing at all, dropless (MegaBlocks-style) so imbalance costs time but not tokens.
- **Instrumentation.** Log $\rho_i$ directly: for each expert, the count of tokens within $\pm 0.01$ of its decision boundary. Log the **override rate**: fraction of tokens whose selected set differs from the unbiased top-$k$.
- **The deciding number.** Validation perplexity gap between *Treatment* and *Control B*, in nats, against the seed standard deviation. If $|\Delta| < 1\sigma$, the benefit is the bias schedule (balance) and any open-loop schedule suffices — the guarantee problem reduces to schedule design. If Treatment beats Control B by $> 2\sigma$, the feedback loop itself is doing the work and a genuine online guarantee is required. Secondary: does the null arm's perplexity match Treatment's? If yes at $N{=}64$, balancing is a systems constraint only, and the whole quality argument collapses.

## 9. Key References

- **[SOTA]** Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, Damai Dai. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.* 2024. — arXiv:2408.15664
- **[SOTA]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Foundational]** William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[Foundational]** Dmitry Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[Foundational]** Noam Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Balance-by-assignment]** Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, Luke Zettlemoyer. *BASE Layers: Simplifying Training of Large, Sparse Models.* ICML, 2021. — arXiv:2103.16716
- **[Balance-by-construction]** Yanqi Zhou et al. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS, 2022. — arXiv:2202.09368
- **[Stability]** Barret Zoph et al. *ST-MoE: Designing Stable and Transferable Sparse Expert Models.* 2022. — arXiv:2202.08906
- **[Systems]** Trevor Gale, Deepak Narayanan, Cliff Young, Matei Zaharia. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys, 2023. — arXiv:2211.15841
- **[Theory, adjacent]** Dimitri P. Bertsekas. *The Auction Algorithm: A Distributed Relaxation Method for the Assignment Problem.* Annals of Operations Research, 1988.
- **[Open-model diagnostics]** Niklas Muennighoff et al. *OLMoE: Open Mixture-of-Experts Language Models.* 2024. — arXiv:2409.02060

## 10. Worked Example

$N = 8$ experts, top-1, $T = 4096$ tokens per micro-batch, $\bar{c} = 512$, $u = 10^{-3}$.

Expert 3 is overloaded at $c_3 = 700$, so
$$\mathrm{MaxVio} = \frac{700 - 512}{512} = 0.367 .$$

Suppose 3000 of the 4096 tokens have a margin to expert 3 spread roughly uniformly over $[-0.7, 0.7]$, giving sensitivity $\rho_3 \approx 3000 / 1.4 \approx 2100$ tokens per unit bias. Shedding 188 tokens needs
$$\Delta b_3 \approx -188 / 2100 \approx -0.090 .$$
At $u = 10^{-3}$ per step, that is **90 steps** — about 370k tokens at this batch size. Steady-state ripple is then $\pm u \rho_3 \approx \pm 2.1$ tokens, i.e. $\mathrm{MaxVio} \approx 0.004$. This is the regime the published results live in, and it looks fine.

Now change one number. Expert 3 has specialized: only 40 tokens sit near its boundary, spread over $[-0.7, 0.7]$, so $\rho_3 \approx 28$. The same 188-token correction now needs $\Delta b_3 \approx -6.7$ — **6700 steps**, and a bias six times larger than the entire range of the sigmoid scores $s_{i,t} \in (0,1)$.

That is the obstruction, made visible. Once $|b_i| \gg 1$, the selection $\arg\max_i (s_{i,t} + b_i)$ is dominated by the bias, not the affinity: routing is decided by the load controller and the router's learned preference is a tie-breaker. The gate weight $g_{i,t} \propto s_{i,t}$ can then be near zero for the expert actually doing the computation — the layer is spending FLOPs on an expert it has down-weighted to nothing. Meanwhile 6700 steps of drift is long enough for the score landscape to move, so the controller is chasing a target that has already left.

Both regimes report the same load $c_3$. Nothing in the standard logs distinguishes $\rho_3 = 2100$ from $\rho_3 = 28$. A guarantee needs the second number, and no published run records it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*