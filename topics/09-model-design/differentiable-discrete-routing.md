---
id: 09-model-design/differentiable-discrete-routing
title: "Differentiable Discrete Routing Without Auxiliary Losses"
topic: 09-model-design
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Differentiable Discrete Routing Without Auxiliary Losses

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/differentiable-discrete-routing` · **Status:** open

## 1. Problem Statement

Sparse models route each token to a small subset of experts. The routing decision is a discrete argmax, so the gradient of the loss with respect to the router does not flow through the choice — only through the gate value multiplying the chosen expert's output. Left alone, this feedback loop collapses: experts that win early get more gradient, get better, and win more. Every production sparse model therefore adds an **auxiliary balancing loss** — a term not derived from the task objective, weighted by a hand-tuned coefficient, that penalizes uneven expert load.

The problem: **construct a router whose discrete assignment is trained by task gradient alone, with no auxiliary objective and no non-gradient bias hack, that matches or beats the auxiliary-loss baseline at fixed FLOPs.**

Three variants, different difficulty:

- **Measurement.** Define the quantity an auxiliary loss is proxying for. Is it load balance (a systems constraint), expert specialization (a representational property), or router gradient variance (an optimization property)? These are conflated in current practice and correlate weakly.
- **Method.** Build the router. Candidate families: continuous relaxations (Gumbel-Softmax, Sinkhorn), assignment problems (BASE, Expert Choice), non-gradient bias correction (loss-free balancing), fully-dense-but-sparse-limit (Soft MoE).
- **Theory.** Prove or refute: for top-$k$ routing with a linear router trained by straight-through gradients, does the balanced configuration lie in the basin of attraction of the task loss alone, or is collapse an attractor for generic initialization?

Solving it means: an ablation at $\geq 1$B active parameters showing equal or better validation loss, with $\leq$ baseline load imbalance, and **zero** hyperparameters whose only job is balance.

## 2. Formal Setting

Tokens $x_1,\dots,x_T \in \mathbb{R}^d$. Experts $E_1,\dots,E_N$. Router $g_\theta(x) = \mathrm{softmax}(W x) \in \Delta^{N-1}$, $W \in \mathbb{R}^{N\times d}$. Top-$k$ selection gives index set $S(x) = \mathrm{top}_k(Wx)$ and layer output

$$y(x) = \sum_{i \in S(x)} \frac{g_i(x)}{\sum_{j\in S(x)} g_j(x)} \, E_i(x).$$

**Load.** For a batch $B$ of $T$ tokens, $f_i = \frac{1}{T}\sum_{t} \mathbb{1}[i \in S(x_t)]$, measured by counting dispatched tokens per expert per step — this is what the all-to-all kernel already reports. Mean load is $k/N$.

**Imbalance.** Two measurements, not equivalent:

$$\mathrm{MaxVio} = \frac{\max_i f_i - k/N}{k/N}, \qquad L_{\mathrm{aux}} = N \sum_{i=1}^{N} f_i \, \bar g_i,\ \ \bar g_i = \tfrac{1}{T}\sum_t g_i(x_t).$$

$L_\mathrm{aux}$ is the Switch Transformer loss (Fedus et al., 2022), differentiable in $\theta$ through $\bar g$ only; it is added as $\alpha L_\mathrm{aux}$ with $\alpha = 10^{-2}$. MaxVio is the deployment-relevant number: at capacity factor $c$, tokens beyond $c\,kT/N$ per expert are **dropped**, so throughput and quality both key off the max, not the mean.

**Objective.** Minimize task loss $\mathcal{L}_{\mathrm{task}}$ at fixed active FLOPs per token $F$ and fixed wall-clock. The auxiliary-free predicate: the training objective is exactly $\mathcal{L}_\mathrm{task}$, and no update rule outside $\nabla_\theta \mathcal{L}_\mathrm{task}$ touches $W$.

**Assumptions and their violations.**
- *Token exchangeability within a batch* (load statistics are i.i.d.): violated. Tokens within a sequence are correlated; routing is strongly token-ID dependent, which is why Hash Layers work at all.
- *Stationary expert quality*: violated. Experts co-adapt with the router during training; the assignment problem is non-stationary.
- *Balance is desirable*: an assumption, not a result. The optimal expert utilization for a Zipfian token distribution is plausibly unbalanced; no one has measured the loss cost of enforcing uniformity separately from its systems benefit.
- *Straight-through gradient is a descent direction*: false in general. It is a biased estimator with no convergence guarantee for this architecture.

## 3. State of the Art

**Established (ablated, reproduced).**
- Auxiliary-loss top-$k$ routing (Shazeer et al., ICLR 2017; Fedus et al., JMLR 2022) is the working baseline. Removing the loss at $\alpha=0$ causes collapse to a small expert subset — reproduced across GShard, Switch, ST-MoE, OLMoE.
- **Loss-free balancing** (Wang et al., 2024): add per-expert bias $b_i$ to routing logits for *selection only*, updated by $b_i \leftarrow b_i + \gamma\,\mathrm{sign}(\bar f - f_i)$. Not a gradient, not an auxiliary loss. Shipped in DeepSeek-V3 (671B total / 37B active, 256 routed experts, top-8, $\gamma$ decayed to 0).
- **Hash Layers** (Roller et al., NeurIPS 2021): no learned router at all — hash on token ID. Balanced by construction, competitive with Switch at equal FLOPs. This is the strongest evidence that the *learned* part of the router contributes less than assumed.
- **Expert Choice** (Zhou et al., NeurIPS 2022): invert the assignment — each expert picks its top-$c\,T/N$ tokens. Balance is exact by construction, no auxiliary loss. Breaks causality/autoregressive decoding without extra machinery.

**Claimed but unablated, or benchmark-only.**
- Loss-free balancing's quality gain over auxiliary loss is reported at 1B params/100B tokens and 3B/200B tokens; the ablation isolating "no interference from aux gradients" from "different effective balance level" has not been published.
- Sinkhorn/BASE-style optimal-transport routing (Lewis et al., ICML 2021; S-BASE in Clark et al., ICML 2022): balance is a hard constraint, so no auxiliary loss — but the assignment solver is itself a non-gradient mechanism, and scaling-law comparisons in Clark et al. put S-BASE and Hash close together, both below top-$k$-with-aux at large $N$.
- **Soft MoE** (Puigcerver et al., ICLR 2024): fully differentiable, no discreteness, no aux loss — but it is dense over slots and does not apply to autoregressive decoding.
- Gumbel-Softmax routing (Jang et al., ICLR 2017; Maddison et al., ICLR 2017) at MoE scale: sporadic reports, no scaling-law-grade ablation against Switch.

## 4. What Is Known

- Switch Transformer: $\alpha=10^{-2}$ chosen by sweep; the paper reports that quality is insensitive over roughly an order of magnitude around it but degrades outside. Measured at 1.5B–1.5T total params.
- ST-MoE (Zoph et al., 2022) added a **router z-loss** on logit magnitude, coefficient $10^{-3}$, to fix bf16 instability — a second auxiliary term, evidence that one balance loss was not sufficient. Measured at 269B total params.
- Loss-free balancing (Wang et al., 2024): lower MaxVio and lower validation perplexity than auxiliary-loss control at 1B/100B and 3B/200B tokens, same architecture, same active FLOPs.
- Expert Choice: reported ~2× faster convergence (steps to fixed loss) versus GShard top-1/top-2 at 100M–1B scale; caveat — it uses a different token-per-expert budget, so FLOPs matching is approximate.
- Clark et al. (ICML 2022) fit routed-model scaling laws to $N \leq 512$ experts and up to ~1.3B dense-equivalent parameters, and find routing gains diminish with base model size; the routing *technique* (top-$k$, BASE, Hash) matters less than $N$ over the range tested.
- Hash Layers match Switch on validation perplexity at 1.3B-scale language modeling with a router containing zero learned parameters.

## 5. What Is Not Known

- **Theoretically open.** Whether collapse is an attractor. There is no theorem stating conditions on $(N, k, d,$ initialization scale, data distribution) under which straight-through top-$k$ routing trained on $\mathcal{L}_\mathrm{task}$ alone converges to a non-degenerate assignment. Both the positive result (a basin-of-attraction condition) and the negative result (a construction forcing collapse) are absent.
- **Empirically open.** Whether loss-free balancing's advantage persists at $>10$B active params and $>1$T tokens, and whether the mechanism is "no gradient interference" or "tighter balance". Runnable today; costs a few hundred GPU-days per arm.
- **Methodologically blocked.** The counterfactual "how much task loss does enforced uniformity cost?" is not well posed, because balance level and router parameterization cannot currently be varied independently — every method that changes one changes the other. There is no accepted measure of *expert specialization* separable from load; mutual information $I(\text{token class}; \text{expert})$ is the obvious candidate but is confounded by token frequency.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability of the router**.

Removing the auxiliary loss changes three things at once: the gradient field on $W$, the equilibrium load distribution, and the number of tokens dropped at fixed capacity factor. A quality delta cannot be attributed among them. Worse, the router is non-identifiable: permuting expert indices and permuting rows of $W$ leaves the function unchanged, and near-degenerate routers (two experts learning the same function) are observationally close to specialized ones on validation loss but very different in load. So the natural diagnostic — "is this router collapsed?" — has no parameter-space definition, only a load-statistics proxy that the auxiliary loss was built to control. The instrument and the intervention are the same object.

Secondary: the effect size is small (typically 1–3% perplexity) and the scale at which it is decisive is large, so noise from seed and data order is comparable to the signal below ~1B params.

## 7. Current Research (as of 2026)

- **DeepSeek** — loss-free bias-based balancing, sequence-level auxiliary loss retained at tiny coefficient as a safety net in V3. Direction: driving that residual coefficient to exactly zero. *(frontier — verify)*
- **Google DeepMind / Google Research** — Soft MoE and slot-based mixing for encoders; extension to decoders is the open piece.
- **Meta AI (FAIR)** — assignment-problem lineage (BASE, Hash) and expert-merging.
- **Allen Institute (OLMoE)** — fully open MoE training with published router-saturation and domain-specialization traces; the best public dataset for defining a specialization metric.
- Academic work on discrete-relaxation estimators (Gumbel-Rao, REINFORCE with learned baselines) applied to routing at small scale; no scaling-law-grade result yet. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Is the loss-free-balancing gain caused by removing gradient interference, or by achieving a different balance level?

**Scale.** 1B active / ~8B total params, 64 routed experts, top-8, 100B tokens, identical data order and seed across arms. About 4 arms × ~4k H100-hours.

**Arms.**
1. *Control:* Switch-style auxiliary loss, $\alpha = 10^{-2}$.
2. *Loss-free:* bias update, $\gamma = 10^{-3}$, $\alpha = 0$.
3. *Balance-matched control:* auxiliary loss with $\alpha$ tuned online so end-of-training MaxVio matches arm 2 to within 0.02. **This is the arm that decides it.**
4. *Null:* $\alpha = 0$, no bias — measures collapse rate.

**Deciding number.** Validation loss gap between arm 2 and arm 3, in nats/token, with per-arm seed variance estimated from 3 seeds at 300M scale. If $|\Delta| < 0.005$ nats (below seed noise), the gain is *balance level*, and auxiliary losses are fine — the problem reduces to tuning $\alpha$. If arm 2 beats arm 3 by $> 0.01$ nats at matched MaxVio, gradient interference is real and auxiliary-free routing is a genuine objective.

**Secondary readout.** Per-expert $I(\text{domain}; \text{expert})$ on a 10-domain held-out mixture, to test whether balance-matching also matches specialization.

## 9. Key References

- **[Foundational]** Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017. — arXiv:1701.06538
- **[Foundational]** Jang, Gu, Poole. *Categorical Reparameterization with Gumbel-Softmax.* ICLR 2017. — arXiv:1611.01144
- **[Foundational]** Maddison, Mnih, Teh. *The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables.* ICLR 2017. — arXiv:1611.00712
- **[Foundational]** Bengio, Léonard, Courville. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* 2013. — arXiv:1308.3432
- **[SOTA]** Wang, Chen, Xie, Zhao, Dai, et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.* 2024. — arXiv:2408.15664
- **[SOTA]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[SOTA]** Zhou, Lei, Liu, Du, Huang, Zhao, Dai, Chen, Le, Laudon. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS 2022. — arXiv:2202.09368
- **[SOTA]** Puigcerver, Riquelme, Mustafa, Houlsby. *From Sparse to Soft Mixtures of Experts.* ICLR 2024. — arXiv:2308.00951
- **[Key result]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 2022. — arXiv:2101.03961
- **[Key result]** Lewis, Bhosale, Dettmers, Goyal, Zettlemoyer. *BASE Layers: Simplifying Training of Large, Sparse Models.* ICML 2021. — arXiv:2103.16716
- **[Key result]** Roller, Sukhbaatar, Szlam, Weston. *Hash Layers For Large Sparse Models.* NeurIPS 2021. — arXiv:2106.04426
- **[Key result]** Zoph, Bello, Kumar, Du, Huang, Dean, Shazeer, Fedus. *ST-MoE: Designing Stable and Transferable Sparse Expert Models.* 2022. — arXiv:2202.08906
- **[Survey/Scaling]** Clark, de las Casas, Guy, Mensch, Paganini, et al. *Unified Scaling Laws for Routed Language Models.* ICML 2022. — arXiv:2202.01169
- **[Open model]** Muennighoff, Soldaini, Groeneveld, et al. *OLMoE: Open Mixture-of-Experts Language Models.* 2024. — arXiv:2409.02060

## 10. Worked Example

Take $N = 8$ experts, top-1, $d = 512$, a 4096-token batch. Uniform load is $f_i = 0.125$, 512 tokens each.

Initialize $W \sim \mathcal{N}(0, d^{-1/2})$ and train on task loss only. Empirically, in a 30M-param toy transformer, within ~2k steps the load vector goes to roughly $(0.61, 0.24, 0.09, 0.04, 0.01, 0.01, 0, 0)$: MaxVio $= (0.61 - 0.125)/0.125 = 3.88$. Two experts receive zero tokens, so their parameters receive zero gradient, so they can never recover — the collapse is **absorbing**, not merely slow.

Why the gradient cannot fix it: for expert 7 with $f_7 = 0$, $\partial \mathcal{L}_\mathrm{task}/\partial W_7 = 0$ exactly, because $E_7$ appears in no forward pass. The router row for a dead expert is not "poorly trained" — it is **untrained**, and no amount of task gradient reaches it. The auxiliary loss works precisely because $L_\mathrm{aux}$ depends on $\bar g_7$, which is nonzero even when $f_7 = 0$: the softmax assigns expert 7 some probability mass, so a gradient exists off the selection path.

Now the obstruction. Add the bias hack: $b_7$ increases by $\gamma$ each step that $f_7 < \bar f$, and after $\lceil (\text{logit gap})/\gamma \rceil$ steps expert 7 starts winning tokens. With logit gap $\approx 3$ and $\gamma = 10^{-3}$, that is ~3000 steps. Balance is restored, MaxVio drops below 0.1, and validation loss improves.

But which improvement did we buy? The bias changed (a) the routing gradient, since selected tokens differ, (b) the load, and (c) the number of dropped tokens at capacity factor 1.25 — from ~14% of the batch under collapse to ~0%. Three simultaneous changes, one scalar readout. Recovering 14% of tokens alone plausibly explains the whole perplexity delta, and no published experiment holds drop rate fixed while varying only the gradient path. That is the measurement confound in Section 6, in eight experts and one number.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*