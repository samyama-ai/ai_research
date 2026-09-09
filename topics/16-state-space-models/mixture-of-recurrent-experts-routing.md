---
id: 16-state-space-models/mixture-of-recurrent-experts-routing
title: "Sparse Mixture of Recurrent Experts Routing Stability"
topic: 16-state-space-models
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Sparse Mixture of Recurrent Experts Routing Stability

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/mixture-of-recurrent-experts-routing` · **Status:** empirically-open

## 1. Problem Statement

In a standard sparse MoE, experts are stateless MLPs: a token routed to expert $e$ at step $t$ and away at $t+1$ loses nothing, because the expert had no memory of $t$. In a **mixture of recurrent experts** (MoRE) — experts that are SSM/RNN layers, each with its own hidden state — routing decisions and state continuity are coupled. An expert only advances its state on the tokens it is given, so its recurrence runs over a *gappy subsequence* whose gap structure is chosen by a router that is itself being trained.

The problem: **does a sparse router over recurrent experts preserve usable long-range state, and under what conditions?**

- **Measurement variant.** Define a quantity that says whether expert state carries information *across* routing gaps, separable from the expert simply acting as a per-token nonlinearity. No standard metric exists.
- **Method variant.** Find a router/parameterization (gap-aware discretization, routing at segment rather than token granularity, frozen routers, shared state) that keeps validation loss at or below a dense-recurrent control matched on active parameters, at equal wall-clock.
- **Theory variant.** Bound the loss in effective memory horizon induced by sparsity $k/E$, as a function of the expert's state decay rate.

Solved means: a router for which the state-reset ablation of §8 costs measurable perplexity, load balance is stable, and the model beats the active-parameter-matched dense control.

## 2. Formal Setting

Input $x_{1:T}$, hidden $u_t \in \mathbb{R}^d$ at a MoRE layer. $E$ experts, top-$k$ routing. Router logits $r_t = W_r u_t$, $W_r \in \mathbb{R}^{E \times d}$; $p_t = \mathrm{softmax}(r_t)$; $\mathcal{S}_t = \mathrm{top}\text{-}k(p_t)$; $m_{e,t} = \mathbb{1}[e \in \mathcal{S}_t]$.

Each expert is a selective SSM with per-expert $(A_e, B_e, C_e)$ and input-dependent step $\Delta_e(u_t)$, discretized $\bar{A}_{e,t} = \exp(\Delta_e(u_t) A_e)$ (Gu & Dao, 2024). Two non-equivalent recurrences, routinely conflated:

$$\text{(a) input-masked:}\quad h^{(e)}_t = \bar{A}_{e,t}\,h^{(e)}_{t-1} + m_{e,t}\,\bar{B}_{e,t} u_t$$
$$\text{(b) time-skipped:}\quad h^{(e)}_t = m_{e,t}\bigl(\bar{A}_{e,t} h^{(e)}_{t-1} + \bar{B}_{e,t} u_t\bigr) + (1-m_{e,t})\,h^{(e)}_{t-1}$$

(a) costs $O(ET)$ state updates — **no FLOP saving**, so it is not a sparse model. (b) is genuinely sparse but the state decays over the expert's *own* visit index, not over token position: after a gap of $g$ tokens the state is undecayed, whereas the dense counterfactual would have applied $\exp(\sum_{j=1}^{g}\Delta A)$. Papers reporting MoE+SSM speedups almost always place the MoE on a separate MLP block and leave the SSM dense, sidestepping the choice entirely.

Measured quantities:

- **Within-sequence flip rate** $\Phi = \frac{1}{T-1}\sum_{t\ge2} \mathbb{1}[\mathcal{S}_t \neq \mathcal{S}_{t-1}]$, on a held-out probe batch.
- **Checkpoint churn** $\Phi_{\delta}$: fraction of probe tokens whose top-1 expert changes between step $s$ and $s+\delta$, probe batch frozen.
- **Gap statistics** $g_{e}$: token distance between consecutive assignments to $e$; report $\bar g_e$ and $P_{90}$.
- **Memory horizon** $\tau_e = -1/\max_i \mathrm{Re}\,\lambda_i(\bar\Delta_e A_e)$ in tokens, $\bar\Delta_e$ the mean step size over assigned tokens.
- **Continuity ratio** $\rho_e = \tau_e / \bar g_e$. $\rho_e \ll 1$: state is dead between visits, the expert is effectively stateless.
- **Balance** $\mathcal{L}_{\text{bal}} = E\sum_e f_e \bar p_e$ (Fedus et al., 2022), $f_e$ = token fraction to $e$.

Assumptions and their status: (i) router decisions are approximately i.i.d. across $t$ — **violated**, language tokens are bursty and $\Phi$ is far below chance in practice; (ii) $\Delta_e$ is well estimated by its mean — **violated**, selective SSMs make $\Delta$ vary orders of magnitude within a sequence; (iii) expert states are independent — **violated** whenever a shared block or residual stream couples them (Zamba, Jamba).

## 3. State of the Art

**Established.** MoE on the *MLP* branch of an SSM backbone works and is reproduced. MoE-Mamba (Pióro et al., 2024) interleaves Mamba blocks with switch-MoE MLPs and reaches Mamba's loss in ~2.35× fewer training steps at sub-1B scale. BlackMamba (Anthony et al., 2024) trains 1.5B and 2.8B Mamba+MoE models on ~300B tokens with open weights. Jamba (Lieber et al., 2024) ships 52B total / 12B active, 16 experts top-2, MoE every other layer, 256K context. Zamba (Glorioso et al., 2024) uses a shared attention block with a Mamba backbone at 7B. All of these keep the recurrent operator **dense**; none routes the state.

**Claimed but unablated.** That these hybrids "combine MoE with SSMs" — true only of the MLP branch. No published ablation isolates a sparsely routed *recurrent* operator against a dense-recurrent control at matched active parameters. The routing-stability machinery imported from Transformer MoE (router z-loss, ST-MoE, Zoph et al. 2022; auxiliary-loss-free bias balancing, Wang et al. 2024, used in DeepSeek-V3) has been validated only on stateless experts.

**Benchmark-number-only.** Downstream scores for Jamba/BlackMamba/Zamba are reported as aggregate benchmark tables. They do not separate the contribution of routing from the contribution of the hybrid attention/SSM mix, and no routing-stability diagnostics ($\Phi$, $\Phi_\delta$) are published for any of them.

**Theory SOTA.** Clark et al. (ICML 2022) give unified scaling laws for routed LMs — routing gain shrinks with dense-parameter count, measured to 1.3B and 64 experts, all stateless. Chi et al. (NeurIPS 2022) prove/measure representation collapse of the router's dot-product similarity. Neither covers recurrent experts.

## 4. What Is Known

- Router churn is large and does not vanish. StableMoE (Dai et al., ACL 2022) reports that routing "fluctuates" throughout training and fixes it by freezing a distilled router after stage 1, improving both perplexity and downstream scores at ~700M scale on multilingual MT and LM.
- Sparse experts are training-unstable at scale; ST-MoE's router z-loss was introduced to stabilize a 269B-parameter sparse model and is the standard fix (Zoph et al., 2022).
- Load balancing is not free: DeepSeek-V3 (671B total, 37B active, 2024) reports that replacing the auxiliary balance loss with a per-expert bias term improves loss, i.e. the standard balance loss actively costs quality.
- Selective SSM $\Delta$ spans a wide dynamic range by design; Mamba's $\Delta$ is initialized so $\tau$ spans roughly $10^0$–$10^2$ tokens (Gu & Dao, 2024). With $E=8, k=2$, uniform balance gives $\bar g \approx E/k = 4$ tokens — so at the short end $\rho_e \sim 0.25$ and at the long end $\rho_e \sim 25$. Both regimes exist in the same layer.
- Sparse upcycling from a dense checkpoint (Komatsuzaki et al., ICLR 2023) recovers most of the gap at a fraction of from-scratch cost — for stateless experts.

## 5. What Is Not Known

- **Empirically open (the main gap).** Whether a sparsely routed recurrent operator beats a dense-recurrent, active-parameter-matched control. The experiment is a single 1–3B pretraining run pair; nobody has published it. Also open: whether frozen-router (StableMoE-style) training helps *more* for recurrent experts than stateless ones, as the state-continuity argument predicts.
- **Methodologically blocked.** "Routing stability" has no agreed measurement for stateful experts. $\Phi$ conflates useful input-dependent specialization with harmful churn; $\Phi_\delta$ conflates learning with drift. Nothing published reports $\rho_e$ or any state-continuity quantity.
- **Theoretically open.** No bound relating $k/E$ and the expert decay spectrum to effective memory horizon. No proof either way that gap-aware discretization ($\bar A$ raised to the gap length) is necessary for the sparse recurrence to approximate the dense one. No identifiability result: given only loss curves, expert states and routing patterns are jointly unidentified.

## 6. Why It Is Hard

**Confounded measurement, compounded by non-identifiability.** A MoRE that trains well admits two incompatible explanations: (i) the experts carry cross-gap state and routing found a good partition, or (ii) $\rho_e \ll 1$, the state is dead between visits, and the layer degenerated to a sparse token-wise nonlinearity — a Switch MLP with extra kernels. These fit the same loss curve, the same benchmark table, and the same balance statistics. Distinguishing them requires an intervention on the state, not an observation of the loss.

Second obstruction: **cost asymmetry**. The variant that is cheap to measure (a, input-masked) is not sparse; the variant that is sparse (b, time-skipped) needs a gather/scatter chunked scan and cannot reuse the fused Mamba-2 SSD kernel (Dao & Gu, ICML 2024) unmodified. So the honest control arm costs a custom kernel before the first number exists — which is exactly why every shipped hybrid routes the MLP instead.

## 7. Current Research (as of 2026)

- Hybrid SSM+MoE production models (AI21/Jamba line, Zyphra/Zamba and BlackMamba) continue to scale MLP-branch routing; Zyphra has been the most open about architecture ablations. *(frontier — verify current model versions.)*
- Auxiliary-loss-free balancing (DeepSeek line) is displacing the aux-loss, and its interaction with stateful experts is untested. *(frontier — verify.)*
- Very-high-$E$ routing (PEER, "Mixture of a Million Experts", He 2024) makes $\bar g_e$ enormous, pushing $\rho_e \to 0$; whether recurrence survives there is unasked.
- Segment-level / chunk-level routing, where a whole chunk of $C$ tokens goes to one expert (analogous to Mixture-of-Depths block routing, Raposo et al. 2024), is the obvious fix and is not systematically evaluated for SSMs. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Scale.** Two 1.3B-parameter Mamba-2 models, 100B tokens of FineWeb-Edu, 4096-token sequences, identical data order and optimizer. Roughly 2–3k A100-hours per arm.

- **Arm A (treatment):** every other block replaced by a MoRE layer — $E=8$ recurrent experts, top-$k=2$, time-skipped recurrence (variant b), router z-loss + aux balance loss.
- **Control arm:** dense Mamba-2 with the *same active parameter count* and same wall-clock budget — this is the arm the literature omits.
- **Secondary arm (cheap):** Arm A with gap-aware discretization, $\bar A_{e,t} \leftarrow \exp\!\big(g_{e,t}\,\bar\Delta_e A_e\big)$ where $g_{e,t}$ is the gap since the last visit.

**The deciding number.** At evaluation, run the **state-reset ablation**: zero $h^{(e)}$ whenever the gap since the last visit exceeds $\bar g_e$. Report $\delta_{\text{reset}} = \mathcal{L}_{\text{reset}} - \mathcal{L}$ in nats/token on held-out data.

- $\delta_{\text{reset}} < 0.01$ nats → expert state carries no cross-gap information; the MoRE is a sparse pointwise layer and the recurrent-expert framing is empty.
- $\delta_{\text{reset}} > 0.05$ nats **and** Arm A beats the control on validation loss → sparse recurrent routing is real, and $\rho_e$ becomes the design knob.

Report alongside: $\Phi$, $\Phi_{\delta}$ at $\delta = 1000$ steps, and the $\rho_e$ histogram over all experts and layers.

## 9. Key References

- **[Foundational]** Gu, A., Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[Foundational]** Dao, T., Gu, A. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML, 2024. — arXiv:2405.21060
- **[Foundational]** Fedus, W., Zoph, B., Shazeer, N. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[Foundational]** Lepikhin, D., et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[SOTA]** Zoph, B., et al. *ST-MoE: Designing Stable and Transferable Sparse Expert Models.* 2022. — arXiv:2202.08906
- **[SOTA]** Dai, D., et al. *StableMoE: Stable Routing Strategy for Mixture of Experts.* ACL, 2022. — arXiv:2204.08396
- **[SOTA]** Lieber, O., et al. *Jamba: A Hybrid Transformer-Mamba Language Model.* 2024. — arXiv:2403.19887
- **[SOTA]** Anthony, Q., et al. *BlackMamba: Mixture of Experts for State-Space Models.* 2024. — arXiv:2402.01771
- **[SOTA]** Pióro, M., et al. *MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts.* 2024. — arXiv:2401.04081
- **[SOTA]** Glorioso, P., et al. *Zamba: A Compact 7B SSM Hybrid Model.* 2024. — arXiv:2405.16712
- **[SOTA]** Wang, L., et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.* 2024. — arXiv:2408.15664
- **[Theory]** Clark, A., et al. *Unified Scaling Laws for Routed Language Models.* ICML, 2022. — arXiv:2202.01169
- **[Theory]** Chi, Z., et al. *On the Representation Collapse of Sparse Mixture of Experts.* NeurIPS, 2022. — arXiv:2204.09179
- **[Related]** Zhou, Y., et al. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS, 2022. — arXiv:2202.09368
- **[Related]** Raposo, D., et al. *Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models.* 2024. — arXiv:2404.02258
- **[Survey]** Cai, W., et al. *A Survey on Mixture of Experts.* 2024. — arXiv:2407.06204

## 10. Worked Example

One MoRE layer, $E=8$, $k=2$, $d_{\text{state}}=64$, sequence $T=4096$. Balanced routing gives each expert $kT/E = 1024$ tokens, mean gap $\bar g_e = E/k = 4$.

Take a real Mamba-style $\Delta$ range. Suppose expert 3 has mean $\bar\Delta_3 = 0.02$ and slowest mode $\lambda = -1$, so $\tau_3 = 1/(0.02 \cdot 1) = 50$ tokens. Then

$$\rho_3 = \tau_3/\bar g_3 = 50/4 = 12.5$$

State survives the typical gap: after 4 skipped tokens the naive time-skipped state is *undecayed*, while the dense counterfactual would have retained $\exp(-4 \cdot 0.02) = 0.923$. Per gap the error is 7.7%; over 1024 visits the sparse expert's effective horizon is $\bar g_3 \cdot \tau_3 = 200$ tokens of *wall* position — a 4× inflation of its intended timescale. Gap-aware discretization removes exactly this.

Now expert 6, a fast expert: $\bar\Delta_6 = 0.5$, $\tau_6 = 2$ tokens, $\rho_6 = 0.5$. Between visits the dense factor would be $\exp(-4 \cdot 0.5) = 0.135$; even undecayed, its own recurrence multiplies by $\exp(-0.5)=0.61$ per visit, so information from 4 visits back is attenuated to $0.61^4 = 0.14$. Expert 6 is, functionally, a pointwise nonlinearity.

**Where the obstruction becomes visible.** Both experts sit in the same layer and contribute to the same loss curve. The layer's average $\rho$ over the eight experts might be a healthy 5, while half the experts carry no state at all. A perplexity number cannot tell these apart, and neither can $\Phi$, $\mathcal{L}_{\text{bal}}$, or any benchmark table published for Jamba, BlackMamba or Zamba. Only the §8 state-reset intervention separates them: reset expert 3 and the loss should move; reset expert 6 and $\delta_{\text{reset}} \approx 0.14^{\,} \cdot$ (its share) $\to$ noise. If the whole-layer $\delta_{\text{reset}}$ comes back under 0.01 nats, the sparse recurrent expert was never recurrent.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*