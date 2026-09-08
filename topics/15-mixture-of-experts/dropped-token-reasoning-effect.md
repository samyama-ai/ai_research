---
id: 15-mixture-of-experts/dropped-token-reasoning-effect
title: "Dropped-Token Effect on Downstream Reasoning"
topic: 15-mixture-of-experts
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Dropped-Token Effect on Downstream Reasoning

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/dropped-token-reasoning-effect` · **Status:** empirically-open

## 1. Problem Statement

In a capacity-constrained Mixture-of-Experts (MoE) layer, each expert accepts at most $C$ tokens per batch. Tokens routed to a full expert are **dropped**: their expert branch is skipped and only the residual stream passes through. Dropping is a function of the *whole batch*, not of the token, so the same prompt can produce different outputs depending on which other sequences share its microbatch.

The problem: **does token dropping degrade multi-step reasoning more than it degrades next-token perplexity, and if so, by how much and where?**

Three variants, of different difficulty:

- **Measurement.** Define a drop-attributable degradation $\Delta$ on a reasoning task that is not confounded by capacity's effect on training dynamics, and estimate it with controlled error bars. Currently the weakest link.
- **Method.** Given a fixed drop budget (a real systems constraint — capacity factor $c$ buys throughput), choose *which* tokens to drop so that $\Delta$ is minimised. Priority routing by router score is the incumbent; nothing token-role-aware is deployed.
- **Theory.** Bound the end-to-end output perturbation of dropping a set $S$ of token-layer pairs, as a function of $|S|$, depth, and the router-gate magnitudes on the dropped entries.

A solution to the measurement variant looks like: a reproducible protocol reporting $\Delta$ per drop rate on chain-of-thought (CoT) benchmarks, with a dense or dropless control that isolates dropping from every other capacity-linked change.

## 2. Formal Setting

Model with $L$ MoE layers, $E$ experts per layer, top-$k$ routing. Batch $B$ of $T$ tokens total (sequences flattened). Router at layer $\ell$ produces gates $g^{\ell}(x) \in \Delta^{E-1}$, typically softmax of a linear map. Top-$k$ index set $K^\ell(x)$.

**Capacity.** Per-expert capacity
$$C = \left\lceil c \cdot \frac{k\,T}{E} \right\rceil,$$
with capacity factor $c$ (measured: read from config; $c \in [1.0, 1.25]$ in GShard/Switch, $c \to \infty$ for dropless kernels). Tokens assigned to expert $e$ are ordered by a priority rule $\pi$ (position order in GShard; gate magnitude in Switch/ST-MoE variants); rank $> C$ are dropped.

**Drop indicator.** $d^\ell_i \in \{0,1\}$ for token $i$ at layer $\ell$, computed by instrumenting the dispatch mask — the count of `True` entries discarded by the top-$C$ truncation, logged per layer per step. Drop rate
$$\rho = \frac{1}{LT}\sum_{\ell,i} d^\ell_i .$$
Measured empirically, not derived from $c$: at fixed $c$, $\rho$ varies with sequence-length mix and domain.

**Dropped mass.** Dropping is not uniformly severe; weight it by the gate that was lost:
$$m = \frac{\sum_{\ell,i} d^\ell_i \, g^\ell_{e(i)}(x_i)}{\sum_{\ell,i} \sum_{e \in K^\ell(x_i)} g^\ell_e(x_i)} \in [0,1].$$

**Degradation.** For task $\mathcal{T}$ with metric $A$ (exact-match accuracy on a final answer), and a model $M$ evaluated at drop rate $\rho$ against a dropless evaluation of *the same weights*:
$$\Delta_{\mathcal{T}}(\rho) = A(M_{\rho=0}) - A(M_\rho), \qquad \Delta_{\mathrm{ppl}}(\rho) = \log \mathrm{PPL}(M_\rho) - \log \mathrm{PPL}(M_{\rho=0}).$$
The object of interest is the **reasoning excess**
$$R(\rho) = \frac{\Delta_{\mathcal{T}}(\rho)}{\Delta_{\mathrm{ppl}}(\rho)},$$
i.e. accuracy lost per nat of perplexity lost. The claim "dropping hurts reasoning specifically" is the claim $R_{\text{CoT}} > R_{\text{single-step}}$.

**Assumptions, and which fail.**
1. *Dropping is i.i.d. across tokens.* **False.** Drops concentrate on the tokens of long sequences and on positions late in a microbatch under position-priority; they are correlated across layers because routing is correlated across depth.
2. *Evaluation batch composition is fixed.* **False in deployment.** Serving batches are dynamic, so $\rho$ for a given request is a random variable determined by co-tenant traffic.
3. *Train-time and inference-time capacity match.* Usually false — models are frequently trained at $c=1.25$ and served dropless, or vice versa.
4. *The residual path is a benign fallback.* Untested. Skipping the MoE branch is not the identity for the layer's contribution; it is a zeroed FFN update with the attention update retained.

## 3. State of the Art

**Systems/empirical SOTA — established.**
- Capacity-based dropping originates with GShard (Lepikhin et al., ICLR 2021) and Switch Transformer (Fedus, Zoph, Shazeer, JMLR 2022), both of which report *pretraining loss* sensitivity to $c$, not reasoning-task sensitivity.
- **Dropless MoE is solved as a systems problem.** MegaBlocks (Gale, Zoph, Fedus, Casson, MLSys 2023) reformulates MoE as block-sparse matmuls, removing the capacity constraint at competitive throughput. Consequence: modern open MoEs — Mixtral 8x7B (Jiang et al., 2024), DeepSeekMoE (Dai et al., ACL 2024), OLMoE (Muennighoff et al., 2025) — are trained dropless. This *removed the incentive to study* dropping rather than answering the question.
- Expert Choice routing (Zhou et al., NeurIPS 2022) inverts the assignment: experts pick top-$C$ tokens, giving exact load balance, but a token can be selected by **zero** experts — dropping by another name, now unavoidable by construction.

**Claimed but unablated.**
- The folk claim that dropped tokens are "unimportant" because priority routing drops the low-gate ones. The gate magnitude is a router confidence score, not an importance score for the downstream answer; no ablation ties low gate to low causal effect on a final answer.
- The claim that residual skip makes dropping graceful. Stated in passing in several systems papers; no controlled study.

**Benchmark-number-only results.** Every published comparison of capacity factors reports pretraining perplexity or a fine-tuned GLUE/SuperGLUE-style score. Reasoning-specific numbers at matched weights and varied inference-time $\rho$ are, to our knowledge, absent from the literature.

**Theory SOTA.** No perturbation bound specific to MoE dropping. The nearest results are MoE learnability analyses (Chen, Deng, Li, Gu, ICLR 2023) which assume no capacity constraint.

## 4. What Is Known

- **Capacity affects pretraining quality at fixed compute.** Switch Transformer (JMLR 2022) reports that raising $c$ from 1.0 to 1.25 and 2.0 improves quality; the paper's headline configuration uses low $c$ because the throughput trade favours it. Scale: T5-Base/Large-class encoder–decoders, up to 1.6T sparse params.
- **Drop rates at $c \approx 1.0$ are non-trivial.** GShard and Switch report that a small but material fraction of tokens overflow per layer under top-2/top-1 routing at $c \in [1.0, 1.25]$; with $L$ MoE layers the per-*sequence* probability that at least one token is dropped somewhere approaches 1 for long sequences. Scale: 600B-param translation models (GShard), 128–2048 experts.
- **Load imbalance is the driver, and auxiliary losses only partly fix it.** ST-MoE (Zoph et al., 2022) adds a router $z$-loss for stability; imbalance persists and is domain-dependent.
- **Expert routing correlates with shallow token identity, not semantics.** Mixtral 8x7B (Jiang et al., 2024) reports expert assignment showing structure by syntactic/positional features rather than clean topical specialisation, measured on The Pile. This weakens the assumption that a dropped token loses a semantically dedicated computation.
- **Reasoning chains are fragile to small perturbations generally.** CoT accuracy is sensitive to prompt and intermediate-step corruption (Wei et al., NeurIPS 2022; Lanham et al., 2023, on truncated/corrupted CoT). This makes a large $R_{\text{CoT}}$ plausible but does not establish it for dropping.

## 5. What Is Not Known

- **Empirically open (principal gap).** $R(\rho)$ itself. Nobody has taken a single set of MoE weights, swept inference-time $\rho \in \{0, 0.01, 0.05, 0.15\}$, and reported GSM8K/MATH/BBH accuracy against matched perplexity. The experiment is runnable today on Mixtral or OLMoE with a patched dispatch kernel; it costs GPU-days, not GPU-months.
- **Empirically open.** Whether drops on CoT *reasoning* tokens hurt more than drops on the prompt, and whether early-chain drops dominate late-chain drops (error propagation would predict yes).
- **Methodologically blocked.** A batch-invariant definition of per-request degradation. Because $\rho_i$ depends on co-tenant sequences, "the accuracy of this model at capacity $c$" is not well defined without specifying the traffic distribution. No standard exists.
- **Theoretically open.** Any bound of the form $\|h^L_\rho - h^L_0\| \le f(m, L, \text{Lipschitz constants})$ that is non-vacuous at realistic depth. Naive layerwise composition gives an exponential-in-$L$ bound that says nothing.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement.** Varying $c$ during *training* changes gradient noise, effective capacity, and load-balancing pressure simultaneously. Any accuracy delta from a train-time sweep is unattributable. The only clean design is fixed weights with inference-time drop injection — which then measures a train/test mismatch, a different quantity.
2. **Non-identifiability of "the counterfactual token."** Dropping is batch-coupled: removing the drop for token $i$ changes which token occupies the freed capacity slot, so the single-token counterfactual is not realisable inside a fixed batch. Isolating it requires synthetic single-sequence batches, which have a drop distribution unlike deployment.
3. **The evaluation does not measure what it names.** Aggregate CoT accuracy at drop rate $\rho$ mixes (a) sequences with zero drops, (b) sequences with drops on inert tokens, (c) sequences with a drop on a load-bearing arithmetic step. Averaging over a distribution where most sequences are in class (a) or (b) can hide a large per-incident effect. The needed statistic is conditional on a drop event, not marginal.

Compute is *not* the obstruction here; the design is.

## 7. Current Research (as of 2026)

- **Dropless-by-default kernels.** MegaBlocks-descended grouped-GEMM MoE kernels are standard in vLLM/SGLang and in DeepSeek/Qwen/OLMoE training stacks. The field's practical answer is "avoid dropping," which leaves the scientific question unanswered and reopens it whenever capacity constraints return (expert parallelism across slow interconnects, on-device MoE, prefill/decode disaggregation). *(frontier — verify current kernel defaults per stack.)*
- **Expert-choice and hybrid routers** in production systems, where zero-expert tokens recur structurally (Google Brain lineage, Zhou et al.).
- **Batch-invariant inference.** Work on deterministic serving (nondeterminism from batch-size-dependent kernels) is directly adjacent: MoE dropping is the largest batch-coupling in the stack. *(frontier — verify.)*
- **Routing interpretability** — per-expert causal ablation and expert-pruning studies on Mixtral/OLMoE — supplies the tooling for the conditional-on-drop analysis but has not been pointed at capacity. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Scale.** OLMoE-1B-7B (64 experts, top-8, 16 MoE layers, open weights and data) plus Mixtral-8x7B for replication. Single 8×H100 node. Estimated cost: ~200 GPU-hours total.

**Design.** Patch the dispatch to reintroduce a capacity mask at inference only. Evaluate the *same weights* at $\rho \in \{0, 0.01, 0.03, 0.10\}$ (calibrate $c$ per setting to hit the target measured $\rho$). Use single-sequence batches with *synthetic* capacity — capacity computed as if the sequence were part of a nominal batch — so $\rho$ is controlled and reproducible.

Tasks: GSM8K (8-shot CoT), BBH, and a matched non-reasoning control (HellaSwag, ARC-easy — single-step, no chain). Also log $\log$PPL on 10M held-out tokens at each $\rho$.

**Control arms.** (i) $\rho = 0$ dropless, same weights. (ii) **Random-drop arm**: drop the same *number* of token-layer pairs, chosen uniformly at random instead of by capacity overflow. This separates "dropping hurts" from "priority routing drops the wrong things." (iii) **Position-restricted arm**: drops confined to prompt tokens vs confined to generated CoT tokens.

**Deciding number.** The reasoning excess ratio at $\rho = 0.03$:
$$\hat{R} = \frac{\Delta_{\text{GSM8K}}}{\Delta_{\text{HellaSwag}}}.$$
$\hat R \le 1.5$ (with 95% CI excluding 3) means dropping is a uniform quality tax and the reasoning-specific concern is unfounded. $\hat R \ge 3$ means capacity-constrained serving silently degrades reasoning far beyond what perplexity monitoring detects, and drop rate must become a served SLO. Power: 1319 GSM8K test items give ~±2.5 pp at 95%; use 5 seeds of batch composition for arm (ii).

**Secondary number.** Conditional degradation: accuracy on the subset of items with $\ge 1$ drop on a generated CoT token, minus accuracy on the zero-drop subset. This is the per-incident effect the marginal average hides.

## 9. Key References

- **[Foundational]** Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[Foundational]** William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 23(120), 2022. — arXiv:2101.03961
- **[SOTA]** Trevor Gale, Deepak Narayanan, Cliff Young, Matei Zaharia. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys, 2023. — arXiv:2211.15841
- **[SOTA]** Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew Dai, Zhifeng Chen, Quoc Le, James Laudon. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS, 2022. — arXiv:2202.09368
- **[SOTA]** Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, William Fedus. *ST-MoE: Designing Stable and Transferable Sparse Expert Models.* 2022. — arXiv:2202.08906
- **[SOTA]** Albert Q. Jiang et al. *Mixtral of Experts.* 2024. — arXiv:2401.04088
- **[SOTA]** Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, et al. *OLMoE: Open Mixture-of-Experts Language Models.* ICLR, 2025. — arXiv:2409.02060
- **[SOTA]** Damai Dai, Chenggang Zhao, Jiashi Li, Deli Chen, Zhiping Xu, et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL, 2024. — arXiv:2401.06066
- **[Context]** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS, 2022. — arXiv:2201.11903
- **[Context]** Tamera Lanham et al. *Measuring Faithfulness in Chain-of-Thought Reasoning.* Anthropic technical report, 2023. — arXiv:2307.13702
- **[Theory]** Zixiang Chen, Yihe Deng, Yue Wu, Quanquan Gu, Yuanzhi Li. *Towards Understanding the Mixture-of-Experts Layer in Deep Learning.* NeurIPS, 2022.
- **[Survey]** Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, Jiayi Huang. *A Survey on Mixture of Experts in Large Language Models.* IEEE TKDE, 2025. — arXiv:2407.06204

## 10. Worked Example

Take OLMoE-1B-7B: $L = 16$ MoE layers, $E = 64$, $k = 8$. Serve a GSM8K item with a 400-token prompt and a 150-token CoT: $n = 550$ tokens.

Suppose measured per-token-per-layer drop probability at $c = 1.0$ is $p = 0.01$ (1%, a mild setting; GShard-class configs sit near or above this under imbalance).

Expected drops on the generated chain alone:
$$\mathbb{E}[\text{drops}] = 150 \times 16 \times 0.01 = 24 .$$

Probability the chain is untouched, if drops were independent:
$$(1-0.01)^{2400} \approx e^{-24} \approx 4\times 10^{-11}.$$

So at a 1% per-slot drop rate, **essentially every reasoning chain is dropped on, many times.** Yet reported perplexity degradation at $c=1.0$ versus dropless is small — a fraction of a percent. Both facts are true at once.

Here is the obstruction, made concrete. Two hypotheses fit the same perplexity number:

| | Per-incident effect | Predicted GSM8K $\Delta$ |
|---|---|---|
| **H1: drops are inert** | ~0; residual path carries it | ≈ 0 pp |
| **H2: drops are load-bearing but rare-on-the-critical-token** | 1 in 200 drops flips a digit | $24/200 \approx 12\%$ of chains corrupted → ~8–10 pp |

An 8–10 pp GSM8K drop is a serious regression. A 0.3% perplexity change is invisible on a training dashboard. Perplexity, averaged over 550 tokens of which perhaps three carry the arithmetic, **cannot distinguish H1 from H2** — the discriminating signal is 3 tokens out of 550, diluted by a factor of ~180.

That dilution is why the question is empirically open despite the experiment being cheap: the standard instrument (loss) is roughly two orders of magnitude too coarse, and the fix is not more compute but a conditional-on-drop-event evaluation of the kind specified in §8.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*