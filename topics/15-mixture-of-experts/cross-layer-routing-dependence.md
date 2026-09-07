---
id: 15-mixture-of-experts/cross-layer-routing-dependence
title: "Cross-Layer Routing Dependence"
topic: 15-mixture-of-experts
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Layer Routing Dependence

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/cross-layer-routing-dependence` · **Status:** open

## 1. Problem Statement

In a sparse MoE transformer, each MoE layer routes each token independently: the router at layer $\ell$ sees only the layer-$\ell$ hidden state and emits its own top-$k$ expert selection. The question is whether the resulting sequence of selections across depth — the token's **path** — carries structure that the independent-per-layer parameterization fails to exploit.

Three variants, with different difficulty:

- **Measurement.** Given a trained MoE and a token stream, how much statistical dependence exists between routing decisions at different layers *beyond* what is explained by the token's identity and context? Solving this means a dependence estimate with a stated null and a confidence interval.
- **Method.** Does making the router at layer $\ell$ explicitly conditional on selections at layers $<\ell$ improve the quality/FLOP frontier at matched active parameters? Solving this means a loss delta at fixed compute, replicated across seeds and scales.
- **Theory.** Is there a class of functions that a depth-$L$ MoE with path-conditional routing represents with fewer active parameters than any MoE with per-layer-independent routing? Solving this means a separation theorem, or a proof that no separation exists.

Status is `open` on all three; the measurement variant is the one currently blocking the other two.

## 2. Formal Setting

Let the model have $L$ MoE layers, $E$ experts per layer, top-$k$ routing. For token $x$ at position $i$ in context $c$, write the layer-$\ell$ hidden state $h_\ell \in \mathbb{R}^d$ and the router logits $g_\ell = W_\ell h_\ell \in \mathbb{R}^E$. The selection is the random variable

$$R_\ell = \mathrm{TopK}_k(g_\ell) \in \binom{[E]}{k}, \qquad \text{path } P = (R_1,\dots,R_L).$$

**Path entropy.** $H(P) \le L \log_2 \binom{E}{k}$. For Mixtral ($L=32$, $E=8$, $k=2$) the ceiling is $32\log_2 28 = 153.8$ bits.

**Dependence, measured.** The naive statistic is the pairwise mutual information $I(R_\ell; R_{\ell'})$, estimated by the plug-in estimator over a corpus of $N$ tokens with a Miller–Madow or bootstrap bias correction (plug-in MI is biased upward by $\approx (|\mathcal{R}|-1)^2/(2N\ln 2)$ bits). This statistic is **wrong for the question**: $R_\ell$ and $R_{\ell'}$ both depend on the token, so $I(R_\ell;R_{\ell'})>0$ even under strictly independent routers. The quantity the problem is about is the conditional MI

$$I(R_\ell; R_{\ell'} \mid X=x, C=c),$$

conditioning on the full token-in-context. Since $h_\ell$ is a deterministic function of $(x,c)$, this quantity is exactly $0$ for any deterministic router — the conditioning set determines both variables. The measurement is therefore only well posed under an explicit noise or intervention model.

**The interventional definition.** Define the counterfactual routing cost. Let $\mathcal{L}(x,c)$ be the token NLL. Replace $R_{\ell'}$ with a draw $\tilde R_{\ell'}$ from a reference distribution $q$ and measure

$$\Delta_{q}(\ell') = \mathbb{E}\big[\mathcal{L}(x,c \mid \mathrm{do}(R_{\ell'} = \tilde R_{\ell'}))\big] - \mathbb{E}[\mathcal{L}(x,c)].$$

Cross-layer dependence is the gap between resampling from the **marginal** $q_{\mathrm{marg}}(\cdot)$ and from the **path-conditional** $q_{\mathrm{cond}}(\cdot \mid R_{<\ell'})$:

$$D(\ell') = \Delta_{q_{\mathrm{marg}}}(\ell') - \Delta_{q_{\mathrm{cond}}}(\ell') \ \ \text{nats}.$$

$D>0$ means knowing the earlier path tells you something about which experts are safe to substitute at $\ell'$ — dependence that is causal for the loss, not merely correlational.

**Assumptions, and which fail.**
1. *Expert marginals are near-uniform.* Enforced by auxiliary load-balancing losses; violated in practice — trained MoEs show persistent 2–5× imbalance across experts even with balancing.
2. *Experts are exchangeable within a layer.* False: experts specialize, and the per-layer permutation symmetry makes "which path" non-identifiable across seeds while making $H(P)$ and $D(\ell')$ identifiable.
3. *Routing is deterministic at inference.* True for top-$k$ argmax, false during training with noisy/jitter routing and false under expert-capacity dropping, where a token's realized expert depends on the other tokens in the batch.

## 3. State of the Art

**Established.**
- Per-layer independent routing is the default in every large deployed MoE: Switch Transformer (Fedus, Zoph, Shazeer, JMLR 2022), GShard (Lepikhin et al., ICLR 2021), ST-MoE (Zoph et al., 2022), Mixtral 8x7B (Jiang et al., 2024), DeepSeekMoE (Dai et al., ACL 2024), OLMoE (Muennighoff et al., ICLR 2025). No frontier system conditions layer $\ell$'s router on layer $\ell-1$'s choice.
- Routing at adjacent layers is **predictable above chance**. Mixtral's routing analysis reports first-choice expert repetition across consecutive layers well above the $1/8 = 12.5\%$ uniform rate, with the effect strongest at mid-depth. This is a descriptive statistic, not an interventional one.
- Systems work monetizes the predictability without claiming it is useful for quality: Pre-gated MoE (Hwang et al., ISCA 2024) predicts layer-$\ell{+}1$ experts from layer-$\ell$ activations to prefetch weights; Eliseev & Mazur (2023) use LRU caching plus speculative expert loading for offloaded inference. Both report latency/memory wins with near-neutral accuracy.

**Claimed but unablated.** Layerwise recurrent routing — RMoE (Qiu et al., 2024) threads a GRU across layer routers so router $\ell$ sees router $\ell-1$'s state — reports perplexity gains at sub-billion scale. The ablation that separates "cross-layer information" from "more router parameters and a smoother gradient path" has not been published; the control arm should be an equal-parameter, equal-gradient-path router with the cross-layer input zeroed, and it is absent.

**Benchmark-number-only.** Downstream-task deltas for any path-conditional router are single-run MMLU/HellaSwag numbers at ≤1B active parameters. Treat them as unmeasured.

**Theory SOTA.** Chen, Deng, Wu, Gu, Li (*Towards Understanding the Mixture-of-Experts Layer in Deep Learning*, NeurIPS 2022) prove a single MoE layer with a nonlinear router learns cluster structure that a single dense expert cannot. The result is one layer deep. There is no depth-$L$ analogue, and therefore no separation theorem for path-conditional routing.

## 4. What Is Known

- **Scaling of routing itself.** Clark et al. (*Unified Scaling Laws for Routed Language Models*, ICML 2022) fit routed-model scaling across 15M–200B parameters and find the benefit of routing shrinks with model size and is well described without any cross-layer term; three routing algorithms (S-BASE, HASH, RL-routing) scale near-identically. Implication: the routing *rule* matters less than the sparsity budget, at least for per-layer rules.
- **Routing is not semantic.** Mixtral 8x7B (2024) reports no clear domain specialization of experts on The Pile subsets (ArXiv, PubMed, GitHub) but clear syntactic/positional locality — consecutive tokens repeat experts far above chance.
- **Fixed, non-learned routing is competitive.** Hash Layers (Roller et al., NeurIPS 2021) route by token-ID hash — zero cross-layer information by construction — and match learned routing at 1–2B scale. This is the strongest existing evidence that cross-layer structure is not load-bearing.
- **Routers saturate early.** OLMoE (1B active / 7B total, 5T tokens) reports router assignments largely fixed within roughly the first 1% of training and stable thereafter; so any cross-layer dependence is a property of an early-frozen discrete map, not a slowly-learned one.
- **Fine-grained expert counts change the arithmetic.** DeepSeekMoE-16B uses 64 fine-grained experts with top-6 plus shared experts; path space per layer rises from $\binom{8}{2}=28$ to $\binom{64}{6}\approx 7.5\times10^7$, so dependence estimates from $E=8$ models do not transfer.

## 5. What Is Not Known

- **Methodologically blocked.** There is no agreed estimator for cross-layer dependence. The observational statistic ($I(R_\ell;R_{\ell'})$) is confounded by token identity and by shared residual-stream content; the interventional statistic $D(\ell')$ is well defined but has never been computed at scale, and its reference distribution $q$ is a free choice that changes the answer.
- **Empirically open.** Whether a path-conditional router beats an independent router at matched active FLOPs and matched router parameters, at ≥1B active parameters and ≥100B tokens. Runnable today for a few thousand GPU-hours; unrun with an adequate control.
- **Theoretically open.** Whether depth-$L$ path-conditional MoE is strictly more expressive per active parameter than depth-$L$ independent MoE. No separation and no collapse theorem exists in either direction.
- **Empirically open.** Whether measured dependence increases, decreases, or is scale-invariant from $10^8$ to $10^{11}$ parameters. All published routing analyses are at a single scale.

## 6. Why It Is Hard

The core obstruction is **confounded measurement compounded by non-identifiability**.

1. *Confounding.* Every router reads the residual stream, which carries the token embedding forward. Any two routers therefore agree above chance for reasons that have nothing to do with routing coordination. Zipfian token frequency makes this severe: a small set of high-frequency tokens dominates the corpus, and a per-token routing preference at both layers manufactures apparent cross-layer dependence (see §10).
2. *Non-identifiability.* Expert indices are arbitrary up to per-layer permutation. Path statistics are comparable across seeds only after a matching step, and no canonical matching exists when experts partially overlap in function.
3. *No ground truth.* There is no reference "correct path" for a token, so a router modification can only be scored by end loss — which mixes routing quality with the optimization side effects of changing the router's parameterization.
4. *Capacity coupling.* Under expert-capacity limits, a token's realized expert depends on the rest of its batch, so $R_\ell$ is not a function of the token alone and the interventional estimand becomes batch-dependent.

Compute is *not* the primary obstruction: routing traces are ~24 bytes/token for a Mixtral-shaped model, so 10B tokens of traces is ~240 GB.

## 7. Current Research (as of 2026)

- **Systems exploitation.** Expert prefetch and caching that predict next-layer experts from current-layer state — the ISCA/MLSys line following Pre-gated MoE. Active and well validated on latency, silent on quality.
- **Recurrent and hierarchical routers.** RMoE-style GRU-linked routers, and two-stage routers that pick a group then an expert (DeepSeek-V2/V3 device-limited routing is a bandwidth-motivated cousin). *(frontier — verify)* whether any 2025–2026 frontier model ships a genuinely path-conditional router; public technical reports still describe per-layer routers.
- **Routing interpretability.** Work relating expert selection to token identity, position, and induction-head-like circuits, mostly at OLMoE and Mixtral scale, from Allen Institute for AI and academic interpretability groups.
- **Mixture-of-Depths** (Raposo et al., 2024) — per-layer capacity routing over the residual stream — is the nearest neighbour: it makes the depth-wise skip pattern itself the routed object, but keeps per-layer independence.

## 8. Concrete Next Experiment

**Scale.** Train three MoE language models, 1.3B active / 8B total, $E=64$ fine-grained experts, top-8, $L=24$ MoE layers, 100B tokens of a public corpus (DCLM or FineWeb-Edu), 3 seeds each. Roughly 3–5k H100-hours per arm.

**Arms.**
- *Control A (independent):* standard linear router, $W_\ell h_\ell$.
- *Control B (parameter-matched placebo):* router input is $[h_\ell ; z_\ell]$ where $z_\ell$ is a fixed random $E$-dimensional vector, permuted per step. Identical parameter count and gradient path to the treatment; zero cross-layer information. **This arm is the one missing from all published work.**
- *Treatment (path-conditional):* router input is $[h_\ell ; m_{\ell-1}]$ where $m_{\ell-1}\in\{0,1\}^E$ is the previous MoE layer's selection mask.

**Deciding number.** Validation NLL of Treatment minus Control B, at matched tokens and matched active FLOPs, averaged over 3 seeds:

$$\delta = \mathrm{NLL}_{\text{treat}} - \mathrm{NLL}_{\text{ctrl-B}}.$$

Decide **yes, cross-layer conditioning is load-bearing** if $\delta \le -0.01$ nats/token with the 3-seed 95% interval excluding 0. Decide **no** if $|\delta| < 0.005$ nats. A gain against Control A but not against Control B is a parameterization artifact, not cross-layer information — and this is the outcome the existing literature cannot rule out.

**Secondary readout.** On the trained Control A, compute the interventional gap $D(\ell)$ of §2 for every layer, resampling from marginal vs. path-conditional at temperature-matched entropy, over 10M held-out tokens. Report $D(\ell)$ in nats. If $\max_\ell D(\ell) < 0.01$ nats, the observational repetition statistics are confounded artifacts and the method variant is dead.

## 9. Key References

- **[Foundational]** Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017. — arXiv:1701.06538
- **[Foundational]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 23(120), 2022. — arXiv:2101.03961
- **[Foundational]** Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer, Chen. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR 2021. — arXiv:2006.16668
- **[SOTA]** Jiang, Sablayrolles, Roux, Mensch, et al. *Mixtral of Experts.* 2024. — arXiv:2401.04088 (contains the consecutive-layer routing repetition analysis)
- **[SOTA]** Dai, Deng, Zhao, Xu, Gao, Chen, et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL 2024. — arXiv:2401.06066
- **[SOTA]** Muennighoff, Soldaini, Groeneveld, Lo, Morrison, et al. *OLMoE: Open Mixture-of-Experts Language Models.* ICLR 2025. — arXiv:2409.02060
- **[SOTA]** Clark, de las Casas, Guy, Mensch, Paganini, et al. *Unified Scaling Laws for Routed Language Models.* ICML 2022. — arXiv:2202.01169
- **[Contrast]** Roller, Sukhbaatar, Szlam, Weston. *Hash Layers For Large Sparse Models.* NeurIPS 2021. — arXiv:2106.04426
- **[Contrast]** Lewis, Bhosale, Dettmers, Goyal, Zettlemoyer. *BASE Layers: Simplifying Training of Large, Sparse Models.* ICML 2021. — arXiv:2103.16716
- **[Theory]** Chen, Deng, Wu, Gu, Li. *Towards Understanding the Mixture-of-Experts Layer in Deep Learning.* NeurIPS 2022.
- **[Systems]** Hwang, Wei, Zheng, Kim, et al. *Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference.* ISCA 2024. — arXiv:2308.12066
- **[Systems]** Eliseev, Mazur. *Fast Inference of Mixture-of-Experts Language Models with Offloading.* 2023. — arXiv:2312.17238
- **[Related]** Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro. *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models.* 2024. — arXiv:2404.02258
- **[Related]** Zhou, Lei, Liu, Du, Huang, Zhao, Dai, Chen, Le, Laudon. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS 2022. — arXiv:2202.09368

## 10. Worked Example

Take Mixtral 8x7B: $L=32$, $E=8$, $k=2$. Consider only the first-choice expert, so each layer contributes at most $\log_2 8 = 3$ bits.

**Step 1 — turn the reported repetition into bits.** Mixtral reports adjacent-layer first-choice repetition above the $12.5\%$ chance rate. Take a representative mid-depth value of $25\%$, with the remaining $75\%$ spread uniformly over the other 7 experts ($10.71\%$ each). Then

$$H(R_{\ell+1}\mid R_\ell) = -\big(0.25\log_2 0.25 + 7 \cdot 0.1071\log_2 0.1071\big) = 0.500 + 2.417 = 2.917 \text{ bits},$$

$$I(R_\ell;R_{\ell+1}) = 3.000 - 2.917 = 0.083 \text{ bits}.$$

Across all 31 adjacent pairs that is at most $31 \times 0.083 = 2.6$ bits, against a first-choice path entropy of $32 \times 3 = 96$ bits. **Adjacent-layer dependence accounts for 2.7% of the path information.** The headline "experts repeat at twice the chance rate" survives as a fact and evaporates as a quantity.

**Step 2 — make the confound visible.** Suppose routing is genuinely independent across layers *given the token*, but a sticky 10% of corpus positions (frequent function words, code punctuation) route deterministically to one fixed expert at both layers, while the other 90% route independently and uniformly. Then

$$P(R_{\ell+1}=R_\ell) = 0.10 \cdot 1 + 0.90 \cdot 0.125 = 0.2125.$$

A 10% sticky-token fraction reproduces $21.3\%$ of the observed $25\%$ repetition with **zero** cross-layer coordination. Zipf makes 10% a conservative figure — in most web corpora the top ~30 token types alone exceed that share.

**Step 3 — read the obstruction.** The residual dependence not explained by the token-identity null is $25\% - 21.3\% = 3.7$ percentage points, corresponding to roughly $0.005$ bits per adjacent pair, or $0.16$ bits over the full depth. That is inside the plug-in MI bias for any corpus smaller than $\sim 10^7$ tokens per layer pair, and it says nothing about whether those bits are causal for the loss. Only the interventional gap $D(\ell)$ of §2, measured against Control B of §8, separates "the routers agree because they read the same token" from "the routers coordinate". Until that number exists, every claim in this problem area — including the systems claim that next-layer experts are predictable, which is true and useful for prefetch — is compatible with cross-layer routing dependence being an artifact of token frequency.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*