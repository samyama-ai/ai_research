---
id: 15-mixture-of-experts/compute-optimal-expert-granularity
title: "Compute-Optimal Granularity of Experts"
topic: 15-mixture-of-experts
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Granularity of Experts

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/compute-optimal-expert-granularity` · **Status:** empirically-open

## 1. Problem Statement

A sparse MoE layer replaces one dense feed-forward block of width $d_{ff}$ with $E$ experts of width $d_e$, of which $k$ are routed to per token. **Granularity** is $G = d_{ff}/d_e$: how finely the same activated FLOP budget is chopped. $G=1$ is the classic Switch/Mixtral shape; $G=8$–$16$ is the DeepSeekMoE/Qwen shape.

The question: **given a compute budget $C$, what is the optimal $G$, and how does $G^\star$ scale with $C$?**

Three variants that are routinely conflated:

- **Measurement.** Is the right budget FLOPs, or wall-clock on a specified accelerator topology? These give different $G^\star$, and the gap is the whole problem. FLOP-optimal $G$ appears to grow without bound; time-optimal $G$ cannot.
- **Method.** Produce a fitted law $G^\star(C)$ that transfers — predicts held-out loss at a budget $10\times$ beyond the fitting range, on a different codebase, without refitting.
- **Theory.** Prove why loss improves with $G$ at fixed FLOPs. No mechanism is established; the candidates (combinatorial routing capacity $\binom{E}{k}$, reduced expert-internal interference, better parameter–FLOP decoupling) are not distinguished by any experiment.

Solved would mean: a law with stated error bars that predicts $G^\star$ under a *time* budget, validated out-of-range.

## 2. Formal Setting

Per MoE layer, with model width $d$, SwiGLU-style FFN (3 matrices):

- $N_{\text{act}}$ — **active** non-embedding parameters per token. Measured by counting parameters touched on a forward pass, not by dividing totals: $N_{\text{act}}^{(\ell)} = 3\,d\,d_e\,(k + k_s)$ for $k$ routed and $k_s$ shared experts.
- $N_{\text{tot}}$ — all parameters: $3\,d\,d_e\,(E + k_s)$ per layer. **Sparsity** $S = 1 - N_{\text{act}}/N_{\text{tot}}$.
- $G = d_{ff}/d_e$, with $d_{ff}$ the width of the *dense reference* model at the same $d$. Measured, not assumed — $d_{ff}/d$ ratios vary from 2.7 (Llama) to 4 (GPT).
- $C_{\text{FLOP}} \approx 6 N_{\text{act}} D$, $D$ tokens. This ignores attention, router, and all-to-all.
- $C_{\text{time}} = D \cdot t_{\text{tok}}$, measured as steady-state step time on a fixed mesh (e.g. 64 H100s, expert-parallel degree $p$), excluding the first 100 steps.

Holding $C_{\text{FLOP}}$ and $N_{\text{act}}$ fixed while varying $G$ requires scaling $E \propto G$ and $k \propto G$ together. The empirical law fitted by Krajewski et al. (2024) has the form

$$\mathcal{L}(N,D,G) \;=\; c \;+\; \Big(a + \frac{g}{G^{\gamma}}\Big) N^{-\alpha} \;+\; b\,D^{-\beta},$$

so granularity enters as a *multiplicative discount on the parameter term*: its benefit shrinks as $N$ grows, and the law is monotone increasing in $G$ — it contains no term that ever makes larger $G$ worse.

**Assumptions, and which are violated:**

1. *All experts see equal token load.* Violated: load imbalance grows with $E$; measured max/mean expert load is routinely 1.5–3× even with auxiliary losses.
2. *FLOPs are a proxy for cost.* Violated at $G \gg 1$: all-to-all volume scales with $k$, and per-expert GEMM rows fall as $1/G$, so arithmetic intensity drops.
3. *The dense reference $d_{ff}$ is well defined.* Violated — $G$ is a ratio to a model nobody trained.
4. *Router quality is $G$-independent.* Unverified; top-$k$ over 256 logits is a harder estimation problem than over 8.

## 3. State of the Art

**Established.**
- Fine-grained MoE beats $G=1$ at matched active FLOPs and matched total parameters, at small scale. Krajewski et al., *Scaling Laws for Fine-Grained Mixture of Experts* (2024, arXiv:2402.07871) fit the law above over models up to ~1.3B active parameters and budgets to ~$10^{20}$ FLOPs.
- MoE beats dense at equal FLOPs, and the gain shrinks as models grow. Clark et al., *Unified Scaling Laws for Routed Language Models* (ICML 2022, arXiv:2202.01169) fit this across three routing techniques up to 900M parameters and report the benefit vanishing around $\sim$900M — a prediction that later fine-grained models contradicted, which is itself the cautionary datum.

**Claimed but unablated.**
- That $G^\star$ *increases* with compute. This follows from the fitted law's functional form, which is monotone in $G$ by construction; it is not an independent measurement. Extrapolating a monotone fit to argue for unbounded $G$ is circular.
- DeepSeekMoE (Dai et al., ACL 2024, arXiv:2401.06066) attributes its gains to fine granularity *plus* shared-expert isolation. The two are ablated at 2B scale only; at 16B and beyond the choice is asserted.

**Benchmark-number only.** DeepSeek-V3 (arXiv:2412.19437), Qwen3-MoE, and Kimi-family models all ship $G \approx 8$–$12$ with $E \geq 128$. Their quality is real; no controlled $G$ sweep at those scales has been published, so these are existence proofs, not measurements of $G^\star$.

**Systems SOTA** is what makes large $G$ viable at all: MegaBlocks (Gale et al., MLSys 2023, arXiv:2211.15841) block-sparse grouped GEMMs, and Tutel (Hwang et al., MLSys 2023) adaptive all-to-all.

## 4. What Is Known

- **Granularity gain, small scale.** Krajewski et al. report best-fit $G$ in the range 8–16 for budgets around $10^{19}$–$10^{20}$ FLOPs, with $G=1$ strictly dominated. Measured at $\leq$1.3B active parameters.
- **Sparsity, not granularity, is the cleaner axis.** Abnar et al., *Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models* (2025, arXiv:2501.12370) find optimal sparsity rises with compute budget, holding at fixed $G$ — up to ~$10^{21}$ FLOPs. Sparsity and granularity are separately identifiable in principle; most papers vary both.
- **Memory/parameter trade.** Ludziejewski et al., *Joint MoE Scaling Laws* (2025, arXiv:2502.05172) fit loss jointly in $N_{\text{act}}$, $N_{\text{tot}}$, $D$ and granularity, finding MoE can be memory-optimal, not just FLOP-optimal.
- **Shipped configurations.** DeepSeek-V3: $d=7168$, $E=256$ routed at $d_e=2048$, $k=8$, one shared expert, $G=9$. Mixtral 8×7B (arXiv:2401.04088): $E=8$, $k=2$, $d_e = d_{ff} = 14336$, $G=1$. OLMoE-1B-7B (Muennighoff et al., 2024, arXiv:2409.02060): $E=64$, $k=8$, and its ablations found more, smaller experts better at 1B active / 7B total.
- **Load imbalance is the binding systems cost**, not the router FLOPs. The router is $<0.1\%$ of layer FLOPs at $E=256$.

## 5. What Is Not Known

- **Empirically open (primary).** Does $G^\star$ under a *wall-clock* budget saturate, and at what value? Runnable today — a $G \in \{1,2,4,8,16,32\}$ sweep at 10–20B total parameters, matched active FLOPs and matched total parameters, with step time logged. Nobody has published it. Cost is the only barrier: ~6 runs × $10^{21}$ FLOPs.
- **Empirically open.** Is granularity separable from shared experts, from $k$, and from expert-parallel topology? Current evidence varies all four together.
- **Methodologically blocked.** $G$ has no operational definition without a dense reference. Two labs reporting "$G=8$" may differ 1.5× in $d_e/d$. Until the field reports $(E, k, d_e/d)$ instead of $G$, cross-paper comparison is not well posed.
- **Theoretically open.** No proof that finer experts increase representable-function capacity at fixed activated FLOPs. The $\binom{E}{k}$ "combinatorial expressivity" argument counts routing configurations, not achievable functions, and no lower bound rules out that gains come entirely from improved optimization conditioning.

## 6. Why It Is Hard

**The obstruction is confounded measurement, not compute alone.** The quantity everyone reports (loss at fixed FLOPs) is not the quantity anyone wants (loss at fixed dollars). The two diverge *in the direction of the effect being studied*: increasing $G$ raises the FLOP-metric score while lowering achieved FLOP utilization, because per-expert GEMM row counts fall as $1/G$ and all-to-all volume rises with $k \propto G$. A FLOP-denominated sweep therefore has a built-in bias toward large $G$ of unknown magnitude — the error term is the thing being measured.

Secondary: the fitted laws are monotone in $G$ by functional form, so they cannot express a turning point. A law that cannot represent the answer cannot find it.

## 7. Current Research (as of 2026)

- **Joint laws over $(N_{\text{act}}, N_{\text{tot}}, D, G)$** — Ludziejewski, Krajewski et al. (IDEAS NCBR / Warsaw), extending the 2024 granularity law to memory-constrained regimes.
- **Optimal-sparsity laws** — Abnar et al. (Apple), treating sparsity as the primary axis and granularity as nuisance.
- **Ultra-fine granularity with $E \geq 512$** and near-zero-compute routing *(frontier — verify)*; motivated by inference-time expert offload, where small experts fetch cheaply from CPU memory.
- **Hardware-aware co-design**: choosing $d_e$ to match tensor-core tile sizes (multiples of 128 on Hopper/Blackwell) rather than to satisfy a scaling law *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Six pretraining runs, each 3B active / 24B total parameters, 300B tokens (~$5\times10^{21}$ FLOPs each), fixed 64×H100 mesh, identical data order, identical optimizer and LR schedule.

**Arms.** $G \in \{1, 2, 4, 8, 16, 32\}$ with $(E,k)$ scaled as $(24G,\,G)$ so $N_{\text{act}}$, $N_{\text{tot}}$, and top-$k$ *fraction* are all held constant. **Control arm:** $G=1$, $E=24$, $k=1$, plus a dense 3B run at the same token count to anchor the FLOP-equivalence claim.

**Decision number.** Report validation loss against **measured wall-clock**, not FLOPs:

$$\Delta(G) \;=\; \mathcal{L}\big(G,\; D_{\text{eff}}(G)\big) - \mathcal{L}(1,\, D_{\text{eff}}(1)), \qquad D_{\text{eff}}(G) = \frac{T_{\text{budget}}}{t_{\text{tok}}(G)}.$$

The question is decided by $\arg\min_G \Delta(G)$ under equal wall-clock $T_{\text{budget}}$. If it lands at $G \leq 8$ while the FLOP-denominated minimum is at $G \geq 16$, the published laws are confirmed biased and by how much. One additional required number: MFU per arm, to attribute the gap.

## 9. Key References

- **[Foundational]** Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017. — arXiv:1701.06538
- **[Foundational]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 2022. — arXiv:2101.03961
- **[SOTA]** Krajewski, Ludziejewski, et al. *Scaling Laws for Fine-Grained Mixture of Experts.* 2024. — arXiv:2402.07871
- **[SOTA]** Abnar, Shah, Busbridge, et al. *Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models.* 2025. — arXiv:2501.12370
- **[SOTA]** Ludziejewski, Krajewski, et al. *Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient.* 2025. — arXiv:2502.05172
- **[SOTA]** Dai, Deng, Zhao, et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL 2024. — arXiv:2401.06066
- **[Survey/Scaling]** Clark, de las Casas, Guy, et al. *Unified Scaling Laws for Routed Language Models.* ICML 2022. — arXiv:2202.01169
- **[Systems]** Gale, Narayanan, Young, Zaharia. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys 2023. — arXiv:2211.15841
- **[Empirical]** Muennighoff, Soldaini, Groeneveld, et al. *OLMoE: Open Mixture-of-Experts Language Models.* 2024. — arXiv:2409.02060
- **[Baseline]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556

## 10. Worked Example

Take the DeepSeek-V3 MoE layer: $d=7168$, $d_e=2048$, $E=256$, $k=8$, one shared expert, SwiGLU.

Active FLOPs per token:
$$6 \cdot d \cdot d_e \cdot (k+k_s) = 6 \cdot 7168 \cdot 2048 \cdot 9 \approx 7.93\times10^{8}.$$

A dense FFN at $d_{ff}=18432$ (the standard $2.57d$ ratio for this width) costs $6 \cdot 7168 \cdot 18432 \approx 7.93\times10^{8}$ — **identical**. So $G = 18432/2048 = 9$, and the layer is exactly FLOP-matched to its dense reference while holding $256/9 \approx 28\times$ the parameters. This is the clean case for granularity.

Now count what FLOPs omit. With 4096 tokens per device and $k=8$ over $E=256$, the expected rows per expert GEMM is $4096 \cdot 8 / 256 = 128$. The $G=1$ control ($E=28$, $k=1$, $d_e=18432$) gives $4096/28 \approx 146$ rows but each GEMM is $9\times$ wider — total work equal, arithmetic intensity far higher, and all-to-all token volume $8\times$ lower. On an H100, a grouped GEMM with $M=128$ and load imbalance of 2× (so real groups range 64–256 rows) runs well below peak; *assumed* penalty 15–30% MFU relative to the $G=1$ arm. That range is an estimate, not a measurement — which is precisely the point.

Predicted loss gain from the fitted law at $G=9$ versus $G=1$, at 3B active parameters: on the order of a 1–3% reduction in the $N^{-\alpha}$ term, i.e. a few hundredths of a nat — *derived* by extrapolating a law fitted below 1.3B, more than $2\times$ outside its range.

**The obstruction, visible:** a claimed benefit of a few percent, extrapolated out of range, set against an unmeasured throughput cost of 15–30%. The two numbers are the same order of magnitude and have never been measured in the same experiment. Until they are, $G^\star$ is not known — it is inherited from whichever lab shipped last.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*