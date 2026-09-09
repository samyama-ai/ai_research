---
id: 14-long-context/learned-sparse-attention-patterns
title: "Sparse Attention Pattern Learnability"
topic: 14-long-context
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Sparse Attention Pattern Learnability

> **Topic:** Long Context · **ID:** `14-long-context/learned-sparse-attention-patterns` · **Status:** open

## 1. Problem Statement

Sparse attention restricts each query to a subset of keys. The subset can be **fixed** (local window, strided, global sinks), **content-derived by a heuristic** (top-$k$ on a cheap score), or **learned** (a selector trained jointly with the model). The question:

> Does end-to-end training discover attention sparsity patterns that outperform a fixed pattern of the same density, and does the advantage survive to long-context tasks that require exact retrieval?

Three variants, different difficulty:

- **Measurement.** Given a trained sparse model, decide whether its selector carries task-relevant information beyond position. Requires a metric that separates "recovers most attention mass" from "recovers the tokens that matter" — these come apart badly (§10).
- **Method.** Build a selector whose gradient signal is strong enough to learn a discrete pattern, and whose pattern is hardware-realizable (contiguous blocks, uniform per-query budget).
- **Theory.** Characterize the function classes where a learnable pattern with density $s$ matches dense attention, and where no static pattern of density $s$ can.

Solving it means: a training recipe where the learned selector beats a density-matched static baseline by a stated margin on retrieval-hard evaluations, with an ablation showing the margin comes from learning and not from added parameters or extra compute.

## 2. Formal Setting

Sequence length $N$, queries $Q \in \mathbb{R}^{N \times d}$, keys/values $K, V \in \mathbb{R}^{N \times d}$. A pattern is a binary mask $M \in \{0,1\}^{N \times N}$, causal ($M_{ij}=0$ for $j>i$). Sparse attention output:

$$O_i = \sum_{j} \frac{M_{ij}\exp(q_i^\top k_j/\sqrt{d})}{\sum_{l} M_{il}\exp(q_i^\top k_l/\sqrt{d})} v_j$$

**Density** $s = \mathbb{E}_i[\|M_{i\cdot}\|_1] / \mathbb{E}_i[i]$, measured as the fraction of causally-legal pairs kept — not of $N^2$, or local-window methods look artificially sparse.

**Learned selector.** $M = \Pi_k(g_\phi(Q,K))$, where $g_\phi$ scores blocks and $\Pi_k$ keeps the top $k$ per query. With block size $B$, there are $\lceil N/B \rceil$ blocks; $\Pi_k$ has zero gradient almost everywhere, so $\phi$ is trained by straight-through (Bengio et al. 2013), Gumbel relaxation (Jang et al., ICLR 2017), or — as in Native Sparse Attention — by making selected blocks differentiable through the attention output itself.

**Attention-mass recovery**, the standard proxy:
$$R_i = \sum_{j: M_{ij}=1} a_{ij}^{\text{dense}}, \qquad a^{\text{dense}} = \mathrm{softmax}(QK^\top/\sqrt d)$$
measured by running the dense model on the same inputs and averaging $R_i$ over queries, heads, layers. **This is the quantity that is easy to measure and is not the quantity that matters** (§10).

**Learnability gap.** For a task family $\mathcal{T}$ and budget $s$:
$$\Delta(s) = \min_{\text{static } M: \text{dens}(M)=s} \mathcal{L}(M) \;-\; \min_{\phi: \mathbb{E}[\text{dens}]=s} \mathcal{L}(g_\phi)$$
$\Delta(s) > 0$ is the claim under test.

**Assumptions, and which are violated.**
- *Sparsity is intrinsic to the trained dense model.* Violated in part: attention sinks are an artifact of softmax's forced normalization (Xiao et al., ICLR 2024), not a semantic pattern.
- *Density is uniform across heads and layers.* Violated: MoA (Fu et al. 2024) and FastGen (Ge et al., ICLR 2024) both find per-head optimal budgets spanning an order of magnitude.
- *Optimal pattern is query-independent.* Violated: Quest (Zhang et al., ICML 2024) shows the critical KV pages change per decoding step.
- *Block-level scoring approximates token-level relevance.* Violated when the needle is a single token in a block of 64 distractors.

## 3. State of the Art

**Systems/empirical SOTA — established.**
- **Native Sparse Attention** (Yuan et al., DeepSeek, ACL 2025; arXiv:2502.11089): hierarchical compress + block-select + sliding window, trained from scratch at 27B (3B active) on 260B tokens. Matches or exceeds full attention on general and reasoning benchmarks; ~9× forward, ~6× backward, ~11× decode speedup at 64k. This is the strongest *pretrained-sparse* result and the strongest evidence that a learned selector is not a handicap.
- **MoBA** (Lu et al., Moonshot AI, 2025; arXiv:2502.13189): mixture-of-block-attention with a top-$k$ gate, deployed in Kimi at 1M context, with a documented full↔sparse switch.
- **MInference** (Jiang et al., NeurIPS 2024; arXiv:2407.02490): three offline-classified patterns (A-shape, vertical-slash, block-sparse), ~10× prefill speedup at 1M on A100. Pattern is *searched offline per head*, not learned by gradient.
- **StreamingLLM** (Xiao et al., ICLR 2024): 4 sink tokens + rolling window streams to 4M tokens with stable perplexity — but explicitly does *not* extend the model's effective context.

**Claimed but unablated.** Almost every learned-selector paper reports "matches dense at $s\approx0.1$" on aggregate benchmark suites. The ablation that is usually missing is a **density-matched static control tuned with the same budget**. Where that control has been run (Nawrot et al. 2025), a large part of the claimed gain is absorbed.

**Benchmark-number-only.** NSA's and MoBA's parity claims rest on single training runs at one scale with one static baseline family. No independent reproduction of NSA at 27B exists publicly as of 2026-09.

**Theory SOTA.** Sanford, Hsu & Telgarsky (NeurIPS 2023; arXiv:2306.02896) give the sparse-averaging separation: a one-layer attention needs width roughly linear in $N$, while depth-2 solves it with near-logarithmic size — sparsity structure is depth-dependent, so a per-layer density budget is the wrong object. Alman & Song (NeurIPS 2023; arXiv:2302.13214) show no subquadratic *approximation* of attention exists under SETH once entries exceed $\Theta(\sqrt{\log n})$ — so any sparse method must be exploiting data structure, not worst-case structure.

## 4. What Is Known

- **Attention mass is empirically concentrated.** H2O (Zhang et al., NeurIPS 2023) shows >95% of attention mass in accumulated-score terms sits on ~20% of tokens at 7B scale (OPT/LLaMA-class), with 20% heavy-hitter + 20% recent giving near-lossless generation on their suite.
- **Sparsity scales with sequence length, not with model quality.** The Sparse Frontier (Nawrot et al., 2025; arXiv:2504.17768), isoFLOP study over Qwen-2.5 7B–72B and up to 128k: at fixed FLOPs, a larger highly-sparse model beats a smaller dense one for long sequences. Two further findings matter here: (a) **no single sparse pattern wins across tasks** — the best method is task-dependent; (b) even at densities where *average* accuracy is preserved, at least one task degrades significantly. Independent, multi-model, and the most reliable regularity in the area.
- **Learned patterns beat naive fixed ones at fixed density in-domain.** Routing Transformer (Roy et al., TACL 2021) and Reformer (Kitaev et al., ICLR 2020) both improve over local+strided at matched budget on autoregressive image/text density modelling.
- **LRA rankings do not transfer.** Long Range Arena (Tay et al., ICLR 2021) rankings of efficient attention variants correlate poorly with downstream LLM long-context quality; the benchmark is now used mainly as a negative control.
- **Effective context ≪ claimed context.** RULER (Hsieh et al., COLM 2024): most models advertising 32k hold well below it; this floor applies to dense baselines, so sparse-vs-dense comparisons above it measure noise.
- **Sinks are structural.** Removing the first few tokens collapses perplexity by orders of magnitude regardless of content (Xiao et al., ICLR 2024) at 7B–70B.

## 5. What Is Not Known

- **Theoretically open.** No separation theorem showing a *learnable* content-based pattern of density $s$ realizes a function class that no static pattern of density $s$ can, under any realistic training assumption. The Sanford et al. separations are representational, not about what SGD finds.
- **Theoretically open.** No sample-complexity or optimization result for discrete top-$k$ selectors: whether straight-through gradients on $\Pi_k$ converge to a good pattern, or merely to whatever pattern the initialization biased toward.
- **Empirically open.** Whether NSA-style pretrained sparsity holds parity at $\geq$100B params and $\geq$1T tokens, or whether the gap opens with scale. Runnable; nobody has published it. Cost is the reason.
- **Empirically open.** The density-matched, budget-matched static control for NSA/MoBA. Cheap relative to the original runs. Unrun.
- **Methodologically blocked.** There is no accepted metric for "the selector learned something." Attention-mass recovery is the default and is provably insensitive to exactly the failure mode that matters (§10). Until a retrieval-sensitive selector metric exists, "learnability" claims are not falsifiable.

## 6. Why It Is Hard

**The primary obstruction is confounded measurement.** A learned-sparse model differs from its static control in at least four ways simultaneously: pattern, selector parameter count, training FLOPs spent on the selector, and — because the pattern shapes gradients — the *optimization trajectory of the whole model*. Attributing a benchmark delta to "learning the pattern" requires holding three of these fixed, which no published comparison does.

**Secondary: the proxy metric is insensitive by construction.** Attention-mass recovery is an $\ell_1$ statistic over a distribution whose tail carries the retrieval signal. A block holding 0.8% of the mass can hold 100% of the answer.

**Secondary: non-identifiability from sinks.** A selector that learns "always take block 0 and the last 4 blocks" reproduces most of the mass-recovery number of a sophisticated selector, so the two are indistinguishable under the standard metric.

**Compute is a real but not the binding constraint.** The decisive control experiments are 100× cheaper than the flagship runs that already happened.

## 7. Current Research (as of 2026)

- **Natively-trained sparse pretraining.** DeepSeek (NSA) and Moonshot (MoBA) are the reference lines; both are production-motivated. Follow-on work on making block selection hardware-aligned under GQA continues *(frontier — verify current variants)*.
- **Post-hoc selector distillation.** SeerAttention (Gao et al., 2024; arXiv:2410.13276) learns a lightweight gate against the dense model's own block-max attention as ground truth — the cleanest existing formulation of "learn the pattern," because it has a supervised target.
- **Per-head budget allocation.** MoA (Fu et al., 2024), FastGen (ICLR 2024): treat density as a search variable, not a hyperparameter.
- **Trade-off characterization.** Nawrot, Ponti et al. (Edinburgh/Cohere) on isoFLOP sparse frontiers — the group most likely to run the controls in §8.
- **Benchmark reform.** RULER and successors pushing evaluations that are retrieval-sensitive rather than perplexity-sensitive.

## 8. Concrete Next Experiment

**Question.** Does a learned selector recover the *oracle* pattern's task accuracy, or merely the *static* pattern's?

**Scale.** 1.3B dense-equivalent transformer, 100B tokens, 32k train context, block size $B=64$, held at density $s = 0.05$ for every arm. Roughly 3k A100-hours per arm; four arms.

**Arms.**
1. **Dense** — upper reference, $s=1$.
2. **Static control** — sink blocks + local window + fixed stride, budget-tuned to exactly $s=0.05$.
3. **Learned** — NSA/MoBA-style top-$k$ block gate, $k$ set so $\mathbb{E}[s]=0.05$.
4. **Oracle** — at each step, blocks chosen by the *dense* arm's true block-max attention (teacher forcing; not deployable, defines the ceiling of pattern choice at this density).

Arms 2–4 share identical parameter count (selector params added as dead weight to 2 and 4) and identical training FLOPs.

**Decision number.** On RULER at 32k (multi-key needle and variable tracking subsets — the retrieval-sensitive half), compute the **recovery fraction**

$$\rho = \frac{\mathrm{Acc}(\text{learned}) - \mathrm{Acc}(\text{static})}{\mathrm{Acc}(\text{oracle}) - \mathrm{Acc}(\text{static})}$$

- $\rho > 0.8$: the selector learns a genuinely content-dependent pattern. Learnability is established at this scale.
- $\rho < 0.2$: the learned gate is a re-parameterized static pattern; all reported gains are budget tuning.
- Denominator $< 5$ accuracy points: the *task* cannot discriminate patterns at this density — report as methodologically blocked and redesign the eval, do not report $\rho$.

Report $\rho$ with a bootstrap CI over 3 seeds. Also report attention-mass recovery for arms 2–4; the expected finding is that it fails to order them.

## 9. Key References

- **[Foundational]** Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever. *Generating Long Sequences with Sparse Transformers.* 2019. — arXiv:1904.10509
- **[Foundational]** Manzil Zaheer et al. *Big Bird: Transformers for Longer Sequences.* NeurIPS, 2020. — arXiv:2007.14062
- **[Foundational]** Iz Beltagy, Matthew E. Peters, Arman Cohan. *Longformer: The Long-Document Transformer.* 2020. — arXiv:2004.05150
- **[Foundational]** Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya. *Reformer: The Efficient Transformer.* ICLR, 2020. — arXiv:2001.04451
- **[Foundational]** Aurko Roy, Mohammad Saffar, Ashish Vaswani, David Grangier. *Efficient Content-Based Sparse Attention with Routing Transformers.* TACL, 2021. — arXiv:2003.05997
- **[SOTA]** Jingyang Yuan et al. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.* ACL, 2025. — arXiv:2502.11089
- **[SOTA]** Enzhe Lu et al. *MoBA: Mixture of Block Attention for Long-Context LLMs.* 2025. — arXiv:2502.13189
- **[SOTA]** Huiqiang Jiang et al. *MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention.* NeurIPS, 2024. — arXiv:2407.02490
- **[SOTA]** Yizhao Gao et al. *SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs.* 2024. — arXiv:2410.13276
- **[Empirical]** Piotr Nawrot et al. *The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs.* 2025. — arXiv:2504.17768
- **[Empirical]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[Empirical]** Zhenyu Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS, 2023. — arXiv:2306.14048
- **[Theory]** Clayton Sanford, Daniel Hsu, Matus Telgarsky. *Representational Strengths and Limitations of Transformers.* NeurIPS, 2023. — arXiv:2306.02896
- **[Theory]** Josh Alman, Zhao Song. *Fast Attention Requires Bounded Entries.* NeurIPS, 2023. — arXiv:2302.13214
- **[Eval]** Cheng-Ping Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Eval]** Yi Tay et al. *Long Range Arena: A Benchmark for Efficient Transformers.* ICLR, 2021. — arXiv:2011.04006
- **[Survey]** Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler. *Efficient Transformers: A Survey.* ACM Computing Surveys, 2022. — arXiv:2009.06732

## 10. Worked Example

**Setup.** $N = 32{,}768$, block size $B = 64$ → 512 blocks. Selector keeps $k = 16$ blocks per query, $s = 16/512 = 3.1\%$. Task: a single needle in one block; the model must copy a 20-token answer.

**Mass accounting.** In a 7B-class model at 32k, the answer block typically carries about **0.8%** of a retrieval head's attention mass; sinks plus the local window carry roughly **92%**.

Two selectors:

| | picks needle block | attention-mass recovery $R$ |
|---|---|---|
| Static (sinks + local + stride) | never | $0.920$ |
| Learned, per-query miss rate $\epsilon = 0.02$ | 98% of steps | $0.920 + 0.98\times0.008 = 0.928$ |

**The gap the metric shows:** $0.928 - 0.920 = 0.008$, i.e. **0.8 points of recovery** — well inside run-to-run variance, and smaller than the difference you get by changing the local window by one block.

**The gap the task shows:** the answer needs the block present at all 20 decode steps.
$$P(\text{correct}) = (1-\epsilon)^{20} = 0.98^{20} = 0.668$$
Static: $0$. So **66.8 accuracy points** separate the two selectors while the standard metric separates them by **0.8**.

**Sensitivity, the part that bites.** Push the selector from $\epsilon = 0.02$ to $\epsilon = 0.10$ — a fivefold worse selector. Recovery moves from $0.9278$ to $0.9272$ (0.06 points, unmeasurable). Accuracy moves from $0.668$ to $0.9^{20} = 0.122$ — a **55-point** collapse.

**The obstruction, made visible.** Attention-mass recovery is a monotone function of selector quality with a derivative near zero in exactly the regime that decides the task, because the discriminating mass lives in the tail. Any paper reporting $R \approx 0.93$ for its learned selector has reported a number that is compatible with both a working retriever and a broken one. This is why §5 classifies the core gap as *methodologically blocked* rather than merely empirically open: the field's default measurement cannot tell the two apart, and §8's $\rho$ is built to route around it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*