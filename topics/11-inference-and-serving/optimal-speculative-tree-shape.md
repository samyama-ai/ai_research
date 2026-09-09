---
id: 11-inference-and-serving/optimal-speculative-tree-shape
title: "Optimal Speculative Tree Shape"
topic: 11-inference-and-serving
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Speculative Tree Shape

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/optimal-speculative-tree-shape` · **Status:** partially-solved

## 1. Problem Statement

Speculative decoding proposes candidate continuations with a cheap drafter and verifies them in one forward pass of the target model. Tree-based speculation proposes a *branching* set of candidates instead of a single chain, verified together under a block-diagonal ("tree") attention mask. The question: **given a token budget, what tree shape maximizes throughput?**

- **Input:** target model $M_t$, drafter $M_d$, current prefix $x_{<i}$, hardware cost curves, a node budget $n$.
- **Output:** a rooted tree $\mathcal{T}$ over draft tokens — depth, branching factor at each node, total size.
- **Objective:** maximize generated tokens per second, not expected accepted tokens per step.

Three variants, with different difficulty:

- **Measurement:** does a given tree actually pay off on this hardware, at this batch size? Requires a cost model for the verification pass that is fit, not assumed.
- **Method:** construct a good tree online, per request, per position, without spending more than it saves.
- **Theory:** characterize the optimal tree under a stated acceptance model, and prove the search is tractable. This is the part that is *solved* — under assumptions that are known to be false.

## 2. Formal Setting

Let $p(\cdot \mid x)$ be the target distribution and $q(\cdot \mid x)$ the drafter's. A draft tree $\mathcal{T}$ is a rooted tree whose nodes $v$ carry tokens $t_v$; the path from the root to $v$ is a candidate continuation $\pi(v)$. Write $n = |\mathcal{T}|$ and $D = \mathrm{depth}(\mathcal{T})$.

**Accepted length.** Verification (SpecInfer-style multi-round rejection sampling, or block verification) returns the longest prefix of some root path accepted under a scheme that is *lossless*: the marginal output law equals $p$. Define
$$A(\mathcal{T}) = \max\{\,|\pi(v)| : \pi(v) \text{ accepted}\,\} + 1,$$
the $+1$ being the bonus token sampled from the residual at the first rejection. **Measured as:** total generated tokens divided by number of target forward passes, over a fixed decode workload — the acceptance length $\tau$.

**Cost.** Let $c_d$ be one drafter forward pass and $c_t(n, B)$ the target forward-pass latency for $n$ draft tokens per sequence at batch size $B$. Both are wall clock on the deployed kernel, not FLOP counts. Throughput is
$$R(\mathcal{T}) = \frac{\mathbb{E}[A(\mathcal{T})]}{D\,c_d + c_t(n, B) + o(\mathcal{T})},\qquad \mathcal{T}^\star = \arg\max_{|\mathcal{T}|\le n} R(\mathcal{T}),$$
with $o(\mathcal{T})$ the tree-mask construction, KV-cache gather, and rollback overhead — measured, typically $50$–$300\,\mu s$ and not negligible at small $n$.

**The acceptance model.** Tractable optimizers assume a *position-only* acceptance vector: the $k$-th child of any node is accepted with probability $\alpha_k$, independent across nodes, where $\alpha_k$ is estimated on a calibration set. Under this model $\mathbb{E}[A(\mathcal{T})] = \sum_{v \in \mathcal{T}} \prod_{u \preceq v} \alpha_{k(u)}$ is additive over nodes, so a dynamic program over $(\text{budget}, \text{depth})$ finds the optimal shape.

For the single-chain special case with uniform acceptance $\alpha$ and $\gamma$ draft tokens, Leviathan et al. give
$$\mathbb{E}[A] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}.$$

**Assumptions known to be violated:**

1. *Position-only, context-independent $\alpha_k$.* Acceptance is strongly bimodal — near 1 inside boilerplate or code indentation, near $1/|V|$ at content words. A single $\alpha_k$ averages two regimes and is optimal for neither.
2. *Independence across siblings and across depth.* Drafter errors are correlated; one bad hidden state poisons an entire subtree.
3. *$c_t(n,B)$ flat in $n$.* True only while the target pass is memory-bound. It has a knee (§10).
4. *$B = 1$.* Nearly all published tree shapes were tuned at batch 1. Serving runs $B \gg 1$, where the knee moves by $1/B$.
5. *Stationarity.* $\alpha_k$ is calibrated offline, then used on a different serving distribution.

## 3. State of the Art

**Theory SOTA.** *Sequoia* (Chen, May, Svirschevski, Huang, Ryabinin, Jia, Chen; NeurIPS 2024) is the strongest positive result: under the position-only acceptance model it gives a dynamic program that constructs the optimal tree for a given budget, proves its sampling-without-replacement verification is *scalable* (expected accepted tokens grows without bound as tree size grows, whereas naive top-$k$ with replacement saturates) and *robust* (acceptance is monotone in drafter temperature), and adds a hardware-aware outer loop that picks $n$ and $D$ from measured latency. **Established**, but only inside its acceptance model. *SpecTr* (Sun et al., NeurIPS 2023) frames multi-draft verification as optimal transport with membership cost and gives a $\;$provably near-optimal accept scheme; *Optimal Block-Level Draft Verification* (Sun, Ro, Beirami, Suresh, 2024) proves a block-level verifier strictly dominates token-level rejection sampling in expected accepted length. Both bound the *verifier*, not the *shape*.

**Systems SOTA.** *EAGLE-2* (Li, Wei, Zhang, Zhang; EMNLP 2024) is the dominant practical answer: a static tree is replaced by a two-stage dynamic tree — expand by drafter confidence, then rerank globally and keep the top-$m$ nodes. *EAGLE-3* (2025) drops the feature-prediction constraint via training-time test and reports higher acceptance at the same shape. *OPT-Tree* (Wang et al., TACL 2025) greedily grows the tree to maximize a per-step expected-accept estimate. *SpecExec* (Svirschevski et al., NeurIPS 2024) shows that in the offloading regime, trees of $10^2$–$10^3$ tokens are worth it — the optimum is regime-dependent by two orders of magnitude.

**Claimed but unablated.** The headline speedups — SpecInfer $1.5$–$2.8\times$, Medusa-2 $2.3$–$3.6\times$, EAGLE-2 $3.05$–$4.26\times$, Sequoia up to $4.04\times$ on-GPU and $9.96\times$ offloaded — are end-to-end wall-clock numbers on batch 1, MT-Bench-like workloads. They conflate drafter quality, verifier scheme, kernel quality, and tree shape. **No published ablation isolates tree shape at fixed drafter, fixed verifier, fixed kernel, across batch sizes.** That is the gap.

## 4. What Is Known

- **Trees beat chains at batch 1.** SpecInfer (ASPLOS 2024) established this on Llama-65B/OPT-30B: multi-candidate verification raises acceptance length over single-chain speculation at nearly identical target-pass cost.
- **Diminishing returns are logarithmic-ish.** Sequoia reports acceptance length continuing to rise with tree size but with sharply falling marginal gain; measured at Llama2-7B/13B/70B on A100 and L40.
- **Dynamic beats static at fixed budget.** EAGLE-2 reports $20$–$40\%$ speedup over EAGLE's fixed tree at equal node count, on Vicuna and LLaMA2-Chat 7B/13B/70B, batch 1. Reproduced by third parties in the Spec-Bench harness.
- **Depth saturates early.** Across EAGLE-2, Medusa, and Sequoia, mean acceptance length sits at $\tau \approx 3$–$5$ tokens/step for 7B–70B targets on chat workloads; trees deeper than $6$–$8$ contribute almost nothing.
- **Batch size destroys the advantage.** MagicDec (2024) and *Decoding Speculative Decoding* (Yan, Agarwal, Venkataraman; NAACL 2025) both show speedup collapsing as $B$ grows in the short-context regime, because verification stops being free. Yan et al. report that the best draft length falls monotonically with batch size, and that many published configurations are net *losses* at serving batch sizes.

## 5. What Is Not Known

- **Theoretically open.** No optimality result for tree shape under *context-dependent* acceptance. Once $\alpha$ depends on the prefix and siblings are correlated, expected accepted length is no longer additive over nodes and the DP argument fails. Whether the problem is NP-hard under a general correlated acceptance model is unproven either way.
- **Empirically open.** The clean ablation — tree shape swept at fixed everything else, across $B \in \{1,8,32,64\}$ and context lengths $\{1\text{K}, 32\text{K}\}$ — is runnable on 8 GPUs in days and has not been published. Nor has the *oracle gap*: how much per-request adaptive shaping could gain over the best single static shape.
- **Methodologically blocked.** "Acceptance rate" is not a well-defined per-node quantity: measured acceptance at node $k$ depends on the shape that produced the sample, so calibration is endogenous. There is no agreed protocol for estimating $\alpha_k$ free of the tree that generated it.

## 6. Why It Is Hard

Three named obstructions, in order of severity.

1. **The objective is a ratio of a combinatorial quantity to a measured hardware curve.** $c_t(n,B)$ is not analytic — it has a knee where the pass switches from memory-bound to compute-bound, and the knee location depends on kernel, sequence length, and KV-cache layout. Any optimizer that assumes verification is free is optimizing the wrong numerator/denominator pair.
2. **Confounded measurement.** Reported speedups move when the drafter, the verifier, the CUDA kernel, or the batch size changes. Tree shape is the smallest of these effects, so it is systematically buried by the others in every published comparison.
3. **Endogenous calibration / non-identifiability.** $\alpha_k$ is estimated from traces produced by some tree, then used to design a tree. The fixed point is not unique and nobody checks whether the iteration converges to the same shape from different starts.

## 7. Current Research (as of 2026)

- **Dynamic trees conditioned on drafter confidence** — EAGLE line (Li et al., Peking University / Microsoft), OPT-Tree. Mature.
- **Batch-aware and throughput-oriented shaping** — MagicDec (CMU/Meta), Yan et al. (Wisconsin), plus vLLM and SGLang engine work on making the tree budget a function of the current running batch. *(frontier — verify)* Production engines appear to be converging on shrinking the tree as the batch fills, but the policy is heuristic and undocumented.
- **Better verifiers rather than better shapes** — block-level verification, optimal-transport multi-draft (Google Research).
- **Learned schedulers** that predict, from the current hidden state, how many tokens are worth drafting. *(frontier — verify)*
- **Long-context regimes**, where the target pass is dominated by KV-cache reads and large trees are again cheap — the one setting where the batch-1 conclusions may survive.

## 8. Concrete Next Experiment

**Question:** how much throughput is left on the table by using a single static tree shape, at real serving batch sizes?

- **Scale.** Llama-3.1-8B target with the released EAGLE-3 drafter, one H100 80GB, vLLM with tree attention. Workloads: MT-Bench (chat), HumanEval (code), and a 32K-context summarization set. Batch sizes $B \in \{1, 8, 32, 64\}$.
- **Sweep.** For each $B$ and workload, enumerate tree shapes from the Sequoia DP family over budgets $n \in \{1, 4, 8, 16, 32, 64, 128\}$ and depths $D \in \{2,\dots,8\}$ — roughly 40 shapes — holding drafter, verifier, and kernel fixed. Record output tokens/s and $\tau$.
- **Control arm.** Two: (a) the single best static shape per $(B, \text{workload})$ from the sweep; (b) EAGLE-2's dynamic top-$m$ tree at the same node budget.
- **Oracle arm.** Post hoc, per decode step, take the best-performing shape from the sweep for that step. This upper-bounds any online adaptive policy.
- **Deciding number.** The **oracle-minus-best-static throughput gap, in percent, at $B = 32$.** If it is under $5\%$, per-request adaptive tree shaping is not worth its overhead in serving and the practical problem closes at "pick $n$ from batch size." If it is over $20\%$, adaptivity is the open frontier and a learned shape scheduler is justified. Secondary number: the batch size at which the best static $n$ falls below 8.

## 9. Key References

- **[Foundational]** Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. — arXiv:2211.17192
- **[Foundational]** Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023. — arXiv:2302.01318
- **[Foundational]** Miao et al. *SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification.* ASPLOS 2024. — arXiv:2305.09781
- **[SOTA — theory]** Chen, May, Svirschevski, Huang, Ryabinin, Jia, Chen. *Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding.* NeurIPS 2024. — arXiv:2402.12374
- **[SOTA — systems]** Li, Wei, Zhang, Zhang. *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees.* EMNLP 2024. — arXiv:2406.16858
- **[SOTA — systems]** Li, Wei, Zhang, Zhang. *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test.* 2025. — arXiv:2503.01840
- **[Verifier]** Sun, Suresh, Ro, Beirami, Jain, Yu. *SpecTr: Fast Speculative Decoding via Optimal Transport.* NeurIPS 2023. — arXiv:2310.15141
- **[Verifier]** Sun, Ro, Beirami, Suresh. *Optimal Block-Level Draft Verification for Accelerating Speculative Decoding.* 2024.
- **[Regime]** Svirschevski, May, Chen, Chen, Jia, Ryabinin. *SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices.* NeurIPS 2024. — arXiv:2406.02532
- **[Batching]** Yan, Agarwal, Venkataraman. *Decoding Speculative Decoding.* NAACL 2025.
- **[Heads]** Cai, Li, Geng, Peng, Lee, Chen, Dao. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML 2024. — arXiv:2401.10774
- **[Survey]** Xia et al. *Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding.* Findings of ACL 2024. — arXiv:2401.07851

## 10. Worked Example

Llama-3.1-8B in bf16 on one H100 SXM. Weights $\approx 16$ GB; HBM bandwidth $\approx 3.35$ TB/s.

**Memory-bound floor per decode step:** $16 / 3350 \approx 4.8$ ms of pure weight streaming.
**Compute per token:** $2 \times 8\text{B} = 16$ GFLOP. At $\sim 50\%$ MFU of $\sim 990$ TFLOP/s bf16, that is $16 \times 10^9 / (495 \times 10^{12}) \approx 32\,\mu s$ per token.

The pass stays memory-bound while total tokens $N_{\text{tot}} = n \times B$ satisfies $32\,\mu s \cdot N_{\text{tot}} \lesssim 4.8$ ms, i.e.

$$N_{\text{tot}} \lesssim 150.$$

Now apply it.

| $B$ | free tokens/sequence | published tree size | verdict |
|---|---|---|---|
| 1 | ~150 | 64–128 (EAGLE-2, Sequoia) | fits — tree is nearly free |
| 8 | ~19 | 64 | $3\times$ over budget |
| 32 | ~4.7 | 64 | $14\times$ over budget |
| 64 | ~2.3 | 64 | $28\times$ over budget |

**Where it bites.** Take $\alpha = 0.8$ and a depth-5 chain: $\mathbb{E}[A] = (1 - 0.8^6)/0.2 = 3.69$ tokens per target pass. A 64-node tree might lift this to $\tau \approx 4.6$ — a $25\%$ gain in the numerator of $R$. At $B=1$ the denominator is unchanged, so it is a genuine $25\%$ win. At $B=32$ the tree pushes $N_{\text{tot}}$ from $32$ to $2048$, so $c_t \approx \max(4.8,\, 0.032 \times 2048) = 65.5$ ms against $4.8$ ms — a $13.6\times$ denominator increase against a $1.25\times$ numerator increase. Net: a $10\times$ **slowdown**.

**The obstruction, made visible.** Sequoia's DP is provably optimal for its objective, and its answer at $B=1$ is a large tree. The same DP with the same calibrated $\alpha_k$ gives a near-degenerate tree at $B=32$ — the shape is decided almost entirely by $c_t(n,B)$, a measured hardware curve, and almost not at all by the acceptance statistics the theory is about. The optimization is solved; the cost model it consumes is the open part.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*