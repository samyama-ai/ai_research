---
id: 15-mixture-of-experts/moe-dense-expressivity-separation
title: "Sparse Model Expressivity Separation From Dense"
topic: 15-mixture-of-experts
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Sparse Model Expressivity Separation From Dense

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/moe-dense-expressivity-separation` · **Status:** open

## 1. Problem Statement

A sparse Mixture-of-Experts (MoE) layer holds $P$ parameters but activates only $P_a \ll P$ per token. The empirical claim that justifies MoE is a *resource separation*: for a fixed budget, sparse beats dense. The open problem is that no one has pinned down which budget the separation is real under, or what function class carries it.

Three variants, routinely conflated:

- **Measurement variant.** Given a matched-resource protocol, does an MoE reach lower loss than the best dense model at the same budget, and does the gap grow or shrink with scale? Empirically open — the sign of the gap at matched *total parameters* is not settled above ~10B.
- **Method variant.** Construct an architecture/training recipe whose advantage survives all three matched-resource controls (active parameters, total parameters, training compute) simultaneously. Currently no recipe does.
- **Theory variant.** Exhibit a function class $\mathcal{F}$ and a proof that MoEs with active width $m$ represent $\mathcal{F}$ to error $\epsilon$ while every dense network of width $\mathrm{poly}(m)$ and equal depth incurs error $\Omega(1)$ — *and* that the separation is not purely a parameter-count artifact. Theoretically open in the non-trivial (matched-total-parameter) regime.

Solving it means: naming the budget, naming the task family, and producing a number that flips sign when the budget changes.

## 2. Formal Setting

Let $x \in \mathcal{X}^T$ be a token sequence. A dense transformer block computes $y_t = W_2\,\sigma(W_1 h_t)$ with $W_1 \in \mathbb{R}^{d_{ff}\times d}$. An MoE block with $E$ experts, top-$k$ routing, router $g: \mathbb{R}^d \to \mathbb{R}^E$:

$$y_t = \sum_{e \in \mathrm{TopK}(g(h_t))} \frac{\exp g_e(h_t)}{\sum_{e' \in \mathrm{TopK}} \exp g_{e'}(h_t)}\; W_2^{(e)}\sigma\!\left(W_1^{(e)} h_t\right).$$

Measured quantities:

- **Total parameters** $P$ — count of stored floats, including router and all experts.
- **Active parameters** $P_a$ — expected floats touched per token; for top-$k$ of $E$, the FFN part is $k/E$ of the expert store. Measure by instrumenting the forward pass, not by formula: shared experts, attention, and embeddings are dense and often dominate at small $d$.
- **Sparsity factor** $s = P/P_a$.
- **Granularity** $G$ — expert hidden width relative to the dense FFN width, $G = d_{ff}^{\text{dense}}/d_{ff}^{\text{expert}}$ (Krajewski et al., 2024).
- **Training compute** $C \approx 6\,P_a\,D$ FLOPs for $D$ tokens. Measured as wall-clock FLOPs from a profiler, since routing, all-to-all, and capacity-factor drops make $6P_aD$ optimistic by 5–20%.
- **Loss** $L$ — validation cross-entropy in nats/token on a held-out corpus fixed *before* any architecture search.

The separation predicate, under budget $B \in \{P_a, P, C\}$:

$$\Delta_B(B) \;=\; \min_{\theta \in \text{dense}(B)} L(\theta) \;-\; \min_{\phi \in \text{MoE}(B)} L(\phi).$$

MoE "wins under $B$" iff $\Delta_B > 0$. The interesting question is $\lim_{B\to\infty} \Delta_B$ and its sign.

Assumptions, with the violated ones flagged:

1. *Both families are trained to their own compute-optimal token count.* **Violated** in nearly all published comparisons, which fix $D$ across arms and thereby hand the advantage to whichever family has fewer active parameters.
2. *Hyperparameters are separately tuned per arm.* **Violated** — MoE papers usually inherit the dense learning rate and batch size.
3. *Routing is a fixed function of $h_t$.* **Violated** by expert-choice and auxiliary-loss-free balancing, where a token's output depends on the rest of the batch — so the model is not a function of $x$ alone at train time.
4. *Loss is the objective.* **Violated** as a proxy for the thing people care about: reasoning benchmarks and loss dissociate for MoEs specifically (§4).

## 3. State of the Art

**Theory SOTA — established.**
- Chen, Deng, Wu, Gu, Li (NeurIPS 2022) prove a *learning* separation: for a mixture-of-classification data model with cluster structure, an MoE with nonlinear experts and gradient-trained router reaches near-zero test error while a single expert of comparable capacity is stuck near-random. The separation is about optimization and cluster structure, not raw representation power, and holds at matched *total* parameters only in a narrow constructed setting.
- Jelassi, Brandfonbrener, Kakade, Malach (*Mixture of Parrots*, ICLR 2025) give the sharpest two-sided result: on memory-like tasks MoEs match dense networks of equal *total* parameters (so sparsity is nearly free), while on a reasoning task defined via graph structure they prove MoEs need width scaling with the problem size regardless of expert count. This is the strongest evidence that the separation is task-class dependent, not architectural.
- Sanford, Hsu, Telgarsky (NeurIPS 2023) supply the communication-complexity machinery used to lower-bound what a bounded-width attention/FFN stack can compute; it has not been fully instantiated for routed layers.

**Empirical SOTA — established.** Clark et al. (ICML 2022) fit unified scaling laws over routed models to 900M dense-equivalent and $E \le 512$, and report that the routing benefit *shrinks* with model size and their fitted law predicts the advantage vanishing around ~900M–1B dense parameters. Krajewski et al. (2024) contradict the vanishing point by adding granularity $G$ as a free variable and find the MoE advantage persists and widens; both fits are over the same rough scale range.

**Claimed but unablated.** Production reports — Switch Transformer (JMLR 2022), Mixtral 8x7B (2024), DeepSeek-V3 (2024, 671B total / 37B active) — quote large speedups ("7x pretraining speedup", "matches Llama-2 70B at 13B active") against dense baselines that were trained by other groups on other data with other hyperparameters. These are benchmark numbers, not controlled ablations, and none of them controls total parameters. OLMoE (Muennighoff et al., 2024) is the closest to a clean open ablation (1.3B active / 6.9B total, 5T tokens, released data and intermediate checkpoints) but its dense control is matched on active parameters only.

## 4. What Is Known

- **Matched active parameters: MoE wins, robustly.** Switch-Base reached the quality of T5-Base in ~1/7 the training steps at equal FLOPs/token (JMLR 2022). Reproduced in form by every subsequent MoE paper. This is the *uninteresting* separation — it compares a model with $P$ stored parameters to one with $P/s$.
- **Matched total parameters: dense wins on reasoning, ties on memory.** *Mixture of Parrots* shows both theoretically and empirically that at equal total parameters, MoE memorization capacity tracks dense, while performance on compositional/graph tasks degrades as $E$ grows at fixed active width.
- **Knowledge storage is parameter-bound, not FLOP-bound.** Allen-Zhu & Li (2024) measure ~2 bits of factual knowledge per parameter for dense transformers at 10M–1B scale, and report MoE models retain most of that per *total* parameter despite sparse activation. This is the mechanism behind the memory-vs-reasoning split.
- **Granularity matters and is monotone up to a point.** Krajewski et al. (2024) fit MoE scaling laws with $G$ up to 32 at compute budgets to ~10^20 FLOPs and find optimal $G>1$ (many small experts) across the range — consistent with DeepSeek-V3's 256 fine-grained experts, top-8, plus 1 shared.
- **Routing carries real signal.** Dikkala et al. (EMNLP 2023) show learned routing beats hash/random routing by a measurable margin, ruling out "MoE is just a bigger parameter bank with random access".
- **The advantage is scale-dependent with disputed sign.** Clark et al.: shrinking, extrapolated to zero near 1B. Krajewski et al.: persistent under granularity tuning. Both fits stop at roughly the same scale; no controlled study resolves them above 10B.

## 5. What Is Not Known

- **Theoretically open.** No proof of a super-polynomial separation in favor of MoE at *matched total parameters* for any natural function class, nor a proof that none exists. The known positive results (Chen et al. 2022) are learning-theoretic under a planted cluster model; the known negative results (Jelassi et al. 2025) cover a specific reasoning family. The gap between them — most of the space — is unproven either way.
- **Empirically open.** Whether $\Delta_{P_a}$ grows or shrinks past 10B active parameters under separately-tuned, compute-optimal-token dense and MoE arms. Runnable today for ~10^22–10^23 FLOPs; nobody has published it with both arms tuned.
- **Methodologically blocked.** "Expressivity" itself. There is no agreed measurement that separates *representable* from *learnable* from *reachable by Adam under this LR schedule* for routed models. Validation loss aggregates memorization and reasoning, which move in opposite directions for MoE — so a single loss number cannot detect the known separation.

## 6. Why It Is Hard

The obstruction is **confounded measurement with no neutral budget**. Every matched-resource protocol embeds an answer:

- Match active parameters → MoE wins by construction (it stores $s\times$ more).
- Match total parameters → dense wins by construction (it activates $s\times$ more per token).
- Match training compute → the comparison depends entirely on the token budget, and the compute-optimal $D$ differs between families (Clark et al. 2022 vs. Chinchilla), so choosing one $D$ picks a winner.

There is no budget under which the two families are obviously commensurable, because they trade the two axes — memory and FLOPs — that the field usually collapses into "size". Compounding it: the aggregate metric (loss) is known to average over two sub-behaviors with opposite signs, so improving the protocol without also decomposing the metric does not help.

## 7. Current Research (as of 2026)

- **Joint memory–compute scaling laws.** Ludziejewski et al. (2025) fit MoE laws with memory as an explicit third axis rather than treating total parameters as free, which is the right shape for the problem. *(frontier — verify the exact optimal-sparsity numbers.)*
- **Task-decomposed evaluation.** Following *Mixture of Parrots*, several groups score MoE vs. dense separately on synthetic memory tasks and synthetic reasoning tasks instead of on aggregate loss (Harvard/Kempner; Google DeepMind). *(frontier — verify.)*
- **Extreme granularity and shared experts.** DeepSeek-lineage and Qwen-lineage models push $E \ge 256$ with shared always-on experts, which partially converts an MoE back into a dense model plus a sparse residual — quietly hedging the reasoning deficit.
- **Routing as computation, not selection.** Work on expert-choice, dynamic-$k$, and depth-recurrent routing asks whether adaptive *compute per token* rather than adaptive *parameters per token* is where the separation actually lives. Largely unmeasured at scale.

## 8. Concrete Next Experiment

**Question:** does the MoE advantage at matched training compute survive when both arms are separately tuned and the metric is decomposed?

- **Scale.** $C = 6\times10^{21}$ FLOPs per arm (~4 A100-months each at 40% MFU; ~8 arms total).
- **Arms.**
  1. Dense, compute-optimal $D$ per Chinchilla, LR/batch swept over 3×3.
  2. MoE at $s=8$ ($E=64$, top-8, $G=8$), same sweep, $D$ re-optimized for the MoE law.
  3. **Control arm:** dense model at the MoE's *total* parameter count, trained on the MoE's token budget — the arm almost always omitted.
- **Metric decomposition.** Three numbers per arm, not one: (a) validation loss; (b) closed-book factual recall on a held-out synthetic knowledge corpus injected at fixed rate into pretraining, scored in bits recovered per total parameter; (c) accuracy on a compositional task with controlled depth (e.g. $k$-hop graph connectivity at $k=2,3,4$).
- **The deciding number.** $\Delta_{\text{reason}} = \text{acc}_{\text{MoE}}(k{=}4) - \text{acc}_{\text{dense-}P_a}(k{=}4)$ at matched compute. If $\Delta_{\text{reason}} \le 0$ while $\Delta_{\text{loss}} > 0$, the aggregate-loss separation is a memorization artifact and the field's headline claim is mis-specified. If $\Delta_{\text{reason}} > 0.05$ absolute, the separation is genuine and extends beyond memory.

## 9. Key References

- **[Foundational]** N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, J. Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017. — arXiv:1701.06538
- **[Foundational]** W. Fedus, B. Zoph, N. Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 23, 2022. — arXiv:2101.03961
- **[SOTA — theory]** S. Jelassi, D. Brandfonbrener, S. Kakade, E. Malach. *Mixture of Parrots: Experts Improve Memorization More Than Reasoning.* ICLR 2025. — arXiv:2410.19034
- **[SOTA — theory]** Z. Chen, Y. Deng, Y. Wu, Q. Gu, Y. Li. *Towards Understanding the Mixture-of-Experts Layer in Deep Learning.* NeurIPS 2022.
- **[SOTA — scaling]** A. Clark, D. de las Casas, A. Guy, et al. *Unified Scaling Laws for Routed Language Models.* ICML 2022. — arXiv:2202.01169
- **[SOTA — scaling]** J. Krajewski, J. Ludziejewski, K. Adamczewski, et al. *Scaling Laws for Fine-Grained Mixture of Experts.* ICML 2024. — arXiv:2402.07871
- **[Supporting]** N. Dikkala, N. Parikh, et al. *On the Benefits of Learning to Route in Mixture-of-Experts Models.* EMNLP 2023.
- **[Supporting]** Z. Allen-Zhu, Y. Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* 2024. — arXiv:2404.05405
- **[Supporting]** L. Sanford, D. Hsu, M. Telgarsky. *Representational Strengths and Limitations of Transformers.* NeurIPS 2023.
- **[Systems]** N. Muennighoff, L. Soldaini, D. Groeneveld, et al. *OLMoE: Open Mixture-of-Experts Language Models.* ICLR 2025. — arXiv:2409.02060
- **[Systems]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Survey]** W. Cai, J. Jiang, F. Wang, J. Tang, S. Kim, J. Huang. *A Survey on Mixture of Experts.* 2024. — arXiv:2407.06204

## 10. Worked Example

Take DeepSeek-V3's published shape: $P = 671$B total, $P_a = 37$B active, so $s \approx 18$.

Construct the three matched-budget dense counterparts and ask which comparison the marketing claim makes.

| Budget held fixed | Dense counterpart | What it implies |
|---|---|---|
| Active params $P_a = 37$B | 37B dense | MoE stores 18× more; wins trivially |
| Total params $P = 671$B | 671B dense | Dense does 18× the FLOPs/token; wins trivially |
| Training compute $C$ | depends on $D$ | Undetermined — see below |

Now make the compute arm concrete. At $C = 6 P_a D$ with $D = 14.8$T tokens, $C \approx 3.3\times10^{24}$ FLOPs. A dense model at the same compute, trained Chinchilla-optimally ($D \approx 20P$), solves $6P \cdot 20P = 3.3\times10^{24}$ giving $P \approx 166$B dense parameters on ~3.3T tokens.

So the honest compute-matched comparison is **671B/37B MoE on 14.8T tokens vs. a 166B dense model on 3.3T tokens**. Which wins?

- On factual recall, apply the ~2 bits/parameter regularity: MoE stores $\sim 671\text{B}\times 2 = 1.34$ Tbit of headroom, dense $\sim 332$ Gbit. MoE wins by ~4×, *provided* it sees enough tokens to fill it — and at 14.8T tokens it plausibly does.
- On 4-hop compositional reasoning, *Mixture of Parrots* predicts the MoE behaves closer to a 37B-active model than a 166B one, because the bound scales with active width, not expert count. Dense should win.

Neither prediction has been measured on this pair, and no released benchmark separates the two. The obstruction is visible here: the same two models are 4× better and worse than each other depending on which sub-behavior you score, and the reported aggregate — a single MMLU or loss number — is a weighted average whose weights nobody has stated. Until the weights are fixed in advance, "MoE beats dense" is not a falsifiable claim.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*