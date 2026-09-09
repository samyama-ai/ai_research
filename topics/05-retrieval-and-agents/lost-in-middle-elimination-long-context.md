---
id: 05-retrieval-and-agents/lost-in-middle-elimination-long-context
title: "Lost-in-the-Middle Elimination in Long-Context Readers"
topic: 05-retrieval-and-agents
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Lost-in-the-Middle Elimination in Long-Context Readers

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/lost-in-middle-elimination-long-context` · **Status:** open

## 1. Problem Statement

A long-context reader is given a context $c$ containing an evidence span $e$ at some position, plus a query $q$. **Position bias** is the dependence of task accuracy on *where* $e$ sits in $c$, holding the content of $c$ fixed. The "lost-in-the-middle" pattern is the specific U-shape: high accuracy when $e$ is near the start or the end, a trough in the middle (Liu et al., TACL 2024).

The problem: **build a reader whose accuracy is invariant to evidence position, at fixed context length, without losing accuracy at the favoured positions.**

Three variants, routinely conflated:

- **Measurement.** Define a positional-invariance statistic that is not an artifact of the probe. Synthetic needle-in-a-haystack (NIAH) probes saturate at 100% on frontier models and report zero bias while real multi-document QA still degrades — so the measurement variant is not settled.
- **Method.** Produce a reader (architecture, positional encoding, decoding, or training recipe) with a measured max–min positional gap below a stated $\epsilon$ on a non-saturated benchmark. Cheap workarounds exist — reorder documents so evidence lands at the edges — but they assume a relevance signal that is the retrieval problem itself.
- **Theory.** Prove whether position invariance is achievable by a causal decoder with any position-dependent encoding, or whether causal masking forces a nonuniform prior. No such theorem exists in either direction.

Solving it means: for a family of tasks with controlled evidence placement, $\max_p \mathrm{Acc}(p) - \min_p \mathrm{Acc}(p) \le 2$ points at $128$K tokens, with mean accuracy no worse than the unmodified baseline's *best* position.

## 2. Formal Setting

Let the context be a token sequence $c = (t_1,\dots,t_n)$ built from $m$ distractor units $d_1,\dots,d_{m-1}$ and one gold unit $e$, inserted at slot index $p \in \{1,\dots,m\}$. Write $c_p$ for the context with $e$ at slot $p$. The reader is $f_\theta$, and correctness is a task-specific predicate $\mathbb{1}[f_\theta(q, c_p) \models a]$.

**Positional accuracy profile**, measured over $N$ query instances (not one query with shuffled slots):

$$\mathrm{Acc}(p) \;=\; \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[f_\theta(q_i, c_p^{(i)}) \models a_i\right].$$

**Positional gap** $\Delta = \max_p \mathrm{Acc}(p) - \min_p \mathrm{Acc}(p)$, with Wilson binomial confidence intervals; $N = 500$ per slot gives a half-width of about $\pm 4.4$ points at $\mathrm{Acc}=0.5$, so $\Delta$ below $\sim 6$ points is unresolvable at that $N$. This is the reason most reported "elimination" results are underpowered.

**Attention mass on evidence** at layer $\ell$, head $h$, decode step $s$:

$$A^{(\ell,h)}_s(e) \;=\; \sum_{j \in \mathrm{idx}(e)} \alpha^{(\ell,h)}_{s,j}, \qquad \sum_j \alpha^{(\ell,h)}_{s,j} = 1 .$$

Averaging $A$ over retrieval heads (Wu et al., ICLR 2025) gives a mechanistic correlate of $\mathrm{Acc}(p)$, but it is a correlate, not the quantity of interest.

**Length control.** $n$ must be held constant across $p$ to the token, otherwise $\Delta$ mixes position bias with length degradation (Levy et al., ACL 2024, show accuracy falls with input length even when the reasoning task is fixed).

Assumptions, and their status:

1. *Slot exchangeability* — swapping $e$ with $d_p$ leaves the context distribution unchanged. **Violated**: discourse coherence, entity first-mention, and topic drift make position-$1$ text distributionally special.
2. *Single sufficient evidence span.* **Violated** in multi-hop and aggregation tasks, where the relevant "position" is a set.
3. *Distractors are non-informative.* **Violated**: hard distractors from the same retriever partially answer the query.
4. *Position is the only varying factor.* **Violated** by tokenizer-level length jitter — different documents tokenize to different lengths, so a naive swap changes $n$ by tens of tokens.

## 3. State of the Art

**Established (reproduced, ablated):**

- The U-shape itself. Liu et al. (TACL 2024) on multi-document QA with 10/20/30 documents: accuracy drops by roughly $20$ points from best to worst position on GPT-3.5-Turbo, and closed-book performance can exceed mid-position performance. Replicated across model families.
- Causal masking plus positional encoding produces a systematic recency/primacy prior. Attention sinks — near-constant mass on the first tokens — are a robust, independently reproduced phenomenon (Xiao et al., ICLR 2024).
- Retrieval heads: a small, sparse, stable set of heads accounts for factual copying from context; ablating them collapses NIAH accuracy while leaving perplexity roughly intact (Wu et al., ICLR 2025).

**Claimed but under-ablated:**

- *Ms-PoE* (Zhang et al., NeurIPS 2024) — per-head rescaling of RoPE position indices, plug-and-play, reported average gains of a few points on multi-doc QA. Reduces $\Delta$; does not report a powered $\Delta \le 2$ result at $128$K.
- *PINE* (Wang et al., 2024) — bidirectional inter-segment attention with importance-based re-ordering, giving exact permutation invariance across segments. Cost is quadratic in segment count and it assumes a clean segmentation; long-context reasoning ablations are thin.
- *IN2 / FILM-7B* (An et al., 2024) — synthetic training data with evidence deliberately placed mid-context. Reported flattening of the U-shape on the authors' own probe (VaL Probing); independent replication at $\ge 128$K is not established.
- *Found in the Middle* (Hsieh et al., ACL Findings 2024) — explicit calibration of the estimated positional attention bias.

**Benchmark-number-only results:** frontier vendor reports of $>99\%$ NIAH recall at $1$M tokens (Gemini 1.5 technical report, 2024) are a benchmark number on a saturated probe, not evidence of position invariance. RULER (Hsieh et al., COLM 2024) and NoLiMa (Modarressi et al., ICML 2025) both show the gap: models near-perfect on literal NIAH fall sharply once matching is non-literal.

## 4. What Is Known

- **Magnitude, at 4K–16K contexts.** Liu et al.: 20-document multi-doc QA, GPT-3.5-Turbo, accuracy about $75\%$ at position 1 versus about $54\%$ mid-context — a $\Delta$ near $20$ points at $N \approx 2{,}600$ queries.
- **Effective context $\ll$ claimed context.** RULER, 13 tasks, models claiming $32$K+: only a handful hold performance to their advertised length; several degrade below their $4$K baseline at $32$K.
- **Non-literal matching collapses first.** NoLiMa: of 12 models scoring $\ge 90\%$ on short contexts, 10 fall below $50\%$ of that baseline at $32$K.
- **Length hurts independently of position.** FLenQA (Levy et al., ACL 2024): identical reasoning task padded from $\sim250$ to $\sim3{,}000$ tokens costs many models over $20$ points, with padding that carries no information.
- **Reordering works when relevance is known.** Attention sorting (Peysakhovich & Lerer, 2023): iteratively moving high-attention documents to the end raises accuracy — an upper bound available only with an oracle-ish relevance signal.
- **Theory of the prior.** Representational collapse and over-squashing in decoder-only transformers give a mechanism for late-token information loss (Barbero et al., NeurIPS 2024); RoPE's frequency structure explains part of the local/recency preference (Barbero et al., ICLR 2025).

## 5. What Is Not Known

- **Theoretically open.** No separation theorem. Nobody has proved that a causal decoder with fixed parameter count cannot achieve $\Delta = 0$ on a non-trivial task family, nor exhibited a construction that does. The candidate obstruction — causal masking makes token $j$'s representation depend on a prefix of size $j$, so the *computational budget* per position is inherently unequal — has not been turned into a lower bound.
- **Empirically open.** Whether the position-debiasing methods of 2024–2025 (Ms-PoE, PINE, IN2-style training, calibration) survive at $128$K–$1$M on non-saturating tasks with $N$ large enough to resolve $\Delta \le 5$. The experiment is runnable today for well under $\$50$K of inference; nobody has published it as a matched comparison.
- **Methodologically blocked.** A position-bias metric that is not confounded by content. Moving $e$ changes discourse, entity introduction order, and token count simultaneously. There is no accepted construction of a *content-identical, position-varied* context set for realistic tasks — only for synthetic needles, which saturate.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by benchmark saturation**.

- Every realistic instantiation of "same context, different position" changes something else: token count, coherence, or the distractor set. So the estimand $\Delta$ is not identified from the observed accuracy difference — it is a mixture of position effect, length effect, and coherence effect.
- The one construction that *is* clean — inserting a synthetic needle into filler — has been optimized against to the point of saturation. A model at $100\%$ NIAH has $\Delta = 0$ by construction and tells you nothing. So the clean measurement has no resolving power and the powerful measurement is not clean.
- Consequence: a method can flatten $\Delta$ by lowering the peak rather than raising the trough, and current reporting (average accuracy, or $\Delta$ alone) cannot distinguish the two. This is the reason to require *both* $\Delta \le \epsilon$ *and* mean accuracy $\ge$ baseline best-position accuracy.

## 7. Current Research (as of 2026)

- **Positional-encoding surgery.** Per-head RoPE index rescaling and frequency-band editing, following Ms-PoE and the YaRN/Position-Interpolation line. Groups: UT Austin, NVIDIA, academic-industry mixes.
- **Permutation-invariant attention over segments.** PINE-style bidirectional inter-document attention; open question is whether invariance survives when segments must be compared, not just read *(frontier — verify)*.
- **Mechanistic targeting of retrieval heads.** Direct up-weighting or fine-tuning of the sparse retrieval-head set to remove positional preference *(frontier — verify)*.
- **Data-side fixes.** IN2-style mid-context supervision at pretraining scale rather than fine-tune scale.
- **Evaluation.** HELMET (Yen et al., ICLR 2025), NoLiMa, and NoCha (Karpinska et al., EMNLP 2024) as non-saturating substrates; the position-controlled variants of these are the missing piece.
- **Agentic sidestep.** Context compaction, sub-agent sharding, and re-retrieval in agent frameworks avoid the problem rather than solve it — and shift the failure to the compaction policy.

## 8. Concrete Next Experiment

**Position-controlled NoLiMa-style QA at 128K with matched token counts.**

- **Scale.** One open-weight model at $\sim70$B with a $128$K window, plus one frontier API model. Context lengths $\{8\text{K}, 32\text{K}, 128\text{K}\}$. Nine evidence slots at relative depths $\{0, 0.125, \dots, 1.0\}$. $N = 800$ queries per (slot, length) cell $\Rightarrow$ $9 \times 3 \times 800 = 21{,}600$ calls per arm. Half-width at $\mathrm{Acc}=0.5$ is $\pm 3.5$ points, resolving $\Delta \ge 5$.
- **Length control.** Pad each candidate slot's distractor to the gold unit's exact token count, so swapping $e$ into slot $p$ leaves $n$ bit-identical. Report the padding overhead.
- **Control arms.** (a) Unmodified model. (b) Oracle reorder — gold at position 1 — as the ceiling. (c) Shuffled-distractor placebo: move a *non-gold* unit through the same slots; its accuracy profile must be flat, or the harness itself is biased.
- **Treatment arms.** Ms-PoE; PINE; an IN2-style fine-tune of the open model; attention-sorting reorder.
- **Deciding number.** $\Delta_{128\text{K}}$ for each arm, paired with $\overline{\mathrm{Acc}}$ against arm (b). A method counts as solving the problem iff $\Delta_{128\text{K}} \le 2$ points **and** $\overline{\mathrm{Acc}} \ge \mathrm{Acc}_{\text{baseline}}(p{=}1) - 1$ point. Any method that hits the first condition alone has flattened by degrading the peak.

## 9. Key References

- **[Foundational]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Foundational]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[SOTA]** Zhenyu Zhang, Runjin Chen, Shiwei Liu, Zhewei Yao, Olatunji Ruwase, Beidi Chen, Xiaoxia Wu, Zhangyang Wang. *Found in the Middle: How Language Models Use Long Contexts Better via Plug-and-Play Positional Encoding.* NeurIPS, 2024. — arXiv:2403.04797
- **[SOTA]** Ziqi Wang, Hanlin Zhang, Xiner Li, Kuan-Hao Huang, Chi Han, Shuiwang Ji, Sham Kakade, Hao Peng, Heng Ji. *Eliminating Position Bias of Language Models: A Mechanistic Approach.* 2024. — arXiv:2407.01100
- **[SOTA]** Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, Weizhu Chen. *Make Your LLM Fully Utilize the Context.* 2024. — arXiv:2404.16811
- **[Mechanism]** Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, Yao Fu. *Retrieval Head Mechanistically Explains Long-Context Factuality.* ICLR, 2025. — arXiv:2404.15574
- **[Theory]** Federico Barbero, Andrea Banino, Steven Kapturowski, Dharshan Kumaran, João Madeira Araújo, Alex Vitvitskyi, Razvan Pascanu, Petar Veličković. *Transformers Need Glasses! Information Over-squashing in Language Tasks.* NeurIPS, 2024. — arXiv:2406.04267
- **[Evaluation]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Evaluation]** Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A. Rossi, Seunghyun Yoon, Hinrich Schütze. *NoLiMa: Long-Context Evaluation Beyond Literal Matching.* ICML, 2025. — arXiv:2502.05167
- **[Evaluation]** Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, Danqi Chen. *HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly.* ICLR, 2025. — arXiv:2410.02694
- **[Survey]** Mo Li, Songyang Zhang, Yunxin Liu, Kai Chen. *NeedleBench: Can LLMs Do Retrieval and Reasoning in Information-Dense Contexts?* 2024. — arXiv:2407.11963

## 10. Worked Example

Take 20 Wikipedia paragraphs, one gold, $n \approx 4{,}000$ tokens, and a reader whose profile matches the Liu et al. shape: $\mathrm{Acc}(1)=0.75$, $\mathrm{Acc}(10)=0.54$, $\mathrm{Acc}(20)=0.63$. So $\Delta = 21$ points.

Now apply a debiasing method and observe the new profile $\mathrm{Acc}'(1)=0.64$, $\mathrm{Acc}'(10)=0.60$, $\mathrm{Acc}'(20)=0.62$. Reported as "$\Delta$ reduced from $21$ to $4$" — a $5\times$ improvement, and mean accuracy is up: $0.62$ versus $0.61$.

But the peak fell $11$ points. On a deployment where the retriever already ranks the gold document first — the normal case — the method costs $11$ points of accuracy. It flattened the curve by breaking primacy, not by fixing the middle. The headline number and the deployed number move in opposite directions.

Now the statistics. At $N=200$ per slot, the Wilson half-width at $\mathrm{Acc}=0.6$ is about $\pm 6.8$ points. The observed $\Delta' = 4$ is inside the noise: it is not distinguishable from $\Delta' = 0$, and also not distinguishable from $\Delta' = 10$. Most published position-bias ablations report $N$ in the low hundreds per slot.

Two failures stack. The metric cannot tell "fixed the trough" from "destroyed the peak," and the sample size cannot tell either from noise. This is why the deciding criterion in §8 is a conjunction — $\Delta \le 2$ **and** mean accuracy $\ge$ baseline best-position — measured at $N=800$ per cell. Neither half is sufficient alone, and no published result satisfies both at $128$K.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*