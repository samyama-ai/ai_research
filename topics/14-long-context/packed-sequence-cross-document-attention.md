---
id: 14-long-context/packed-sequence-cross-document-attention
title: "Cross-Document Attention Contamination in Packed Training"
topic: 14-long-context
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Document Attention Contamination in Packed Training

> **Topic:** Long Context · **ID:** `14-long-context/packed-sequence-cross-document-attention` · **Status:** empirically-open

## 1. Problem Statement

Pretraining corpora are shipped to the GPU as fixed-length sequences. Documents are shorter than the context window, so they are concatenated end to end and cut at the window boundary — "packing". Under a plain causal mask, a token in document $j$ can attend to every token of documents $1..j-1$ that share its packed sequence. Those documents are unrelated by construction (the shuffle is random). This is **cross-document attention contamination**.

Three variants, with different difficulty:

- **Measurement.** Given a trained model, quantify how much of its attention mass, and how much of its loss, is attributable to out-of-document context. No agreed estimator exists.
- **Method.** Choose the packing/masking policy — plain causal, block-diagonal intra-document masking, best-fit packing, retrieval-ordered packing — that maximises downstream quality at fixed token budget and fixed wall-clock. Partially answered; the answer appears to depend on target context length.
- **Theory.** Does training under contaminated context induce a bias in the learned conditional $p_\theta(x_t \mid x_{<t})$ that persists at inference, when inference sequences are single-document? Open, no proof either way.

Solving it means: a policy $\pi^\star$ with a stated regret bound or a reproduced ablation showing it dominates alternatives at $\geq 7$B parameters and $\geq 1$T tokens, plus an estimator of contamination that predicts the downstream gap before the downstream eval is run.

## 2. Formal Setting

Corpus $\mathcal{D} = \{d_1,\dots,d_N\}$, document $d_i$ of length $n_i$ tokens. A packing policy $\pi$ produces sequences $x^{(1..M)} \in \mathcal{V}^L$ of fixed length $L$ (measured: `seq_len` in the dataloader, e.g. $L = 8192$). Each position $t \in [L]$ carries a **document id** $\mathrm{doc}(t) \in [N]$, materialised in practice as the cumulative-sequence-length vector `cu_seqlens` passed to a varlen attention kernel.

**Masks.** Plain causal: $m_{ts} = \mathbb{1}[s \le t]$. Intra-document (block-diagonal) causal: $m^{\mathrm{intra}}_{ts} = \mathbb{1}[s \le t]\cdot\mathbb{1}[\mathrm{doc}(s) = \mathrm{doc}(t)]$.

**Contamination (attention side).** For layer $\ell$, head $h$, post-softmax weights $A^{(\ell,h)}_{ts}$:

$$\Lambda^{(\ell,h)}(t) \;=\; \sum_{s\,:\,\mathrm{doc}(s)\neq\mathrm{doc}(t)} A^{(\ell,h)}_{ts}, \qquad \bar\Lambda \;=\; \frac{1}{LHL_{\text{layers}}}\sum_{t,h,\ell} \Lambda^{(\ell,h)}(t) .$$

Measured by running one forward pass with attention weights materialised (not FlashAttention; a reference eager-attention path) on a held-out packed batch, and summing weights outside the block diagonal. Cost: $O(L^2)$ memory per head, so it is done on $\le 100$ batches.

**Contamination (loss side).** The quantity that actually matters is the counterfactual:

$$\Delta(t) \;=\; -\log p_\theta\big(x_t \mid x_{<t}^{\text{packed}}\big) \;+\; \log p_\theta\big(x_t \mid x_{<t}^{\text{intra}}\big),$$

i.e. the same model, same weights, evaluated twice with the two masks. $\Delta(t) > 0$ means the foreign prefix *hurt*; $\Delta(t) < 0$ means it helped (it usually does slightly, via bag-of-tokens priming and attention-sink absorption).

**Training objective.** $\mathcal{L}(\theta;\pi,m) = \mathbb{E}_{x\sim\pi}\big[\tfrac{1}{L}\sum_t -\log p_\theta(x_t\mid x_{<t}, m)\big]$, at fixed token budget $T$ (measured: tokens seen, excluding padding) and fixed step time (measured: ms/step on the same hardware).

**Assumptions, and which are violated.**
1. *Documents are i.i.d. and independent.* Violated: web corpora contain near-duplicates and split articles, so "foreign" context is sometimes genuinely relevant. This is the load-bearing violation — it is why the sign of $\Delta$ is not obvious.
2. *Inference sequences are single-document.* Violated for RAG, few-shot prompting, and agent transcripts, where the prompt *is* a pack of unrelated documents. A model trained with strict intra-document masking has never seen that distribution.
3. *Masking is free.* Violated: block-diagonal attention with many short documents produces small ragged blocks, and varlen kernels lose occupancy; the throughput cost is not zero and is workload-dependent.
4. *Positional encodings restart per document.* Usually false — RoPE position ids typically run $0..L-1$ across the whole pack even when the mask is block-diagonal, so document $j$ sees a large positional offset it will never see at inference.

## 3. State of the Art

**Established.**
- Packing itself is universal since T5 (Raffel et al., JMLR 2020); the alternative, pad-to-max, wastes 30–70% of tokens on typical web-text length distributions.
- Krell et al. (2021, Graphcore) gave the first explicit "packing without cross-contamination" treatment: a bin-packing algorithm (SPFHP/NNLSHP) plus per-document masking and per-document loss normalisation, on BERT-style pretraining. Reported ~2$\times$ throughput over padded batches at unchanged downstream accuracy.
- Llama 3 (Grattafiori et al., 2024) states document masking is used and reports it as *unimportant during standard pretraining but important for continued pretraining at long context*. This is the single most-cited empirical datum and it is a report, not an ablation table with confidence intervals.

**Claimed but unablated at scale.**
- Zhao et al. (ACL 2024), *Analysing the Impact of Sequence Composition on Language Model Pre-Training*, is the closest thing to a controlled study: intra-document causal masking plus retrieval-based packing (BM25Chunk) improves in-context learning, reading comprehension and factuality. Measured at 350M–1.7B parameters, tens of billions of tokens. Not reproduced at 7B+.
- Ding et al. (ICML 2024), *Fewer Truncations Improve Language Modeling*: best-fit packing eliminates unnecessary document truncation and is reported to reduce closed-domain hallucination by up to ~58% and improve reading comprehension, at 7B/13B on ~2T tokens. Confounds truncation with contamination — the two are changed together.
- Shi et al. (ICLR 2024), *In-Context Pretraining*, goes the other way: deliberately pack **related** documents so cross-document attention is informative. Reported gains on in-context learning and reading comprehension at 7B. If contamination were purely harmful, this should not work.
- Staniszewski et al. (SPLiCe, AAAI 2025) and Pouransari et al. (Dataset Decomposition, NeurIPS 2024) each change packing composition and report long-context gains; neither isolates the mask from the composition.

**Systems SOTA.** `flash-attn` varlen (`cu_seqlens`) and FlexAttention block masks make intra-document masking approximately free for long documents and measurably lossy for short ones. No published throughput-vs-length-distribution curve.

## 4. What Is Known

- Attention sinks absorb much of the nominal cross-document mass. Xiao et al. (ICLR 2024) show the BOS/first token carries a large share of attention in every layer; Gu et al. (ICLR 2025) tie sink emergence to the packing and optimisation regime. Consequence: raw $\bar\Lambda$ overstates functional contamination, because a large fraction of the off-block mass sits on one no-op token.
- The effect is length-dependent. At $L = 2048$ with median web-document length ~600 tokens, a pack holds ~3 documents; at $L = 32768$ it holds ~50. Contamination pressure grows roughly linearly in $L/\mathbb{E}[n]$, which is exactly why Llama 3 reports it mattering only in the long-context phase.
- Truncation and contamination are separable in principle and were not separated in practice: best-fit packing (ICML 2024) reduces both, and its 58% hallucination number cannot be attributed to either alone.
- Related-document packing helps (In-Context Pretraining, 7B, ICLR 2024), which establishes that the *content* of foreign context, not its foreignness, drives the sign of the effect.
- Measured at 350M–1.7B (Zhao et al., ACL 2024): intra-document masking alone gives consistent but small gains; combining it with retrieval-ordered packing gives the larger gains. The mask is not the dominant term at that scale.

## 5. What Is Not Known

- **Empirically open (primary).** Whether intra-document masking changes downstream quality at $\ge 7$B parameters and $\ge 1$T tokens with $L \ge 32$k, holding composition, data order, and step count fixed. The experiment is runnable today on ~$10^{22}$ FLOP; nobody has published the control arm.
- **Empirically open.** Whether models trained with strict masking are *worse* at RAG and long few-shot prompting — the inference-time distribution where unrelated documents share a window. Never measured as an ablation; it is the natural cost of the fix.
- **Methodologically blocked.** There is no accepted contamination estimator. $\bar\Lambda$ is confounded by sinks; $\Delta(t)$ is a post-hoc counterfactual on a model already trained one way, so it does not measure what training under the other policy would produce. No estimator is known to predict the downstream gap.
- **Theoretically open.** No result characterising the bias of $p_\theta$ trained under a mixture of clean and contaminated contexts, and no identifiability statement separating "the model learned to ignore foreign context" from "the model learned a document-boundary detector that generalises to inference".
- **Open.** The interaction with position ids: whether restarting RoPE positions per document, resetting nothing, or both, is correct. Nobody has run the 2$\times$2.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by an untestable counterfactual at scale**.

- Every published intervention changes at least two things at once: the mask, the document composition, the truncation rate, and often the sequence-length distribution. The gains are 0.5–2 points on noisy benchmarks whose seed-to-seed variance at 1.7B is of the same order.
- The clean experiment requires two pretraining runs differing in one boolean. At 7B/1T tokens that is ~$8 \times 10^{22}$ FLOP per arm, roughly 50k H100-hours — affordable to five labs, none of whom get a paper out of a null result.
- Attention-mass metrics do not measure what they name. $\bar\Lambda = 0.15$ tells you nothing if 0.13 of it sits on the sink token, and sink location is itself a function of the packing policy being compared.
- The effect is plausibly *negative-then-positive*: contamination hurts token prediction and simultaneously trains robustness to distractors, which is precisely the RULER/lost-in-the-middle skill. A single scalar cannot separate them.

## 7. Current Research (as of 2026)

- Intra-document masking is now the default in most open recipes (OLMo, Llama 3-lineage, Qwen-lineage) — adopted on the strength of the Llama 3 report rather than a public ablation.
- Composition-aware packing (SPLiCe, In-Context Pretraining, dataset decomposition) is the active line: the question has shifted from "mask or not" to "what should share a window". EleutherAI, Allen AI, Apple, Meta all publish here.
- Kernel-side, FlexAttention block masks have made arbitrary per-document masks cheap enough that the throughput argument against masking is weakening *(frontier — verify: no published occupancy benchmark across realistic length distributions)*.
- Long-context continued-pretraining recipes ($L: 8$k $\to 128$k) are where the disagreement is live, because the pack holds hundreds of documents *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** 1.4B parameters, 300B tokens, $L = 32768$, RoPE base scaled for the window, identical data order and identical step count across arms. ~$3.5\times10^{21}$ FLOP per arm; four arms fit in ~6k H100-hours.

**Arms.**
1. *Control:* plain causal mask, random packing, position ids $0..L-1$.
2. Intra-document mask, position ids $0..L-1$.
3. Intra-document mask, position ids reset per document.
4. Plain causal mask, BM25-ordered packing (composition changed, mask not).

Three seeds per arm — mandatory, because the expected effect size is near seed noise.

**Deciding number.** RULER average accuracy at 32k, arm 2 minus arm 1, with the paired-seed 95% CI. If the interval excludes zero and the gap is $\ge 3$ points, masking matters at long context and the field's default is justified. If the interval contains zero while arm 4 minus arm 1 is $\ge 3$ points, the mask is a red herring and composition is the real variable. Secondary readout, reported but not decisive: RAG accuracy on Natural Questions with 20 retrieved passages, to detect whether masking costs distractor-robustness.

## 9. Key References

- **[Foundational]** Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR, 2020. — arXiv:1910.10683
- **[Foundational]** Mario Michael Krell, Matej Kosec, Sergio P. Perez, Andrew Fitzgibbon. *Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance.* arXiv preprint, 2021. — arXiv:2107.02027
- **[SOTA]** Yu Zhao, Yuanbin Qu, Konrad Staniszewski, Szymon Tworkowski, Wei Liu, Piotr Miłoś, Yuxiang Wu, Pasquale Minervini. *Analysing the Impact of Sequence Composition on Language Model Pre-Training.* ACL, 2024. — arXiv:2402.13991
- **[SOTA]** Weijia Shi, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Gergely Szilvasy, Rich James, Xi Victoria Lin, Noah A. Smith, Luke Zettlemoyer, Scott Yih, Mike Lewis. *In-Context Pretraining: Language Modeling Beyond Document Boundaries.* ICLR, 2024. — arXiv:2310.10638
- **[SOTA]** Hantian Ding et al. *Fewer Truncations Improve Language Modeling.* ICML, 2024. — arXiv:2404.10830
- **[Systems]** Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR, 2024. — arXiv:2307.08691
- **[Context]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[Context]** Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, Min Lin. *When Attention Sink Emerges in Language Models: An Empirical View.* ICLR, 2025. — arXiv:2410.10781
- **[Eval]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Report]** Aaron Grattafiori et al. (Llama Team, Meta AI). *The Llama 3 Herd of Models.* 2024. — arXiv:2407.21783
- **[Related]** Hadi Pouransari, Chun-Liang Li, Jen-Hao Rick Chang, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, Oncel Tuzel. *Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum.* NeurIPS, 2024. — arXiv:2405.13226

## 10. Worked Example

Take $L = 8192$ and a web corpus with median document length 480 tokens, mean 1{,}100. A pack holds on average $8192/1100 \approx 7.4$ documents. For a token at position $t = 6000$ sitting in the last document, the fraction of its causal prefix that is foreign is roughly $(6000 - 550)/6000 \approx 0.91$. Naively: 91% of available context is noise.

Now measure $\bar\Lambda$ on a 1.4B model trained with a plain causal mask. Typical observation: off-block attention mass averages ~0.20 per head, of which ~0.17 lands on position 0 (the sink) and near-duplicate boilerplate. Functional foreign mass is ~0.03. The "91% of context is noise" framing collapses to a 3% effect.

Then run the counterfactual on the *same* weights: re-evaluate the held-out pack under $m^{\mathrm{intra}}$. Perplexity typically *rises* by a small amount — the model has learned to use the sink positions that masking removes, so the isolated-context score is worse. $\Delta(t) < 0$ on average.

That is the obstruction, visible in three numbers. The attention metric (0.20) says contamination is large. The sink-corrected metric (0.03) says it is small. The loss counterfactual ($\Delta < 0$) says removing it hurts — but only because the model was trained under it, so the counterfactual measures adaptation, not harm. None of the three predicts what a model trained *from scratch* under $m^{\mathrm{intra}}$ would score on RULER at 32k. Only §8's paired pretraining run does, and it costs 6k H100-hours to find out.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*