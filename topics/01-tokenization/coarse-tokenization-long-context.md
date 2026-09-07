---
id: 01-tokenization/coarse-tokenization-long-context
title: "Long-Context Cost Reduction by Coarser Tokenization"
topic: 01-tokenization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Long-Context Cost Reduction by Coarser Tokenization

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/coarse-tokenization-long-context` · **Status:** empirically-open

## 1. Problem Statement

Long-context inference cost is driven by sequence length in tokens, not by information content. Attention is $O(L^2)$ in time and the KV cache is $O(L)$ in memory. Coarser tokenization — larger vocabularies, larger byte patches, or learned chunking that merges several words into one representation — shortens $L$ for the same document. The question is whether that shortening is free.

- **Measurement variant.** Given two models that segment the same corpus at different granularities, how do you compare them at all? Cross-entropy per token is not comparable across tokenizers; bits-per-byte is, but it does not price the compute the model spent. The measurement problem is defining a single quantity that jointly prices quality and cost across segmentations.
- **Method variant.** Build a model whose average bytes-per-token $b$ is $2\text{–}8\times$ that of a standard BPE tokenizer while matching a BPE baseline at equal training FLOPs *and* equal inference FLOPs on long-context tasks that require token-level precision (exact retrieval, copying, arithmetic, code edits).
- **Theory variant.** Is there a granularity–capability tradeoff theorem? I.e., does there exist a task family and a coarseness threshold $b^\star$ above which any model with a fixed per-step compute budget must incur error, independent of parameter count?

Solving it means: a coarseness setting that cuts end-to-end long-context serving cost by $\ge 2\times$ at matched downstream quality, with the ablation isolating coarseness from the confounds in §6.

## 2. Formal Setting

Let a document be a byte string $x \in \Sigma^*$, $|x| = N$ bytes. A segmenter $\tau$ maps $x$ to $T_\tau(x)$ units. Define **coarseness**

$$b_\tau = \mathbb{E}_{x\sim\mathcal{D}}\!\left[\frac{|x|}{|T_\tau(x)|}\right] \quad \text{(bytes per token, measured on a held-out corpus, not on the tokenizer's training set)}.$$

Standard BPE at 32K–128K vocabulary gives $b_\tau \approx 3.5\text{–}4.5$ on English web text; measure it, don't assume it, because $b_\tau$ drops sharply on code, non-Latin scripts, and digits.

**Quality**, measured tokenizer-independently as bits per byte:

$$\mathrm{BPB}(\theta,\tau) = -\frac{1}{N\ln 2}\sum_{i=1}^{|T_\tau(x)|}\ln p_\theta(t_i \mid t_{<i}).$$

Valid only if $\tau$ is a lossless, deterministic encoding of $x$ (violated by lossy chunkers and by any pipeline that normalizes Unicode).

**Cost.** Per generated token of output, with $d$ model width, $L$ layers, $P$ non-embedding parameters, context $S$ tokens:

$$C_{\text{infer}} \approx \underbrace{2P}_{\text{dense}} + \underbrace{4 L d S}_{\text{attention over cache}}\ \text{FLOPs}, \qquad M_{\text{KV}} = 2 L d S \cdot \text{bytes}_{\text{dtype}}.$$

For a fixed document, $S \propto 1/b_\tau$, so the attention term falls as $b_\tau^{-1}$ per step and $b_\tau^{-2}$ per document; KV memory falls as $b_\tau^{-1}$. But coarse schemes pay a **local encoder/decoder** overhead $C_{\text{local}}$ per byte, so the honest figure of merit is FLOPs per *byte*, not per token:

$$\Phi(\theta,\tau) = \frac{C_{\text{infer}}}{b_\tau} + C_{\text{local}}.$$

**The objective.** Minimize $\mathrm{BPB}$ subject to $\Phi \le \Phi_0$ and $M_{\text{KV}} \le M_0$; or, for capability, maximize downstream accuracy $A$ at fixed $\Phi_0$.

Assumptions and their status:
- *Constant $b_\tau$ across the corpus* — **violated**. $b_\tau$ has heavy variance by domain; entropy-based patchers deliberately make it input-dependent, which breaks static batching and makes throughput, not FLOPs, the binding constraint.
- *BPB comparability* — holds only under losslessness; **violated** by concept-level and latent-chunk models, whose likelihood is not a density over bytes.
- *Compute-optimality transfers across $b_\tau$* — **unverified**. Chinchilla-style token budgets are stated in tokens; changing $b_\tau$ changes the meaning of the $x$-axis.
- *Softmax cost is negligible* — **violated at large vocabulary**. A 1M-entry vocabulary costs $2 d |V|$ FLOPs per step, which at $d{=}4096$ is $\sim 8$ GFLOP — comparable to a 4B-parameter forward pass.

## 3. State of the Art

**Established (with ablations).**
- *Byte Latent Transformer* (Pagnoni et al., Meta, 2024; arXiv:2412.09871). Entropy-based dynamic byte patching, average patch $\approx 6\text{–}8$ bytes, trained to 8B parameters on 4T bytes. Reports matching Llama-3-class BPE training-FLOP-controlled scaling while using up to $\approx 50\%$ fewer inference FLOPs at fixed quality, plus improved robustness on noised and character-manipulation tasks. This is the strongest existing evidence that coarsening is not automatically lossy.
- *MegaByte* (Yu et al., NeurIPS 2023; arXiv:2305.07185). Fixed patches of 8 bytes with a local decoder; established the two-level global/local factorization at sub-1.5B scale.
- *Scaling Laws with Vocabulary* (Tao et al., NeurIPS 2024; arXiv:2407.13623). Compute-optimal vocabulary grows sublinearly with $N_{\text{params}}$; a 3B model's optimal vocabulary is $\approx 35$K, not the $\sim 200$K some frontier models ship. Direct evidence that "coarser via bigger vocabulary" saturates.
- *Dynamic Token Pooling* (Nawrot et al., ACL 2023) and *H-Net* (Hwang, Wang, Gu, 2025; arXiv:2507.07955): learned, end-to-end differentiable chunking beats fixed patching at matched compute at sub-2B scale.

**Claimed but unablated.**
- That coarse tokenization *preserves* long-context retrieval. Nearly all coarse-model papers report perplexity/BPB and short-form benchmarks; needle-in-a-haystack and long-code-edit results at $\ge 128$K context are largely absent.
- *Large Concept Models* (LCM team, Meta, 2024; arXiv:2412.08821) predict in a sentence-embedding space ($b \approx 100+$ bytes). Reported as benchmark numbers on summarization; there is no FLOP-matched comparison against a token LLM at equal quality, and the objective is not a byte likelihood, so BPB comparison is unavailable.
- Multi-token prediction (Gloeckle et al., ICML 2024) speeds decoding without changing $b_\tau$ — often conflated with coarsening; it is not.

## 4. What Is Known

- **Attention share is what coarsening buys.** At $d{=}4096$, $L{=}32$, $P{=}7$B: dense term $1.4\times10^{10}$ FLOPs/token; attention term at $S{=}128$K is $4\cdot32\cdot4096\cdot131072 \approx 6.9\times10^{10}$. Attention is $\approx 83\%$ of per-step compute. Halving $S$ cuts per-step cost $\approx 41\%$; per-document cost $\approx 70\%$.
- **KV memory scales exactly linearly.** Same model, fp16: $2\cdot32\cdot4096\cdot131072\cdot2 = 68.7$ GB. A $4\times$ coarsening yields 17.2 GB — the difference between multi-GPU and single-H100 serving.
- **Coarsening degrades character-level tasks under standard BPE.** Established at 7B–70B scale: digit-grouping choices change arithmetic accuracy by tens of points, and BPE models fail character counting that byte models pass (BLT reports large gains on CUTE-style character manipulation at 8B).
- **Vocabulary coarsening saturates.** Beyond compute-optimal $|V|$, loss worsens at fixed compute (Tao et al. 2024) — measured at 33M–1.13B parameters, extrapolated to 3B.
- **Fixed-size patching underperforms entropy-adaptive patching** at matched FLOPs — measured at $\le 8$B (BLT ablations) and $\le 1.3$B (H-Net).

## 5. What Is Not Known

- **Empirically open.** Whether the BLT-style $\approx 50\%$ inference-FLOP saving survives at $\ge 128$K context on retrieval-precise tasks. Every published coarse-model evaluation is at $\le 8$K–32K context. The experiment is runnable today on a few hundred H100-days; nobody has published it.
- **Empirically open.** Whether coarseness benefit continues past $b_\tau \approx 8$. No paper reports a controlled $b_\tau \in \{4, 8, 16, 32\}$ sweep at fixed $\Phi$ and fixed data.
- **Theoretically open.** No lower bound of the form "a two-level model with global patch length $b$ and local decoder of width $w$ cannot solve exact substring copy over a $\Theta(b)$-length span". Related expressivity results exist for one-layer attention and for chain-of-thought length, but not for the granularity axis.
- **Methodologically blocked.** Comparing latent/concept-level models to token models. Without a byte likelihood there is no common quality axis, and downstream benchmarks confound the segmentation change with the training corpus and the decoder.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a variable-length throughput cliff**.

1. Changing $b_\tau$ changes, simultaneously: sequence length, effective batch composition, the parameter count in embedding/output layers, the number of gradient updates per byte, and the position-encoding extrapolation regime. A single "coarse vs. fine" comparison moves five variables. Almost no published comparison holds all of $\{$data bytes, training FLOPs, non-embedding parameters, inference FLOPs per byte$\}$ fixed at once, because the first two and the last cannot generally be matched by one baseline — you need two control arms.
2. Adaptive coarseness makes $b_\tau$ input-dependent. Batches of variable-patch sequences pad to the worst case, so measured wall-clock savings can be far below the FLOP savings. A FLOP-only result is not a cost result.
3. Long-context evaluation frequently does not measure what it names: passkey retrieval is solvable by attention-sink shortcuts and is insensitive to exactly the token-level precision coarsening threatens.

## 7. Current Research (as of 2026)

- Meta FAIR: BLT follow-ups scaling entropy patching past 8B and into multimodal byte streams *(frontier — verify)*.
- CMU / Cartesia (Gu, Hwang): H-Net hierarchical dynamic chunking, multi-stage hierarchies, state-space local encoders.
- Vocabulary-scaling work following Tao et al., including tokenizer-free output heads (T-FREE, Deiseroth et al., EMNLP 2024) that decouple $|V|$ from the softmax cost.
- Context compression as an orthogonal coarsening axis: AutoCompressor (Chevalier et al., EMNLP 2023), In-Context Autoencoder (Ge et al., ICLR 2024) — lossy, so not BPB-comparable.
- Patch-level *training* (Shao et al., 2024) as a cheap-training-only use of coarseness.

## 8. Concrete Next Experiment

**Scale.** 1.5B non-embedding parameters, 300B training bytes, context 128K.

**Arms.** Two-level global/local architecture held fixed; sweep global-stream coarseness $b \in \{4, 8, 16, 32\}$ bytes/patch (entropy-adaptive, target mean).

**Control arms (two, both required).**
1. *Equal-quality control:* Llama-3-tokenizer BPE model ($b_\tau \approx 4.4$), same 300B bytes, same non-embedding parameters.
2. *Equal-inference-FLOP control:* the same BPE model scaled in width so its $\Phi$ at 128K matches each coarse arm's $\Phi$.

**Evaluation.** (a) BPB on held-out web, code, and a non-Latin split; (b) exact-match on 128K-context tasks that need token precision: verbatim span copy of a randomly located 200-byte string, multi-file code-edit application, and 6-digit arithmetic embedded at random depth; (c) measured tokens/s and peak KV bytes on one H100, batch 8, not just FLOPs.

**The deciding number.** $\Delta A_{\text{copy}}$ — exact-match accuracy on 128K verbatim span copy for the coarse arm minus the equal-inference-FLOP BPE control, at the largest $b$ whose measured serving throughput is $\ge 2\times$ the equal-quality control. If $\Delta A_{\text{copy}} \ge 0$ at $b \ge 16$, coarsening is close to free and the field should move; if $\Delta A_{\text{copy}} \le -5$ points, the granularity–precision tradeoff is real and the theory variant becomes the priority.

## 9. Key References

- **[Foundational]** Lili Yu, Dániel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, Mike Lewis. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.* NeurIPS, 2023. — arXiv:2305.07185
- **[SOTA]** Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* Meta AI, 2024. — arXiv:2412.09871
- **[SOTA]** Sukjun Hwang, Brandon Wang, Albert Gu. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling.* 2025. — arXiv:2507.07955
- **[Foundational]** Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, Ngai Wong. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS, 2024. — arXiv:2407.13623
- **[Foundational]** Piotr Nawrot, Jan Chorowski, Adrian Łańcucki, Edoardo M. Ponti. *Efficient Transformers with Dynamic Token Pooling.* ACL, 2023. — arXiv:2211.09761
- **[Related]** LCM team, Meta AI. *Large Concept Models: Language Modeling in a Sentence Representation Space.* 2024. — arXiv:2412.08821
- **[Related]** Alexis Chevalier, Alexander Wettig, Anirudh Ajith, Danqi Chen. *Adapting Language Models to Compress Contexts.* EMNLP, 2023. — arXiv:2305.14788
- **[Related]** Fabian Deiseroth et al. *T-FREE: Subword Tokenizer-Free Generative LLMs via Sparse Representations over Character Triplets.* EMNLP, 2024. — arXiv:2406.19223
- **[Survey]** Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting. *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation.* TACL, 2022. — arXiv:2103.06874

## 10. Worked Example

Serve a 7B-class model ($d{=}4096$, $L{=}32$) over a 500 KB document (a 12-file codebase).

| | BPE, $b_\tau{=}4.4$ | Coarse, $b{=}16$ |
|---|---|---|
| Context tokens $S$ | 113,600 | 31,250 |
| Attention FLOPs/step | $5.96\times10^{10}$ | $1.64\times10^{10}$ |
| Dense FLOPs/step | $1.4\times10^{10}$ | $1.4\times10^{10}$ |
| Local encoder/decoder | — | $\approx 0.6\times10^{10}$ |
| Total/step | $7.36\times10^{10}$ | $3.64\times10^{10}$ |
| KV cache, fp16 | 59.6 GB | 16.4 GB |

Per *step*, coarsening wins $2.0\times$ and fits on one 80 GB GPU. But each coarse step emits $16$ bytes versus $4.4$, so per byte the coarse arm costs $2.3\times10^{9}$ FLOPs against $1.67\times10^{10}$ — nominally $7.3\times$ cheaper.

Now the obstruction. The task is "reproduce line 8,412 verbatim." Those bytes live inside one 16-byte patch plus its neighbours; the global stream sees a single vector per patch, and the local decoder must reconstruct 16 exact bytes conditioned on that vector alone. The information the decoder needs is $\approx 16 \times 8 = 128$ bits of near-incompressible identifier text, and the patch embedding is trained under a next-byte objective that assigns it almost no gradient pressure — average-case BPB is dominated by predictable bytes. So BPB can *improve* while exact copy accuracy *falls*, and the $7.3\times$ number is real while the capability is not. That divergence — the metric moving opposite to the capability — is exactly why §8 makes $\Delta A_{\text{copy}}$, not BPB, the deciding number.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*