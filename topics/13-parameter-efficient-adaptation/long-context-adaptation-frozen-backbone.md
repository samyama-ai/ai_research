---
id: 13-parameter-efficient-adaptation/long-context-adaptation-frozen-backbone
title: "Long-Context Adaptation with Frozen Backbones"
topic: 13-parameter-efficient-adaptation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Long-Context Adaptation with Frozen Backbones

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/long-context-adaptation-frozen-backbone` · **Status:** empirically-open

## 1. Problem Statement

A pretrained decoder-only LM $f_\theta$ was trained at context length $L_0$ (say 4K–8K). We want it to work at $L_1 \gg L_0$ (128K–1M) while **freezing $\theta$** and training only a small adapter $\Delta$ with $|\Delta| \ll |\theta|$ — typically $|\Delta|/|\theta| < 10^{-2}$.

Three variants, routinely conflated:

- **Method.** Does there exist an adapter family (LoRA on attention projections, RoPE-rescaling parameters, side networks, learned KV compressors) that recovers full-finetuning long-context quality at a fixed adapter budget? Currently the answer is *partly*, with the gap concentrated in retrieval-and-aggregate tasks.
- **Measurement.** Is "long-context quality" even measured? Perplexity at 128K falls monotonically with context and is nearly insensitive to whether the model uses distant tokens. Needle-in-a-haystack saturates. The measurement variant asks for a metric where a model that ignores tokens beyond $L_0$ scores at chance.
- **Theory.** Is length generalization a *low-rank* update to $\theta$? If the required change in attention behaviour has rank $\Theta(d)$ per layer, no rank-8 adapter can express it, regardless of data or compute. No proof exists either way.

Solving it: an adapter with $|\Delta|/|\theta| \le 10^{-2}$ that matches full continued pretraining on RULER-style synthetic aggregation at 128K to within 2 points, without regression on short-context evals.

## 2. Formal Setting

Backbone $f_\theta: \mathcal{V}^{\le L} \to \Delta(\mathcal{V})$, pretrained by minimizing $\mathbb{E}_{x\sim \mathcal{D}_0}[-\log p_\theta(x)]$ on sequences of length $\le L_0$. Target distribution $\mathcal{D}_1$ over sequences of length $L_1$.

**Adapter.** $\Delta = \{(A_\ell, B_\ell)\}_{\ell=1}^{N} \cup \{\text{RoPE params}\}$, applied as $W_\ell \mapsto W_\ell + \frac{\alpha}{r} B_\ell A_\ell$ with $A_\ell \in \mathbb{R}^{r\times d}$, $B_\ell \in \mathbb{R}^{d\times r}$. **Measured** as: count of trainable scalars reported by the optimizer state, not nominal rank. Embedding and LayerNorm unfreezing must be counted; LongLoRA's reported "LoRA rank 8" also trains embeddings and norms, which at 7B adds $\approx 1.3\times10^{8}$ parameters — two orders of magnitude more than the LoRA matrices.

**Positional rescaling.** RoPE frequencies $\theta_i = b^{-2i/d}$. Position interpolation: $m \mapsto m/s$, $s = L_1/L_0$. YaRN: per-dimension $\theta_i \mapsto \theta_i \cdot g(\lambda_i)$ with wavelength $\lambda_i = 2\pi/\theta_i$, interpolating only dimensions whose wavelength exceeds $L_0$. Measured as: the scalar $s$ and base $b$ actually loaded at inference, which frequently differs from the training-time value.

**Effective context length.** For task family $T$ and threshold $\tau$,
$$L_{\mathrm{eff}}(f, T, \tau) = \max\{L : \mathrm{acc}_T(f, L') \ge \tau \ \ \forall L' \le L\}.$$
Measured by evaluating at a geometric ladder of lengths $L' \in \{4\text{K}, 8\text{K}, \ldots, L_1\}$ with $\ge 500$ samples per cell; $\tau$ set by a short-context reference model (RULER uses Llama-2-7B at 4K).

**Use-of-distance.** The quantity most papers claim but never measure:
$$U(f, L) = \mathbb{E}_x\big[\ell(f(x_{<L})) - \ell(f(\tilde{x}_{<L}))\big],$$
where $\tilde{x}$ corrupts tokens at distance $> L_0$ from the query. $U \approx 0$ means the long context is decorative.

**Assumptions and their violations.**
1. *Low intrinsic rank of the adaptation.* Assumed by every LoRA variant; unproven for length generalization, and contradicted by the need to unfreeze embeddings/norms in practice.
2. *$\mathcal{D}_1$ is available.* Violated — genuinely long, coherent, dependency-rich 128K documents are scarce; most long-context corpora are concatenations, for which $U \approx 0$ is the *correct* behaviour.
3. *Frozen backbone attention entropy is stable under length scaling.* Violated: softmax over $L_1$ keys has entropy growing like $\log L_1$ absent recalibration; the attention-sink phenomenon (Xiao et al., 2024) is direct evidence of pathological reallocation.
4. *Perplexity is monotone in capability.* Violated: PI-rescaled models improve perplexity at 32K while losing short-context accuracy.

## 3. State of the Art

**Established (ablated, independently reproduced).**
- **Position Interpolation** (Chen et al., 2023): linear position rescaling plus 1000 steps of fine-tuning extends Llama 7B–65B to 32K. The core claim — that interpolation is vastly more stable than extrapolation — is reproduced everywhere.
- **YaRN** (Peng, Quesnelle, Fan, Shippole, ICLR 2024): NTK-by-parts rescaling plus attention temperature; reported to reach PI-level quality with ~10× fewer tokens and 2.5× fewer steps. Widely reproduced in open-weight releases.
- **StreamingLLM** (Xiao et al., ICLR 2024): keeping 4 initial "sink" tokens plus a sliding window gives stable perplexity to 4M tokens with *zero* training. Established, and established to *not* extend $L_{\mathrm{eff}}$ — it is a streaming fix, not a long-context fix.

**Claimed but under-ablated.**
- **LongLoRA** (Chen et al., ICLR 2024): shifted sparse attention ($S^2$-Attn) plus LoRA plus trainable embeddings/norms extends Llama-2 7B to 100K and 70B to 32K on one 8×A100 node. The parameter-efficiency claim is not cleanly ablated: the embedding/norm unfreezing is load-bearing, and the paper's own ablation shows LoRA-only fails. What fraction of the gain comes from $S^2$-Attn versus the unfrozen norms is not resolved.
- **LongRoPE** (Ding et al., ICML 2024): evolutionary search over per-dimension rescaling factors, claimed to 2048K. Reported largely as perplexity and needle retrieval.
- **Benchmark-number-only results.** Most 128K+ claims in model cards are a needle-in-a-haystack heatmap. RULER (Hsieh et al., COLM 2024) showed this is not evidence: of 10 models advertising $\ge$ 32K, most had $L_{\mathrm{eff}}$ well below the advertised length.

**Full-finetuning SOTA (the control arm).** Xiong et al. (NAACL 2024) continued-pretrain Llama-2 to 32K with 400B tokens; Fu et al. (ICML 2024) reach 128K with ~500M tokens of *domain-balanced* upsampled long data — the key ablated finding being that data mixture, not volume, carries the gain.

## 4. What Is Known

- **Numbers, 7B scale.** PI: Llama-7B, 32K, 1000 steps at batch 64. LongLoRA: Llama-2-7B to 100K, 13B to 64K, 70B to 32K, all on 8×A100, LoRA $r=8$ + embeddings + norms.
- **Data beats volume.** Fu et al. (ICML 2024): 128K capability from ~500M tokens with per-domain length upsampling, at 7B — three orders of magnitude below Xiong et al.'s 400B.
- **LoRA learns less and forgets less** (Biderman et al., TMLR 2024): on continued pretraining at 7B/13B, LoRA underperforms full finetuning on code and math by large margins but preserves base-model behaviour better. Directly relevant: long-context adaptation *is* continued pretraining.
- **Effective $<$ advertised.** RULER, 2024: needle-retrieval accuracy near 100% coexists with multi-hop/aggregation accuracy collapse at the same length.
- **Position bias is not fixed by extension.** "Lost in the Middle" (Liu et al., TACL 2024): U-shaped accuracy over document position persists in extended models at 4K–20K.
- **Attention sinks are structural.** Xiao et al., 2024: removing the first few tokens' KV destroys perplexity at any length; sinks are an artifact of softmax normalization, not semantics.

## 5. What Is Not Known

- **Theoretically open.** Whether the parameter update required for length generalization is low-rank. No lower bound of the form "any rank-$r$ adapter with $r < g(L_1/L_0)$ cannot represent correct attention rescaling at length $L_1$" exists, nor an upper bound. The associated question — whether RoPE rescaling is expressible as a rank-1 update to $W_Q, W_K$ — is answerable and unanswered.
- **Empirically open.** The clean head-to-head at matched token budget: frozen backbone + rank-$r$ adapter versus full continued pretraining, same data mixture, same 500M tokens, evaluated on RULER at 7B and 70B. Runnable today on ~2k A100-hours. Nobody has published it with a matched control.
- **Empirically open.** Whether adapter rank must scale with $\log(L_1/L_0)$, $(L_1/L_0)$, or not at all. No rank sweep at fixed data exists past 128K.
- **Methodologically blocked.** $U(f,L)$ — use-of-distance — has no standard estimator. Corrupting distant tokens changes the input distribution, so the delta confounds "used the information" with "was surprised by nonsense". Until this is settled, "the model uses its 128K context" is not a measurable claim.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by an absent control arm**.

Every long-context result mixes four interventions: positional rescaling, attention sparsification, data mixture, and adapter parameters. Papers report the bundle. Position rescaling alone is training-free and provides most of the perplexity gain, so any bundle including it shows a large number regardless of whether the adapter contributed. Perplexity is the wrong readout because a model that attends only to the last 4K still improves its next-token loss when the prefix grows (more in-domain conditioning, shorter effective document). Retrieval benchmarks are the wrong readout because a single-key lookup is solvable by an induction head that needs no length adaptation at all.

The secondary obstruction is cost asymmetry. The control arm — full continued pretraining at 128K — costs $O(L^2)$ attention at 70B and is precisely the thing PEFT exists to avoid. So the paper that would settle the question is the one nobody has budget to run, and the field publishes adapter results with no full-finetuning comparison at the same data.

## 7. Current Research (as of 2026)

- **Rescaling search.** LongRoPE-style evolutionary/per-dimension frequency search, now standard in open-weight releases. Mature.
- **KV compression as adaptation.** Training a small compressor to summarize distant KV (AutoCompressor, Chevalier et al., EMNLP 2023; gist tokens, Mu et al., NeurIPS 2023) rather than adapting attention. Reframes the problem as learned memory. *(frontier — verify current scale claims.)*
- **Diagnostic benchmarks.** RULER's synthetic-and-controllable design is being extended to variable-hop aggregation and to distractor density sweeps. This is the line most likely to unblock the measurement variant.
- **Rank-adaptive PEFT.** DoRA (Liu et al., ICML 2024) and rank-allocation methods applied to the long-context setting specifically. *(frontier — verify.)*
- **Retrieval-head interpretability.** Locating the small set of heads responsible for long-range copying, then adapting only those. If a handful of heads carry the behaviour, the low-rank hypothesis gains support. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** At matched data and matched tokens, how much long-context capability does freezing the backbone cost?

**Scale.** Llama-2-7B (or any 7B base with $L_0 = 4$K). Target $L_1 = 128$K. Training budget fixed at 500M tokens, using the Fu et al. per-domain length-upsampled mixture (published recipe). Approximately 400–600 A100-hours per arm; 5 arms.

**Arms.**
1. **Control:** full continued pretraining, all $\theta$ trainable, YaRN rescaling.
2. Frozen backbone, LoRA $r=8$ on $\{W_Q,W_K,W_V,W_O\}$, YaRN rescaling, **norms and embeddings frozen**.
3. Same as (2) with $r=64$.
4. Same as (2) with $r=256$.
5. **Null arm:** YaRN rescaling only, no training, no adapter.

**Decision number.** RULER aggregate accuracy at 128K, averaged over the 13 task categories, $\ge 500$ samples per cell. Report the single scalar
$$G(r) = \mathrm{acc}_{\text{control}} - \mathrm{acc}_{\text{LoRA}(r)}.$$

**Reading it.** If $G(256) \le 2$ points while $G(8) \ge 10$, the adaptation is real but not low-rank, and the field's rank-8 defaults are wrong. If $G(r) \le 2$ for all $r$ including $r=8$, freezing is free and the low-rank hypothesis is supported. If $G(r) \ge 10$ for all $r$, frozen-backbone long-context adaptation is a dead end and the reported successes are the null arm plus benchmark leakage. Arm 5 bounds the whole thing: if arm 5 is within 2 points of the control, the experiment shows training contributed nothing and the entire literature's ablation is missing.

## 9. Key References

- **[Foundational]** Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[Foundational]** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing, 2024. — arXiv:2104.09864
- **[Foundational]** Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian. *Extending Context Window of Large Language Models via Positional Interpolation.* 2023. — arXiv:2306.15595
- **[SOTA]** Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia. *LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models.* ICLR, 2024. — arXiv:2309.12307
- **[SOTA]** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR, 2024. — arXiv:2309.00071
- **[SOTA]** Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, Mao Yang. *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens.* ICML, 2024. — arXiv:2402.13753
- **[Evaluation]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Evaluation]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Empirical]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[Empirical]** Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, Hao Peng. *Data Engineering for Scaling Language Models to 128K Context.* ICML, 2024. — arXiv:2402.10171
- **[Empirical]** Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John P. Cunningham. *LoRA Learns Less and Forgets Less.* TMLR, 2024. — arXiv:2405.09673
- **[Empirical]** Wenhan Xiong et al. *Effective Long-Context Scaling of Foundation Models.* NAACL, 2024. — arXiv:2309.16039
- **[Survey]** Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, Sai Qian Zhang. *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* TMLR, 2024. — arXiv:2403.14608

## 10. Worked Example

Take Llama-2-7B: $d = 4096$, $N = 32$ layers, RoPE base $b = 10^4$, $L_0 = 4096$. Extend to $L_1 = 131072$, so $s = 32$.

**Step 1 — how many parameters is the "efficient" adapter, really?**

LoRA $r=8$ on four projections per layer: $4 \times 32 \times 2 \times 8 \times 4096 = 8.4\times 10^{6}$, or 0.12% of 6.7B. Add trainable embeddings ($32000 \times 4096 = 1.31\times 10^{8}$, input and output tied or not) and RMSNorm weights ($\approx 2.7\times 10^{5}$). Total trainable becomes $\approx 1.4\times 10^{8}$ — **1.7 to 4 percent** of the model, and 94% of it is the embedding table, which is not low-rank at all.

**Step 2 — the ablation nobody publishes.** LongLoRA's own ablation reports that LoRA alone (norms and embeddings frozen) fails to extend context; adding norms and embeddings recovers it. So the load-bearing component is a dense $32000\times 4096$ matrix. The headline "0.12% of parameters" is not the number that produced the result.

**Step 3 — make the confound visible.** Compare two configurations, no training in either:

| Config | 128K perplexity (PG19-style) | 128K single-needle | 128K multi-hop aggregation |
|---|---|---|---|
| No rescaling, $s=1$ | diverges ($>10^3$) | ~0% | ~0% |
| YaRN rescaling only, $s=32$, zero training | ~8–9, stable | high (often $>$90%) | collapses |

The middle column is the number papers report. It is obtained with **zero trainable parameters**. Any adapter evaluated on top of rescaling inherits that number and can claim it.

**Step 4 — the obstruction, stated numerically.** Suppose an adapter arm reports 92% needle accuracy at 128K and the null arm (rescaling only, no training) reports 89%. The adapter's marginal contribution is 3 points on a metric where an untrained induction head already scores 89. Meanwhile on multi-hop aggregation both arms may sit near 20%, and the full-finetuning control — the only arm that would say whether 20% is a ceiling of the method or of the frozen backbone — was never run because it costs 10× the adapter arm. That missing control, not the adapter design, is what keeps this problem open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*