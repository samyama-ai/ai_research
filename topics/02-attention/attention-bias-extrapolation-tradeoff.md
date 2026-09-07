---
id: 02-attention/attention-bias-extrapolation-tradeoff
title: "Attention Bias Terms and Extrapolation Tradeoff"
topic: 02-attention
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Bias Terms and Extrapolation Tradeoff

> **Topic:** Attention Mechanisms · **ID:** `02-attention/attention-bias-extrapolation-tradeoff` · **Status:** partially-solved

## 1. Problem Statement

A decoder-only Transformer trained on sequences of length $L$ is often evaluated at length $L' \gg L$. Additive attention bias terms — a scalar $b_{ij}$ added to the query–key logit before softmax, as in T5's learned relative bucket bias, ALiBi's $-m\,(i-j)$, KERPLE, Sandwich, and FIRE — reliably keep *perplexity* from exploding at $L' > L$. The open problem is that the mechanism which buys that stability appears to be **monotone recency decay**, which is also the mechanism that discards distant tokens. So the same knob that fixes extrapolation may cap the model's usable context.

Three variants, different difficulty:

- **Measurement.** Define a scalar that separates "does not diverge at $L'$" from "actually reads tokens at distance $L'$", and show the two are dissociable. Partly done (RULER, needle/passkey suites); no accepted continuous metric.
- **Method.** Build a bias family whose extrapolation is not purchased by decay — extrapolates to $8\times$ *and* retrieves at $8\times$. FIRE and NoPE are the strongest candidates; neither is demonstrated at frontier scale.
- **Theory.** Prove or refute: any bias $b_{ij}$ that guarantees bounded logit statistics for all $L'$ must induce an effective attention window of width $O(f(L))$ independent of $L'$. Open.

**Solved** would mean: a bias family with a proof of length-uniform logit control *and* a measured retrieval accuracy at $L'=8L$ within 5 points of a model natively trained at $L'$, at $\geq 7$B parameters.

## 2. Formal Setting

Tokens $x_{1:n}$, head $h$, queries/keys $q_i, k_j \in \mathbb{R}^d$. Biased attention:

$$a_{ij} = \operatorname{softmax}_j\!\left( \frac{q_i^\top k_j}{\sqrt{d}} + b^{(h)}_{ij} \right), \quad j \le i .$$

Bias families, as implemented:
- **ALiBi:** $b^{(h)}_{ij} = -m_h (i-j)$, $m_h$ fixed geometric, $m_h = 2^{-8h/H}$ for $H$ heads.
- **KERPLE (log):** $b_{ij} = -r_1 \log(1 + r_2 (i-j))$, $r_1,r_2>0$ learned.
- **T5:** $b_{ij} = \beta_{\phi(i-j)}$, $\phi$ a log-spaced bucketing into 32 buckets, learned, clipped past distance 128.
- **FIRE:** $b_{ij} = f_\theta\!\big(\psi(i-j)/\psi(\max(i, L_c))\big)$, $f_\theta$ an MLP, $\psi$ monotone — the normalizer is what makes it length-invariant.

Measured quantities:

- **Extrapolation gap.** $\Delta(L') = \mathrm{PPL}_{L}(L') - \mathrm{PPL}_{L'}(L')$, where $\mathrm{PPL}_{L}(L')$ is perplexity of an $L$-trained model on non-overlapping windows of length $L'$. Non-overlapping matters: sliding-window evaluation with stride $\ll L$ hides divergence.
- **Effective receptive field.** $\rho_i(w) = \sum_{j > i-w} a_{ij}$, and $w^*(\epsilon) = \min\{w : \mathbb{E}_i[\rho_i(w)] \ge 1-\epsilon\}$ at $\epsilon=0.05$, averaged over heads and a held-out corpus. This is the decay measurement.
- **Retrieval.** $R(L', p)$ = exact-match accuracy on a needle inserted at relative depth $p \in [0,1]$ in a context of length $L'$; report $\min_p R$, not the mean, since depth-averaged scores mask the failure.
- **Usable context.** $L_{\text{eff}} = \max\{L' : \min_p R(L',p) \ge 0.85\}$ (the RULER convention, threshold set by the Llama-2-7B baseline at its train length).

The tradeoff conjecture, stated measurably: for a bias family $\mathcal{B}$, is $\sup_{b\in\mathcal{B}} L_{\text{eff}}$ bounded by $c \cdot w^*(0.05)$ with $c=O(1)$, while $\Delta(L')$ small for all $L'$ forces $w^*$ to be $L'$-independent?

Assumptions, with the ones known violated flagged:
1. $b_{ij}$ depends only on $i-j$ — **violated by FIRE** (depends on $i$ too) and by any length-normalized scheme; this is exactly the escape hatch.
2. Bias is content-independent — **violated in practice** by attention sinks, where the first few token *contents* dominate the logit budget (Xiao et al., ICLR 2024).
3. Perplexity is monotone in context usage — **violated**: most perplexity reduction comes from the nearest $\sim$few hundred tokens, so $\Delta(L')\approx 0$ is compatible with $L_{\text{eff}} = L$.
4. Heads are exchangeable, so a single $w^*$ describes the model — violated; retrieval concentrates in a small number of induction/retrieval heads.

## 3. State of the Art

**Established (ablated, reproduced):**
- ALiBi (Press, Smith, Lewis, ICLR 2022) gives non-diverging perplexity beyond train length, with ablations showing the *slopes* — not the linear form per se — carry the effect.
- KERPLE (Chi et al., NeurIPS 2022) and Sandwich (Chi et al., ACL 2023) generalize it; the ACL 2023 receptive-field analysis is the strongest evidence for the tradeoff: ALiBi-style extrapolation is explained by a bounded receptive field, i.e. the model behaves like a sliding window.
- Position Interpolation (Chen et al., 2023) and YaRN (Peng et al., ICLR 2024) show that *fine-tuning* with rescaled RoPE beats zero-shot bias extrapolation for real long-context use — the field's practical answer.

**Claimed but unablated at scale:**
- FIRE (Li et al., ICLR 2024) reports both extrapolation and downstream long-context gains, but the language-model experiments are $\le$ 1B-ish and the retrieval evaluation is not the RULER suite.
- NoPE (Kazemnejad et al., NeurIPS 2023): removing positional information entirely beats ALiBi/RoPE/T5-bias on length generalization — established on small algorithmic tasks, **not** established for natural-language retrieval at scale.

**Benchmark-number-only:** most vendor claims of "128k context" are a training-length statement. RULER (Hsieh et al., COLM 2024) measured that several models advertising 32k have $L_{\text{eff}}$ well below it. No frontier model ships an ALiBi-style monotone bias as its long-context mechanism.

## 4. What Is Known

- ALiBi at **1.3B params**, trained $L=1024$, evaluated at $L'=2048$, reaches perplexity better than a sinusoidal model *trained* at 2048; training is ~11% faster and uses ~11% less memory than the sinusoidal 2048 baseline (Press et al., ICLR 2022).
- Sinusoidal and rotary embeddings without adaptation degrade sharply past $L$ — the divergence is reproduced in every paper above.
- The receptive-field explanation: ALiBi's per-head slopes $m_h$ produce $w^*$ values spanning roughly $10^0$–$10^3$ tokens, and truncating attention to those windows changes perplexity little (Chi et al., ACL 2023, ~125M scale). Whatever ALiBi is doing beyond a window is small.
- StreamingLLM (Xiao et al., ICLR 2024): keeping **4** initial "sink" tokens plus a rolling window gives stable perplexity over **4M** tokens on Llama-2/Falcon/MPT/Pythia, with up to **22.2$\times$** speedup over recomputation — and explicitly *does not* extend usable context. This is the cleanest existing dissociation of $\Delta(L')$ from $L_{\text{eff}}$.
- YaRN extends Llama-2 7B/13B to 64k–128k with ~400 fine-tuning steps, i.e. $<0.1\%$ of pretraining tokens; retrieval at the new length is real, not just perplexity.
- Depth-dependent retrieval failure ("lost in the middle", Liu et al., TACL 2024) at 7B–175B scale: accuracy is U-shaped in $p$, so $\mathbb{E}_p R$ overstates capability.

## 5. What Is Not Known

- **Theoretically open.** No theorem states that length-uniform control of the softmax logit distribution requires an $L'$-independent effective window. Neither direction is proved. A negative result — exhibiting a bias family with unbounded $w^*$ and bounded $\Delta$ — would also settle it; FIRE is a candidate but has no proof.
- **Empirically open.** Whether a bias-only model (FIRE, KERPLE with learned decay, NoPE) trained at $\geq$7B and $L=8$k reaches $L_{\text{eff}} \geq 32$k. The experiment is runnable — roughly $10^{22}$ FLOPs per arm — and has not been run with a matched control.
- **Methodologically blocked.** There is no continuous, task-free measure of "context actually used". $w^*(\epsilon)$ is attention-weight-based and attention weight is a poor proxy for causal influence; needle tests are discrete, saturating, and confounded with instruction-following. Until this is fixed, the tradeoff can only be probed at pass/fail resolution.

## 6. Why It Is Hard

**The evaluation does not measure what it names.** "Length extrapolation" is reported as perplexity at $L' > L$, but perplexity is dominated by short-range statistics: a model that hard-truncates to the last 1k tokens loses almost nothing in next-token loss on natural text, which is exactly why StreamingLLM's 4M-token curve is flat. So the headline metric is nearly insensitive to the quantity in dispute. Second obstruction: **non-identifiability** — the bias, the learned key/query geometry, and the emergent attention sink all shape the same logit, so an ablation that swaps $b_{ij}$ also changes the sink structure the model had adapted to. Third: **compute**. The dissociation only appears at scales where retrieval behavior exists at all ($\gtrsim$1B, arguably $\gtrsim$7B), so every arm is a pretraining run, and the literature's bias comparisons live at 125M–1B where retrieval heads are weak.

## 7. Current Research (as of 2026)

- Length-invariant *learned* biases (FIRE lineage, Google DeepMind / UT Austin), aiming to break assumption 1 by conditioning on $i$ as well as $i-j$.
- Hybrid stacks: a minority of full-attention layers over a majority of local/decayed layers — the practical answer in several open-weight releases; the decay/retrieval split is moved from within a head to across layers.
- Attention-sink theory: analyses treating the sink as a learned no-op that stabilizes logit scale, which reframes bias design as variance control rather than position encoding *(frontier — verify)*.
- Measurement work: RULER-style synthetic suites with controlled distractor counts, and causal-mediation ("activation patching") estimates of long-range influence as a replacement for $w^*$ *(frontier — verify)*.
- Surveys: Zhao et al., *Length Extrapolation of Transformers: A Survey from the Perspective of Positional Encoding*, Findings of EMNLP 2024.

## 8. Concrete Next Experiment

**Scale.** Four 1.4B-parameter decoder-only models, identical data (~100B tokens), identical everything except the bias, trained at $L=4096$.

**Arms.** (a) ALiBi; (b) FIRE; (c) NoPE; (d) **control:** RoPE trained at $L=4096$ then YaRN-fine-tuned to 32k for 400 steps. The control is what makes the result interpretable — it is the method practitioners would actually use.

**Evaluation.** At $L' \in \{4\text{k}, 8\text{k}, 16\text{k}, 32\text{k}\}$: non-overlapping perplexity $\Delta(L')$; RULER single-needle and multi-key at 4 depths; and $w^*(0.05)$ per head.

**Deciding number.** $\min_p R(32\text{k}, p)$ for arm (b) minus the same for arm (d). If FIRE lands within 5 points of the YaRN control while its ALiBi sibling (a) sits below 20% — the tradeoff is a property of *monotone* biases, not of bias terms as such, and the theory question becomes "what does length-normalization buy". If all bias arms fall below 20% while $\Delta(32\text{k}) < 0.3$ nats for each, the tradeoff is real for the whole family and the theoretical conjecture in §1 is worth attacking directly. Cost: ~4 × 3k A100-hours.

## 9. Key References

- **[Foundational]** Ofir Press, Noah A. Smith, Mike Lewis. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.* ICLR 2022. — arXiv:2108.12409
- **[Foundational]** Colin Raffel et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR 21(140), 2020. — arXiv:1910.10683
- **[Foundational]** Jianlin Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing, 2024. — arXiv:2104.09864
- **[SOTA]** Ta-Chung Chi, Ting-Han Fan, Peter J. Ramadge, Alexander I. Rudnicky. *KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation.* NeurIPS 2022. — arXiv:2205.09921
- **[SOTA]** Ta-Chung Chi, Ting-Han Fan, Alexander I. Rudnicky, Peter J. Ramadge. *Dissecting Transformer Length Extrapolation via the Lens of Receptive Field Analysis.* ACL 2023.
- **[SOTA]** Shanda Li, Chong You, Guru Guruganesh, et al. *Functional Interpolation for Relative Positions Improves Long Context Transformers.* ICLR 2024. — arXiv:2310.04418
- **[SOTA]** Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, Siva Reddy. *The Impact of Positional Encoding on Length Generalization in Transformers.* NeurIPS 2023. — arXiv:2305.19466
- **[SOTA]** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR 2024. — arXiv:2309.00071
- **[SOTA]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. — arXiv:2309.17453
- **[Measurement]** Cheng-Ping Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024. — arXiv:2404.06654
- **[Measurement]** Nelson F. Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Survey]** Liang Zhao et al. *Length Extrapolation of Transformers: A Survey from the Perspective of Positional Encoding.* Findings of EMNLP 2024.

## 10. Worked Example

Take ALiBi with $H=16$ heads, $m_h = 2^{-8h/16} = 2^{-h/2}$, $h=1..16$. The gentlest head has $m_1 = 2^{-0.5} \approx 0.707$; the steepest $m_{16} = 2^{-8} \approx 0.0039$. Wait — the ordering runs the other way: the *smallest* slope is $m_1$ under the standard indexing $m_h = 2^{-8h/H}$ only if $h$ counts down. Take the extremes as given: slopes span $2^{-0.5}$ to $2^{-8}$.

Consider the flattest head, $m=2^{-8}=0.0039$, and a query at position $i=32{,}768$. Compare a token at distance $1$ against one at distance $16{,}384$. The bias difference is

$$\Delta b = -m(16384 - 1) \approx -0.0039 \times 16383 \approx -64 \text{ nats}.$$

For the distant token to receive equal attention, its content logit must exceed the near token's by 64. Logits are $q^\top k/\sqrt d$; in a trained model these have standard deviation of order 1–3 and are effectively bounded within roughly $\pm 15$ by LayerNorm and weight decay. A 64-nat deficit is unreachable — by a factor of four, in the *most* permissive head. The steep heads are worse by three orders of magnitude.

Now the other side of the ledger. Evaluate the same model at $L'=32$k with non-overlapping windows: $\Delta(32\text{k})$ stays small, well under a nat, because the next-token distribution at position $i$ is nearly determined by the previous few hundred tokens. The perplexity metric is flat while the retrieval capacity at 16k is provably (by the arithmetic above) zero.

That is the obstruction in one calculation: the reported success metric and the disputed capability are numerically decoupled, and the coupling constant between them — how much perplexity a token at distance $10^4$ is worth — is roughly $10^{-3}$ nats, far below the noise floor of any pretraining comparison. Any experiment that decides this question must measure retrieval directly.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*