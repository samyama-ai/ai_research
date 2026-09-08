---
id: 09-model-design/encoder-decoder-vs-decoder-only
title: "Encoder-Decoder Versus Decoder-Only at Matched Compute"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Encoder-Decoder Versus Decoder-Only at Matched Compute

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/encoder-decoder-vs-decoder-only` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed training compute budget $C$ and a fixed inference budget, does a Transformer **encoder-decoder** (separate bidirectional encoder over the conditioning text, causal decoder with cross-attention) reach lower loss and higher downstream accuracy than a **decoder-only** Transformer that concatenates conditioning and target into one causal stream? The field standardized on decoder-only after 2020 without a matched-compute settlement of this question.

Three variants, with different difficulty:

- **Measurement.** Define "matched compute" so that the comparison is not rigged. Parameters, training FLOPs, inference FLOPs per generated token, and serving memory rank the two families *differently*; a claim is only meaningful relative to a stated budget.
- **Method.** Build the compute-optimal recipe for each family — depth/width split between encoder and decoder, objective (span corruption vs. causal LM vs. UL2 mixture), and data ratio — and compare at the frontier of each, not at the frontier of one and a default for the other.
- **Theory.** Prove or refute a separation: is there a task family on which bidirectional encoding gives an asymptotic advantage that no causal decoder with equal compute can match, or is the decoder-only model a strict superset up to constants?

Solving it means: a scaling-law statement of the form "for input-to-output length ratio $\rho$ above threshold $\rho^*$ and budget $C$, family $A$ dominates family $B$ by $\Delta$ nats," reproduced independently.

## 2. Formal Setting

Let a task be a distribution over pairs $(x, y)$, $x \in V^{m}$ (conditioning tokens), $y \in V^{n}$ (target tokens). Define the **length ratio** $\rho = \mathbb{E}[m]/\mathbb{E}[n]$.

Both families model $p_\theta(y \mid x) = \prod_{t=1}^{n} p_\theta(y_t \mid y_{<t}, x)$. They differ in the attention mask: the decoder-only model applies a causal mask over the concatenation $[x; y]$; the encoder-decoder applies a full mask within $x$, a causal mask within $y$, and cross-attention $y \to x$.

**Objective, as measured.** Per-target-token loss on a held-out set:
$$\mathcal{L}(\theta) = -\frac{1}{\sum_i n_i}\sum_i \log p_\theta(y^{(i)} \mid x^{(i)}).$$
Loss over $x$ tokens is excluded, because a decoder-only model is normally trained to predict them and an encoder-decoder is not — including them makes the numbers non-comparable.

**Compute, as measured.** Use the $6ND$ approximation only after separating the stacks. For an encoder-decoder with $N_e$ encoder and $N_d$ decoder non-embedding parameters, training FLOPs per example are
$$C_{\text{train}} \approx 6\,(N_e m + N_d n) + \text{attn}(m,n),$$
against $6 N (m+n)$ for a decoder-only model of $N$ parameters. Hence the T5 convention: an encoder-decoder with $N_e + N_d = 2N$ costs roughly the same per token as a decoder-only model with $N$ parameters. **Inference** is different again: after the encoder runs once, each generated token costs $\approx 2N_d + $ cross-attention over $m$ keys, versus $2N$ plus self-attention over $m+t$ keys. KV-cache bytes: encoder-decoder caches $m$ cross-attention keys/values once plus $t$ decoder keys; decoder-only caches $m+t$ at full width.

**Budget-matched comparison.** Fix $C$; for each family let $\mathcal{L}^*_F(C) = \min_{\theta \in F, \, C(\theta) \le C} \mathcal{L}(\theta)$ over architecture shape, tokens, and objective. The quantity of interest is $\Delta(C, \rho) = \mathcal{L}^*_{\text{dec}}(C) - \mathcal{L}^*_{\text{encdec}}(C)$.

**Assumptions, and which are violated.**
1. *The $6ND$ FLOP count dominates.* Violated at long context, where attention is $O(m^2)$ and the encoder-decoder's quadratic term over $x$ is paid once rather than $n$ times.
2. *FLOPs predict wall-clock.* Violated: decoder-only inference is memory-bandwidth-bound, so KV-cache size, not FLOPs, sets throughput.
3. *Each family is trained at its own optimum.* Badly violated in the literature — decoder-only models get 2020–2026 of recipe tuning; encoder-decoders are usually run with T5-era hyperparameters.
4. *Held-out loss ranks downstream ability.* Violated; see §4.

## 3. State of the Art

**Established (ablated, controlled).**
- Wang et al., *What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?* (ICML 2022) ran a 3×3 grid of architecture (causal decoder, non-causal prefix-LM decoder, encoder-decoder) × objective (full LM, prefix-LM, span corruption) at ~4.8B parameters and 168B tokens, with matched compute. Result: causal decoder + full LM is best for pure zero-shot after pretraining; **encoder-decoder + span corruption is best after multitask finetuning**; adaptation between regimes is cheap. This is the single cleanest matched-compute experiment in the literature — and it is one scale, one data mixture, four years old.
- Tay et al., *Scaling Laws vs Model Architectures* (Findings of EMNLP 2023) trained ten architectures across scales and found that architectural ranking is **not** scale-invariant: models that win at small scale lose at large, and upstream perplexity is a poor predictor of downstream rank.

**Claimed but unablated.**
- UL2 (Tay et al., ICLR 2023) reports a 20B encoder-decoder trained on a mixture-of-denoisers beating comparable decoder-only baselines, but the baselines are external, not compute-matched re-trains.
- AlexaTM 20B (Soltan et al., 2022) claims a 20B seq2seq model beats 175B GPT-3 on some few-shot translation and summarization. Different data, different tokenizer — a benchmark number, not an ablation.
- T5Gemma / *Encoder-Decoder Gemma* (Google, 2025) claims adapting pretrained decoder-only checkpoints into encoder-decoder form improves the quality/inference-cost trade-off. The adaptation baseline is the source decoder-only model, so the comparison is fair in inference FLOPs but not in total training compute.

**Theory SOTA.** Fu et al. (*Decoder-only or Encoder-Decoder? Interpreting Language Model as a Regularized Encoder-Decoder*, 2023) recast the decoder-only model as an encoder-decoder with tied parameters and shared representation space — a reduction, not a separation. No lower bound distinguishing the families is known.

## 4. What Is Known

- **Machine translation, up to ~2B encoder-decoder params (Zhang et al., ICML 2022; Ghorbani et al., ICLR 2022):** encoder capacity matters more than decoder capacity for translation quality; decoder-heavy scaling raises BLEU less per FLOP. Ghorbani et al. fit separate power laws in $N_e$ and $N_d$ and show the exponents differ.
- **Wang et al. (ICML 2022), 4.8B / 168B tokens:** the best architecture flips with the finetuning regime. Same compute, opposite conclusion — this is the load-bearing empirical fact for this page.
- **Objective, not mask, carries much of the gap.** T5's own ablation (Raffel et al., JMLR 2020) found span corruption beats plain LM by roughly 1–2 GLUE points at ~220M–3B under the same encoder-decoder backbone; UL2 showed a decoder-only model trained on a denoiser mixture recovers much of the infilling advantage.
- **Inference asymmetry is arithmetic, not empirical.** At $m = 4096$, $n = 64$, a decoder-only model attends over up to 4160 positions per generated token; an encoder-decoder with a 12-layer decoder attends over 64 self plus 4096 cross keys that were computed once. For summarization-shaped workloads ($\rho \gg 1$) the encoder-decoder's per-token cost is strictly lower at equal $N_e+N_d = 2N$.
- **Nothing above was measured above ~20B parameters with both arms trained under 2026 recipes** (RoPE, SwiGLU, grouped-query attention, µP-tuned LR, 20+ tokens/param).

## 5. What Is Not Known

- **Empirically open (dominant).** Whether $\Delta(C,\rho) > 0$ for $\rho \gtrsim 10$ at $C \ge 10^{22}$ FLOPs with both arms tuned. The experiment is runnable today for roughly the cost of one 8B-class pretraining run per arm; nobody has published it. Every large encoder-decoder result is confounded by data or recipe.
- **Empirically open.** Whether encoder-decoders are compatible with the properties that made decoder-only models win in deployment: prompt-prefix KV reuse across turns, long-context extension, and mixture-of-experts routing. Each is engineered for causal streams and untested at scale in cross-attention form.
- **Theoretically open.** No proof that bidirectional encoding gives a separation. Fu et al.'s reduction suggests the families are equivalent up to constants, but the constants are the whole question, and there is no lower-bound theorem for either direction.
- **Methodologically blocked.** "Matched compute" itself. Training FLOPs, inference FLOPs/token, KV bytes, and parameters give four different rankings for the same pair of models, and no accepted convention says which to fix. Papers pick the one that favors their arm.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by asymmetric recipe maturity**. A fair test requires tuning both arms to their own optimum, but a decade of hyperparameter, data-mixture, and kernel work is decoder-only-specific. Any observed gap is inseparable from "we know how to train one of these." The tuning budget needed to remove the confound is itself a multi-run pretraining sweep per arm — a $10^{23}$-FLOP experiment to answer a question worth one architecture decision, which is why no lab has run it and instead each simply ships decoder-only. Second obstruction: **the evaluation does not measure what it names.** Held-out loss on $y$ is the natural metric, but Tay et al. showed upstream perplexity does not preserve downstream rank across architectures, so the cheap measurement is not the one that decides.

## 7. Current Research (as of 2026)

- **Adaptation over pretraining.** Google's T5Gemma line converts pretrained decoder-only checkpoints into encoder-decoders, sidestepping the cost of a matched-compute pretrain. This makes the inference argument testable and leaves the pretraining question untouched. *(frontier — verify current model sizes.)*
- **Long-context serving economics.** Prefill-heavy workloads (RAG, agent traces, document QA) have pushed $\rho$ to $10^2$–$10^4$, which is exactly the regime where the encoder-decoder cost model wins on paper. Prefix caching in decoder-only serving stacks is the competing answer. *(frontier — verify.)*
- **Objective unification.** Mixture-of-denoisers pretraining for decoder-only models continues, which if successful removes the strongest historical argument for encoder-decoders (infilling and multitask finetuning quality).
- **Encoder reuse in multimodal stacks.** Vision-language models are de facto encoder-decoders with a non-text encoder; whether the same argument transfers to text conditioning is untested.

## 8. Concrete Next Experiment

**Scale.** Two arms at $C \approx 6\times10^{21}$ FLOPs each (≈1.4B-parameter decoder-only equivalent, 20 tokens/param ≈ 28B tokens), identical corpus, tokenizer, optimizer, and 2026 recipe (RoPE, SwiGLU, GQA, µP-transferred LR).

- **Arm A (control):** decoder-only, $N = 1.4$B, prefix-LM loss masked so gradients come only from $y$ tokens.
- **Arm B:** encoder-decoder, $N_e + N_d = 2.8$B, swept over the split $N_e/(N_e+N_d) \in \{0.33, 0.5, 0.67\}$ (three runs), span-corruption pretraining plus the same $(x,y)$ objective.
- **Data shaping:** three $\rho$ buckets — $\rho \in \{0.5, 4, 32\}$ — evaluated separately on a held-out mixture of translation, summarization, and long-document QA.

**Deciding number.** $\Delta(C, \rho=32)$ in nats per target token, with the encoder split chosen by validation. **Decision rule:** if $\Delta > 0.02$ nats/token at $\rho = 32$ *and* the encoder-decoder arm's measured tokens/second at batch 64, $m=4096$, $n=64$ on one A100/H100 node is at least 1.5× the control's, the encoder-decoder is established as the better family for prefill-heavy workloads at this compute. If $|\Delta| < 0.01$ nats at all three $\rho$ values, the families are compute-equivalent for text and the remaining argument is purely a serving-cost one.

Cost estimate: 4 runs × ~$6\times10^{21}$ FLOPs ≈ 3–4k H100-days total. This is a single-lab experiment, not a moonshot.

## 9. Key References

- **[Foundational]** Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR 21(140), 2020. — arXiv:1910.10683
- **[Foundational]** Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer. *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* ACL 2020. — arXiv:1910.13461
- **[SOTA]** Thomas Wang, Adam Roberts, Daniel Hesslow, Teven Le Scao, Hyung Won Chung, Iz Beltagy, Julien Launay, Colin Raffel. *What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?* ICML 2022. — arXiv:2204.05832
- **[SOTA]** Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia, Jason Wei, Xuezhi Wang, Hyung Won Chung, Siamak Shakeri, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Denny Zhou, Neil Houlsby, Donald Metzler. *UL2: Unifying Language Learning Paradigms.* ICLR 2023. — arXiv:2205.05131
- **[SOTA]** Yi Tay, Mostafa Dehghani, Samira Abnar, Hyung Won Chung, William Fedus, Jinfeng Rao, Sharan Narang, Vinh Q. Tran, Dani Yogatama, Donald Metzler. *Scaling Laws vs Model Architectures: How Does Inductive Bias Influence Scaling?* Findings of EMNLP 2023. — arXiv:2207.10551
- **[Analysis]** Biao Zhang, Behrooz Ghorbani, Ankur Bapna, Yong Cheng, Xavier Garcia, Jonathan Shen, Orhan Firat. *Examining Scaling and Transfer of Language Model Architectures for Machine Translation.* ICML 2022. — arXiv:2202.00528
- **[Analysis]** Behrooz Ghorbani, Orhan Firat, Markus Freitag, Ankur Bapna, Maxim Krikun, Xavier Garcia, Ciprian Chelba, Colin Cherry. *Scaling Laws for Neural Machine Translation.* ICLR 2022. — arXiv:2109.07740
- **[Theory]** Zihao Fu, Wai Lam, Qian Yu, Anthony Man-Cho So, Shengding Hu, Zhiyuan Liu, Nigel Collier. *Decoder-Only or Encoder-Decoder? Interpreting Language Model as a Regularized Encoder-Decoder.* 2023. — arXiv:2304.04052
- **[Compute]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Claimed]** Saleh Soltan et al. *AlexaTM 20B: Few-Shot Learning Using a Large-Scale Multilingual Seq2Seq Model.* 2022. — arXiv:2208.01448
- **[Recent]** Google DeepMind. *Encoder-Decoder Gemma (T5Gemma): improving the quality–efficiency trade-off via adaptation.* 2025. (Identifier not verified here; cite by title.)

## 10. Worked Example

Take a summarization workload: $m = 4096$ source tokens, $n = 64$ summary tokens, $\rho = 64$. Budget: decoder-only at $N = 1$B.

**Arm A, decoder-only ($N = 10^9$).** Training FLOPs per example $\approx 6N(m+n) = 6\times10^9\times4160 = 2.50\times10^{13}$. Per generated token at inference: $2N = 2\times10^9$ FLOPs, plus attention over up to 4160 cached positions.

**Arm B, encoder-decoder, $N_e = N_d = 10^9$ (total 2B).** Training FLOPs per example $\approx 6(N_e m + N_d n) = 6(10^9\!\cdot\!4096 + 10^9\!\cdot\!64) = 2.50\times10^{13}$. Identical. Per generated token: $2N_d = 2\times10^9$ FLOPs plus cross-attention over 4096 keys computed once during a single encoder pass.

So at matched *training* compute Arm B carries **twice the parameters**. That looks like a free win — and here the obstruction becomes visible three ways:

1. **Memory, not FLOPs, decides serving.** Arm B holds 2B weights (4 GB in bf16) versus 1 GB… no: 2 GB versus 4 GB. Arm B needs 2× the weight bandwidth per decode step, but its KV cache per sequence is smaller, because the 4096 cross-attention entries are shared across all decode steps rather than growing. Which model is faster depends on batch size: at batch 1, Arm B loses on weight bandwidth; at batch 256, it wins on cache. **The same pair of models ranks in opposite orders under two defensible "matched compute" definitions** — this is the methodological block of §5.
2. **The parameter doubling is not free at fixed data.** Chinchilla says a 2B-parameter model wants ~40B tokens; Arm B sees the same 28B example-tokens as Arm A, but only 64 of every 4160 tokens produce a decoder gradient. Arm B's decoder is data-starved by a factor of ~64 relative to its parameter count, and no published scaling law covers this asymmetric regime.
3. **The measurement that is cheap is not the one that decides.** Suppose Arm B wins by 0.03 nats/target-token. Tay et al. (2023) showed upstream perplexity does not preserve downstream ranking across architecture families — so the 0.03 nats does not license a claim about summarization ROUGE or instruction-following without running the downstream evaluation too, at both scales, with both recipes tuned.

The arithmetic that makes encoder-decoders look obviously better at $\rho = 64$ is correct and easy. It is also not sufficient to decide the question, which is precisely why the question is still open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*