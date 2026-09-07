---
id: 14-long-context/attention-entropy-collapse-long-context
title: "Attention Entropy Collapse at Extreme Lengths"
topic: 14-long-context
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Entropy Collapse at Extreme Lengths

> **Topic:** Long Context · **ID:** `14-long-context/attention-entropy-collapse-long-context` · **Status:** open

## 1. Problem Statement

As the context length $n$ of a decoder-only transformer grows from $10^3$ to $10^6$ tokens, the per-head distribution of attention weights degenerates in one of two directions, and it is not established which degeneration causes the observed loss of long-context retrieval accuracy.

- **Dispersion.** With logits bounded independent of $n$, softmax spreads mass over more keys; entropy drifts toward $\log n$ and no single key can be selected.
- **Sink concentration.** In practice most heads dump the bulk of their mass on a handful of positions (token 0, delimiters), so raw entropy *falls* toward $O(1)$ while the entropy over the remaining tokens still disperses.

The name "entropy collapse" is used for both, and for a third thing (Zhai et al., ICML 2023, where entropy $\to 0$ during *training* and destabilizes optimization). The page treats all three as one problem because they share a measurement.

Three variants, of different difficulty:

- **Measurement.** Define a length-comparable statistic of the attention distribution that predicts long-context task failure. Currently unsettled: raw entropy is dominated by sinks and by the $\log n$ baseline.
- **Method.** Find an intervention (logit temperature, learned scaling, sink handling) that restores the statistic *and* improves accuracy at $n \ge 128\text{K}$ without retraining from scratch.
- **Theory.** Prove or refute: for fixed weights and bounded key/query norms, retrieval accuracy of a softmax head must decay in $n$; and characterize the logit growth rate needed to avoid it.

**Solved** means: a statistic $S(n)$ computable from a forward pass such that intervening on $S$ (holding perplexity fixed) moves RULER-style retrieval accuracy in the predicted direction, with a matched control arm ruling out the intervention's other effects.

## 2. Formal Setting

Sequence $x_{1:n}$, layer $\ell$, head $h$, query position $i$. With causal masking,

$$a^{(\ell,h)}_{ij} = \frac{\exp\!\big(\langle q_i, k_j\rangle/\sqrt{d_h} + \beta_{ij}\big)}{\sum_{j'\le i}\exp\!\big(\langle q_i,k_{j'}\rangle/\sqrt{d_h}+\beta_{ij'}\big)},\quad j \le i,$$

where $\beta$ absorbs any additive positional bias (ALiBi); RoPE instead rotates $q,k$ before the inner product.

**Measured quantities.**

- Raw entropy: $H^{(\ell,h)}_i = -\sum_{j\le i} a_{ij}\log a_{ij} \in [0,\log i]$.
- Normalized entropy: $\tilde H^{(\ell,h)}_i = H^{(\ell,h)}_i/\log i$. This is the only form comparable across $n$; unnormalized entropy trivially grows.
- Sink mass: $\sigma^{(\ell,h)}_i = \sum_{j\in\mathcal{S}} a_{ij}$ with $\mathcal{S}$ the first $k=4$ positions plus BOS-like delimiters.
- Sink-excluded entropy: renormalize $a$ over $j\notin\mathcal{S}$, then take $\tilde H$. This is the quantity that decouples the two failure modes and is the one most papers do *not* report.
- Effective support: $N^{(\ell,h)}_i=\exp(H^{(\ell,h)}_i)$, the perplexity of the attention distribution — number of keys effectively read.
- Logit gap: $\Delta_i = \max_j \ell_{ij} - \operatorname{quantile}_{0.99}(\ell_{ij})$, where $\ell_{ij}$ is the pre-softmax score.

**Practical measurement notes.** FlashAttention never materializes $a$, so any entropy readout requires a recompute pass (or a fused entropy kernel), roughly doubling attention cost. Softmax must be accumulated in fp32; in bf16 the tail terms of $-a\log a$ for $a\sim 10^{-6}$ are the ones that carry the dispersion signal, and they are exactly where bf16 loses them. Entropy is a function of the *data*, so any cross-length comparison must fix the token distribution, not just the token count.

**Assumptions, and which fail.**

1. *Bounded logits:* $\|q\|,\|k\| \le B$ independent of $n$. **Violated** — massive activations (Sun et al., COLM 2024) give a few coordinates magnitudes $10^3$–$10^4$, which is precisely the mechanism that lets a head beat the $\log n$ dispersion bound.
2. *Length-stationary data:* the marginal token distribution at position $10^6$ matches that at $10^3$. **Violated** — long-context corpora are code, books and repeated boilerplate, not scaled-up short text.
3. *Head specialization is stable across $n$:* retrieval heads at 4K are still retrieval heads at 128K. **Untested at $\ge$256K.**

## 3. State of the Art

**Theory (established).** Veličković et al. (2024, *Softmax is not enough for sharp out-of-distribution*) prove that for a fixed-parameter softmax head with bounded inputs, attention coefficients must disperse as $n\to\infty$: sharpness is not preserved out of the training length distribution. Chiang & Cholak (ACL 2022) show, in the opposite direction, that logits scaling as $\Theta(\log n)$ suffice to make attention effectively hard and to get exact length generalization on PARITY/FIRST. Together these bound the problem: dispersion is unavoidable at fixed logit scale, and $\log n$ scaling is enough to avoid it.

**Method (established by ablation).** YaRN (Peng et al., ICLR 2024) adds an attention temperature $1/\sqrt{t}=0.1\ln s+1$ ($s$ = context extension factor) and ablates it: the temperature term alone accounts for a measurable perplexity gain at extended lengths, independent of the RoPE interpolation. StreamingLLM (Xiao et al., ICLR 2024) establishes that a handful of initial tokens act as an attention sink and that keeping 4 of them in the KV cache is necessary and largely sufficient for stable streaming perplexity to millions of tokens.

**Method (claimed, weakly ablated).** Scalable-Softmax (Nakanishi, 2025) replaces $\exp(\ell)$ with $n^{s\ell}$, making the effective temperature $\log n$-dependent by construction; the reported long-context gains are from a small-scale pretraining run and the causal claim ("because entropy no longer grows") is asserted, not isolated. Differential Transformer (Ye et al., ICLR 2025) subtracts two softmax maps to cancel "attention noise" and reports better needle retrieval; the entropy account is a post-hoc explanation, and no arm holds entropy fixed. Adaptive-temperature heads (Veličković et al., 2024) are demonstrated on synthetic max-retrieval and a small LM, not at 128K.

**Benchmark-only results.** RULER (Hsieh et al., COLM 2024) shows most models advertising 128K degrade sharply well before it. That is a number about *accuracy*, not about entropy; no published run measures $\tilde H$ and RULER accuracy on the same forward passes.

## 4. What Is Known

- Uniform attention over $i$ keys has $H=\log i$: $\log(4096)=8.32$ nats vs $\log(131072)=11.78$ nats. Any raw-entropy comparison across those lengths that does not divide by $\log i$ is measuring the axis, not the model.
- Sinks are large and universal. In Llama-2-7B, removing the first token from a sliding window sends perplexity from single digits to the tens — order $10\times$ (Xiao et al., ICLR 2024). Massive activations in Llama-2-7B/13B concentrate in ~4 coordinates with magnitudes ~$10^3$–$10^4$ against a typical ~$0.1$–$1$ (Sun et al., COLM 2024).
- Sink strength scales with pretraining context length: Barbero et al. (2025) show, across LLaMA 3.1 8B/70B/405B and Gemma, that models trained on longer contexts form stronger sinks, and argue sinks act as a no-op that limits over-mixing.
- Over-squashing is real and length-dependent: Barbero et al. (NeurIPS 2024, *Transformers need glasses!*) show representational collapse where distinct long sequences map to near-identical final representations, with failures on copying/counting at modest lengths.
- Training-time entropy collapse ($H\to 0$) causes instability, and $\sigma$Reparam (spectral reparameterization of $W_Q,W_K$) fixes it (Zhai et al., ICML 2023) — measured on ViT and LM training at up to ~1B scale. This is a *different regime* from inference at $n=10^6$; the shared name is a hazard.
- Position matters independently of entropy: "lost in the middle" U-shaped accuracy (Liu et al., TACL 2024) at 20-document contexts, far below any dispersion threshold.

## 5. What Is Not Known

- **Theoretically open.** Whether the required logit growth is $\Theta(\log n)$ *per head* for retrieval in a trained multi-layer model, or whether layer composition plus sinks admits an $o(\log n)$ construction. Chiang–Cholak gives sufficiency for a hand-built single head; no necessity result for depth-$L$ trained networks.
- **Empirically open.** Whether sink-excluded $\tilde H$ at $n=10^6$ predicts retrieval failure *after* controlling for length. Runnable today on any open 1M-context model; nobody has published entropy and RULER accuracy from the same passes at that scale.
- **Methodologically blocked.** Which statistic to use. Raw $H$, $\tilde H$, sink-excluded $\tilde H$ and $\exp(H)$ can move in opposite directions on the same run, and there is no agreed head-selection rule (all heads? retrieval heads only? which layer?). Until the statistic is fixed, "entropy collapse" is not falsifiable.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability**, not compute.

Any intervention that changes attention entropy also changes the logit scale, hence the loss surface, hence what the head reads — so the observed accuracy change is not attributable to entropy. YaRN's temperature, SSMax, and DIFF Transformer each change entropy *and* effective capacity *and* effective RoPE frequency response. There is no known intervention that moves $\tilde H$ while holding the ranking of attended keys fixed, so the causal arrow between entropy and accuracy is unidentified.

Second, the statistic is not scale-free. Because $H \le \log i$ mechanically, and because sinks absorb a length-dependent share of mass, "entropy fell" and "entropy rose" are both true of the same model depending on normalization. Third, measurement costs a second attention pass under FlashAttention, so entropy is rarely logged at the lengths where it matters.

## 7. Current Research (as of 2026)

- **Sink mechanism.** Google DeepMind (Barbero, Veličković and colleagues) on why sinks form and what they do to mixing; adaptive-temperature softmax as a sharpness fix. *(frontier — verify current status)*
- **Softmax replacements.** Scalable-Softmax and $\log n$-scaled logits; selective attention (Leviathan et al., ICLR 2025) which masks out tokens rather than reweighting them.
- **Activation outliers.** Follow-on work to massive activations linking sinks to quantization failure — outlier coordinates are exactly what int8/int4 KV-cache quantization destroys. *(frontier — verify)*
- **Retrieval-head localization.** Identifying the small set of heads responsible for needle retrieval and studying only their entropy, rather than the layer average. *(frontier — verify)*
- **Benchmark hardening.** RULER-style synthetic tasks with distractor scaling, to separate dispersion failure from position bias.

## 8. Concrete Next Experiment

**Question.** Is long-context accuracy loss *caused* by attention dispersion, or merely correlated with it?

**Scale.** One open 128K model (Llama-3.1-8B-Instruct) plus one 1M model (Qwen2.5-7B-1M class). Lengths $n \in \{4\text{K}, 16\text{K}, 64\text{K}, 128\text{K}, 512\text{K}\}$. 500 RULER NIAH-multikey items per length. Recompute-pass entropy logging in fp32.

**Arms.**
1. *Base* — unmodified.
2. *Intervention* — per-head logit temperature $\tau^{(\ell,h)}(n)$ fitted so that sink-excluded $\tilde H^{(\ell,h)}$ at length $n$ equals its measured value at 4K. No weight updates.
3. *Control (the arm that makes it a real experiment)* — a **rank-preserving sham**: apply a monotone logit transform matched to arm 2 in norm and in per-head perplexity change, but chosen so $\tilde H$ is left at its uncorrected value. Any accuracy change in arm 3 is the intervention's side effect, not entropy.

**Deciding number.** $\Delta = \text{acc}(\text{arm 2}) - \text{acc}(\text{arm 3})$ at $n=128\text{K}$, on matched validation perplexity ($\pm 1\%$).

- $\Delta \ge 5$ accuracy points → entropy dispersion is causal; the statistic is sink-excluded $\tilde H$; the fix is a temperature schedule.
- $\Delta \le 1$ point → entropy is epiphenomenal at this scale, and the field should stop reporting it as an explanation.

Cost estimate: ~$10^3$ GPU-hours on 8×H100, dominated by the 512K passes.

## 9. Key References

- **[Foundational]** Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. — arXiv:1706.03762
- **[Foundational]** Zhai, Likhomanenko, Littwin, Busbridge, Ramapuram, Zhang, Gu, Susskind. *Stabilizing Transformer Training by Preventing Attention Entropy Collapse.* ICML 2023. — arXiv:2303.06296
- **[Theory]** Chiang, Cholak. *Overcoming a Theoretical Limitation of Self-Attention.* ACL 2022. — arXiv:2202.12172
- **[Theory/SOTA]** Veličković, Perivolaropoulos, Barbero, Pascanu. *softmax is not enough (for sharp out-of-distribution).* 2024. — arXiv:2410.01104
- **[SOTA]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. — arXiv:2309.17453
- **[SOTA]** Peng, Quesnelle, Fan, Shippole. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR 2024. — arXiv:2309.00071
- **[Mechanism]** Sun, Chen, Bhojanapalli, Kolter et al. *Massive Activations in Large Language Models.* COLM 2024. — arXiv:2402.17762
- **[Mechanism]** Barbero, Arroyo, Gu, Perivolaropoulos, Bronstein, Veličković et al. *Why do LLMs attend to the first token?* 2025. — arXiv:2504.02732
- **[Mechanism]** Barbero, Banino, Kapturowski, Pascanu, Veličković et al. *Transformers need glasses! Information over-squashing in language tasks.* NeurIPS 2024. — arXiv:2406.04267
- **[Method]** Ye, Dong, Xia, Wang, Wei et al. *Differential Transformer.* ICLR 2025. — arXiv:2410.05258
- **[Method]** Nakanishi. *Scalable-Softmax Is Superior for Attention.* 2025. — arXiv:2501.19399
- **[Evaluation]** Hsieh, Sun, Kriman, Ginsburg et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024. — arXiv:2404.06654
- **[Evaluation]** Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024.

## 10. Worked Example

One retrieval head, one needle, two lengths.

Take a head whose maximum logit on the needle key exceeds the bulk by $\Delta=6$ nats, with the other $i-1$ keys roughly exchangeable. Then the needle weight is

$$a_{\text{needle}} \approx \frac{e^{6}}{e^{6}+(i-1)} .$$

- At $i=4096$: $403/(403+4095)=0.090$.
- At $i=131072$: $403/(403+131071)=0.0031$.
- At $i=1{,}048{,}576$: $0.00038$.

A fixed logit gap loses a factor of 32 in needle weight from 4K to 128K. To hold $a_{\text{needle}}$ constant, the gap must grow as $\log i$: from 6 to $6+\log(32)=9.47$ nats. That is Chiang–Cholak in one line, and it is why a $\log n$ temperature is the obvious fix.

Now the obstruction. Measure raw entropy on the same head. At 4K, sink mass $\sigma=0.62$ and $H=2.1$ nats, so $\tilde H = 2.1/8.32 = 0.25$. At 128K, sink mass rises to $\sigma=0.86$ and $H$ *falls* to $1.4$ nats, $\tilde H = 0.12$. Raw and normalized entropy both say the head got **sharper**. But strip the sinks and renormalize the remaining $0.14$ of mass: sink-excluded $\tilde H$ goes from $0.71$ to $0.93$ — the head got **flatter** over the tokens it could actually retrieve from, exactly as the $0.090 \to 0.0031$ calculation predicts.

Same forward pass, same head, opposite conclusion depending on the normalization. Any paper reporting "attention entropy collapses at long context" without stating the sink treatment has reported the sink growth curve, not the dispersion curve — and the two demand opposite interventions (suppress the sink vs. sharpen the temperature). That ambiguity, not compute, is what keeps the problem open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*