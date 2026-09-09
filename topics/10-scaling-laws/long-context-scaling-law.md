---
id: 10-scaling-laws/long-context-scaling-law
title: "Scaling Laws for Long-Context Training"
topic: 10-scaling-laws
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Laws for Long-Context Training

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/long-context-scaling-law` · **Status:** empirically-open

## 1. Problem Statement

Chinchilla-style scaling laws allocate a compute budget $C$ between parameters $N$ and tokens $D$. They say nothing about a third axis: the **training context length** $c$. Given a fixed budget, how many tokens should be seen at what sequence length, and with what curriculum from short to long?

Three variants, with different difficulty:

- **Measurement.** Define a target quantity that is monotone in "long-context capability" and estimable from a single training run. Average next-token loss over a long-context corpus is estimable but nearly flat in $c$; downstream retrieval benchmarks are sensitive but discrete, saturating, and not smooth in compute.
- **Method.** Find the compute-optimal frontier $(N^\*, D^\*, c^\*, \text{schedule})(C)$ for a stated target. Solving it means: given $C$ and a target context $c_{\max}$, predicting the optimal allocation to within a few percent of loss, verified on a held-out scale.
- **Theory.** Explain why the exponent on $c$ is what it is — derive it from a data property (the decay of mutual information between distant tokens) rather than fitting it.

Solved would mean: a fitted law that extrapolates one order of magnitude in $C$ and predicts long-context *behaviour*, not just per-token loss.

## 2. Formal Setting

Let $\mathcal{D}$ be a document distribution, $x_{1:c}$ a sequence of $c$ tokens. Model $p_\theta$ has $N$ non-embedding parameters, $L$ layers, width $d$.

**Per-position loss.** Measured by evaluating on documents of length at least $c$ and bucketing by position $i$:
$$\ell(i) = \mathbb{E}_{x\sim\mathcal{D}}\big[-\log p_\theta(x_i \mid x_{<i})\big], \qquad L(c) = \frac{1}{c}\sum_{i=1}^{c}\ell(i).$$
The context-length law proposed by Xiong et al. (2024) fits the *per-position* curve:
$$\ell(i) \approx \left(\frac{\alpha}{i}\right)^{\beta} + \gamma,$$
with $\gamma$ the irreducible term. $\beta$ is the quantity of interest: it measures how much a token is worth as context.

**Compute.** Per training token, including the quadratic attention term (Kaplan et al., 2020):
$$C_{\text{tok}} \approx 6N + 12\,L\,d\,c, \qquad \frac{\text{attention}}{\text{dense}} = \frac{12Ldc}{6N} \approx \frac{c}{6d}$$
using $N \approx 12Ld^2$. So $c = 6d$ is the crossover where attention doubles per-token cost. Total budget $C = D \cdot C_{\text{tok}}$.

**The joint law to be fitted.** The natural extension of Hoffmann et al. (2022):
$$L(N, D, c) = E + \frac{A}{N^{a}} + \frac{B}{D^{b}} + \frac{F}{c^{\,\beta}},$$
which assumes the context term is **separable** from $N$ and $D$.

**Assumptions, and which are violated.**
1. *Separability.* Violated: $\beta$ measurably depends on $N$ — larger models extract more from distant context.
2. *I.i.d. documents of length $\geq c$.* Violated: web corpora are dominated by short documents; long batches are built by concatenation, so much "long context" is cross-document noise. Fu et al. (2024) show per-source length distribution changes the result.
3. *Loss is the target.* Violated in the sense that matters: $L(c)$ is dominated by short-range prediction. Position-averaged loss compresses the effect of interest into the last few percent.
4. *Position encoding is fixed.* Violated: RoPE base frequency, interpolation scheme, and attention-sink handling all change $\beta$, so any fitted law is conditional on a positional-encoding choice.
5. *Dense attention.* Violated by every production long-context system (sliding windows, sparse/linear layers), which changes the $c$ term in $C_{\text{tok}}$ from linear to constant.

## 3. State of the Art

**Established.**
- Hoffmann et al. (NeurIPS 2022) fit $L(N,D) = 1.69 + 406.4/N^{0.34} + 410.7/D^{0.28}$ at fixed short context (2048). $N^\* \propto C^{0.5}$, $D^\* \propto C^{0.5}$. This is the reference law any long-context extension must reduce to.
- Position-interpolation methods extend a short-context model's window cheaply: Chen et al. (2023) PI, Peng et al. (YaRN, ICLR 2024), Ding et al. (LongRoPE, ICML 2024). Established: window extension needs orders of magnitude less compute than pretraining long.
- Xiong et al. (NAACL 2024, Llama 2 Long) show continued pretraining on ~400B tokens at 32k recovers short-context quality and fits the power-law-plus-constant form above. This is the closest thing to a published context-length scaling law.

**Claimed but unablated.** That "long-context ability emerges from data mixture, not architecture" (Fu et al., ICML 2024) rests on a single 80k-token continued-pretraining setting with ~5B tokens; it is not a compute-matched comparison against pretraining long from scratch. Vendor claims of 1M–10M context (Gemini 1.5, 2024) are benchmark numbers on synthetic retrieval, with no released loss-vs-compute curves.

**Benchmark-only results.** RULER (Hsieh et al., COLM 2024) and BABILong (Kuratov et al., NeurIPS 2024) are the standard effective-length probes. RULER's headline — many models advertising 32k+ fail to hold performance well before their claimed length — is a benchmark ranking, not a scaling law: it has no compute axis.

## 4. What Is Known

- **The context term is small in loss.** Per-position loss keeps falling out to at least 32k tokens with no visible plateau (Xiong et al., 2024, 7B–70B, 32k), but the *average* loss improvement from 4k → 32k on general web text is on the order of a few hundredths of a nat.
- **Attention cost crossover.** For $d = 4096$ (an 8B-class model), $c = 6d \approx 24.6$k. At $c = 128$k, per-token cost is $\approx 6.3\times$ the dense-parameter cost — measured arithmetic from $C_{\text{tok}}$, not an empirical fit.
- **Extension is cheap relative to pretraining.** LongRoPE and YaRN report 128k–2M windows from a few thousand fine-tuning steps; Llama 2 Long used ~5% of pretraining compute for 32k.
- **Data length distribution dominates.** Fu et al. (2024) hold compute fixed and vary only the source mixture at 80k context; per-domain upsampling changes needle-retrieval accuracy from near-chance to near-perfect. Scale: 7B, ~5B tokens.
- **Position use is non-uniform.** "Lost in the middle" (Liu et al., TACL 2024) — retrieval accuracy is U-shaped in the target's position, so average-loss laws hide a large positional effect.
- **Effective $\ll$ claimed.** An et al. (ICLR 2025) attribute the gap to left-skewed relative-position frequency in training, and recover much of it at inference with no training.

## 5. What Is Not Known

- **Empirically open (the main gap).** Nobody has published a compute-matched IsoFLOP sweep over $(N, D, c)$ where $c$ is a swept axis rather than a fixed configuration. All the ingredients exist; the sweep costs 10⁵ GPU-hours and nobody has spent it as a scaling study.
- **Empirically open.** Whether $\beta$ is scale-invariant. If $\beta = \beta(N)$, the separable form is wrong and every extrapolation from small proxies fails.
- **Empirically open.** The short-to-long curriculum. Is one-shot extension at 95% of budget optimal, or a staged ramp? No compute-matched comparison exists.
- **Methodologically blocked.** The target itself. There is no metric that is (i) smooth in compute, (ii) sensitive to long-range structure, and (iii) not a synthetic retrieval task. Until one exists, "compute-optimal for long context" has no well-posed objective.
- **Theoretically open.** No derivation of $\beta$ from the source's long-range statistics. Natural-language mutual information decays roughly as a power law in distance; connecting that exponent to $\beta$ is unproved either way.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure the thing it names**, compounded by cost.

Average loss $L(c)$ is dominated by tokens whose prediction needs a few hundred tokens of context. Extra context moves it by ~1%. A scaling law fitted to a signal that small is dominated by data-mixture and tokenization confounds. Meanwhile the sensitive metrics — needle retrieval, variable tracking — are near-binary, saturate, and are not differentiable functions of compute, so they cannot be fitted with a power law at all.

Second obstruction: **confounded measurement**. Changing $c$ changes the batch composition (fewer documents per batch, more cross-document concatenation), the effective batch size in tokens, the optimal learning rate, and the positional-encoding regime — all at once. An observed $\Delta L$ cannot be attributed to $c$ without controlling four other variables.

Third: **cost**. A proper sweep needs $\geq 4$ model sizes $\times$ $\geq 4$ context lengths $\times$ $\geq 3$ token budgets, with the $c=128$k arms costing $6\times$ per token.

## 7. Current Research (as of 2026)

- **Data-centric long-context training.** Princeton NLP's ProLong line (Gao et al., 2024) argues length-aware mixture and short-context retention are the dominant factors; it publishes a recipe, not a law.
- **Efficient attention as a way to change the exponent.** Hybrid sliding-window/full-attention stacks (Character.AI, Mistral, Google) make the $c$ term in $C_{\text{tok}}$ nearly constant, which would reshape the frontier entirely. Whether hybrid models obey the same $\beta$ is untested *(frontier — verify)*.
- **Effective-context diagnostics.** An et al. (ICLR 2025) and follow-ups treat relative-position frequency as the controllable variable.
- **RL and reasoning-length interaction.** Long-context pretraining budget vs. long-chain-of-thought RL budget is now the live allocation question at frontier labs *(frontier — verify)*; no public numbers.

## 8. Concrete Next Experiment

**Scale.** Four models: 150M, 400M, 1.1B, 3B (non-embedding), decoder-only, RoPE with base tuned per context length. Four training contexts: 4k, 16k, 64k, 256k. Two token budgets per cell on a shared IsoFLOP grid. Fixed corpus with a fixed per-source *document*-length distribution — every arm sees identical documents, only the packing window changes. Estimated cost: ~$1.5\times10^{5}$ A100-hours.

**Control arm.** Train at 4k to the full budget, then spend the *same total FLOPs* by continued pretraining at 256k for the final 5%, using YaRN interpolation. This is the "extend late, cheaply" hypothesis and the industry default.

**The deciding number.** Fit $\ell(i) = (\alpha/i)^\beta + \gamma$ separately for each $(N, c)$ cell and report
$$\frac{\partial \log \beta}{\partial \log N}.$$
If it is statistically indistinguishable from 0 across 150M→3B, the separable law holds and small-scale proxies extrapolate: publish $L(N,D,c)$ and the frontier. If $|\partial\log\beta/\partial\log N| > 0.05$ with the sign positive, long-context value grows with model size, the control arm is systematically undertrained for long context, and every extend-late recipe is leaving capability on the table at the frontier. Secondary readout: RULER effective length at 128k for the swept arms vs. the control, to check whether $\beta$ and effective length even move together.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, Brown, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Xiong, Liu, Molybog, et al. *Effective Long-Context Scaling of Foundation Models.* NAACL, 2024. — arXiv:2309.16039
- **[SOTA]** Fu, Panda, Niu, et al. *Data Engineering for Scaling Language Models to 128K Context.* ICML, 2024. — arXiv:2402.10171
- **[SOTA]** Peng, Quesnelle, Fan, Shippole. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR, 2024. — arXiv:2309.00071
- **[SOTA]** Ding, Zhang, Zhang, et al. *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens.* ICML, 2024. — arXiv:2402.13753
- **[Measurement]** Hsieh, Sun, Kriman, et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Measurement]** Liu, Lin, Hewitt, et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Measurement]** Kuratov, Bulatov, Anokhin, et al. *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack.* NeurIPS Datasets & Benchmarks, 2024.
- **[Analysis]** An, Zhang, Zhou, et al. *Why Does the Effective Context Length of LLMs Fall Short?* ICLR, 2025. — arXiv:2410.18745
- **[Recipe]** Gao, Wettig, Yen, Chen. *How to Train Long-Context Language Models (Effectively).* ACL, 2025. — arXiv:2410.02660
- **[Foundational]** Chen, Wong, Chen, Tian. *Extending Context Window of Large Language Models via Positional Interpolation.* 2023. — arXiv:2306.15595

## 10. Worked Example

Take an 8B-class model, $d = 4096$, fixed FLOP budget $C$. Compare training at $c = 8$k against $c = 128$k.

Per-token cost multiplier relative to dense-only, $1 + c/(6d)$:

```
c =   8,192 :  1 + 8192/24576   = 1.33
c = 131,072 :  1 + 131072/24576 = 6.33
```

Same $C$ buys $6.33/1.33 = 4.76\times$ fewer tokens at 128k.

Price that in Chinchilla terms. The data term is $B/D^{0.28}$ with $B = 410.7$. At $D = 10^{12}$ that term is $410.7 / (10^{12})^{0.28} = 410.7/e^{7.74} \approx 0.178$ nats. Cutting $D$ by $4.76\times$ multiplies it by $4.76^{0.28} = 1.55$:

```
loss penalty = 0.178 x (1.55 - 1) = +0.098 nats
```

Now the credit side. Using $\ell(i) = (\alpha/i)^\beta + \gamma$ with the shallow exponents reported for continued long-context pretraining, the *position-averaged* gain from extending the window 8k → 128k on general web text is on the order of $0.01$–$0.03$ nats — the last 94% of positions each gain a little, and short-range-predictable tokens gain nothing.

```
net: -0.098 (fewer tokens)  +0.02 (longer context)  = -0.08 nats
```

**The obstruction, made visible.** Under the metric the scaling-law literature actually uses, long-context pretraining is a clear loss — by about 4× more than the gain. Yet a model trained only at 8k cannot do the tasks people buy long-context models for. So either the metric is wrong for the purpose, or long context is not worth compute. The literature has quietly assumed the first and never measured it: no published study reports a compute-matched arm where the decision flips. That is what makes this empirically open rather than settled.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*