---
id: 02-attention/depth-vs-attention-width-composition
title: "Depth versus Attention Width for Composition Tasks"
topic: 02-attention
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Depth versus Attention Width for Composition Tasks

> **Topic:** Attention Mechanisms · **ID:** `02-attention/depth-vs-attention-width-composition` · **Status:** partially-solved

## 1. Problem Statement

A *composition task* asks a model to apply a relation $k$ times: given a pointer table and a start token, return the $k$-th successor ($k$-hop induction), or evaluate $f_k \circ \cdots \circ f_1(x)$ from a table of the $f_i$. Attention is a parallel operator, so each layer performs roughly one round of routing. The question is how the two resources trade off:

- **Theory variant.** For a decoder-only transformer with $L$ layers, $H$ heads per layer, head dimension $d_h$, embedding width $d$, and $p$-bit precision, what is the exact frontier of $(L, H, d, p)$ that admits an exact solver for $k$-hop on context length $N$? Is depth ever *replaceable* by width — is there a width $d(k)$, polynomial in $k$, that lets a fixed-depth transformer do $k$-hop?
- **Method variant.** Given a fixed parameter or FLOP budget $C$, choose the aspect ratio $\rho = d/L$ that maximises composition accuracy. Is the optimal $\rho$ for composition different from the optimal $\rho$ for next-token loss?
- **Measurement variant.** Does *any* natural-language benchmark measure serial composition depth, as opposed to memorised shortcuts plus a shallow lookup?

Solved would mean: a matching upper and lower bound in $(L, d, H, p)$ for $k$-hop, plus a trained-model experiment showing that the empirically optimal depth tracks the bound rather than the loss-scaling optimum.

## 2. Formal Setting

Input: a sequence $x_{1:N} \in \Sigma^N$ encoding a function table and a query. For $k$-hop, $x_i \in \Sigma$ and the task is $\mathrm{hop}^k(x, i)$, where $\mathrm{hop}^1(x,i) = \max\{ j < i : x_j = x_i \} + 1$ and $\mathrm{hop}^{k}= \mathrm{hop}^1 \circ \mathrm{hop}^{k-1}$.

A transformer layer is $\mathrm{Attn}(X) = \sum_{h=1}^{H} \mathrm{softmax}(X W_Q^h (X W_K^h)^\top / \sqrt{d_h}) X W_V^h W_O^h$ followed by an MLP. Measured quantities:

- **Depth** $L$: number of blocks. Measured by counting; the effective serial budget is $L$ (one routing hop per block), not $L \cdot H$.
- **Attention width**: report all three, they are not interchangeable. Embedding width $d$; per-layer head budget $H d_h$ (usually $= d$); and the *communication width* $m = H d_h p$ bits, the quantity lower bounds actually constrain.
- **Precision** $p$: bits per activation entry, measured at inference (bf16 $\Rightarrow p=16$, but the *usable* $p$ after softmax saturation is smaller and is what bounds assume).
- **Budget** $C$: non-embedding parameters $\approx 12 L d^2$, or FLOPs $\approx 6 C_{\text{param}} T$ per token; state which.
- **Aspect ratio** $\rho = d/L$. GPT-3 175B: $d=12288$, $L=96$, $\rho=128$. Llama-3 70B: $d=8192$, $L=80$, $\rho=102$.
- **Composition accuracy** $A_k$: exact-match on held-out tables at hop count $k$, with tables resampled per example so memorisation is impossible.

Assumptions and their status: (i) *one hop per layer* — violated, because a layer with $H$ heads can perform several independent hops in parallel and the MLP can precompute a partial composition table when the domain is small; (ii) *log-precision* — the standard $\mathsf{TC}^0$ results assume $p = O(\log N)$, which is violated by fp32 at short $N$ and vacuous at long $N$; (iii) *no chain of thought* — violated by every deployed system, and CoT changes the class outright.

## 3. State of the Art

**Theory SOTA (established).** Sanford, Hsu & Telgarsky (ICML 2024) give a matched pair for $k$-hop: an $O(\log k)$-depth construction with width independent of $N$, and a lower bound showing a *single* layer needs embedding width $\tilde\Omega(N)$ (equivalently, $L=1$ with subpolynomial width fails). Peng, Narayanan & Papadimitriou (COLM 2024) prove, by a communication-complexity reduction, that a one-layer multi-head transformer cannot compose two functions on a domain of size $n$ unless $H d_h p = \tilde\Omega(n)$ — width must be *linear* in the domain to buy one hop. Merrill & Sabharwal (TACL 2023) place log-precision transformers in uniform $\mathsf{TC}^0$, so no constant-depth model computes an $\mathsf{NC}^1$-hard composition (e.g. $S_5$ word problems) unless $\mathsf{TC}^0 = \mathsf{NC}^1$.

**Established, complementary.** Liu et al. (ICLR 2023) show semiautomata over $T$ steps are simulable in $O(\log T)$ depth — the "shortcut" — so depth requirement is logarithmic, not linear, in the composition length.

**Empirical SOTA.** Tay et al. (ICLR 2022) find DeepNarrow T5 variants match T5-Base with ~50% fewer parameters. Petty et al. (NAACL 2024) isolate depth at fixed parameter count on compositional generalization. Both are *benchmark numbers under one training recipe*, not ablated against learning-rate/initialisation scaling; deep-narrow models are harder to optimise, and neither study used $\mu$P-style width transfer, so the optimiser is a confound.

**Claimed but unablated.** That production aspect ratios ($\rho \approx 100$) are near-optimal for reasoning. The scaling-law evidence (Kaplan et al. 2020) says loss is *insensitive* to shape over $0.5 \le \log_{10}\rho \le 2.5$ — it says nothing about composition depth, and no one has re-run that sweep with a composition metric as the target.

## 4. What Is Known

- $k$-hop is solvable at depth $\lceil \log_2 k \rceil + O(1)$ with width polylogarithmic in $N$ (Sanford et al., ICML 2024). Verified empirically on synthetic tasks with $N$ up to $10^2$–$10^3$; the same paper reports GPT-4 degrading sharply beyond small $k$ on $k$-hop prompts.
- Depth $1$ is provably insufficient at any subpolynomial width for two-function composition (Peng et al., COLM 2024): the threshold is $Hd_hp \gtrsim n$ for domain size $n$. At $n = 10^4$ and $p=16$, that demands $Hd_h \gtrsim 625$ — achievable, which is exactly why small-domain composition looks "solved" in one layer and misleads.
- Induction (the $k=1$ case) needs two layers: the induction-head circuit is a previous-token head composed with a matching head (Elhage et al. 2021; Olsson et al. 2022), observed at every scale from 2-layer attention-only models to 13B.
- Depth helps compositional generalization at fixed parameters, with saturation. Petty et al. (NAACL 2024) vary depth from 2 to 32 at roughly constant parameter count on COGS/GeoQuery-style splits: out-of-distribution compositional accuracy rises with depth, most of the gain is realised by the first handful of layers, and in-distribution accuracy is nearly depth-insensitive.
- Levine et al. (NeurIPS 2020) derive a depth-efficiency threshold $L \lesssim \log_3 d$: below it depth buys expressivity exponentially, above it width is the better marginal spend. For $d = 12288$, $\log_3 d \approx 8.6$ — far below the deployed $L=96$, i.e. real models sit deep in the width-favouring regime by that criterion.
- Chain of thought raises the ceiling: $T$ decoding steps give constant-depth transformers power up to $\mathsf{P}$-completeness territory for polynomial $T$ (Li et al., ICLR 2024; Merrill & Sabharwal, ICLR 2024). Serial depth can be bought in *time* instead of layers.

## 5. What Is Not Known

- **Theoretically open.** Whether the $O(\log k)$ depth upper bound is tight for $k \ge 3$ at polylogarithmic width. No lower bound rules out a depth-2, $\mathrm{poly}(k)$-width solver for $k$-hop; the known lower bounds cover $L=1$, and multi-layer unconditional lower bounds (Chen, Peng & Wu, 2024) apply only to restricted regimes. The general depth hierarchy for softmax attention is unproven.
- **Empirically open.** The fixed-FLOP aspect-ratio sweep with composition accuracy as the objective. Runnable today at 1B parameters for well under $10^{21}$ FLOPs; nobody has published it with $\mu$P-controlled optimisation, so depth and trainability remain entangled.
- **Empirically open.** Whether pretrained LLMs realise the $\log k$ construction or a memorised shortcut. Circuit-level evidence exists only for $k\le 2$.
- **Methodologically blocked.** "Composition depth" of a natural benchmark. GSM8K, BBH and MMLU have no annotated serial-hop count, and any measured depth is confounded with pretraining exposure. Without ground-truth $k$ per item, the depth–accuracy curve on real tasks is uninterpretable.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus non-identifiability**, not compute.

1. Changing $L$ at fixed $C$ changes optimisation, not just expressivity. Deep-narrow models need different learning rates and warmup; a depth ablation without width-transfer parameterisation measures the optimiser as much as the architecture.
2. Width can *simulate* depth by tabulation. If the domain is small enough, an MLP memorises $f_2 \circ f_1$ and the hop disappears. Every "wide shallow model succeeds" result is ambiguous between real parallel routing and table lookup, and the two are not distinguishable from accuracy alone.
3. Chain of thought makes the layer count non-binding. Any depth deficit is patchable with more decoding steps, so the deployed system never exhibits the architectural limit the theory describes.
4. The lower-bound technique (communication complexity across an attention layer) is naturally a *one-round* argument. Extending it to $L$ rounds is the same obstacle as multi-round communication lower bounds — a known hard problem borrowed, not created, by this field.

## 7. Current Research (as of 2026)

- **Depth-parameterised expressivity.** Follow-ups to Sanford–Hsu–Telgarsky pushing $L$-layer lower bounds; Chen, Peng & Wu (2024) give the first unconditional multi-layer decoder lower bounds via an indistinguishability argument. Columbia/NYU, Berkeley theory groups.
- **Formal-language depth hierarchies.** Merrill & Sabharwal (AI2/NYU) mapping fixed-depth, CoT-length and precision classes; the $\mathsf{TC}^0$/$\mathsf{NC}^1$ boundary is the live frontier.
- **Mechanistic verification of hop circuits.** Extending induction-head analysis to $k>2$ in mid-size open models *(frontier — verify)*.
- **Architectures that decouple serial depth from parameters**: universal/looped transformers and recurrent-depth models, which run $L_{\text{eff}} \gg L_{\text{params}}$ passes at inference — the cleanest test of whether composition wants depth or wants parameters *(frontier — verify)*.
- **Shape-aware scaling laws.** Whether the shape-insensitivity of the loss survives when the target is a reasoning metric. Not settled.

## 8. Concrete Next Experiment

**Question:** at fixed compute, is composition accuracy governed by $L$ alone or by $Hd_hp$?

- **Scale.** Six models at ~400M non-embedding parameters, iso-FLOP ($\approx 8\times10^{20}$, 8B tokens), aspect ratios $\rho = d/L \in \{1024/8,\ 768/14,\ 640/20,\ 512/32,\ 384/56,\ 256/128\}$. Pretrain on the same corpus; use $\mu$P so the learning rate transfers across width and depth. Cost: roughly 300–500 A100-days total.
- **Probe.** Held-out $k$-hop with tables resampled per example, domain size $n = 8192$ (large enough that MLP tabulation is ruled out by the $Hd_hp \gtrsim n$ threshold for the widest model), $k \in \{1,2,4,8,16\}$, no chain of thought, single forward pass.
- **Control arms.** (a) Width-matched, depth-matched *shuffled-table* task, which needs no composition — any depth trend here is optimisation artefact, not composition. (b) The same models with CoT allowed; if the depth effect vanishes, layer count is not the binding resource in practice.
- **Deciding number.** $k^\star(L)$ — the largest $k$ with $A_k \ge 90\%$ — regressed against $\log_2 L$ and against $\log_2 (Hd_h)$. If $k^\star \approx 2^{cL}$ with $c \in [0.5, 1]$ and no significant width coefficient, the $\log$-depth theory governs trained models. If the width coefficient is significant after the shuffled-table control, depth is substitutable and the practical problem is open in the other direction. Report a single scalar: the partial $R^2$ of $\log_2 L$ over $\log_2(Hd_h)$ in predicting $k^\star$; $>0.7$ settles it for depth.

## 9. Key References

- **[Foundational]** Sanford, C., Hsu, D., Telgarsky, M. *Representational Strengths and Limitations of Transformers.* NeurIPS 2023. — arXiv:2306.02896
- **[SOTA]** Sanford, C., Hsu, D., Telgarsky, M. *Transformers, Parallel Computation, and Logarithmic Depth.* ICML 2024. — arXiv:2402.09268
- **[SOTA]** Peng, B., Narayanan, S., Papadimitriou, C. *On Limitations of the Transformer Architecture.* COLM 2024. — arXiv:2402.08164
- **[Foundational]** Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A., Zhang, C. *Transformers Learn Shortcuts to Automata.* ICLR 2023. — arXiv:2210.10749
- **[Foundational]** Merrill, W., Sabharwal, A. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL, 2023. — arXiv:2207.00729
- **[SOTA]** Li, Z., Liu, H., Zhou, D., Ma, T. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR 2024. — arXiv:2402.12875
- **[Empirical]** Petty, J., van Steenkiste, S., Dasgupta, I., Sha, F., Garrette, D., Linzen, T. *The Impact of Depth on Compositional Generalization in Transformer Language Models.* NAACL 2024. — arXiv:2310.19956
- **[Empirical]** Tay, Y., Dehghani, M., Rao, J., Fedus, W., Abnar, S., Chung, H. W., Narang, S., Yogatama, D., Vaswani, A., Metzler, D. *Scale Efficiently: Insights from Pretraining and Finetuning Transformers.* ICLR 2022. — arXiv:2109.10686
- **[Foundational]** Levine, Y., Wies, N., Sharir, O., Bata, H., Shashua, A. *Limits to Depth Efficiencies of Self-Attention.* NeurIPS 2020.
- **[Empirical]** Dziri, N., et al. *Faith and Fate: Limits of Transformers on Compositionality.* NeurIPS 2023. — arXiv:2305.18654
- **[Theory]** Chen, L., Peng, B., Wu, H. *Theoretical Limitations of Multi-Layer Transformer.* 2024.
- **[Survey]** Strobl, L., Merrill, W., Weiss, G., Chiang, D., Angluin, D. *What Formal Languages Can Transformers Express? A Survey.* TACL, 2024.

## 10. Worked Example

Take $k=4$-hop on a table with domain size $n = 8192$, context $N = 1024$, bf16 ($p=16$).

**Depth route.** The pointer-doubling construction composes $\mathrm{hop}^1 \to \mathrm{hop}^2 \to \mathrm{hop}^4$: $\lceil \log_2 4 \rceil = 2$ routing layers plus one for the base hop, so $L = 3$ suffices with $d$ polylog in $N$ — say $d = 256$. Parameter cost $\approx 12Ld^2 = 12\cdot 3\cdot 256^2 \approx 2.4$M.

**Width route.** To do it in one layer, the communication bound requires $Hd_hp = \tilde\Omega(n)$, i.e. $Hd_h \gtrsim 8192/16 = 512$ *per hop composed*, and the honest reading of the argument for 4 hops needs the layer to carry the full composed table, $Hd_h \gtrsim 8192 \cdot 3/16 \approx 1536$. At $L=1$ that costs $\approx 12 \cdot 1 \cdot 1536^2 \approx 28$M parameters — an order of magnitude more, and it scales *linearly in $n$* while the depth route scales as $\log k$.

**Where the obstruction becomes visible.** Now shrink the domain to $n = 64$. The width threshold drops to $Hd_h \gtrsim 4$, so a single wide layer's MLP can memorise all $64^2 = 4096$ composed pairs outright. A 1-layer, $d=512$ model will hit ~100% on 4-hop at $n=64$ — and this is exactly the configuration most published synthetic composition experiments use. The measurement cannot tell "learned parallel routing" from "tabulated the answer": both give $A_4 = 1.0$. Unless the probe fixes $n$ above the $Hd_hp$ threshold of the widest arm (the reason §8 sets $n=8192$), the depth–width experiment returns a number that is real, reproducible, and about the wrong mechanism.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*