---
id: 14-long-context/context-length-loss-scaling-law
title: "Context Length Scaling Laws for Loss"
topic: 14-long-context
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Context Length Scaling Laws for Loss

> **Topic:** Long Context · **ID:** `14-long-context/context-length-loss-scaling-law` · **Status:** open

## 1. Problem Statement

Parameter count $N$ and token count $D$ have well-fit joint scaling laws for language-model cross-entropy. Context length $c$ does not. The question: **is there a stable functional form $L(c)$, and does it compose with $L(N,D)$?**

Three variants, different difficulties:

- **Measurement.** Given a fixed model and a fixed corpus, what is the loss as a function of position-in-context $t$, and how should it be aggregated into a single "context length" number? This is where most confusion lives — the standard aggregate (mean loss over a $c$-token window) mixes a genuine long-range-information effect with a warm-up artifact at small $t$.
- **Method.** Given a compute budget $C$, what $(N, D, c)$ triple minimises loss at a target deployment context? Attention cost is quadratic in $c$ at fixed $N$, so $c$ trades against $N$ and $D$ in a way $D$ and $N$ do not trade against each other.
- **Theory.** Does a power-law-plus-constant $L(c) = L_\infty + A c^{-\alpha}$ follow from any model of natural-language long-range structure, and is $\alpha$ predicted by the mutual-information decay exponent of the text distribution?

Solving it means: a fitted form that extrapolates — fit on $c \le 8\text{k}$, predict loss at $c = 128\text{k}$ within measurement noise, on held-out corpora and at a model scale not in the fit.

## 2. Formal Setting

Let $x_{1:T} \sim \mathcal{D}$ be a document, model $p_\theta$, and define the **per-position loss**

$$\ell(t) \;=\; \mathbb{E}_{x \sim \mathcal{D}}\!\left[-\log p_\theta\!\left(x_t \mid x_{<t}\right)\right], \qquad t = 1,\dots,c .$$

Measured as: sample $M$ documents of length $\ge c$, run one forward pass each at context $c$, average the token-level NLL at index $t$ across documents. Standard error at $M = 2000$ documents is roughly $0.005$ nats per position — comparable to the effect being measured, which is why $M$ must be reported.

The **windowed loss** usually called "loss at context $c$" is the aggregate

$$L(c) \;=\; \frac{1}{c}\sum_{t=1}^{c} \ell(t) ,$$

and the **marginal** (far more informative) is $\ell(c)$ itself. The two differ: $L(c)$ inherits the large early-position losses forever, so $L(c) - L(\infty)$ decays even when $\ell(t)$ has been flat for decades of $t$.

Candidate forms:

$$\ell(t) = \ell_\infty + A\,t^{-\alpha} \quad\text{(power law)}, \qquad \ell(t) = \ell_\infty + A e^{-t/\tau} \quad\text{(finite correlation length)} .$$

The composed law under test:

$$L(N, D, c) \;=\; E \;+\; \frac{A}{N^{\alpha}} \;+\; \frac{B}{D^{\beta}} \;+\; \frac{G}{c^{\gamma}},$$

with the strong hypothesis being that $\gamma$ is independent of $N$ and $D$ (separability).

Compute accounting must be explicit, since it is what makes $c$ different:

$$C \;\approx\; 6ND \;+\; 12\,L_{\text{layers}}\,d_{\text{model}}\,D\,c ,$$

the second term being attention, negligible when $c \ll d_{\text{model}}\!\cdot\!L/\text{(head count)}$ and dominant beyond.

**Assumptions, and which are violated:**

1. *Documents are longer than $c$.* Violated: web corpora are mostly short. Packing multiple documents into a window injects hard resets and makes $\ell(t)$ measure document-boundary statistics, not long-range structure. Known-violated and rarely controlled.
2. *The evaluation corpus is stationary in $t$.* Violated: books and code have position-correlated structure (front matter, imports) so $\ell(t)$ falls partly because *later text is intrinsically easier*, independent of any conditioning benefit.
3. *Separability of $c$ from $N,D$.* Untested at scale; the $\ell_\infty$ term almost certainly depends on $N$.
4. *One tokenizer, one corpus.* $\alpha$ is a property of $\mathcal{D}$, not the architecture, under the mutual-information hypothesis — so any cross-paper comparison of $\alpha$ across corpora is invalid.

## 3. State of the Art

**Established.**
- Kaplan et al. (2020) measured $\ell(t)$ over a 1024-token context and reported monotone decrease with position, with the improvement concentrated in the first few hundred tokens. The reported curve is not a clean single power law on log-log axes.
- Hoffmann et al. (2022, Chinchilla) fit $L(N,D) = E + A N^{-0.34} + B D^{-0.28}$ with $E \approx 1.69$ nats at fixed context. Context is a constant of that fit, not a variable.
- Olsson et al. (2022) operationalised the long-range component as an **in-context learning score**: $\ell(50) - \ell(500)$, and showed it undergoes a phase change coincident with induction-head formation. This is a two-point difference, not a law.

**Claimed but unablated.**
- Xiong et al. (*Effective Long-Context Scaling of Foundation Models*, NAACL 2024) fit validation loss against context length with a power-law-plus-constant and report that larger models have a steeper exponent — i.e. long context helps big models more. The fit is over one model family, one data mixture, and one context-extension recipe; the confound of assumption (1) above is not ablated.
- Fu et al. (ICML 2024) and Gao et al. (ProLong, 2025) show that data mixture — specifically the fraction of genuinely long documents — dominates the loss achieved at 128k. This makes any published $\gamma$ a joint property of model and mixture.

**Benchmark-only.** RULER (Hsieh et al., COLM 2024) and Liu et al. (*Lost in the Middle*, TACL 2024) are downstream-accuracy results, not loss laws. RULER's finding that "effective" context is often far below advertised context has **no established mapping** to any feature of $\ell(t)$.

## 4. What Is Known

- Loss falls monotonically with position for every autoregressive LM measured; the bulk of the drop is inside the first $\sim 10^3$ tokens (Kaplan et al. 2020, 117M–1.5B; replicated widely).
- The Chinchilla residual $E \approx 1.69$ nats is measured at $c = 2048$. Nobody has shown $E$ is context-independent.
- Position-interpolation methods (Chen et al. 2023; YaRN, Peng et al. ICLR 2024; LongRoPE, Ding et al. ICML 2024) extend usable context $8$–$64\times$ with modest continued pretraining, and perplexity stays flat rather than diverging out to 128k–2M claimed positions. Flat perplexity is a *non-divergence* result; it is not evidence of information use.
- ALiBi (Press et al., ICLR 2022) extrapolates perplexity beyond training length, but its effective attention window is bounded — extrapolating perplexity and using long-range information are demonstrably different properties.
- Downstream accuracy degrades from input lengths as low as a few thousand tokens even when the task is length-invariant (Levy et al., ACL 2024), while windowed perplexity over the same range is essentially flat. **Loss and capability decouple.**

## 5. What Is Not Known

- **Theoretically open.** Whether $\ell(t)$ should be a power law at all. If natural language has mutual information $I(X_{t}; X_{t-k}) \sim k^{-\eta}$, a decay exponent for $\ell$ follows heuristically, but no theorem connects the two for a finite-capacity model. Neither the data-manifold-dimension account (Sharma & Kaplan, 2022) nor the quantization account (Michaud et al., NeurIPS 2023) has been extended to the context axis.
- **Empirically open.** Separability. The experiment — an $(N, D, c)$ grid with $c$ swept $2$k–$256$k at three model scales on a fixed long-document corpus — is runnable at a few hundred thousand GPU-hours. It has not been published.
- **Methodologically blocked.** The definition of "loss at context $c$". Until $\ell(t)$ is decomposed into (i) conditioning benefit, (ii) intrinsic position-dependent difficulty of the corpus, and (iii) document-packing artifacts, no reported $\gamma$ is comparable across papers.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by irreducible-term non-identifiability**.

Fitting $\ell_\infty + A t^{-\alpha}$ over a range of $t$ spanning less than two decades gives a near-flat likelihood ridge in $(\ell_\infty, \alpha)$: a small $\alpha$ with low $\ell_\infty$ and a large $\alpha$ with high $\ell_\infty$ fit the same data to within the $\pm 0.005$-nat measurement noise. Extending the range costs quadratically — going from $8$k to $128$k is $16\times$ in tokens and up to $256\times$ in attention FLOPs — and the extension itself requires long documents, which are scarce, so the corpus changes as $c$ grows. Every attempt to break the degeneracy by widening the range simultaneously changes $\mathcal{D}$.

Secondary: the quantity people care about (does the model *use* position $t$?) is not what $\ell(t)$ measures. A model that ignores everything beyond a 4k window still shows falling $\ell(t)$, because $t$ correlates with within-window document progress.

## 7. Current Research (as of 2026)

- **Data-centric long-context pretraining.** Princeton NLP (ProLong; Gao, Wettig, Yen, Chen, 2025) and the Fu et al. line: the finding is that mixture, not architecture, sets the achievable 128k loss.
- **Effective-context measurement.** NVIDIA (RULER) and follow-on work replacing needle retrieval with length-controlled synthetic tasks.
- **Sub-quadratic architectures.** Mamba/SSM and hybrid-attention families change the compute term in $C$, so the optimal $c$ under a budget is architecture-dependent; joint $(N,D,c)$ laws for hybrids are *(frontier — verify)*.
- **Perplexity-decomposition metrics.** Proposals to weight tokens by how much they depend on distant context, rather than averaging uniformly, are active but not standardised *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Three model sizes — 150M, 600M, 2.4B — each trained Chinchilla-optimally on a *single* corpus filtered to documents $\ge 256$k tokens (books, long code repos, legal), with **no cross-document packing**. Sweep $c \in \{2\text{k}, 8\text{k}, 32\text{k}, 128\text{k}\}$ at fixed total token budget. Nine to twelve runs; roughly $1$–$3\times10^5$ A100-hours.

**Control arm.** For each $(N, c)$, a **shuffled-context twin**: identical model and tokens, but every position beyond a 2k window is replaced by tokens from a different document. This holds intrinsic position-difficulty and packing structure fixed while destroying long-range information. The conditioning benefit is $\Delta(t) = \ell_{\text{shuffled}}(t) - \ell_{\text{true}}(t)$.

**Deciding number.** Fit $\Delta(t) = A t^{-\gamma}$ on $t \in [2\text{k}, 8\text{k}]$ separately per model size, then extrapolate to $t = 128\text{k}$. The single decisive quantity is the **relative extrapolation error at $t = 128$k**. If it is $< 10\%$ for all three sizes *and* the fitted $\gamma$ values agree within their confidence intervals, the law is separable and extrapolable. If $\gamma$ varies with $N$ by more than its CI, separability is falsified and the composed form in §2 must be replaced.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, Brown, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Xiong, Liu, Molybog, et al. *Effective Long-Context Scaling of Foundation Models.* NAACL, 2024. — arXiv:2309.16039
- **[SOTA]** Fu, Panda, Niu, et al. *Data Engineering for Scaling Language Models to 128K Context.* ICML, 2024. — arXiv:2402.10171
- **[SOTA]** Gao, Wettig, Yen, Chen. *How to Train Long-Context Language Models (Effectively).* 2025. — arXiv:2410.02660
- **[Method]** Press, Smith, Lewis. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.* ICLR, 2022. — arXiv:2108.12409
- **[Method]** Peng, Quesnelle, Fan, Shippole. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR, 2024. — arXiv:2309.00071
- **[Measurement]** Hsieh, Sun, Kriman, et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Measurement]** Liu, Lin, Hewitt, et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Measurement]** Levy, Jacoby, Goldberg. *Same Task, More Tokens: The Impact of Input Length on the Reasoning Performance of Large Language Models.* ACL, 2024. — arXiv:2402.14848
- **[Mechanism]** Olsson, Elhage, Nanda, et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, 2022. — arXiv:2209.11895
- **[Theory]** Bahri, Dyer, Kaplan, Lee, Sharma. *Explaining Neural Scaling Laws.* PNAS, 2024. — arXiv:2102.06701
- **[Theory]** Michaud, Liu, Girit, Tegmark. *The Quantization Model of Neural Scaling.* NeurIPS, 2023. — arXiv:2303.13506

## 10. Worked Example

Take a 1B-parameter model evaluated on 2000 books, each $\ge 32$k tokens, one document per window, $M = 2000$, so per-position standard error $\approx 0.005$ nats. Suppose measured marginals:

| $t$ | $\ell(t)$ (nats) |
|---|---|
| 1 000 | 2.410 |
| 4 000 | 2.362 |
| 16 000 | 2.335 |
| 32 000 | 2.326 |

Fit $\ell(t) = \ell_\infty + A t^{-\gamma}$. Two fits are near-indistinguishable:

- **Fit A:** $\ell_\infty = 2.300$, $A = 1.10$, $\gamma = 0.36$ → predicts $\ell(128\text{k}) = 2.313$.
- **Fit B:** $\ell_\infty = 2.250$, $A = 0.53$, $\gamma = 0.19$ → predicts $\ell(128\text{k}) = 2.297$.

Residuals for both are under $0.004$ nats at every measured $t$ — inside the noise floor. Yet at $t = 10^6$ they diverge by $0.03$ nats, which at Chinchilla-scale sensitivity is roughly the gain from a $2\times$ increase in parameters. **Two-decade data cannot separate $\ell_\infty$ from $\gamma$.**

Now the second obstruction. Run the shuffled-context control: replace everything before $t - 2000$ with foreign tokens. Suppose it gives $\ell_{\text{shuffled}}(32\text{k}) = 2.339$, so $\Delta(32\text{k}) = 0.013$ nats. The *total* observed drop from $t=1$k to $t=32$k was $0.084$ nats. So **about 85% of the apparent "long-context benefit" is intrinsic position-difficulty in the corpus, not conditioning**. Fitting a law to the raw curve fits mostly the corpus, not the model — and the genuine signal, $0.013$ nats, is only $2.6\times$ the standard error. That is the state of the problem: the quantity is real, small, and buried under a confound that almost no published fit subtracts.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*