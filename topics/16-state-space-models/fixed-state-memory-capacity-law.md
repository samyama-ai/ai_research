---
id: 16-state-space-models/fixed-state-memory-capacity-law
title: "Fixed-State Memory Capacity Scaling Law"
topic: 16-state-space-models
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Fixed-State Memory Capacity Scaling Law

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/fixed-state-memory-capacity-law` · **Status:** empirically-open

## 1. Problem Statement

A recurrent or state-space model (SSM) compresses an arbitrarily long prefix into a fixed-size state. A Transformer does not: its KV cache grows linearly in context. The question is how much retrievable information a fixed state actually holds, as a function of its size, and whether that function is a stable law.

- **Measurement variant.** Define a state budget $S$ in bits. Define recallable capacity $C$ as the number of distinct key–value associations a trained model can retrieve from context at a fixed accuracy threshold. Is $C = f(S)$ reproducible across architectures, or is it architecture-specific?
- **Method variant.** Given a hard deployment budget $S$ (bytes of recurrent state per token position), what architecture maximises $C$? Larger $N$ per head, more heads, delta-rule updates, hybrid attention layers?
- **Theory variant.** Prove upper and lower bounds. The upper bound is a counting/communication-complexity argument and is near-trivial; the *achievable* bound — what gradient descent on next-token prediction actually reaches — is open.

Solving it means: a published exponent $\beta$ in $C \asymp \alpha S^{\beta}$ (or a proof that no single exponent exists), fitted across at least two architecture families and three orders of magnitude in $S$, with the fit validated out-of-sample.

## 2. Formal Setting

**State budget.** For a model with $L$ recurrent layers, layer $l$ having $H_l$ heads, head dimension $d_l$, and state expansion $N_l$, stored at $b_l$ bits per element:

$$S \;=\; \sum_{l=1}^{L} H_l \, d_l \, N_l \, b_l .$$

Measured, not derived: $S$ is the number of bits that must persist between token $t$ and $t+1$ — read it off the inference implementation's recurrent buffer, not the parameter count. For Mamba-2 with $d=64$, $N=128$, $H=24$, $L=48$, $b=16$: $S = 24\cdot64\cdot128\cdot16\cdot48 \approx 1.5\times10^{8}$ bits $\approx 19$ MB. A Transformer's comparator is $S_{\text{ctx}}(T) = 2 L d_{\text{model}} b T$ at context length $T$.

**Capacity probe.** Multi-query associative recall (MQAR): a sequence contains $K$ key–value pairs $(k_i, v_i)$ drawn from vocabulary $\mathcal{V}$, then $Q$ queries. Accuracy is exact-match on $v$. Define

$$C(S) \;=\; \max\{K : \operatorname{acc}_{\text{MQAR}}(K) \ge \tau\},\qquad \tau = 0.9 .$$

**Information-theoretic ceiling.** Each pair needs $\log_2|\mathcal{V}|$ bits if stored losslessly, so trivially $C \le S/\log_2|\mathcal{V}|$. The empirical claim of interest is the *utilisation ratio* $\rho = C \log_2|\mathcal{V}| / S \in (0,1]$ and whether $\rho$ is constant in $S$.

**Hypothesised law.** $\log C = \log \alpha + \beta \log S$. $\beta = 1$ means constant utilisation; $\beta < 1$ means large states are wasted; $\beta > 1$ is impossible above the ceiling.

**Assumptions, and which fail.**
1. *State bits are spent on recall.* False — the same state carries syntax, induction, and local features. $\rho$ is therefore a lower bound on physical capacity, and the contamination is not measurable separately.
2. *Effective precision equals $b$.* False. Linear-attention states are dominated by a few large eigen-directions; effective bits per element are well below 16.
3. *Layers compose additively.* False. Multi-layer recurrences can route, so $S$ is an upper bound on jointly usable state, not a sum of independent registers.
4. *MQAR recall is representative of "memory".* Unverified — see §6.

## 3. State of the Art

**Established.**
- Jelassi et al., *Repeat After Me: Transformers Are Better than State Space Models at Copying* (ICML 2024): a hard separation. Copying a string of length $\ell$ needs $\Omega(\ell)$ state bits, so any fixed-state model fails beyond a length set by $S$; Transformers with two layers copy with $O(\log \ell)$ width. Verified empirically on Mamba up to 1.4B and on pretrained Pythia/Mamba pairs.
- Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models* (ICLR 2024): the perplexity gap between attention and gated-convolution models is concentrated in "associative recall hits" — tokens whose bigram context appeared earlier. MQAR was introduced here as the diagnostic.
- Arora et al., *Simple Linear Attention Language Models Balance the Recall–Throughput Tradeoff* (Based, ICML 2024): the recall–state-size Pareto frontier. Recall improves monotonically with recurrent state size across linear attention, sliding window, and convolutional variants; Based sits on the frontier by combining a small feature-map linear attention with a 64-token sliding window.
- Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models* (ICML 2024): S4/Mamba-style SSMs sit in $\mathrm{TC}^0$ under standard uniformity assumptions — they cannot do inherently sequential state tracking, independent of $S$.

**Claimed but unablated.** That state expansion is the dominant lever: Mamba-2 (Dao & Gu, ICML 2024) raises $N$ from 16 to 128 and reports better recall, but the change is confounded with the SSD parameterisation, larger head structure, and different training. No clean $N$-only sweep at matched tokens exists at ≥1B scale.

**Benchmark-number-only.** Most "long-context" claims for recurrent models (Needle-in-a-Haystack pass rates, RULER scores) are single numbers at one state size, with no $S$-sweep. They do not constrain $\beta$.

## 4. What Is Known

- **Copying.** Mamba at 130M–1.4B fails to length-generalise on string copying past roughly its training length, while equal-size Transformers generalise several-fold further (Jelassi et al. 2024). Failure onset moves with state size, in the direction the bound predicts.
- **MQAR.** In Zoology, required model dimension for $\tau\approx1.0$ grows with the number of pairs $K$; attention solves $K$ up to hundreds at $d=64$, while gated convolutions need $d$ growing with $K$. Measured at 70M–360M parameters, sequences 64–512.
- **Hybrids.** Waleffe et al., *An Empirical Study of Mamba-based Language Models* (2024): an 8B Mamba-2 matches an 8B Transformer on many tasks at 3.5T tokens but loses on in-context recall (e.g. five-shot phonebook lookup); a hybrid with ~8% attention layers closes the gap. Interpretation: a small number of unbounded-state layers substitutes for a large fixed state.
- **Bounded state is genuinely bounded.** Wen, Dang & Lyu, *RNNs Are Not Transformers (Yet)* (2024): an $o(n)$-memory RNN cannot solve certain in-context retrieval tasks a one-layer Transformer solves; CoT partially rescues it.

No published work reports a fitted exponent $\beta$.

## 5. What Is Not Known

- **Empirically open (primary).** Whether $C(S)$ follows a single power law across architecture families. The sweep is runnable today — 8 state sizes $\times$ 2 families at ~350M parameters is a few thousand GPU-hours — and nobody has published it.
- **Empirically open.** Whether the exponent measured on synthetic MQAR transfers to natural-language recall loss. All existing links between MQAR and perplexity are correlational at one scale.
- **Theoretically open.** Achievable capacity under gradient training. Counting gives $C \le S/\log_2|\mathcal{V}|$; no lower bound says SGD finds an $\Omega(S)$-capacity solution for any concrete recurrent parameterisation. Delta-rule updates (DeltaNet, Yang et al. 2024) plausibly change the constant; unproven.
- **Methodologically blocked.** Separating recall bits from computation bits inside one state. There is no accepted estimator of "bits of the state currently allocated to retrievable associations", so $\rho$ cannot be decomposed.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement.** $S$ cannot be varied alone. Raising $N$ changes parameter count, FLOPs per token, kernel efficiency, and optimal learning rate at once. Any observed $\Delta C$ is a mixture of capacity and optimisation effects, and the standard fix — matching parameters — forces a compensating change elsewhere in the architecture.
2. **The evaluation may not measure what it names.** MQAR is a synthetic uniform-key task. Real recall is Zipfian and partially memorisable in weights, so a model can score on natural-language recall without using state at all. An exponent fitted on MQAR may be an exponent of MQAR, not of memory.
3. **Non-identifiability of effective precision.** The denominator $S$ counts nominal bits. If linear-attention states are effectively rank-limited, the true budget is smaller by an unknown factor, and $\beta$ absorbs it. Without an effective-rank estimator, $\alpha$ and $\beta$ trade off against each other in the fit.

## 7. Current Research (as of 2026)

- **Frontier-tracing.** Hazy Research (Stanford) continues the recall–throughput Pareto line begun in Zoology/Based, including *Just Read Twice* (Arora et al. 2024), which shows recall in fixed-state models depends on data *order* — a state-management effect, not pure capacity.
- **Richer update rules.** DeltaNet / Gated DeltaNet (Yang, Kautz et al.) replace additive outer-product writes with error-correcting ones; the claim is higher capacity at equal $S$. *(frontier — verify the equal-$S$ control in any specific paper.)*
- **Hybrid ratio search.** NVIDIA, AI21 (Jamba), and Falcon-Mamba lines treat "how few attention layers suffice" as the practical form of this question. Reported sweet spots cluster near one attention layer per 6–8 recurrent layers. *(frontier — verify)*
- **Theory.** Merrill/Sabharwal-style circuit-complexity work continues to bound what fixed state cannot do; it does not yet quantify what it can store.

## 8. Concrete Next Experiment

**Scale.** Mamba-2 at ~350M parameters, trained on 20B tokens of a fixed corpus (e.g. FineWeb-Edu), context 8192. Sweep state size over eight points spanning $2^{20}$ to $2^{27}$ bits by varying $N \in \{8,16,32,64,128,256,512,1024\}$, holding $d_{\text{model}}$, $L$, token budget, and tuned LR schedule fixed. Repeat the full sweep for a second family (Gated DeltaNet or Based) to test family-independence. ~3,000 A100-hours total.

**Control arm.** A sliding-window Transformer whose KV cache is truncated to the *same byte budget* at each of the eight points, trained identically. Its capacity is known to scale linearly in cache bytes, so it calibrates the probe: if the control does not fit $\beta \approx 1$, the probe is broken, not the SSM.

**Deciding number.** The fitted exponent $\beta$ in $\log C(S) = \log\alpha + \beta \log S$, with a bootstrap 95% CI over seeds and probe vocabularies.

- CI containing $1.0$ for both families → constant utilisation; state size is the whole story, and the law is architecture-independent.
- CI strictly below $1.0$ (e.g. $\beta \approx 0.5$) → large states are systematically underused; the research target moves from bigger states to better write rules.
- Non-overlapping CIs between families → no single law; capacity is a property of the update rule, and the catalog entry becomes a per-architecture measurement problem.

Secondary readout: correlation between $\log C$ and the natural-language *AR-hit* loss slice (Zoology's decomposition) at each $S$. If $r^2 < 0.5$, obstruction 2 in §6 is confirmed and MQAR should be retired as the probe.

## 9. Key References

- **[Foundational]** Albert Gu, Karan Goel, Christopher Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022. — arXiv:2111.00396
- **[Foundational]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024. — arXiv:2312.00752
- **[SOTA]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[SOTA]** Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, Christopher Ré. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR 2024. — arXiv:2312.04927
- **[SOTA]** Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, Dylan Zinsley, James Zou, Atri Rudra, Christopher Ré. *Simple Linear Attention Language Models Balance the Recall–Throughput Tradeoff.* ICML 2024. — arXiv:2402.18668
- **[Theory]** Samy Jelassi, David Brandfonbrener, Sham M. Kakade, Eran Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024. — arXiv:2402.01032
- **[Theory]** William Merrill, Jackson Petty, Ashish Sabharwal. *The Illusion of State in State-Space Models.* ICML 2024. — arXiv:2404.08819
- **[Theory]** Kaiyue Wen, Xingyu Dang, Kaifeng Lyu. *RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval.* 2024. — arXiv:2402.18510
- **[Empirical]** Roger Waleffe et al. *An Empirical Study of Mamba-based Language Models.* NVIDIA, 2024. — arXiv:2406.07887
- **[Method]** Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024. — arXiv:2406.06484

## 10. Worked Example

Take Mamba-2 130M: $L=24$, $H=8$ heads of $d=64$, $N=128$, bf16.

$$S = 24 \times 8 \times 64 \times 128 \times 16 \approx 2.5\times10^{7}\ \text{bits} \approx 3.1\ \text{MB}.$$

With $|\mathcal{V}| = 50{,}000$, $\log_2|\mathcal{V}| \approx 15.6$ bits, so the counting ceiling is

$$C_{\max} = 2.5\times10^{7}/15.6 \approx 1.6\times10^{6}\ \text{pairs}.$$

Observed MQAR capacity at $\tau=0.9$ for models of this class is on the order of $10^2$–$10^3$ pairs. Take $C = 500$. Then

$$\rho = \frac{500 \times 15.6}{2.5\times10^{7}} \approx 3\times10^{-4}.$$

**The obstruction, made visible.** Utilisation is roughly one part in three thousand. Two incompatible readings fit that number equally well:

1. *Capacity is the binding constraint and the model is a terrible encoder* — the write rule (an additive outer product) destructively interferes, so effective capacity is $O(\sqrt{N})$ per head rather than $O(N)$. Predicts $\beta \approx 0.5$.
2. *Capacity is not binding at all* — the state is almost entirely spent on language modelling features, and MQAR fails for optimisation reasons (the retrieval circuit is not learned), not storage reasons. Predicts $C$ nearly flat in $S$ over this range, $\beta \approx 0$, until a threshold.

Both are consistent with every published number, because no published sweep varies $S$ alone. That is exactly why the exponent, not another benchmark score, is the quantity to measure — and why the sliding-window control arm in §8 is load-bearing: it is the only way to tell a broken probe from a real $\beta < 1$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*