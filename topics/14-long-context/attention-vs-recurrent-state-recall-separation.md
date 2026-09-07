---
id: 14-long-context/attention-vs-recurrent-state-recall-separation
title: "Provable Separation Between Attention and Recurrent State for Long-Range Recall"
topic: 14-long-context
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Provable Separation Between Attention and Recurrent State for Long-Range Recall

> **Topic:** Long Context · **ID:** `14-long-context/attention-vs-recurrent-state-recall-separation` · **Status:** partially-solved

## 1. Problem Statement

A softmax-attention layer keeps every token available at every step: its KV cache grows linearly in context length $n$. A recurrent model (linear attention, SSM, gated RNN) compresses the prefix into a fixed-size state $S$ bits that does not grow with $n$. The question: **for which tasks, and by how much, does the fixed state provably cost you?**

Three variants, of very different difficulty:

- **Theory variant.** Prove a separation: exhibit a task family $\mathcal{T}_n$ solvable by a transformer with $\mathrm{poly}(\log n)$ width and depth, and prove that any recurrent architecture in a stated class with state $S = o(n)$ has error bounded away from zero. *Largely settled for exact recall — see §4.*
- **Method variant.** Given a target recall accuracy on natural text, find the minimal $S$. Equivalently: characterize the achievable frontier $(S, \text{recall})$ and find architectures on it. *Open; this is where the hybrid-model literature lives.*
- **Measurement variant.** Decide whether the separation proved for synthetic copying/associative-recall tasks explains any part of the loss gap on real corpora. *Methodologically blocked — see §5.*

Solving the problem means: a lower bound that (a) applies to the architecture classes actually deployed in 2026 (gated linear attention, delta-rule updates, hybrids with $k$ attention layers), (b) is stated against a *distribution* rather than worst case, and (c) has a matching empirical estimate of the constant at a real model scale.

## 2. Formal Setting

**Input.** A sequence $x_{1:n} \in \Sigma^n$, $|\Sigma| = V$. **Output.** A next-token distribution $p(\cdot \mid x_{1:n})$.

**Recurrent model.** A map $h_t = f_\theta(h_{t-1}, x_t)$, $h_t \in \mathbb{R}^d$ stored at $b$ bits per coordinate, then $y_t = g_\theta(h_t)$. The measured quantity is the **state budget**
$$S = d \cdot b \quad \text{bits per sequence}.$$
Measured in practice as the byte size of the recurrent cache at inference: for Mamba-2, $S = (\text{n\_layers}) \times (\text{d\_state} \times \text{d\_inner}) \times \text{bytes}$; for a transformer the analogue is the KV cache, $S_{\text{attn}}(n) = 2 \cdot n \cdot n_{\text{layers}} \cdot n_{\text{kv heads}} \cdot d_{\text{head}} \cdot \text{bytes}$, which is $\Theta(n)$, not constant.

**Task: multi-query associative recall (MQAR).** The prompt contains $N$ key–value pairs $(k_i, v_i)$ drawn without replacement, followed by $Q$ queries. Accuracy is exact-match on the $Q$ answer tokens. **Task: copying.** Input $w \\# $ with $w \in \Sigma^L$; the model must emit $w$. Accuracy is string-exact.

**The lower-bound instrument.** A recurrent forward pass is a one-way communication protocol: Alice holds $x_{1:m}$, sends $h_m$ ($S$ bits), Bob holds $x_{m+1:n}$ and must answer. The INDEX problem — Alice holds $z \in \{0,1\}^m$, Bob holds $i$, output $z_i$ — has one-way randomized communication complexity $\Omega(m)$ (Kremer, Nisan, Ron 1999). Hence any recurrent model solving INDEX-style recall over $m$ bits of prefix with success $\ge 2/3$ needs
$$S = \Omega(m).$$
Attention needs no such state: it re-reads the prefix.

**Assumptions, and which are violated.**
- *State is the only channel between prefix and suffix.* Holds for pure SSM/RNN stacks. **Violated** by hybrids (any single full-attention layer restores $\Theta(n)$ channel capacity), by sliding-window layers, and by chunked/parallel scan implementations that still read raw tokens locally.
- *Single pass, left-to-right.* **Violated** in practice: "read twice" prompting and encoder-style bidirectional prefixes give the recurrent model two passes, which provably reduces the required $S$ for set-disjointness-like recall.
- *Bounded precision.* Lower bounds count bits; float32 states with unbounded precision can in principle pack more. Real inference is bf16 or int8, so the bit count is honest.
- *Worst-case inputs.* Natural text is far from uniform; the lower bound says nothing about average case under a realistic $p(x)$. This is the load-bearing violated assumption.

## 3. State of the Art

**Theory SOTA (established).**
- Sanford, Hsu, Telgarsky, *Representational Strengths and Limitations of Transformers* (NeurIPS 2023): sparse averaging is solvable by one attention layer of width $\tilde{O}(1)$ but requires $\tilde\Omega(n)$ width for recurrent/fully-connected models — a communication-complexity separation.
- Jelassi, Brandfonbrener, Kakade, Malach, *Repeat After Me: Transformers are Better than State Space Models at Copying* (ICML 2024): a transformer with $O(\log n)$ precision and constant depth copies length-$n$ strings via hash-based induction; any state-space model with state of $S$ bits fails to copy strings longer than $\Theta(S)$. This is the cleanest known separation.
- Merrill, Petty, Sabharwal, *The Illusion of State in State-Space Models* (ICML 2024): SSMs, like log-precision transformers, sit in $\mathrm{TC}^0$ under uniformity assumptions — so the separation is *not* a general expressivity gap, it is specifically about memory-for-recall.

**Empirical SOTA (established, ablated).** Arora et al., *Zoology* (2023) and *Based* (ICML 2024) map the recall–throughput frontier: MQAR accuracy is a sharp function of recurrent state size, and closing the gap to attention requires $S$ growing with the number of key–value pairs. Waleffe et al. (NVIDIA, 2024) trained 8B-parameter Mamba, Mamba-2 and Mamba-2-Hybrid on 3.5T tokens with matched data — the honest large-scale control arm.

**Claimed but unablated.** Vendor claims that gated recurrent architectures (Gated DeltaNet, RWKV-7 and successors) "match attention on long context" rest on aggregate benchmark scores, not on a state-size sweep at fixed data. Where a number exists only as a leaderboard entry — most long-context suite scores for hybrids — treat it as unablated: the state budget, the number of attention layers, and the data mix all move together.

## 4. What Is Known

- **Copying.** At 160M parameters trained on strings of length $\le 50$: transformers copy at high accuracy out to length $\approx 300$; Mamba of matched size collapses near the training length. Pretrained Pythia-2.8B beats Mamba-2.8B on phone-book lookup by a wide margin, and the gap grows with book size (Jelassi et al., ICML 2024).
- **MQAR.** Recall accuracy for gated-convolution and linear-attention models is monotone in recurrent state size and falls off once the number of key–value pairs exceeds the state's capacity; attention is flat in the same sweep. Measured at 100M–1.3B parameters on the Pile (Arora et al., 2023–2024).
- **Hybrids erase most of the gap.** Mamba-2-Hybrid (8B, 3.5T tokens, ~7% attention layers) matches or exceeds a matched transformer on 12 standard tasks, while pure Mamba-2 lags on tasks needing in-context copying — including five-shot MMLU and phone-book retrieval (Waleffe et al., 2024). One or a few attention layers restore the missing channel; this is the strongest evidence that the theoretical mechanism is the operative one.
- **Two passes help.** Arora et al., *Just Read Twice* (2024): repeating the context before the query lifts recurrent-model recall substantially at fixed state, consistent with the two-round communication bound being lower than the one-way bound.
- **The separation is not about circuit class.** Both log-precision transformers (Merrill & Sabharwal, TACL 2023) and SSMs live in $\mathrm{TC}^0$; neither does inherently sequential state tracking. So the recall separation is a *memory* separation, not a *computation* separation.

## 5. What Is Not Known

- **Theoretically open.** (i) A lower bound against **hybrids**: how does required non-attention state scale when $k$ full-attention layers are permitted? No theorem gives the tradeoff curve in $(k, S, n)$. (ii) **Average-case separation** under a realistic text distribution, rather than worst case over $\Sigma^n$. (iii) Lower bounds against **delta-rule / non-diagonal** state updates, which are strictly more expressive than diagonal SSMs; existing proofs assume the diagonal-linear form.
- **Empirically open.** Whether the copying/MQAR gap survives at $\ge 30$B parameters and $\ge 10$T tokens with matched data. The 8B/3.5T run is the largest matched comparison in public; nobody has run the state-size sweep at frontier scale, because it costs a full pretraining run per point.
- **Methodologically blocked.** Attributing any part of the *natural-text* loss gap to recall. There is no accepted decomposition of cross-entropy into a "retrieval-limited" component and everything else. Needle-in-a-haystack and long-context suites are contaminated by position bias, prompt format, and instruction tuning; they measure a mixture, not the quantity named.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by the cost of the control arm**. To claim "the gap is caused by finite state," you must vary $S$ and hold everything else fixed — parameters, data, tokens, optimizer, and the number of attention layers. Each point on that curve is a pretraining run. At 8B/3.5T that is a few hundred thousand GPU-hours per point; a five-point sweep at 30B is out of reach for any academic group and unattractive to labs, whose incentive is to ship the hybrid, not to measure the pure model's deficit.

A second obstruction is **non-identifiability of the mechanism on real data**. Two models with different $S$ differ in effective capacity as well as recall capacity. When the recurrent model's loss is higher, no existing method separates "it forgot the key" from "it has a worse language prior." Synthetic MQAR avoids this by construction, and that is exactly why it does not license conclusions about text.

## 7. Current Research (as of 2026)

- **Architectural**: delta-rule and gated-delta state updates (Yang, Kautz et al., *Gated DeltaNet*, ICLR 2025) aim to raise recall per bit of state rather than raise the bit count. Whether they beat the information-theoretic bound or just improve the constant is unresolved — the bound is about bits, and these keep the bits fixed. *(frontier — verify)*
- **Hybrid layout search**: how many attention layers, and where. Empirical, no theory. Groups at NVIDIA, AI21 (Jamba), Microsoft (Samba), Zyphra (Zamba).
- **Theory of memory–recall tradeoffs**: Stanford Hazy Research (Arora, Ré) on the recall–throughput frontier; Sanford/Telgarsky-line work on communication-complexity separations; Merrill & Sabharwal on formal-language characterizations.
- **Test-time recall augmentation**: repeated passes, retrieval into the state, and chunked re-encoding — all of which change the communication model and therefore sidestep rather than close the one-way bound. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question decided:** does the copying/recall deficit scale as predicted by state bits, once parameters and data are held fixed?

- **Scale.** Five models at 1.3B parameters, 100B tokens each, identical data order and tokenizer. Vary only the recurrent state: $S \in \{0.5, 1, 2, 4, 8\}$ MB per sequence, achieved by scaling `d_state` and compensating parameter count with width so total parameters match to within 1%.
- **Control arms.** (a) A matched 1.3B transformer with full attention. (b) A matched 1.3B hybrid with exactly one attention layer and the smallest state, $S = 0.5$ MB. Arm (b) is the discriminating control: theory says one attention layer should recover most of the gap regardless of $S$.
- **Evaluation.** Synthetic phone-book lookup with $N \in \{10^2, 10^3, 10^4\}$ entries at context lengths 4K–128K, plus held-out Pile loss restricted to tokens whose exact 8-gram appeared earlier in the same document (a proxy for recall-limited tokens).
- **The deciding number.** The fitted exponent $\alpha$ in $\text{error}(N, S) \propto (N / S)^{\alpha}$ on phone-book lookup. Theory predicts a threshold at $N \cdot \log V \approx S$ — i.e. error near zero below the threshold and near chance above it, so a large $\alpha$ with the collapse point tracking $S$ linearly. If instead error is smooth and roughly independent of $S$ over a 16× state range, the bound is not the operative constraint at this scale and the field should stop citing it as an explanation for text-level gaps.
- **Cost.** ~7 runs × 1.3B × 100B tokens ≈ 1.5–2 × $10^{22}$ FLOPs total; feasible on a few hundred GPUs for a few weeks.

## 9. Key References

- **[Foundational]** Kremer, Nisan, Ron. *On randomized one-round communication complexity.* Computational Complexity, 1999. — the INDEX lower bound the separations reduce to.
- **[Foundational]** Sanford, Hsu, Telgarsky. *Representational Strengths and Limitations of Transformers.* NeurIPS, 2023.
- **[SOTA]** Jelassi, Brandfonbrener, Kakade, Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[SOTA]** Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR, 2024. — arXiv:2312.04927
- **[SOTA]** Arora, Eyuboglu, Zhang, Timalsina, Alberti, Zou, Rudra, Ré. *Simple Linear Attention Language Models Balance the Recall–Throughput Tradeoff.* ICML, 2024.
- **[Theory]** Merrill, Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL, 2023.
- **[Theory]** Merrill, Petty, Sabharwal. *The Illusion of State in State-Space Models.* ICML, 2024.
- **[Theory]** Sarrof, Veitsman, Hahn. *The Expressive Capacity of State Space Models: A Formal Language Perspective.* NeurIPS, 2024.
- **[Empirical]** Waleffe et al. *An Empirical Study of Mamba-based Language Models.* NVIDIA technical report, 2024.
- **[Architecture]** Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[Architecture]** Yang, Kautz, Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR, 2025.
- **[Method]** Arora et al. *Just Read Twice: Closing the Recall Gap for Recurrent Language Models.* 2024.

## 10. Worked Example

Take Mamba-2 at 1.3B: 48 layers, `d_state` = 128, `d_inner` = 4096, bf16. State size
$$S = 48 \times 128 \times 4096 \times 2\ \text{bytes} \approx 50\ \text{MB} = 4.0 \times 10^{8}\ \text{bits}.$$

Now the task: memorize a phone book of $N$ entries, each a 10-digit number keyed by a name drawn from a set of $10^6$. The information content of the book is
$$I = N \times (\log_2 10^{6} + \log_2 10^{10}) \approx N \times 53\ \text{bits}.$$
The one-way bound says the model can answer random lookups only if $S \gtrsim I$, i.e.
$$N \lesssim 4.0\times 10^{8} / 53 \approx 7.5 \times 10^{6}\ \text{entries}.$$

Seven million entries. The transformer's KV cache at the same 1.3B scale is ~0.4 MB per token, so a 128K context costs ~50 GB — the transformer is *worse* on memory here, by three orders of magnitude, and yet it is the one that wins the benchmark.

**This is the obstruction made visible.** The information-theoretic bound is not binding at any context length anyone runs: the 50 MB state could in principle hold a book far larger than fits in 128K tokens. Mamba nonetheless fails phone-book lookup at $N = 10^3$ — roughly $5\times10^4$ bits, four orders of magnitude under its budget. So the observed failure is *not* the proved lower bound. It is a learnability or optimization failure: gradient descent does not find the packing that the counting argument permits.

The catalog status is therefore "partially-solved" in a specific sense. The worst-case separation is proved and clean. The gap people actually measure is a different phenomenon wearing the same name, and no one has yet stated a bound — an effective-capacity bound, not a bit-count bound — that predicts where the real collapse happens. Any experiment that reports a recall gap without checking whether it sits above or below $S / \log_2(\text{key} \times \text{value space})$ is not testing the theorem it cites.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*