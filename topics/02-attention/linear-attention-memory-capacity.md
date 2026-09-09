---
id: 02-attention/linear-attention-memory-capacity
title: "Linear Attention Memory Capacity Bound"
topic: 02-attention
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Linear Attention Memory Capacity Bound

> **Topic:** Attention Mechanisms · **ID:** `02-attention/linear-attention-memory-capacity` · **Status:** partially-solved

## 1. Problem Statement

Linear attention replaces the softmax kernel with a feature map, turning attention into a recurrent layer with a fixed-size state $S_t \in \mathbb{R}^{d_k \times d_v}$. The state does not grow with sequence length, so the layer is a lossy memory. The problem: **give a tight bound on how much retrievable information a linear-attention state of a given size holds, and show it is achieved by a trainable update rule.**

Three variants, with different difficulty:

- **Theory.** For a state of $m$ scalars, what is the maximum number $N$ of key–value pairs recoverable with error $\le \epsilon$? Upper bounds from communication complexity are known; matching *achievable* bounds for gradient-trainable update rules are not.
- **Measurement.** Define an operational capacity $N^\*(\epsilon)$ that can be read off a trained model rather than a synthetic probe, and that is invariant to tokenizer, head count, and layer count.
- **Method.** Find an update rule whose realized capacity is within a constant factor of the information-theoretic bound at fixed $m$ and fixed wall-clock throughput.

Solving it means: a bound $N^\*(\epsilon) = \Theta(f(m, d_k, d_v, \epsilon))$ with a construction that attains it and a measurement protocol that recovers $f$ from a trained checkpoint to within a constant.

## 2. Formal Setting

Tokens $x_1,\dots,x_T$; per head, keys $k_t = \phi(W_K x_t) \in \mathbb{R}^{d_k}$, values $v_t = W_V x_t \in \mathbb{R}^{d_v}$, queries $q_t = \phi(W_Q x_t)$. Plain linear attention (Katharopoulos et al., 2020):

$$S_t = S_{t-1} + v_t k_t^\top, \qquad o_t = \frac{S_t q_t}{u_t^\top q_t}, \quad u_t = u_{t-1} + k_t.$$

The delta rule (Schlag et al., 2021) replaces the sum with an error-correcting write:

$$S_t = S_{t-1}\left(I - \beta_t k_t k_t^\top\right) + \beta_t v_t k_t^\top .$$

Gated variants (Mamba-2, GLA, Gated DeltaNet) insert a decay $\alpha_t \in (0,1)$ or a diagonal $A_t$: $S_t = \alpha_t S_{t-1}(\cdot) + \cdots$.

**Quantities, as measured.**

- **State size** $m = H \cdot L \cdot d_k \cdot d_v$ scalars across $H$ heads and $L$ layers. Measured in *bits*: $m \cdot b$ where $b$ is the inference dtype width (bf16: $b=16$). Report bits, not "state dimension" — a $d_k{=}128$ Mamba-2 head and a $d_k{=}128$ DeltaNet head differ by a factor $d_v$.
- **Recall error.** Feed $N$ pairs $(k^{(i)}, v^{(i)})$ then query. $\epsilon = \Pr[\hat v \ne v]$ over the query distribution, measured as exact-token match, not cosine similarity.
- **Capacity** $N^\*(\epsilon) = \max\{N : \epsilon(N) \le \epsilon\}$, with $\epsilon = 0.05$ the usual operating point.
- **Bit efficiency** $\eta = N^\* \log_2 |\mathcal{V}| / (m b)$ — retrieved bits per state bit. $\eta \le 1$ by counting.

**Assumptions, and which break.**

1. *Keys are near-orthogonal / drawn i.i.d.* Violated: trained key distributions are anisotropic and low-effective-rank, so realized capacity is below the random-key bound.
2. *The layer is the memory.* Violated: MLPs, short convolutions, and (in hybrids) a few full-attention layers carry recall, so a layer-level capacity number does not compose into a model-level one.
3. *Retrieval is a single hop.* Violated: multi-hop recall re-reads the state, and error compounds non-linearly in $N$.
4. *Inference in bf16.* Violated under 8-bit state quantization, which changes $m b$ by $2\times$ but capacity by an unmeasured amount.

## 3. State of the Art

**Theory SOTA (established).**

- Communication-complexity lower bound: any recurrent model solving multi-query associative recall over $N$ pairs needs state $\Omega(N)$ bits, up to log factors (Arora et al., *Zoology*, ICLR 2024; extended in *Based*, ICML 2024). This is an *upper* bound on capacity, not an achievability result.
- Copying separation: transformers copy length-$n$ strings with $O(\log n)$-width heads; any fixed-state recurrent model needs state $\Omega(n)$ (Jelassi et al., *Repeat After Me*, ICML 2024).
- Retrieval separation with matching constructions (Wen, Dangovski et al., *RNNs are not Transformers (Yet)*, 2024); representational separations for index-lookup-style tasks (Bhattamishra, Hahn, Blunsom, Kanade, NeurIPS 2024).
- State-tracking limits: linear RNNs with non-negative eigenvalues cannot represent parity/$S_5$ composition (Merrill, Petty, Sabharwal, *The Illusion of State in State-Space Models*, COLM 2024); Grazzi et al. (ICLR 2025) show extending eigenvalues to $[-1,1]$ restores parity.

**Empirical SOTA (established by ablation).** DeltaNet's delta rule beats additive linear attention at fixed state size on MQAR, and the parallel chunked form makes it trainable at scale (Yang, Wang, Zhang, Kim, NeurIPS 2024). Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025) adds decay and improves recall-heavy tasks at matched state. RWKV-7 (Peng et al., 2025) uses a generalized delta update and claims expressivity beyond $\mathsf{TC}^0$.

**Claimed but unablated.** "State size is the capacity" is asserted widely but almost never measured at matched *bits* — comparisons are matched on parameters or on model dimension, which conflates state size with MLP capacity. Recall gains from gating are reported as benchmark deltas (SWDE, FDA, SQuAD in the Based/JRT line) with no isolation of the state-size term. Hybrid recipes (e.g. 1 attention layer per 6–8 linear layers) are tuned empirically; the ratio has no derivation.

## 4. What Is Known

- **Lower bound is linear in pairs.** $\Omega(N)$ state bits for MQAR (Zoology, ICLR 2024). Measured at synthetic scale: sequence lengths 64–512, vocab up to 8192, models of 2 layers.
- **Attention is dramatically more state-efficient on the probe.** Arora et al. report attention solving MQAR at model dimension 64 where gated-convolution baselines need roughly an order of magnitude more dimension for the same pair count — the gap widens with $N$.
- **Copying fails out of distribution.** *Repeat After Me* reports Mamba models at the ~360M scale, trained to copy, degrading sharply past training length while transformers of matched size generalize further.
- **Delta rule buys real capacity at fixed $m$.** DeltaNet at 340M–1.3B params on 100B tokens beats GLA/Mamba on recall-intensive tasks at comparable state, but loses to full attention on the same tasks.
- **Hybrids close most of the gap.** Reported across the Based, Jamba, Samba, and Zamba lines: a small number of full-attention layers recovers most recall while keeping most of the throughput. Established as a repeated observation across independent groups; the *mechanism* (attention layers act as the exact-recall store) is inferred, not proven.
- **Throughput side is solid.** Chunked parallel forms give linear-time training with hardware-efficient kernels; Mamba-2's SSD form (Dao & Gu, ICML 2024) reaches 2–8× faster than the Mamba-1 scan.

## 5. What Is Not Known

- **Theoretically open.** No matching upper bound on *achievable* capacity for the delta rule. Classical outer-product associative memories store $\Theta(d/\log d)$ random patterns; the delta rule stores up to $d_k$ exactly for linearly independent keys. Which regime a trained model occupies, and whether a rule exists with capacity $\Theta(m b / \log|\mathcal{V}|)$ bits — i.e. $\eta = \Theta(1)$ — is unproven either way.
- **Empirically open.** Nobody has run the bits-matched sweep: fix state bits $m b$, vary $(d_k, d_v, H, L)$ and update rule, measure $N^\*(0.05)$ on natural-language recall at $\ge$1B params. Runnable today for well under 10k GPU-hours.
- **Methodologically blocked.** There is no accepted way to read capacity off a *pretrained* model. MQAR is synthetic and its keys are uniform; natural-text recall benchmarks (SWDE, FDA, needle-in-haystack) confound capacity with retrieval-head formation, positional generalization, and MLP-stored knowledge. The measurement, not the experiment, is the bottleneck.

## 6. Why It Is Hard

**Confounded measurement, compounded by non-identifiability.** Recall failure in a trained linear-attention LM has at least four causes — insufficient state bits, key collisions from an anisotropic key distribution, an update rule that overwrites rather than corrects, and the read-out MLP — and the standard benchmarks return one scalar for all four. Two models with identical $m$ can differ 5× in measured recall for reasons entirely outside the state.

Second obstruction: **the bound depends on the key distribution, which is learned.** Capacity for random orthogonal keys is $d_k$ per head; for keys concentrated in a rank-$r$ subspace it is $r$. Since $r$ is a property of the trained checkpoint, capacity is not a function of the architecture alone — which is exactly why an architecture-level theorem has not landed.

## 7. Current Research (as of 2026)

- **Expressive update rules.** DeltaNet → Gated DeltaNet → higher-rank and multi-step delta updates; RWKV-7 (BlinkDL / EleutherAI-adjacent). Direction: raise capacity per state bit by making the write an online regression step rather than an accumulation.
- **Test-time training as memory.** Framing the recurrent state as a fast-weight network optimized online (TTT layers, Sun et al. 2024; Titans, Behrouz et al. 2025) — capacity becomes the expressivity of the fast network. *(frontier — verify claims of matched-state gains.)*
- **Hybrid scaling laws.** Fitting loss and recall jointly against attention-layer fraction (NVIDIA, AI21, Microsoft lines). Not yet a published law with a state-bits term. *(frontier — verify.)*
- **Theory.** Merrill/Sabharwal (AI2, NYU), Sanford/Hsu/Telgarsky-style separations, and the Stanford Hazy Research group's recall–throughput Pareto framing.

## 8. Concrete Next Experiment

**Bits-matched capacity sweep.**

- **Scale.** 1.3B-parameter LMs, 100B tokens of a fixed corpus, context 8192. Six arms, all with the *same total state bits* $m b = 2^{26}$ (67.1 Mbit, i.e. 8 MB in bf16), reached by trading $d_k$, $d_v$, and $H$: {additive linear attention, GLA, Mamba-2, DeltaNet, Gated DeltaNet, DeltaNet with $2\times$ $d_k$ and $\tfrac12$ heads}.
- **Control arm.** A full-attention transformer of identical parameter count and depth, plus a *state-crippled* control: DeltaNet with the state quantized to 4 bits ($m b$ cut $4\times$). The quantized arm calibrates how much measured recall actually tracks bits.
- **Deciding number.** $N^\*(0.05)$ — the number of key–value pairs recoverable at 95% exact-match, measured with keys drawn from *held-out natural text* (entity–attribute pairs from unseen documents), not uniform tokens. Report $\eta = N^\* \cdot \log_2|\mathcal{V}| / (mb)$.
- **Decision rule.** If the best rule reaches $\eta \ge 0.1$ and the 4-bit control drops $N^\*$ by $\approx 4\times$, capacity is bits-limited and the bound is close to achievable. If all arms sit at $\eta \le 0.01$ and the quantized control barely moves, capacity is rank-limited, not bits-limited — and the field's state-size framing is measuring the wrong quantity.

## 9. Key References

- **[Foundational]** Katharopoulos, Vyas, Pappas, Fleuret. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020. — arXiv:2006.16236
- **[Foundational]** Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021. — arXiv:2102.11174
- **[SOTA/Theory]** Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR 2024. — arXiv:2312.04927
- **[SOTA]** Arora, Eyuboglu, Zhang, Timalsina, Alberti, Zinsley, Zou, Rudra, Ré. *Simple Linear Attention Language Models Balance the Recall–Throughput Tradeoff.* ICML 2024. — arXiv:2402.18668
- **[Theory]** Jelassi, Brandfonbrener, Kakade, Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024. — arXiv:2402.01032
- **[SOTA]** Yang, Wang, Zhang, Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024. — arXiv:2406.06484
- **[SOTA]** Yang, Kautz, Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025. — arXiv:2412.06464
- **[SOTA]** Dao, Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[Theory]** Merrill, Petty, Sabharwal. *The Illusion of State in State-Space Models.* COLM 2024. — arXiv:2404.08819
- **[Theory]** Grazzi, Siems, Franke, Zela, Hutter, Pontil. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR 2025. — arXiv:2411.12537
- **[Theory]** Bhattamishra, Hahn, Blunsom, Kanade. *Separations in the Representational Capabilities of Transformers and Recurrent Architectures.* NeurIPS 2024.

## 10. Worked Example

One DeltaNet head, $d_k = d_v = 64$, bf16.

**State bits.** $m b = 64 \times 64 \times 16 = 65{,}536$ bits.

**Information-theoretic ceiling.** Storing pairs whose values are tokens from $|\mathcal{V}| = 32{,}000$ costs $\log_2 32{,}000 \approx 14.97$ bits per value. Ceiling:

$$N_{\text{info}} \le \frac{65{,}536}{14.97} \approx 4{,}378 \text{ pairs.}$$

**Algebraic capacity.** The delta rule writes into $S \in \mathbb{R}^{64\times 64}$, a rank-$\le 64$ object. If keys $k^{(1)},\dots,k^{(N)}$ are linearly independent, $\beta_t = 1$ gives exact retrieval for all $N \le 64$. At $N = 65$ some key is a combination of the others and the read $S q$ returns a blend. So:

$$N_{\text{alg}} = d_k = 64 \text{ pairs.}$$

**The gap.** $4{,}378 / 64 \approx 68\times$. Bit efficiency $\eta = 64 \times 14.97 / 65{,}536 \approx 0.0146$ — the head uses about **1.5%** of its own state bits.

**Why the gap is real, not an artifact.** The state holds $64 \times 64$ bf16 scalars, but retrieval is a linear read $Sq$. Capacity is set by the *rank* of the write, and rank is capped at $d_k$ regardless of how many bits each entry carries. Going to fp32 doubles $m b$ and changes $N_{\text{alg}}$ by zero. Going to 4-bit quantization cuts $m b$ by $4\times$ and, until keys start colliding under quantization noise, also changes $N_{\text{alg}}$ by zero.

**And trained models sit below 64.** Learned keys are anisotropic; if the empirical key covariance has effective rank $r \approx 20$, the head retrieves ~20 pairs, not 64 — a further $3\times$ loss that depends on the checkpoint, not the architecture.

That is the obstruction in one number: the community's capacity story is denominated in state *bits*, the achievable capacity is denominated in state *rank*, and at $d_k = 64$ these differ by 68×. Any bound stated in bits is loose by two orders of magnitude until an update rule is exhibited that writes at more than rank $d_k$ per read.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*