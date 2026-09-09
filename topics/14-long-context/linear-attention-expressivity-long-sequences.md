---
id: 14-long-context/linear-attention-expressivity-long-sequences
title: "Formal Expressivity of Linear Attention Over Long Sequences"
topic: 14-long-context
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Formal Expressivity of Linear Attention Over Long Sequences

> **Topic:** Long Context · **ID:** `14-long-context/linear-attention-expressivity-long-sequences` · **Status:** partially-solved

## 1. Problem Statement

Linear attention replaces the softmax kernel with a factorizable one, turning the $O(L^2)$ attention computation into an $O(L)$ recurrence with a fixed-size state. The question: **which functions of a length-$L$ sequence can a linear-attention model compute that a softmax transformer can, and which can it provably not?**

Three variants, routinely conflated:

- **Theory variant.** Given a family of linear-attention models with state dimension $d_k d_v$, precision $p$ bits, and $\ell$ layers, characterize the class of length-$L$ sequence-to-sequence functions it computes exactly. Solving it means a separation theorem (or collapse) against log-precision softmax attention, uniform in $L$.
- **Measurement variant.** Given a trained model, decide whether an observed failure at length $L$ is a *representational* limit (no parameter setting works), an *optimization* limit (a setting exists, training does not find it), or a *data* limit. Solving it means an estimator that separates the three.
- **Method variant.** Construct a linear-time architecture that matches softmax attention on the tasks where separations are proven — associative recall over $L$ far beyond training length, verbatim copy, state tracking — at equal parameters and tokens. Solving it means the gap closes without a quadratic-cost fallback.

The theory variant is largely settled for the coarse complexity classes. The measurement and method variants are open, and this page treats the *status* as partially-solved for that reason.

## 2. Formal Setting

A causal linear-attention layer over tokens $x_1,\dots,x_L$ produces $q_t,k_t \in \mathbb{R}^{d_k}$, $v_t \in \mathbb{R}^{d_v}$ and maintains a matrix state $S_t \in \mathbb{R}^{d_v \times d_k}$:

$$S_t = A_t \, S_{t-1} + v_t k_t^{\top}, \qquad o_t = S_t\, q_t / z_t .$$

Instantiations differ only in the transition $A_t$ and the normalizer $z_t$:

| Model | $A_t$ |
|---|---|
| Linear attention (Katharopoulos 2020) | $I$ |
| RetNet | $\gamma I$, $\gamma\in(0,1)$ fixed |
| Mamba-2 / GLA | $\alpha_t I$ or $\mathrm{diag}(\alpha_t)$, $\alpha_t = \sigma(\cdot)\in(0,1)$ |
| DeltaNet | $I - \beta_t k_t k_t^{\top}$ (rank-1, non-diagonal) |
| DeltaProduct | $\prod_{i=1}^{n_h}(I-\beta_{t,i} k_{t,i}k_{t,i}^{\top})$ |

**Quantities as measured.**

- **State budget** $M = \ell \cdot h \cdot d_k d_v \cdot b$ bits, with $\ell$ layers, $h$ heads, $b$ bits per entry ($b=16$ for bf16). Measured by reading the config, not by a proxy like "parameter count".
- **Length generalization ratio** $\rho = L_{\text{eval}}/L_{\text{train}}$ at which task accuracy first drops below $0.9$ of in-distribution accuracy. Measured by sweeping $L_{\text{eval}}$ on a fixed checkpoint.
- **Recall capacity** $C$: the largest number of key–value pairs for which multi-query associative recall (MQAR) exceeds $95\%$ exact-match, with keys drawn uniformly from a vocabulary of size $|V|$.
- **Representational gap.** For task $f$, $\Delta(f) = \min_\theta \mathcal{L}_{\text{softmax}}(\theta) - \min_\theta \mathcal{L}_{\text{lin}}(\theta)$. Only measurable via a *constructed* optimum or an exhaustive fit on a tiny instance; on real models it is estimated by trained loss, which confounds it with optimization.

**Assumptions, and where they break.**

1. *Finite precision, $p = O(\log L)$.* Standard in the circuit-complexity results; real models use bf16, i.e. constant precision with a wide exponent. The $O(\log L)$ idealization is **more generous** than practice, so upper bounds carry over; lower bounds constructed in $O(\log L)$ precision may not.
2. *Uniformity of the circuit family.* Theorems assume one parameter setting for all $L$. Practice trains per-context-length with tuned RoPE/decay schedules; the uniform statement is violated.
3. *Exact computation.* Separations are stated for exact function computation; deployment cares about accuracy under a data distribution, where a model can be wrong on a measure-zero hard set and still score well.
4. *State is the only channel.* Hybrid models with a few full-attention layers break the fixed-state assumption entirely, and most deployed "linear" models are hybrids.

## 3. State of the Art

**Theory SOTA (established).**

- Log-precision softmax transformers with constant depth are contained in uniform $\mathrm{TC}^0$ (Merrill & Sabharwal, TACL 2023). They cannot solve $\mathrm{NC}^1$-hard problems unless $\mathrm{TC}^0=\mathrm{NC}^1$.
- The same containment holds for S4/Mamba-style SSMs and diagonal linear attention (Merrill, Petty & Sabharwal, ICML 2024): they cannot track permutation-group state ($S_5$ word problem) either. **Both families sit in the same coarse class** — the separation is not at the level of $\mathrm{TC}^0$.
- Within that class, separations are *resource-based*, not class-based: any recurrent model with $M$ bits of state cannot solve a task whose one-way communication complexity exceeds $M$ (Jelassi et al., ICML 2024, for copying; Wen, Dang & Lyu, 2024, for in-context retrieval; Bhattamishra et al., NeurIPS 2024, for index lookup and nearest-neighbour).
- Non-diagonal transitions strictly help. DeltaNet's rank-1 Householder update escapes the diagonal-SSM limitation; allowing eigenvalues in $[-1,1]$ rather than $[0,1]$ makes parity and modular counting expressible (Grazzi et al., ICLR 2025), and products of $n_h$ Householders extend the reachable state-tracking group (DeltaProduct, Siems et al., 2025).

**Empirical SOTA (claimed, partly unablated).**

- Gated Delta Networks (Yang, Kautz & Hatamizadeh, ICLR 2025) combine gating with the delta rule and beat Mamba2 and DeltaNet at matched scale on language modelling and recall benchmarks.
- RWKV-7 "Goose" (Peng et al., 2025) claims a transition matrix expressive enough to recognize languages beyond $\mathrm{TC}^0$ under the standard assumptions. This is an architectural claim with a construction, not a measured separation on a trained model — **treat as claimed**.
- Hybrid stacks (a small fraction of full-attention layers interleaved with linear layers) are the deployed SOTA. The specific *ratio* and *placement* are tuned per-paper and rarely ablated against a matched-state pure-linear control — **claimed but unablated**.

## 4. What Is Known

- **Copying.** At ~160M parameters trained on identical data, transformers copy strings of length far beyond the training distribution while GSSM/SSM baselines degrade sharply past training length (Jelassi et al., ICML 2024). The theoretical companion: a fixed-state recurrent model needs $\Omega(L)$ state to copy length-$L$ random strings; a transformer needs $O(\log L)$ depth-independent positional machinery.
- **Recall vs. throughput.** MQAR accuracy is governed almost entirely by recurrent state size, not parameters: on synthetic MQAR at 70M–360M scale, accuracy is a clean function of $d_{\text{state}}$, and closing the gap to attention costs proportionally more state (Zoology, ICLR 2024; Based, ICML 2024).
- **State tracking.** Diagonal linear RNNs and SSMs fail $S_5$ word problems at any length; DeltaNet with negative eigenvalues solves parity and $S_3$-type tasks; $n_h$ Householders extend this further (Grazzi et al. 2025; Siems et al. 2025). Measured on synthetic group-word tasks at model scales under 100M.
- **At production scale.** An 8B-parameter, 3.5T-token comparison (Waleffe et al., 2024) found pure Mamba/Mamba-2 matching or beating a matched transformer on most standard zero-shot tasks but lagging on tasks requiring in-context copying and few-shot format tracking (five-shot MMLU, Phonebook retrieval); a hybrid with a small number of attention layers closed the gap and exceeded the transformer by roughly 2.6 points averaged over 12 tasks.
- **Length generalization.** On RULER (COLM 2024), effective context of most models is far below advertised context; this holds for both attention and linear models, so it does not by itself separate them.

## 5. What Is Not Known

- **Theoretically open.** Whether any *fixed-state* architecture in the $S_t = A_tS_{t-1}+v_tk_t^\top$ family with $A_t$ from a polynomially-computable matrix set can match log-precision softmax attention on all $\mathrm{TC}^0$ functions with $\mathrm{poly}(\log L)$ state. Also open: the exact expressivity ladder as a function of $n_h$ (Householder count) — only containments, no matching lower bounds.
- **Empirically open.** Whether the copy/recall separation persists at $\ge 30$B parameters and $\ge 10$T tokens with matched state budgets, or whether it is absorbed by scale. Runnable; not run at that scale with a controlled state budget.
- **Methodologically blocked.** Attributing an observed failure to representation rather than optimization. $\Delta(f)$ as defined in §2 is not estimable on a trained model: nobody can certify $\min_\theta \mathcal{L}_{\text{lin}}$. Every published "linear attention cannot do X" claim on a trained checkpoint is an upper bound on found parameters, not on reachable ones.

## 6. Why It Is Hard

**The binding obstruction is that the proven bounds are loose by three to four orders of magnitude at real scale, so they never bite where the failures are observed.** The information-theoretic argument says a model with $M$ bits of state fails a task requiring more than $M$ bits. Deployed linear models carry $M \sim 10^9$ bits (see §10), while the tasks they empirically fail carry $\sim 10^5$ bits of payload. The lower bound is satisfied with a factor of $10^4$ to spare, and the model still fails. The theory therefore does not explain the observation, and no sharper quantity — an "addressable" or "retrievable" fraction of the state — has an agreed definition or an estimator.

Secondary: benchmark confounding. Needle-in-a-haystack and its relatives measure a single retrieval against uniform distractors; RULER shows this correlates weakly with multi-hop or aggregation performance. An architecture can win the benchmark named "long context" without the property the name asserts.

## 7. Current Research (as of 2026)

- **Richer transitions.** Delta-rule and Householder-product families (MIT/Songlin Yang, NVIDIA, University of Freiburg) — pushing $A_t$ from diagonal to structured non-diagonal while keeping a chunkwise-parallel training kernel.
- **State-tracking-first design.** Deliberately choosing $A_t$ eigenvalue ranges to hit a target group-theoretic expressivity, then checking language-modelling loss does not regress *(frontier — verify at >7B)*.
- **Hybrid ratio science.** Systematic sweeps of attention-layer fraction and placement (NVIDIA, AI21, Mistral lineage). Mostly reported as benchmark numbers rather than controlled state-budget ablations.
- **Test-time-training / fast-weight views.** Treating the recurrent state as an inner-loop learner (Stanford, Sakana-adjacent groups) — reframes capacity as an online-learning regret question *(frontier — verify)*.
- **Circuit-complexity refinement.** Attempts to find a class finer than $\mathrm{TC}^0$ that separates the families *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** is the linear-attention recall gap representational or optimizational at matched state?

- **Scale.** Four models at 1.4B parameters, 100B tokens, identical data order and tokenizer: (a) softmax attention, (b) Mamba-2, (c) Gated DeltaNet, (d) hybrid with 3 attention layers of 24. Configure (b) and (c) so total recurrent state $M$ matches the softmax KV cache at $L=8{,}192$ to within 5%.
- **Control arm.** For each linear model, a *constructed-weights* twin: hand-build the associative-recall circuit (key $\to$ outer-product write, query $\to$ read) into the same architecture and freeze it. This measures $\min_\theta\mathcal{L}_{\text{lin}}$ from above by construction, converting the unmeasurable $\Delta(f)$ into a two-sided estimate.
- **Task.** MQAR with $N \in \{64, 256, 1024, 4096\}$ key–value pairs, evaluated at $L_{\text{eval}}/L_{\text{train}} \in \{1,2,4,8\}$.
- **Deciding number.** The gap $g = \text{acc}(\text{constructed}) - \text{acc}(\text{trained})$ at $N=1024$, $\rho=4$. If $g < 0.05$, the limit is representational and richer $A_t$ is the right lever. If $g > 0.30$, the limit is optimization and the architecture literature is mis-diagnosing its own failures. Cost: roughly $4\times$ a 1.4B/100B run, order $10^4$ A100-hours.

## 9. Key References

- **[Foundational]** Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML, 2020. — arXiv:2006.16236
- **[Foundational]** Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML, 2021. — arXiv:2102.11174
- **[Theory]** William Merrill, Ashish Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL, 2023.
- **[Theory]** William Merrill, Jackson Petty, Ashish Sabharwal. *The Illusion of State in State-Space Models.* ICML, 2024. — arXiv:2404.08819
- **[Theory]** Samy Jelassi, David Brandfonbrener, Sham Kakade, Eran Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Theory]** Satwik Bhattamishra, Michael Hahn, Phil Blunsom, Varun Kanade. *Separations in the Representational Capabilities of Transformers and Recurrent Architectures.* NeurIPS, 2024. — arXiv:2406.09347
- **[Theory]** Kaiyue Wen, Xingyu Dang, Kaifeng Lyu. *RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval.* 2024. — arXiv:2402.18510
- **[SOTA]** Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML, 2024. — arXiv:2312.06635
- **[SOTA]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML, 2024. — arXiv:2405.21060
- **[SOTA]** Songlin Yang, Jan Kautz, Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR, 2025. — arXiv:2412.06464
- **[SOTA]** Riccardo Grazzi, Julien Siems, Jörg Franke, Arber Zela, Frank Hutter, Massimiliano Pontil. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR, 2025. — arXiv:2411.12537
- **[Empirical]** Simran Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR, 2024. — arXiv:2312.04927
- **[Empirical]** Roger Waleffe et al. *An Empirical Study of Mamba-based Language Models.* 2024. — arXiv:2406.07887
- **[Benchmark]** Cheng-Ping Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654

## 10. Worked Example

**Phonebook lookup on an 8B Mamba-2, carried through.**

State budget. Mamba-2 8B: model dim $4096$, expansion $2$ giving inner dim $8192$, head dim $64$ (so $h=128$ heads), $d_{\text{state}}=128$, $\ell=56$ layers. Per layer:

$$h \cdot d_{\text{head}} \cdot d_{\text{state}} = 128 \times 64 \times 128 = 1{,}048{,}576 \text{ entries.}$$

At bf16, $2$ MiB per layer, so $M = 56 \times 2\,\text{MiB} \approx 117\,\text{MiB} \approx 9.4\times10^{8}$ bits.

Task payload. A phonebook of $1{,}000$ entries, each a $10$-character name plus a $7$-digit number:

$$1000 \times \big(10 \times 8 + 7 \times \log_2 10\big) \approx 1000 \times 103 = 1.03\times10^{5}\ \text{bits} \approx 12.9\ \text{kB}.$$

Ratio. $9.4\times10^{8} / 1.03\times10^{5} \approx 9{,}100$. The state exceeds the information content of the task by nearly four orders of magnitude.

Observed. Pure Mamba-2 at this scale degrades on phonebook-style lookup well before $1{,}000$ entries, while a hybrid with a handful of attention layers holds up (Waleffe et al. 2024).

**What this makes visible.** The one-way communication lower bound — the only rigorous tool available — predicts failure only when payload $> M$, i.e. above roughly $9\times10^{6}$ entries. The model fails around $10^{3}$. So the proven bound is not the operative constraint; something about *addressing* the state, not *storing* in it, is. No published quantity measures that something, which is precisely why §5 lists the attribution question as methodologically blocked and why §8's constructed-weights control is the cheapest way to break the tie.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*