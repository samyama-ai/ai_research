---
id: 16-state-space-models/diagonal-vs-dense-recurrence-expressivity
title: "Expressivity Gap Between Diagonal and Dense Linear Recurrences"
topic: 16-state-space-models
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Expressivity Gap Between Diagonal and Dense Linear Recurrences

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/diagonal-vs-dense-recurrence-expressivity` · **Status:** partially-solved

## 1. Problem Statement

Modern state-space models (S4D, Mamba, LRU, GLA) replace a dense state transition matrix $A \in \mathbb{R}^{N\times N}$ with a diagonal one. The question: **what is lost?**

Three variants, with different answers and different difficulty:

- **Theory variant.** For which classes of sequence-to-sequence maps does the diagonal-constrained family fail to contain a function the dense family contains, at equal state size $N$, depth $L$, and precision $p$? Solving it means a separation theorem plus a matching upper bound.
- **Measurement variant.** Given a trained dense-recurrence model, decide whether its behaviour is reachable by a diagonal model of the same budget. This requires a metric that is invariant to the change of basis $A \mapsto V^{-1}AV$, which most reported diagnostics are not.
- **Method variant.** Find the cheapest structured relaxation of diagonality — Householder products, block-diagonal, diagonal-plus-low-rank — that recovers the missing capability while keeping a parallel scan of cost $O(T\log T)$ or $O(T)$.

The theory variant is **settled for time-invariant recurrences** (no gap, generically) and **settled in one direction for time-varying ones** (a real gap, from non-commutativity). What remains open is the quantitative version: how much depth, width, or precision buys back the gap.

## 2. Formal Setting

A single linear recurrence layer maps input $u_{1:T}$, $u_t\in\mathbb{R}^{d}$, to output $y_{1:T}$ via

$$x_t = A_t x_{t-1} + B_t u_t,\qquad y_t = C_t x_t + D u_t,\qquad x_0=0,$$

with $x_t\in\mathbb{K}^N$, $\mathbb{K}\in\{\mathbb{R},\mathbb{C}\}$. Write $\Phi_{t\leftarrow s}=A_t A_{t-1}\cdots A_{s+1}$ for the transition operator, so $y_t = \sum_{s\le t} C_t\Phi_{t\leftarrow s}B_s u_s + Du_t$.

**Families.**
- $\mathcal{D}_N$: $A_t$ diagonal. $\mathcal{D}_N^{+}$: additionally $\mathbb{K}=\mathbb{R}$ with entries in $(0,1)$ (Mamba's $A_t=\exp(-\Delta_t\,\mathrm{softplus})$).
- $\mathcal{F}_N$: $A_t$ unconstrained.
- $\mathcal{H}_{N,k}$: $A_t=\prod_{j=1}^{k}(I-\beta_j v_jv_j^\top)$, products of $k$ Householders (DeltaNet is $k=1$).

**LTI case** ($A_t\equiv A$, $B_t\equiv B$, $C_t\equiv C$). The realized map is the convolution kernel $\bar K = (CA^{s}B)_{s\ge0}$, equivalently the transfer function $H(z)=C(zI-A)^{-1}B$, a rational function of degree $\le N$. **Measured as:** fit $\bar K$ over $s<T$ and report relative $\ell_2$ kernel error $\|\bar K-\bar K^\star\|_2/\|\bar K^\star\|_2$.

**Measured quantities.**
- Eigenvector conditioning $\kappa(V)=\|V\|\|V^{-1}\|$ for $A=V\Lambda V^{-1}$ — measured by `numpy.linalg.eig` in float64 on the initialized $A$, not the trained one.
- Non-normality $\nu(A)=\|A^\top A-AA^\top\|_F/\|A\|_F^2$.
- Effective monoid rank: for time-varying models, $\mathrm{rank}$ of the set $\{\Phi_{t\leftarrow 0}\}$ over sampled inputs, measured as the dimension of the span of vectorized $\Phi$ matrices.
- Capability gap: accuracy on a word problem over a fixed finite monoid $M$ (e.g. $S_5$, $\mathbb{Z}_2$-parity) at train length $T_{\mathrm{tr}}$ and test length $T_{\mathrm{te}}>T_{\mathrm{tr}}$.

**Assumptions, and which break.**
1. *Simple spectrum.* Generic $A$ is diagonalizable over $\mathbb{C}$; then $\mathcal{F}_N$ and $\mathcal{D}_N$ realize identical transfer functions, since $V$ is absorbed into $B,C$. **Violated in practice:** HiPPO-LegS is highly non-normal, and the absorbing change of basis is numerically unrepresentable (§10).
2. *Unconstrained $B,C$ per channel.* Real models share one $A$ across $d$ channels with structured $B,C$; the basis change can then be absorbed only once, not per channel. **Violated** in every multi-head implementation.
3. *Exact arithmetic.* Separations stated over $\mathbb{R}$ can vanish or invert under $p$-bit fixed precision. **Violated** — bf16 training is standard.

## 3. State of the Art

**Theory SOTA (established).**
- Diagonal $\approx$ dense for LTI: eigendecomposition argument, made explicit in Gupta et al. (DSS, NeurIPS 2022) and Gu et al. (S4D, NeurIPS 2022); Orvieto et al. (LRU, ICML 2023) reuse it to justify dropping HiPPO entirely.
- Sarrof, Veitsman & Hahn (NeurIPS 2024): SSMs with diagonal, input-gated recurrences and finite precision recognize exactly the **star-free** regular languages; they cannot recognize PARITY.
- Merrill, Petty & Sabharwal (ICML 2024): S4 and Mamba, as uniformly log-precision circuits, lie in $\mathrm{L}$-uniform $\mathrm{TC}^0$; no state tracking beyond it, so no $S_5$ word problem, under $\mathrm{TC}^0\ne\mathrm{NC}^1$.
- Grazzi et al. (ICLR 2025): the restriction to *positive* eigenvalues, not diagonality itself, is what blocks parity and modular counting; allowing eigenvalues in $[-1,1]$ restores it.
- Cirone et al. (NeurIPS 2024): a rough-path/signature analysis showing dense selective SSMs span a strictly richer set of signature coordinates than diagonal ones.

**Empirical SOTA (claimed, partially unablated).**
- DeltaProduct (Siems et al., 2025) interpolates $\mathcal{H}_{N,k}$ from DeltaNet toward dense and reports full $S_n$ state tracking as $k$ grows *(frontier — verify at LM scale)*.
- RWKV-7 uses a generalized delta rule and claims expressivity beyond $\mathrm{TC}^0$ *(frontier — verify)*.
- Mamba-2 (Dao & Gu, ICML 2024) goes *further* than diagonal — $A_t=a_tI$, scalar — and matches Mamba-1 perplexity while running faster. This is a benchmark number, not an ablation of expressivity: no state-tracking task separates the two in the paper.

## 4. What Is Known

- **LTI: no gap.** S4D-Lin/S4D-Inv reach Long Range Arena averages within about one point of S4's 86.09 (S4 v3, ICLR 2022), at $N=64$ per channel; Path-X is the most sensitive task. Reported as benchmark numbers only.
- **Non-normality is the obstruction to naive diagonalization.** The HiPPO matrix is diagonalizable, but the S4 paper shows the diagonalizing matrix has entries of magnitude $2^{\Theta(N)}$ — order $2^{3N/4}$. At $N=64$ that is $\approx 3\times10^{14}$, past float32's $2^{24}$ mantissa.
- **Parity.** Diagonal recurrences with $a_t\in(0,1)$ (Mamba, GLA, mLSTM) fail parity at any length; extending to $[-1,1]$ gives near-100% length generalization on parity and $\mathbb{Z}_n$ arithmetic (Grazzi et al., ICLR 2025), with no language-modelling regression reported up to 1.3B parameters.
- **$S_5$.** Neither diagonal nor rank-1-delta recurrences solve $S_5$ composition with length generalization; $k\ge2$ Householder products do, on synthetic sequences of length $\sim256$.
- **Depth substitutes for density, partially.** Constant-depth diagonal stacks fail asymptotically but reach >99% in-distribution on short state-tracking sequences; failure appears only under length extrapolation.

## 5. What Is Not Known

- **Theoretically open.** The depth–density trade: is there $c>0$ such that any $S_5$-tracking task solvable by one dense layer needs $\Omega(\log^{c} T)$ diagonal layers? Only $\Omega(1)$-vs-impossible is proved. Also open: a tight lower bound on state size $N$ for diagonal models emulating a $k$-Householder recurrence, if one exists at all.
- **Empirically open.** Whether the separation matters for natural language. No controlled run holds parameters, data, and tokens fixed while varying only $A_t$ structure (scalar → diagonal → signed-diagonal → $k$-Householder → dense) at $\ge$3B parameters on $\ge$300B tokens. The 1.3B evidence is suggestive and underpowered for downstream reasoning deltas.
- **Methodologically blocked.** "Expressivity used by the trained model" has no basis-invariant estimator. $\kappa(V)$ and $\nu(A)$ change under reparameterization that leaves the function identical, so current diagnostics measure the parameterization, not the function.

## 6. Why It Is Hard

**Non-identifiability, then confounded measurement.** The map from $(A,B,C)$ to the realized function is invariant under $ (V^{-1}AV, V^{-1}B, CV)$. Any statistic computed on $A$ alone — eigenvalue histogram, condition number, effective rank — is therefore not a property of the model's behaviour. Two networks with identical outputs on all inputs can have $\kappa(V)=1$ and $\kappa(V)=10^{14}$. This kills the obvious experiment ("look at the spectrum of a trained dense model and check whether diagonal suffices").

Compounding it: the tasks that *do* separate the classes (parity, $S_5$) are synthetic, and the tasks people care about (perplexity, MMLU) are dominated by capacity and data, not by transition structure. So the evaluation does not measure the thing it names. A perplexity tie between Mamba-2 (scalar $A$) and Mamba-1 (diagonal $A$) is evidence about a benchmark, not about expressivity.

## 7. Current Research (as of 2026)

- **Structured non-diagonal transitions.** DeltaNet and DeltaProduct lines (Yang, Kautz, Hatamizadeh; Siems, Hutter and collaborators) — Householder products with a chunkwise-parallel scan. Active question: the throughput cost per additional Householder *(frontier — verify)*.
- **Eigenvalue-range engineering.** Grazzi, Franceschi, Pontil and collaborators; negative and complex eigenvalues in otherwise diagonal recurrences, cheapest known fix for the counting gap.
- **Formal-language characterization.** Hahn's group (Saarland) mapping SSM variants onto the star-free/regular/$\mathrm{TC}^0$ hierarchy; Merrill & Sabharwal (AI2) on circuit-complexity upper bounds.
- **Continuous-time/signature theory.** Cirone, Salvi and collaborators, treating selectivity as a controlled differential equation.
- **Hybridization** — attention layers inserted into diagonal-SSM stacks (Jamba, Zamba, Nemotron-H) sidestep the gap rather than close it; whether the attention layers are doing the state tracking is untested *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does transition-matrix structure change downstream capability once perplexity is matched?

- **Scale.** Five 1.4B-parameter models, identical tokenizer, data order, and 100B tokens from a fixed corpus. Arms differ only in $A_t$: (a) scalar $a_tI$, (b) diagonal positive, (c) diagonal signed ($[-1,1]$), (d) $k=2$ Householder, (e) dense block-diagonal with $16\times16$ blocks. Match parameter count by adjusting $N$ per arm, not width.
- **Control arm.** (b), diagonal positive — the Mamba-standard configuration. Also report a same-token attention baseline to bound the ceiling.
- **Evaluation.** A held-out state-tracking suite embedded in text: $S_5$ composition, bracket matching at depth 8, and parity over 512 tokens, each trained on lengths $\le128$ and evaluated at 512; plus standard LM perplexity.
- **The deciding number.** Length-extrapolated $S_5$ accuracy at $T=512$, arm (d) minus arm (b), conditional on validation perplexity differing by $<0.02$ nats. If that difference is $\ge 20$ points, the expressivity gap is real at LM scale and structure is worth its cost. If it is $<5$ points, the gap is a synthetic-task artifact at this scale and the field should stop paying for density.

## 9. Key References

- **[Foundational]** Albert Gu, Karan Goel, Christopher Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022. — arXiv:2111.00396
- **[Foundational]** Ankit Gupta, Albert Gu, Jonathan Berant. *Diagonal State Spaces are as Effective as Structured State Spaces.* NeurIPS 2022. — arXiv:2203.14343
- **[Foundational]** Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré. *On the Parameterization and Initialization of Diagonal State Space Models.* NeurIPS 2022. — arXiv:2206.11893
- **[Foundational]** Antonio Orvieto, Samuel L. Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, Soham De. *Resurrecting Recurrent Neural Networks for Long Sequences.* ICML 2023. — arXiv:2303.06349
- **[SOTA]** William Merrill, Jackson Petty, Ashish Sabharwal. *The Illusion of State in State-Space Models.* ICML 2024. — arXiv:2404.08819
- **[SOTA]** Yash Sarrof, Yana Veitsman, Michael Hahn. *The Expressive Capacity of State Space Models: A Formal Language Perspective.* NeurIPS 2024. — arXiv:2405.17394
- **[SOTA]** Riccardo Grazzi, Julien Siems, Simon Schrodi, Thomas Brox, Frank Hutter, et al. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR 2025. — arXiv:2411.12537
- **[SOTA]** Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024. — arXiv:2406.06484
- **[SOTA]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[Theory]** Nicola Muca Cirone, Antonio Orvieto, Benjamin Walker, Cristopher Salvi, Terry Lyons. *Theoretical Foundations of Deep Selective State-Space Models.* NeurIPS 2024. — arXiv:2402.19047
- **[Survey]** Bingbing Liu et al. *A Survey on Visual Mamba.* Applied Sciences, 2024. (Architecture-side survey; no formal-expressivity coverage.)

## 10. Worked Example

**Instance.** A 3-state layer over the alphabet $\{a,b\}$, where $a$ applies the transposition $(1\,2)$ and $b$ applies $(2\,3)$, as permutation matrices $P_a,P_b\in\{0,1\}^{3\times3}$. These generate $S_3$ and do not commute: $P_aP_b\ne P_bP_a$. Read out $y_T = e_1^\top \Phi_{T\leftarrow0}e_1$.

**Dense arm.** $\Phi$ is the product of permutations; $ab$ and $ba$ give different states, and $y$ distinguishes them. Exact, $N=3$, one layer, length-independent.

**Diagonal arm.** Any input-selected diagonal $A_t=\mathrm{diag}(\lambda^{(u_t)})$ gives $\Phi_{T\leftarrow0}=\mathrm{diag}\!\left(\prod_t \lambda^{(u_t)}\right)$, which depends only on the **counts** $\\#a,\\#b$ — the generated monoid is abelian. So $ab$ and $ba$ produce byte-identical activations. On a balanced two-token discrimination set the layer is pinned at 50%: chance. No choice of $N$, complex eigenvalues, or gating changes this, because commutativity is a property of the diagonal family, not of $N$.

**Where the obstruction becomes visible.** Now stack 6 diagonal layers with MLPs between them and train on $S_3$ words of length $\le64$. The model reaches 99%+ in-distribution — the MLPs implement a shortcut of the $O(\log T)$ prefix-product circuit, valid up to the depth it was trained at. At $T=256$ accuracy collapses toward the majority-class rate. The separation is real but invisible to any evaluation that tests at the training length.

**The conditioning half.** Take $A$ = HiPPO-LegS, $N=64$. It is diagonalizable, so a diagonal model *exists* realizing the same kernel; the diagonalizing $V$ has entries of order $2^{3N/4}\approx3\times10^{14}$, while float32 resolves $2^{24}\approx1.7\times10^{7}$. The equivalence is a theorem with no float32 witness — which is why S4D needed a different initialization (S4D-Lin) rather than a diagonalization of HiPPO. The gap is not in the function class; it is in the representable parameterizations of it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*