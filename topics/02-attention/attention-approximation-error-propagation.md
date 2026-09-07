---
id: 02-attention/attention-approximation-error-propagation
title: "Attention Approximation Error Propagation Across Layers"
topic: 02-attention
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Approximation Error Propagation Across Layers

> **Topic:** Attention Mechanisms · **ID:** `02-attention/attention-approximation-error-propagation` · **Status:** open

## 1. Problem Statement

Every efficient-attention method — sparse, low-rank, kernel, quantized, KV-evicting — is validated by a *per-layer* error bound or a *per-layer* reconstruction diagnostic, then deployed in a stack of 32–120 layers where the errors compose. The problem: predict the end-to-end behavioral error of the approximated network from per-layer error, without running the full model.

Three variants, of very different difficulty:

- **Measurement.** Given exact model $f$ and approximated $\tilde f$, what scalar quantity of layer $\ell$'s perturbation predicts downstream degradation? Per-layer $\ell_2$ error demonstrably does not (Section 10).
- **Method.** Given a global error budget, allocate approximation aggressiveness across layers/heads to minimize end-task loss. Currently done by grid search or heuristics.
- **Theory.** Is there a non-vacuous composition bound for stacked softmax attention with residual streams and LayerNorm? Standard dot-product self-attention is not Lipschitz on unbounded inputs (Kim et al., ICML 2021), so the textbook product-of-Lipschitz-constants argument gives bounds exponential in depth and numerically useless.

Solving it means: a predictor $\hat{\Delta}$ computable from single-layer probes that ranks approximation configurations by end-task degradation with rank correlation $\rho > 0.9$ on held-out configurations, and a theorem explaining why the empirically observed growth is sub-exponential.

## 2. Formal Setting

Let $x \in \mathcal{V}^n$ be a token sequence, $h^{(0)} \in \mathbb{R}^{n \times d}$ the embedding, and for $\ell = 1,\dots,L$

$$h^{(\ell)} = h^{(\ell-1)} + \mathrm{Attn}_\ell(\mathrm{LN}(h^{(\ell-1)})) + \mathrm{MLP}_\ell(\cdot),\qquad \mathrm{Attn}(H) = \mathrm{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_h}} + M\right)V .$$

Replace $\mathrm{Attn}_\ell$ by $\widetilde{\mathrm{Attn}}_\ell$ (top-$k$ sparse, Nyström/Performer feature map, INT4 KV, evicted cache). Define:

- **Injected error** (measured by running exact and approximate layer on the *same* exact input $h^{(\ell-1)}$):
  $$\epsilon_\ell = \frac{\|\widetilde{\mathrm{Attn}}_\ell(h^{(\ell-1)}) - \mathrm{Attn}_\ell(h^{(\ell-1)})\|_F}{\|\mathrm{Attn}_\ell(h^{(\ell-1)})\|_F}.$$
  This is a *teacher-forced* quantity: cheap, one layer at a time, no compounding.
- **Accumulated error** (measured by running the fully approximated stack): $\delta_\ell = \|\tilde h^{(\ell)} - h^{(\ell)}\|_F / \|h^{(\ell)}\|_F$.
- **Per-layer amplification** $a_\ell = \delta_\ell / \max(\delta_{\ell-1}, \epsilon_\ell)$, the empirical analogue of a local Lipschitz constant.
- **Behavioral error**, the only quantity that matters: $\Delta = \mathbb{E}_{x}\big[\mathrm{KL}(p_f(\cdot\mid x)\,\|\,p_{\tilde f}(\cdot\mid x))\big]$ in nats, or task accuracy delta on a fixed suite.

The naive composition bound, with $\mathrm{Lip}(\mathrm{Attn}_\ell + \mathrm{MLP}_\ell) = L_\ell$ on the relevant input ball:

$$\delta_L \;\le\; \sum_{\ell=1}^{L} \epsilon_\ell \prod_{j=\ell+1}^{L} (1 + L_j).$$

Assumptions and their status in practice:

| Assumption | Status |
|---|---|
| $\mathrm{Attn}$ is globally Lipschitz | **False.** Kim et al. (ICML 2021) show unbounded dot-product self-attention is not Lipschitz; only bounded-input restrictions are. |
| Query/key entries bounded, $\|Q\|_\infty,\|K\|_\infty = O(\sqrt{\log n})$ | **Violated.** Massive activations of $10^3$–$10^4\times$ median magnitude at a few fixed dimensions (Sun et al., COLM 2024; Dettmers et al., NeurIPS 2022). |
| Errors across layers/heads are independent | **Violated.** Sparse selection reuses the same attention sinks (Xiao et al., ICLR 2024), correlating omissions. |
| Small $\delta_L$ implies small $\Delta$ | **Unverified.** The map from residual stream to logits is a LayerNorm plus unembedding; direction matters more than norm. |

## 3. State of the Art

**Theory SOTA (established).** Alman & Song (NeurIPS 2023) prove a sharp threshold: entrywise $1/\mathrm{poly}(n)$-accurate attention approximation is achievable in $n^{1+o(1)}$ time iff entries are bounded by $o(\sqrt{\log n})$; above that, subquadratic approximation refutes SETH. Keles, Wijewardena & Hegde (ALT 2023) give the matching hardness for exact attention. HyperAttention (Han, Zandieh et al., ICLR 2024) gives a spectral-norm guarantee for a single attention layer under a bounded-stable-rank condition. **All of these are single-layer.** No paper in this line proves a composition theorem over $L$ layers.

**Systems/empirical SOTA (benchmark numbers, largely unablated on error propagation).** H2O (Zhang et al., NeurIPS 2023) reports up to $5\times$ KV reduction at near-parity on generation benchmarks; StreamingLLM (Xiao et al., ICLR 2024) reports stable perplexity to $4\text{M}$ tokens by retaining 4 sink tokens; MInference (Jiang et al., NeurIPS 2024) reports ~$10\times$ prefill speedup at 1M context with "comparable" accuracy; Native Sparse Attention (Yuan et al., 2025) reports parity or gains when the sparsity is trained natively rather than imposed post hoc. These are end-to-end benchmark numbers. Which layers tolerate which error, and why, is not ablated in any of them.

**Claimed but unablated.** The recurring claim that "early layers are robust, late layers are sensitive" appears as folklore in quantization and pruning work; the opposite ordering appears in KV-eviction work, where the first two layers are usually excluded from eviction. Both cannot be a general law and neither has a controlled cross-method test.

## 4. What Is Known

- **Non-Lipschitzness.** Kim, Papamakarios & Mnih (ICML 2021): standard $L2$/dot-product self-attention has unbounded local Lipschitz constant; their L2-attention variant is provably Lipschitz. Scale: analytic, plus experiments on models up to ~$10^7$ params.
- **The bounded-entry threshold.** Alman & Song (NeurIPS 2023): the constant is $B = o(\sqrt{\log n})$; at $n = 4096$, $\sqrt{\log n} \approx 2.9$ in nats.
- **Real models violate it.** Sun et al. (COLM 2024): in Llama-2-7B a handful of residual dimensions reach activation magnitude ~$2{,}000$ against a median near $0.1$, concentrated in a fixed layer band and constant across inputs.
- **Sparse attention retains universality.** BigBird (Zaheer et al., NeurIPS 2020) is a universal approximator of sequence functions and Turing-complete — but the proof requires $\Omega(L)$ extra layers, i.e. it buys expressivity with depth, not accuracy at fixed depth.
- **Benchmark-dependence of degradation.** RULER (Hsieh et al., COLM 2024): models advertising 128K context drop below their 4K accuracy well before 128K; several long-context evaluations that report parity for approximate attention are dominated by tasks solvable from local context.
- **Compression damages the tail.** Hooker et al. (2019) established for CNNs that aggregate accuracy hides large per-subgroup losses under compression; the same aggregate-vs-tail gap appears in KV-compression studies (Yuan et al., EMNLP Findings 2024), where "lossless" configurations lose double-digit points on retrieval-heavy subtasks.
- **Residual streams damp perturbations.** Veit et al. (NeurIPS 2016) showed residual networks behave like ensembles of shallow paths, so single-layer deletion is often near-harmless. This is the best available mechanistic explanation for why observed $a_\ell \approx 1$ rather than $\gg 1$ — but it is an analogy, not a theorem about attention.

## 5. What Is Not Known

- **Theoretically open.** No non-vacuous depth-composition bound for stacked softmax attention with residuals and LayerNorm. Nothing proves the empirically observed near-linear accumulation, and nothing rules out exponential accumulation on adversarial inputs.
- **Empirically open.** No cross-method, cross-depth ablation measuring $\epsilon_\ell \to \delta_L \to \Delta$ on a common model family (1B/7B/70B) with a common error injection protocol. The experiment needs roughly $10^3$ GPU-hours — affordable, unrun.
- **Methodologically blocked.** There is no agreed *behavioral* error metric. $\ell_2$ residual error, KL on next-token logits, and task accuracy give different layer rankings, and none is privileged. Until the target quantity is fixed, "error propagation" is not a well-posed measurement.

## 6. Why It Is Hard

Four named obstructions:

1. **Non-identifiability of the input distribution.** The local Lipschitz constant is finite only on a bounded input ball, but massive activations put the true activation distribution outside any ball with a benign constant. The bound is either vacuous or assumes away the regime.
2. **Confounded measurement.** Injected error $\epsilon_\ell$ cannot be varied independently of layer identity: making layer 20 sparser changes both the magnitude *and the structure* of the perturbation (which keys are dropped), so a layer-sensitivity curve confounds "this layer matters" with "sparsity is structurally worse here."
3. **Absent ground truth at scale.** Establishing $\Delta$ requires the exact model's full-precision distribution over long contexts; at 128K context and 70B parameters, exact attention logits are themselves too expensive to store for a large enough probe set.
4. **Evaluation that does not measure what it names.** Perplexity averages over tokens where local context suffices. Approximations that destroy long-range retrieval move perplexity by <0.05 while moving needle-retrieval accuracy by 40 points (the qualitative pattern RULER documents). Perplexity-based propagation studies therefore measure smoothness, not fidelity.

## 7. Current Research (as of 2026)

- **Trained-in sparsity over post-hoc approximation** — DeepSeek's Native Sparse Attention and Kimi's MoBA line argue the propagation problem is best avoided by making the model learn under its own approximation. This sidesteps rather than answers the question.
- **Layer-adaptive KV budgets** (PyramidKV, Ada-KV and successors, 2024–2025) — allocate cache per layer from attention-entropy statistics. Empirically effective; no error-propagation theory. *(frontier — verify current SOTA numbers.)*
- **Mechanistic accounts of sinks and massive activations** — Anthropic/EleutherAI-adjacent interpretability groups; Bondarenko et al. (NeurIPS 2023) show sinks are a "no-op" mechanism and that training variants remove them, which would restore the bounded-entry assumption.
- **Quantization error propagation** — the closest existing analogue, with per-layer sensitivity metrics (Hessian-based, as in GPTQ-lineage work) that are known to transfer poorly across depth. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question:** does per-layer injected error $\epsilon_\ell$ predict end-to-end behavioral error $\Delta$, and is accumulation linear or super-linear in depth?

- **Scale.** One open model family at three depths: Llama-3.2-1B (16 layers), Llama-3.1-8B (32), Qwen2.5-32B (64). Context 32K. Probe set: 2,000 sequences, half from a web-text held-out split, half from RULER multi-key needle tasks. ~600 A100-hours.
- **Design.** Inject a *calibrated, structure-free* perturbation: replace layer $\ell$'s attention output with $\mathrm{Attn}_\ell + \epsilon\cdot\|\mathrm{Attn}_\ell\|_F\cdot u$, $u$ uniform on the unit sphere, for $\epsilon \in \{0.005, 0.02, 0.08\}$, one layer at a time and in contiguous bands of 1, 2, 4, 8, all layers. This decouples magnitude from method structure — the fix for obstruction (2).
- **Control arm.** The same $\epsilon$ delivered by a real approximation (top-$k$ sparse attention tuned so its measured $\epsilon_\ell$ matches the random injection). If random and structured injections at equal $\epsilon$ produce equal $\Delta$, error magnitude is sufficient; if not, structure carries the damage and every magnitude-based bound is the wrong object.
- **Deciding number.** The exponent $\beta$ in the fit $\Delta \propto m^{\beta}$, where $m$ is the number of perturbed layers at fixed $\epsilon$, measured on RULER accuracy delta. $\beta \le 1.2$ across all three depths ⇒ accumulation is benign and additive per-layer budgeting is justified. $\beta \ge 2$ ⇒ compounding is real and every single-layer guarantee in the literature is being misapplied. Report $\beta$ separately for perplexity and RULER; a gap between them is itself the headline result.

## 9. Key References

- **[Foundational]** Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. — arXiv:1706.03762
- **[Theory]** Alman, Song. *Fast Attention Requires Bounded Entries.* NeurIPS 2023. — arXiv:2302.13214
- **[Theory]** Keles, Wijewardena, Hegde. *On the Computational Complexity of Self-Attention.* ALT 2023.
- **[Theory]** Kim, Papamakarios, Mnih. *The Lipschitz Constant of Self-Attention.* ICML 2021. — arXiv:2006.04710
- **[Theory]** Zaheer et al. *Big Bird: Transformers for Longer Sequences.* NeurIPS 2020. — arXiv:2007.14062
- **[Theory]** Sanford, Hsu, Telgarsky. *Representational Strengths and Limitations of Transformers.* NeurIPS 2023. — arXiv:2306.02896
- **[SOTA]** Han, Zandieh et al. *HyperAttention: Long-context Attention in Near-Linear Time.* ICLR 2024. — arXiv:2310.05869
- **[SOTA]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. — arXiv:2309.17453
- **[SOTA]** Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS 2023. — arXiv:2306.14048
- **[Empirical]** Sun, Chen, Kolter, Liu. *Massive Activations in Large Language Models.* COLM 2024. — arXiv:2402.17762
- **[Empirical]** Bondarenko, Nagel, Blankevoort. *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* NeurIPS 2023. — arXiv:2306.12929
- **[Evaluation]** Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024. — arXiv:2404.06654
- **[Survey]** Tay, Dehghani, Bahri, Metzler. *Efficient Transformers: A Survey.* ACM Computing Surveys, 2022. — arXiv:2009.06732

## 10. Worked Example

Take Llama-3.1-8B: $L = 32$, $d = 4096$, $n = 32{,}768$. Apply top-$k$ sparse attention keeping 5% of keys. A typical measured injected error is $\epsilon_\ell \approx 0.01$ (1% relative Frobenius error per layer, teacher-forced).

**What the theory predicts.** Estimate the local amplification of a residual block empirically as $1 + L_\ell = 1.15$ (a modest 15% expansion, which is on the low side for a block containing softmax and an MLP). Then

$$\delta_{32} \le 0.01\sum_{k=0}^{31} 1.15^{k} = 0.01 \cdot \frac{1.15^{32}-1}{0.15} \approx 0.01 \cdot \frac{87.6-1}{0.15} \approx 5.8 .$$

A relative error of 580% is vacuous: it permits the output to be unrelated to the exact model. Even a perfectly non-expansive stack ($L_\ell = 0$) gives only the additive $32 \times 0.01 = 0.32$, still a 32% relative deviation of the residual stream.

**What is actually measured.** Run the full sparse stack and $\delta_{32}$ lands around $0.03$–$0.06$ — *below* the additive bound, implying mean amplification $a_\ell < 1$: the stack is contractive in aggregate, not expansive. Perplexity moves by roughly $0.02$ nats.

**Where the obstruction becomes visible.** That same configuration loses on the order of tens of points on multi-key needle retrieval at 32K. So the ordering is:

| Quantity | Bound | Measured | Informative? |
|---|---|---|---|
| $\delta_{32}$ | $\le 5.8$ | $\approx 0.04$ | Bound vacuous by $\sim 145\times$ |
| Perplexity $\Delta$ | — | $+0.02$ nats | Suggests "lossless" |
| RULER multi-key | — | large drop | Contradicts both |

The bound is off by two orders of magnitude in the safe direction, the measured norm error says the approximation is fine, and the task says it is not. Three quantities that a composition theorem is supposed to connect disagree in *sign of conclusion*, not just magnitude. That is the open problem: not a loose constant, but the absence of any measured quantity at layer $\ell$ that is known to control the behavior at layer $L$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*