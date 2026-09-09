---
id: 16-state-space-models/hybrid-attention-ssm-layer-ratio
title: "Hybrid Attention-SSM Layer Ratio Optimality"
topic: 16-state-space-models
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hybrid Attention-SSM Layer Ratio Optimality

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/hybrid-attention-ssm-layer-ratio` · **Status:** empirically-open

## 1. Problem Statement

Every frontier hybrid language model interleaves a small number of softmax-attention layers among a majority of state-space (SSM) or linear-recurrent layers. The published ratios disagree by an order of magnitude: Jamba uses 1 attention layer per 8 (12.5%), NVIDIA's Mamba-2-Hybrid uses 4 of 28 mixer layers (14%, reported as ~8% of all 56 layers), Samba alternates 1:1, Qwen3-Next uses 1:3, MiniMax-01 uses 1:7. Each was chosen by small-scale sweep or by fiat, then frozen for the large run.

The problem: **determine the optimal attention fraction $\rho^\*$ as a function of parameter count $N$, token budget $D$, and target context length $T$ — and determine whether $\rho^\*$ is scale-invariant.**

Three variants, of different difficulty:

- **Measurement.** Define an objective under which $\rho^\*$ is well posed. Perplexity alone is nearly flat in $\rho$ over $[0.1, 0.5]$; the discriminating capabilities (exact copying, associative recall, long-context format following) are underweighted or absent in the standard evaluation averages that hybrid papers report. *This variant is the bottleneck.*
- **Method.** Given a fixed budget, find $\rho^\*$ and the placement of attention layers empirically. Runnable today; not run at more than one scale with a shared recipe.
- **Theory.** Prove a lower bound on the number of attention layers needed for a task class (e.g. $n$-token exact copy, $k$-key associative recall) as a function of recurrent state size. Partially answered for $\rho = 0$ vs $\rho > 0$; open for the interior.

Solving it means: a predictive rule $\rho^\*(N, D, T)$ validated by held-out extrapolation to a scale not used to fit it.

## 2. Formal Setting

A hybrid has $L$ sequence-mixing layers, indexed $\ell = 1..L$, each of type $t_\ell \in \{\mathrm{ATT}, \mathrm{SSM}\}$, with channel-mixing MLPs interleaved. Let

$$\rho = \frac{1}{L}\sum_{\ell=1}^{L} \mathbf{1}[t_\ell = \mathrm{ATT}], \qquad \pi = (t_1,\dots,t_L)$$

so $\rho$ is the ratio and $\pi$ the *placement*. Note $\rho$ is a function of $\pi$, not a sufficient statistic for it.

**Measured quantities.**

- *Loss.* $\mathcal{L}(\pi; N, D, T)$ = held-out next-token cross-entropy in nats/token on a fixed corpus, measured at $T$-token context with no sliding-window truncation.
- *Parameters.* $N = N_{\text{emb}} + L_{\mathrm{att}}N_{\mathrm{att}} + L_{\mathrm{ssm}}N_{\mathrm{ssm}} + N_{\text{mlp}}$. Critically $N_{\mathrm{att}} \neq N_{\mathrm{ssm}}$: a Mamba-2 layer at hidden size $d$ carries roughly $3d^2$ projection parameters versus $2d^2$ for GQA attention with $n_{kv} < n_h$. Varying $\rho$ at fixed $L$ therefore changes $N$; iso-parameter and iso-layer sweeps are different experiments.
- *Decode memory.* KV cache bytes at context $T$, batch $B$, precision $b$:
$$M_{\mathrm{KV}} = 2\,b\,B\,T\,\rho L\, n_{kv} d_h, \qquad M_{\mathrm{SSM}} = b\,B\,(1-\rho)L\, n_h d_h d_s$$
$M_{\mathrm{KV}}$ is linear in $T$; $M_{\mathrm{SSM}}$ is constant in $T$. This is the entire economic motivation for $\rho < 1$.
- *Throughput.* Tokens/s at batch $B$ and context $T$ on a named accelerator — not FLOPs, since SSM kernels and attention kernels have very different hardware utilization.

**The optimization.** Constrained Pareto selection, not unconstrained loss minimization:
$$\rho^\* (N,D,T) = \arg\min_{\rho}\ \mathcal{L}(\pi_\rho; N, D, T) \quad \text{s.t.}\quad M_{\mathrm{KV}}(\rho, T) \le M_{\max},\ \ \text{tok/s} \ge R_{\min}$$

**Assumptions, and which are violated.**

1. *$\rho$ summarizes the architecture.* **Violated.** Placement matters: Jamba and Nemotron-H both avoid attention in layer 1, and Samba's results depend on attention being *sliding-window*, which changes $M_{\mathrm{KV}}$ from $O(T)$ to $O(w)$ and makes $\rho$ non-comparable across papers.
2. *A single $\rho^\*$ exists per scale.* **Suspect.** The loss surface in $\rho$ is empirically shallow; the argmin may be unidentifiable within seed noise.
3. *Held-out perplexity ranks downstream capability.* **Violated** for exactly the capabilities at issue (copy, recall) — see §6.
4. *Training recipe is $\rho$-invariant.* **Violated.** Optimal learning rate and initialization differ between attention and SSM layers; a single recipe tuned at one $\rho$ handicaps others.

## 3. State of the Art

**Empirical SOTA — established (ablated at a single scale).**

- **Waleffe et al. (NVIDIA, 2024)** ran the only controlled three-way comparison at 8B parameters / 3.5T tokens with a shared recipe: pure Transformer, pure Mamba-2, and a Mamba-2-Hybrid with 4 attention, 24 Mamba-2, and 28 MLP layers. The hybrid exceeded the Transformer by **+2.65 points** averaged over 12 standard tasks and predicted tokens up to **8× faster** at inference. This is the strongest evidence that $\rho^\* \in (0,1)$ strictly.
- **Poli et al. (2024, "MAD")** established the methodology of screening architectures on synthetic tasks then validating with scaling laws, and reported that striped hybrids beat homogeneous stacks at matched compute.

**Empirical — claimed but unablated.**

- Jamba's 1:7 ratio is stated as chosen from small-scale ablation; the ablation grid is not published at the 52B scale of the released model. The 52B/12B-active result is a **benchmark number**, not a ratio ablation.
- Nemotron-H, Falcon-H1, Zamba2, MiniMax-01, Qwen3-Next each report strong benchmark numbers at one ratio. None publishes a same-recipe sweep over $\rho$ at its production scale. Ratio choices spanning 1:1 to 1:7 coexist with comparable reported quality, which is itself evidence the loss surface is flat — or that the benchmarks do not resolve it.
- Distillation results (Wang et al., *The Mamba in the Llama*, NeurIPS 2024) show a Llama checkpoint converted to a hybrid keeping **50%, 25%, and 12.5%** of attention layers, with quality degrading gracefully as attention is removed. This bounds $\rho$ for *conversion*, which need not equal $\rho^\*$ for *pretraining*.

**Theory SOTA.** No bound on $\rho^\*$ exists. What exists bounds the endpoints: Jelassi et al. (ICML 2024) prove a fixed-state recurrent model cannot copy strings whose information content exceeds its state, while a two-layer transformer copies with $O(\log n)$ width; Merrill, Petty & Sabharwal (ICML 2024) place SSM layers in $\mathrm{TC}^0$, so they cannot solve state-tracking problems believed outside $\mathrm{TC}^0$ — but neither is Turing-complete-style separation broken by adding *one* attention layer, and nobody has proved how many are needed.

## 4. What Is Known

- **$\rho = 0$ is strictly worse for recall-heavy tasks at 8B/3.5T.** Pure Mamba-2 lags the Transformer on 5-shot MMLU and on phonebook-style lookup; the hybrid closes the gap (Waleffe et al., 2024).
- **Very small $\rho$ suffices for long-context retrieval.** The 8B Mamba-2-Hybrid with 4 attention layers matched or exceeded the Transformer on 23 long-context tasks at 16K and 32K contexts.
- **Recall scales with recurrent state, not layer count alone.** Arora et al. (ICLR 2024; ICML 2024) show a smooth recall–throughput Pareto frontier governed by state size $d_s$ in linear-attention models; gap-to-attention on multi-query associative recall closes as $d_s$ grows. Scale: 360M–1.3B, synthetic and Pile.
- **Local attention alone is enough at 1.3B–7B.** Griffin (DeepMind, 2024) matches Llama-2 quality at 7B/300B tokens using only local (sliding-window) attention mixed with gated linear recurrences — attention fraction $\approx 1/3$ but with $O(w)$ cache.
- **Placement is not free.** Multiple production reports put attention layers in the middle third and avoid the first layer; no paper isolates placement from ratio with a controlled sweep.

## 5. What Is Not Known

- **Empirically open.** Whether $\rho^\*$ is scale-invariant. Every published ratio was picked at $\le$ 1.5B parameters (or by architectural taste) and reused at 8B–450B. Nobody has run a $\rho$-sweep at two scales separated by $\ge 5\times$ with an identical recipe and reported the argmin shift. The experiment is a few hundred thousand GPU-hours — expensive, not novel.
- **Empirically open.** The interaction $\rho \times T$. If attention exists to serve exact retrieval, $\rho^\*$ should *fall* with context length (fewer, cheaper caches) or *rise* (more retrieval demand). No published curve.
- **Methodologically blocked.** The objective. There is no agreed metric under which $\rho^\*$ is identifiable: perplexity is flat, standard 12-task averages move by less than seed noise across the interesting range, and synthetic recall probes saturate. Until the metric is fixed, "optimal ratio" is not a well-posed quantity.
- **Theoretically open.** A lower bound of the form: any hybrid with $k$ attention layers and recurrent state of $s$ bits requires $k \ge f(n, s)$ to copy $n$ tokens exactly. Only $k = 0$ is settled.
- **Non-identifiability.** $\rho$ and $d_s$ purchase overlapping capability. Whether $(\rho, d_s)$ has a unique optimum or a degenerate valley is unknown.

## 6. Why It Is Hard

The obstruction is **an evaluation that does not measure what it names, compounded by confounded measurement.**

Standard hybrid-model reporting is an average over 12 short-context multiple-choice tasks. Those tasks are the ones the ratio *least* affects: they need little exact copying and fit in 2K tokens. The reported spread across ratios from 1:1 to 1:7 is a point or two — within the seed-to-seed variation of such averages (typically $\pm 0.3$–$0.5$ points; *assumed*, since hybrid papers report single seeds). So the headline metric cannot distinguish the hypotheses.

Second, $\rho$ cannot be varied alone. Changing it changes parameter count (SSM and attention layers differ in parameters per layer), FLOPs per token, memory traffic, and the optimal learning rate. An iso-parameter sweep and an iso-FLOP sweep can rank ratios differently, and both are defensible. A "clean" ablation requires choosing which confound to hold fixed, and that choice partly determines the answer.

Third, cost. A single $\rho$ point at 8B/1T tokens is on the order of $10^5$ GPU-hours. A 6-point sweep at two scales is a frontier-lab pretraining budget spent on an ablation, which is why every lab picks a ratio and ships.

## 7. Current Research (as of 2026)

- **NVIDIA (ADLR)** — the Nemotron-H family continues the ~8% attention design at 8B/47B/56B, with FP8 pretraining and pruning/distillation to smaller hybrids. The public artifacts are model releases, not ratio ablations. *(frontier — verify whether a ratio sweep accompanies the 2025–26 releases.)*
- **Alibaba (Qwen)** — Qwen3-Next uses a 3:1 gated-DeltaNet-to-attention mix, a much higher $\rho$ than NVIDIA's, at MoE scale. The two families disagree by ~3× on $\rho$ with no published head-to-head. *(frontier — verify.)*
- **TII (Falcon-H1)** and **Zyphra (Zamba2)** — parallel rather than sequential hybridization (attention and SSM in the same block, shared attention blocks). Parallel hybrids make $\rho$ ill-defined as a layer fraction and need a different parameterization. *(frontier — verify.)*
- **Stanford Hazy Research / Together** — the recall–throughput frontier line (Zoology, Based, and successors); the closest thing to a principled objective for choosing state versus attention.
- **Theory: NYU / AI2 / Harvard** — circuit-complexity characterizations of SSMs and of attention-augmented recurrences; the natural next result is a separation as a function of attention-layer count.
- **Distillation route** — converting trained transformers to hybrids at several $\rho$ values (MOHAWK, Mamba-in-the-Llama) is two orders of magnitude cheaper than pretraining and is being used as a proxy for the pretraining sweep. Whether the proxy is faithful is itself unestablished.

## 8. Concrete Next Experiment

**Question decided:** does $\rho^\*$ shift with scale?

**Scale.** Two model sizes, $N \approx 400\text{M}$ and $N \approx 2.4\text{B}$ (6× apart), each trained on Chinchilla-proportional tokens (8B and 48B) with one identical recipe, $T = 8192$. Fix $L = 24$ mixer layers at both scales; sweep $\rho \in \{0, 1/24, 2/24, 4/24, 6/24, 12/24, 24/24\}$ (7 points × 2 scales = 14 runs). Hold $N$ constant within each scale by adjusting MLP width, not layer count. Fix placement by a single deterministic rule (uniform spacing, never layer 1). Three seeds at $\rho \in \{2/24, 4/24, 6/24\}$ to measure noise. Estimated cost: ~15–25k A100-hours total — a single-node-cluster month, not a frontier budget.

**Control arms.** (a) $\rho = 1$ pure Transformer and $\rho = 0$ pure SSM at both scales, same recipe. (b) A *placement control*: at $\rho = 4/24$, three placements (early, uniform, late) to bound how much of any observed effect is ratio versus placement.

**Metric — fixed in advance, because §6 says the usual one cannot resolve it.** A composite $\mathcal{S}$ = mean of (i) held-out perplexity at 8K, (ii) exact-match on $k$-key phonebook lookup at $k \in \{16, 64, 256\}$ inserted at 8K context, (iii) 5-shot MMLU. Report each separately as well.

**The deciding number.** $\Delta\rho^\* = \rho^\*_{2.4\text{B}} - \rho^\*_{400\text{M}}$, measured as the argmin of $\mathcal{S}$ with a seed-noise band. **If $|\Delta\rho^\*| < 1/24$ (one layer), the ratio is scale-invariant and 8–17% is a defensible default. If $|\Delta\rho^\*| \ge 2/24$, every published ratio transferred from a small sweep to a large run is mis-set, and ratio must be re-tuned per scale.** A third outcome is informative and likely: if the $\mathcal{S}$ curve is flat within noise over $\rho \in [2/24, 12/24]$ at both scales, the problem is confirmed methodologically blocked and the field should optimize $M_{\mathrm{KV}}$ subject to a quality floor rather than optimize quality over $\rho$.

## 9. Key References

- **[Foundational]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[Foundational]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML, 2024. — arXiv:2405.21060
- **[SOTA]** Roger Waleffe et al. *An Empirical Study of Mamba-based Language Models.* 2024. — arXiv:2406.07887
- **[SOTA]** Opher Lieber et al. *Jamba: A Hybrid Transformer-Mamba Language Model.* 2024. — arXiv:2403.19887
- **[SOTA]** Liliang Ren et al. *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.* ICLR, 2025. — arXiv:2406.07522
- **[SOTA]** Soham De et al. *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.* 2024. — arXiv:2402.19427
- **[SOTA]** NVIDIA. *Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models.* 2025. — arXiv:2504.03624
- **[Theory]** Samy Jelassi, David Brandfonbrener, Sham Kakade, Eran Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Theory]** William Merrill, Jackson Petty, Ashish Sabharwal. *The Illusion of State in State-Space Models.* ICML, 2024. — arXiv:2404.08819
- **[Method]** Simran Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR, 2024. — arXiv:2312.04927
- **[Method]** Simran Arora et al. *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.* ICML, 2024. — arXiv:2402.18668
- **[Method]** Michael Poli et al. *Mechanistic Design and Scaling of Hybrid Architectures.* 2024. — arXiv:2403.17844
- **[Method]** Junxiong Wang et al. *The Mamba in the Llama: Distilling and Accelerating Hybrid Models.* NeurIPS, 2024. — arXiv:2408.15237
- **[Related]** Xin Dong et al. *Hymba: A Hybrid-head Architecture for Small Language Models.* ICLR, 2025. — arXiv:2411.13676
- **[Related]** Paolo Glorioso et al. *Zamba: A Compact 7B SSM Hybrid Model.* 2024. — arXiv:2405.16712
- **[Related]** MiniMax. *MiniMax-01: Scaling Foundation Models with Lightning Attention.* 2025. — arXiv:2501.08313

## 10. Worked Example

Take the 8B Mamba-2-Hybrid geometry: 56 total layers = 4 attention + 24 Mamba-2 + 28 MLP, $d = 4096$, GQA with $n_{kv} = 8$, $d_h = 128$, fp16.

**Memory, decode, $T = 131{,}072$, $B = 1$.** Per attention layer per token:
$$2 \times 8 \times 128 \times 2\ \text{B} = 4096\ \text{B} = 4\ \text{KiB}$$
With 4 attention layers: $16$ KiB/token, so $16 \times 131072 = \mathbf{2.0\ GiB}$. A pure Transformer of the same depth (28 attention layers) would need $28 \times 4\ \text{KiB} \times 131072 = \mathbf{14\ GiB}$ — 7× more, and growing linearly in $T$.

Mamba-2 recurrent state per layer, with $d_{\text{inner}} = 8192$, head dim 64 (128 heads), $d_s = 128$:
$$128 \times 64 \times 128 \times 2\ \text{B} = 2\ \text{MiB} \quad\Rightarrow\quad 24 \times 2\ \text{MiB} = \mathbf{48\ MiB},\ \text{independent of } T.$$

So going from $\rho = 28/28$ to $\rho = 4/28$ buys 12 GiB per sequence at 128K — enough to raise decode batch size by roughly an order of magnitude on an 80 GiB card. That is the whole case for hybrids, and it is unambiguous.

**Now the obstruction.** Suppose we ask whether $\rho = 4/28$ is *optimal* or merely *sufficient*. Compare against $\rho = 2/28$: it saves a further 1.0 GiB (2 GiB → 1 GiB), a 50% cut in cache. The quality question is whether 2 attention layers still support exact retrieval. Waleffe et al.'s 12-task average moved the hybrid $+2.65$ points over the Transformer; a plausible 2-layer variant would land somewhere in that band, and the band is roughly the width of seed noise on such averages. Perplexity would differ in the third decimal place. On needle-in-a-haystack, both would score near 100% and tell us nothing.

The signal lives only in a task the standard suite does not contain: a 256-key phonebook lookup at 128K context, where pure Mamba-2 collapses and the 4-attention hybrid recovers. Whether 2 layers or 3 layers is the knee is not reported by anyone, at any scale. So the decision that saves 1 GiB per sequence in production — a real cost, in dollars — currently rests on no measurement at all. That is the obstruction: the economics are precisely quantified and the quality axis is not measurable with the instruments in use.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*