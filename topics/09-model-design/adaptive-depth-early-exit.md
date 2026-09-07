---
id: 09-model-design/adaptive-depth-early-exit
title: "Adaptive Depth and Early Exit Without Quality Cliffs"
topic: 09-model-design
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Adaptive Depth and Early Exit Without Quality Cliffs

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/adaptive-depth-early-exit` · **Status:** open

## 1. Problem Statement

A fixed-depth transformer spends the same compute on "the capital of France is ___" and on a 4-step arithmetic carry. Adaptive depth aims to spend serial compute per input: exit early, skip layers, or recur more when the input is hard. The engineering result is reliable — average FLOPs drop. The open problem is the **quality cliff**: a small subset of inputs, usually the ones that need the most serial computation, degrades sharply while the aggregate benchmark average stays flat.

Three variants, different difficulty:

- **Measurement.** Given a static model $M$ and an adaptive variant $M_\pi$ matched on *average* cost, decide whether $M_\pi$ has a cliff: a non-negligible input subpopulation with large quality loss. Currently underdefined — mean accuracy on GLUE/MMLU is the wrong statistic, and the subpopulation is not named in advance.
- **Method.** Build a routing policy $\pi$ that attains a target speedup with a *per-input* (not per-corpus) guarantee: for all but $\delta$ of inputs, $M_\pi$'s output is within $\epsilon$ of $M$'s.
- **Theory.** Characterize which function classes admit input-dependent depth reduction at all. A cliff may be forced, not fixable: if depth is the serial-computation budget, halving it on an input that provably needs it must fail.

Solving it means: an adaptive model with $\geq 2\times$ measured wall-clock speedup at batch size $\geq 32$, whose worst-decile quality loss is bounded and reported, and whose router's confidence signal is validated against difficulty ground truth rather than against its own logits.

## 2. Formal Setting

A transformer with $L$ blocks $f_1,\dots,f_L$, hidden states $h^{(\ell)}_t$ for token $t$. Static forward pass: $h^{(L)}_t = (f_L \circ \cdots \circ f_1)(h^{(0)}_t)$.

An **adaptive policy** $\pi$ produces a per-token exit index $e_t \in \{1,\dots,L\}$ (early exit), a per-layer skip mask $m_{t,\ell}\in\{0,1\}$ (layer skipping / mixture-of-depths), or a recurrence count (ACT/PonderNet). Output uses a head $g_{e_t}$ attached at the exit layer.

**Cost, as measured.** Not FLOPs. Define
$$C(\pi) = \mathbb{E}_{x\sim\mathcal D}\big[\text{wall-clock latency of } M_\pi(x)\big]$$
measured at a fixed batch size $B$, sequence length, and hardware. FLOPs and latency diverge sharply here: with $B>1$ a batch finishes when its *slowest* member finishes, so realized speedup is governed by $\mathbb{E}[\max_{i\le B} e^{(i)}]$, not $\mathbb{E}[e]$.

**Quality gap.** For task metric $q$,
$$\Delta(x) = q(M(x)) - q(M_\pi(x)), \qquad \bar\Delta = \mathbb{E}_{\mathcal D}[\Delta(x)].$$
A **cliff** is a tail property, so define the conditional-value-at-risk at level $\alpha$:
$$\mathrm{CVaR}_\alpha(\Delta) = \mathbb{E}\big[\Delta(x) \mid \Delta(x) \geq \mathrm{VaR}_\alpha(\Delta)\big].$$
The problem is: policies are tuned on $\bar\Delta$ and reported on $\bar\Delta$; $\mathrm{CVaR}_{0.1}$ is almost never reported. **Cliff-free at level $(\alpha,\epsilon)$** means $\mathrm{CVaR}_\alpha(\Delta)\le\epsilon$.

**Router signal.** Typical $\pi$ exits when a confidence statistic exceeds a threshold $\tau$: softmax max-probability at layer $\ell$, hidden-state cosine saturation $\cos(h^{(\ell)},h^{(\ell-1)})>\tau$, or a trained meta-classifier. Calibration is the question of whether $\Pr[\,M(x)=M_\pi(x) \mid \text{conf}=c\,]$ tracks $c$.

**Assumptions, and which are violated.**
1. *Exit heads see in-distribution states.* Violated — intermediate states are trained under a full-depth objective unless the model is explicitly co-trained (LayerSkip, CALM), so heads at layer $\ell$ operate off-manifold.
2. *Per-token independence.* Violated in autoregressive decoding — an early exit at position $t$ leaves layers $>e_t$ with no KV entries for $t$, so later tokens attend to fabricated (copied or lazily-computed) state. Error compounds along the sequence.
3. *Difficulty is observable from the prefix.* Violated for the hardest cases: an input can look easy at layer 4 and require layer 30 to resolve.
4. *i.i.d. calibration.* Threshold $\tau$ fitted on a calibration set transfers only under distribution match; long-tail and adversarial inputs are exactly where it fails.

## 3. State of the Art

**Established (ablated, reproduced):**
- **Confident Adaptive Language Modeling (CALM)**, Schuster et al., NeurIPS 2022: early exit in T5 encoder-decoder with a *distributional* guarantee via Learn-then-Test calibration (Angelopoulos et al., 2021) — bounds textual/risk consistency with the full model at confidence $1-\delta$. Reported up to $\sim3\times$ decoder speedup on CNN/DM, WMT, SQuAD. This is the strongest existing result because the guarantee is per-corpus-risk, not per-example.
- **LayerSkip**, Elhoushi et al., ACL 2024: layer-dropout + early-exit-loss training, then *self-speculative decoding* — draft with the early exit, verify with the remaining layers. Reported up to $2.16\times$ on Llama-2 7B CNN/DM. Verification makes the output distribution match the full model, so the cliff is converted into a latency variance problem rather than a quality problem. This is the cleanest existing evasion of the cliff.
- **Mixture-of-Depths**, Raposo et al., 2024: top-$k$ token routing per block with a *static* compute graph (capacity known ahead of time), $\sim$12.5% of tokens through the block. IsoFLOP-matched language-modeling parity with faster steps.
- **Layer pruning depth studies** — Gromov et al., "The Unreasonable Ineffectiveness of the Deeper Layers" (2024); Men et al., "ShortGPT" (2024): large fractions of deep layers can be removed with small MMLU movement. Established as a benchmark number.

**Claimed but unablated / benchmark-only:**
- Early-exit GLUE results (DeeBERT, ACL 2020; FastBERT, ACL 2020; PABEE, NeurIPS 2020) report speedup at near-zero or *improved* average accuracy. These are single-label classification at $\leq$110M–235M parameters with short inputs; the results have not been shown to transfer to generative decoding, and per-subset breakdowns are largely absent.
- Layer-pruning "no degradation" claims rest heavily on multiple-choice benchmarks (MMLU, HellaSwag) which are partially recoverable from shallow lexical priors. Reasoning-heavy and long-generation evaluation of the same pruned checkpoints is thin.
- Wall-clock claims at batch size $>1$ are frequently absent; several early-exit papers report FLOP reduction only.

## 4. What Is Known

- **Depth is a serial-computation budget with a complexity-theoretic ceiling.** Log-precision fixed-depth transformers are contained in uniform $\mathrm{TC}^0$ (Merrill & Sabharwal, TACL 2023). Reducing effective depth on an input strictly reduces the circuit depth available; for problems that need serial steps, this is not recoverable by better routing. Cliffs are partly a theorem, not a bug.
- **Compositional tasks degrade non-gracefully with depth.** Dziri et al., "Faith and Fate" (NeurIPS 2023), show transformer accuracy on multi-digit multiplication and puzzles collapses as compositional depth grows — the failure is a cliff in problem-difficulty space, exactly the axis adaptive depth cuts.
- **Softmax confidence is a poor exit signal for generation.** CALM's own ablations rank a trained early-exit classifier and hidden-state saturation above max-softmax; the calibration gap is why the paper needs a statistical calibration procedure rather than a hand-set threshold.
- **Batching destroys naive early-exit speedups.** At $B=32$, latency tracks $\max_i e^{(i)}$; a policy with mean exit at layer 8 of 32 can deliver near-zero speedup. This is measured routinely in serving practice and is the reason MoD fixes capacity statically.
- **Verification recovers exactness.** Speculative decoding (Leviathan et al., ICML 2023; Chen et al., 2023) gives output distributions identical to the target model. Self-speculative variants (Zhang et al., "Draft & Verify", 2023; LayerSkip) apply this to depth, so a mis-routed token costs latency, not quality.

## 5. What Is Not Known

- **Theoretically open.** No characterization of the function class admitting $(\alpha,\epsilon)$-cliff-free adaptive depth. No lower bound of the form "any policy achieving $2\times$ expected depth reduction on distribution $\mathcal D$ incurs $\mathrm{CVaR}_{0.1}(\Delta)\ge c$." No theory of whether difficulty is *prefix-predictable* — i.e. whether a router at layer $\ell$ can even in principle estimate the required depth.
- **Empirically open.** Whether trained-for-exit models (LayerSkip/CALM-style) at 70B+ scale hold up on long-horizon agentic and multi-step reasoning traces where errors compound over thousands of tokens. Runnable today; not run at that scale with tail reporting.
- **Methodologically blocked.** The cliff itself has no agreed measurement. There is no standard hard-subset, no standard $\alpha$, no requirement to report $\mathrm{CVaR}$, and no ground-truth "required depth" label per input. Without a difficulty oracle, "the router was wrong" and "the model was wrong anyway" are not separable.

## 6. Why It Is Hard

Two specific obstructions.

**Absent ground truth for required depth.** There is no label $\ell^*(x)$ = minimum depth sufficient for $x$. The only proxy is exhaustive: run the model at every truncation depth $1..L$ and record where the answer stabilizes — $O(L)$ forward passes per example, and even that measures *this* model's truncation profile, not intrinsic difficulty. Every router is therefore trained against its own model's agreement signal, which is self-confirming.

**An evaluation that does not measure what it names.** "No quality degradation" is reported as a mean over benchmarks whose hard tail is a few percent of items. A policy that fails 100% of the 3% hardest items moves an MMLU average by $\sim$3 points — within the noise band routinely attributed to prompt formatting. The metric names quality; it measures the head of the difficulty distribution.

Both compound with the KV-cache violation of assumption 2: the cheapest fix (copy $h^{(e_t)}$ into layers above) injects an error that grows with generation length, so cliffs appear late in sequences where short-form benchmarks never look.

## 7. Current Research (as of 2026)

- **Verification-based depth adaptivity** — self-speculative decoding as the dominant practical answer (Meta's LayerSkip line; Draft & Verify). Direction: make the draft path a learned shallow sub-network so acceptance rate rises. Cliff-free by construction; cost is latency variance.
- **Static-capacity routing** — Mixture-of-Depths and successors (Google DeepMind), chosen specifically because dynamic control flow does not batch. *(frontier — verify)* MoD-style routing combined with MoE width routing in production-scale models.
- **Recurrent-depth / latent reasoning** — models that loop a block a variable number of times to add serial compute (Universal Transformers, Dehghani et al., ICLR 2019; PonderNet, Banino et al., 2021; recent depth-recurrent LM work, 2025). Inverts the problem: add depth on hard inputs rather than remove it on easy ones.
- **Conformal / risk-controlling routing** — extending Learn-then-Test guarantees from corpus risk to conditional (per-subgroup) risk. *(frontier — verify)*
- **Depth-pruning skepticism** — follow-ups showing the "deep layers are useless" result weakens under reasoning and long-context evaluation. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** At matched wall-clock speedup, does a confidence-routed early-exit model have a quality cliff that verification-based routing does not?

**Scale.** Llama-3-class 8B, trained with early-exit loss + layer dropout (LayerSkip recipe) so intermediate heads are in-distribution. Single node, 8×H100. Est. 2–4k GPU-hours including the exit-training phase.

**Arms** (all at measured $1.8\times$ end-to-end wall-clock speedup, batch size 32, 512-token generations):
1. **Control:** full-depth model, no adaptivity.
2. Confidence-threshold early exit, $\tau$ calibrated on a held-out i.i.d. set.
3. Self-speculative (draft at exit layer, verify with full stack) — exact-output arm.
4. Uniform static layer pruning to the same mean depth — the cheap baseline every adaptive method must beat.

**Difficulty labels.** Build $\ell^*(x)$ by running the control at every truncation depth on 5,000 held-out items ($O(L)$ passes, budgeted). Stratify the evaluation into deciles of $\ell^*$.

**Deciding number.** $\mathrm{CVaR}_{0.1}(\Delta)$ — mean quality loss over the worst 10% of examples — on a mixed suite (GSM8K, MMLU-Pro, a 4k-token summarization set, a 20-step agentic trace set). **Arm 2 passes if $\mathrm{CVaR}_{0.1}(\Delta) \le 2$ points absolute; it has a cliff if $\mathrm{CVaR}_{0.1}(\Delta) \ge 10$ points while $\bar\Delta \le 1$ point.** The gap between $\bar\Delta$ and $\mathrm{CVaR}_{0.1}$ is the reportable quantity the literature currently omits.

**Secondary number.** Latency variance (p99/p50) for arm 3 — the cost verification pays for exactness.

## 9. Key References

- **[Foundational]** Graves, A. *Adaptive Computation Time for Recurrent Neural Networks.* 2016. — arXiv:1603.08983
- **[Foundational]** Teerapittayanon, S., McDanel, B., Kung, H.T. *BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks.* ICPR, 2016.
- **[Foundational]** Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., Kaiser, Ł. *Universal Transformers.* ICLR, 2019. — arXiv:1807.03819
- **[Foundational]** Elbayad, M., Gu, J., Grave, E., Auli, M. *Depth-Adaptive Transformer.* ICLR, 2020. — arXiv:1910.10073
- **[SOTA]** Schuster, T., Fisch, A., Gupta, J., Dehghani, M., Bahri, D., Tran, V.Q., Tay, Y., Metzler, D. *Confident Adaptive Language Modeling.* NeurIPS, 2022. — arXiv:2207.07061
- **[SOTA]** Elhoushi, M., et al. *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding.* ACL, 2024. — arXiv:2404.16710
- **[SOTA]** Raposo, D., Ritter, S., Richards, B., Lillicrap, T., Humphreys, P.C., Santoro, A. *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models.* 2024. — arXiv:2404.02258
- **[SOTA]** Leviathan, Y., Kalman, M., Matias, Y. *Fast Inference from Transformers via Speculative Decoding.* ICML, 2023. — arXiv:2211.17192
- **[Theory]** Merrill, W., Sabharwal, A. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL, 2023.
- **[Evidence]** Dziri, N., et al. *Faith and Fate: Limits of Transformers on Compositionality.* NeurIPS, 2023. — arXiv:2305.18654
- **[Evidence]** Gromov, A., Tirumala, K., Shapourian, H., Glorioso, P., Roberts, D.A. *The Unreasonable Ineffectiveness of the Deeper Layers.* 2024. — arXiv:2403.17887
- **[Method]** Zhou, W., Xu, C., Ge, T., McAuley, J., Xu, K., Wei, F. *BERT Loses Patience: Fast and Robust Inference with Early Exit.* NeurIPS, 2020. — arXiv:2006.04152
- **[Calibration]** Angelopoulos, A.N., Bates, S., Candès, E.J., Jordan, M.I., Lei, L. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* 2021. — arXiv:2110.01052
- **[Survey]** Han, Y., Huang, G., Song, S., Yang, L., Wang, H., Wang, Y. *Dynamic Neural Networks: A Survey.* IEEE TPAMI, 2022. — arXiv:2102.04906

## 10. Worked Example

**Setup.** 32-layer model, mean exit at layer 12 under a max-softmax threshold $\tau=0.9$. Evaluation: 1,000 MMLU items + 100 3-digit×3-digit multiplication items, weighted as they appear in a typical suite (91% / 9%).

**Aggregate.** Suppose early exit is correct on 99% of the MMLU items ($\Delta=0$) and wrong on 60% of the multiplication items. Then
$$\bar\Delta = 0.91\times(0.01\times 1) + 0.09\times(0.60\times 1) = 0.0091 + 0.054 = 0.063.$$
Reported as **6.3 points**? No — reported the way papers report it, as a benchmark average shift, and the multiplication set is usually not in the suite at all. Drop it: $\bar\Delta = 0.9$ points. **Headline: "1.9× speedup, <1 point degradation."**

**Tail.** Now compute $\mathrm{CVaR}_{0.1}$ over the full mixture. The worst 10% of examples are dominated by the multiplication items (9% of the mass, 60% failure) plus the hardest MMLU tail:
$$\mathrm{CVaR}_{0.1}(\Delta) \approx 0.60 \text{–} 0.65 \Rightarrow \textbf{60+ points}.$$

**The obstruction made visible.** Same policy, same run. $\bar\Delta = 0.9$; $\mathrm{CVaR}_{0.1} = 60$. A factor of $\sim$65 between the number reported and the number a user of the hard subpopulation experiences.

**Why the router cannot fix it.** For 3×3 multiplication the model must carry partial products across serial steps. At layer 12 the partial products are not yet composed, but the softmax over the first output digit is already peaked — the leading digit of a product is often guessable from magnitude alone. So $\text{conf} = 0.94 > \tau$ while the answer is wrong. The confidence signal is high *because* the easy part of the answer resolves early. No threshold on this statistic separates the cases; the signal is measuring the wrong thing.

**What the batching number does to the headline.** At $B=32$ with exits distributed around layer 12 (std $\approx$ 6), $\mathbb{E}[\max_i e^{(i)}] \approx 12 + 6\cdot\sqrt{2\ln 32} \approx 27$. Realized speedup $\approx 32/27 = 1.19\times$, not the $32/12 = 2.7\times$ the FLOP count implies. The measured cliff is large and the measured speedup is small — both invisible in the standard reporting.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*