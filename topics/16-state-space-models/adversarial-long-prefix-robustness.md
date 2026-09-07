---
id: 16-state-space-models/adversarial-long-prefix-robustness
title: "Recurrent Model Robustness to Adversarial Long Prefixes"
topic: 16-state-space-models
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Recurrent Model Robustness to Adversarial Long Prefixes

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/adversarial-long-prefix-robustness` · **Status:** empirically-open

## 1. Problem Statement

A recurrent sequence model — S4, Mamba, Mamba-2, RWKV, Griffin, or any linear-attention variant — compresses an unbounded prefix into a fixed-size state. A Transformer does not: its KV cache grows with context. This asymmetry gives an attacker a lever that has no Transformer analogue. Instead of perturbing tokens near the task, the attacker prepends a long, benign-looking prefix chosen so that the state entering the task region is corrupted.

**Input.** A model $f_\theta$, a benign task instance $(q, y^\star)$, an attacker-controlled prefix $p \in \mathcal{V}^L$.
**Output.** $f_\theta(p \Vert q)$.
**Decision predicate.** Does there exist $p$ with $|p| \le L$, drawn from a semantically innocuous set $\mathcal{P}$ (fluent text, valid documents, retrieved passages), such that $f_\theta(p \Vert q) \ne y^\star$ while $f_\theta(q) = y^\star$?

Three variants, with sharply different difficulty:

- **Measurement.** Define an attack success rate that separates *adversarial* prefix effects from *generic* long-context degradation. Currently under-specified — see §6.
- **Method.** Find $p$ efficiently. Discrete search over $L \sim 10^4$–$10^6$ tokens; gradients through the recurrence are available but the search space is astronomically larger than for GCG-style suffix attacks.
- **Theory.** Prove or refute: for a state of dimension $d$ over $\mathcal{V}$, there exist prefix pairs colliding in state but differing in downstream conditional distribution by $\Omega(1)$ total variation — and that fluent prefixes suffice.

**Solved** would mean: a scaling law $\varepsilon(L, d, N)$ for adversarial prefix damage, with matched Transformer control, plus a defence that flattens the $L$-dependence at fixed inference cost.

## 2. Formal Setting

A selective SSM layer with state $h_t \in \mathbb{R}^{d}$:

$$h_t = A(x_t)\, h_{t-1} + B(x_t)\, x_t, \qquad y_t = C(x_t)^\top h_t,$$

with $A(x_t)$ diagonal (Mamba) or diagonal-plus-low-rank (S4). Total recurrent state is $S = \sum_{\ell} d_\ell$ across layers; **measured** as bytes of the inference-time recurrent cache at batch 1, read directly from the allocator, not from a formula.

**Prefix-induced drift.** With $h^{(0)}$ the state after the empty prefix,

$$\delta(p) = \frac{\lVert h(p) - h^{(0)} \rVert_2}{\lVert h^{(0)} \rVert_2},$$

measured per layer at the last prefix token, before the task tokens enter.

**Task damage.** For metric $m$ (exact match, or $-\log P_\theta(y^\star \mid \cdot)$),

$$\Delta(p) = m\big(f_\theta(q), y^\star\big) - m\big(f_\theta(p \Vert q), y^\star\big).$$

**Adversarial excess.** The quantity that actually matters, because long prefixes hurt even when random:

$$\varepsilon(L) = \mathbb{E}_{q}\Big[\max_{p \in \mathcal{P}, |p|=L} \Delta(p)\Big] - \mathbb{E}_{q, p \sim \mathcal{D}_L}\big[\Delta(p)\big],$$

where $\mathcal{D}_L$ is natural text of the same length. The max is not computable; in practice it is a **lower bound** from $K$ search restarts, and $K$ must be reported.

**State collision.** Prefixes $p, p'$ collide at tolerance $\tau$ if $\lVert h(p) - h(p') \rVert_2 \le \tau$ yet $\mathrm{TV}\big(P_\theta(\cdot \mid p \Vert q),\, P_\theta(\cdot \mid p' \Vert q)\big) \ge \gamma$. Since $h$ is a deterministic function of $p$ and $|\mathcal{V}|^L \gg$ the effective number of distinguishable states at float precision, collisions exist for large $L$ by counting alone; the open question is whether they exist *within fluent text* at $L$ reachable in deployment.

**Assumptions, with the violated ones flagged.**
1. $\mathcal{P}$ is fluent and semantically innocuous — **enforced only by a proxy** (perplexity under a separate LM), which is a weak stand-in for "a human reviewer would not object". *Violated in practice.*
2. Transformer control is matched on parameters, tokens, tokenizer, and data. *Rarely holds* — public Mamba and Transformer checkpoints differ in data mixture.
3. The recurrence is executed at the precision used in the attack. *Violated:* attacks are often searched in fp32/bf16 and deployed against fp16 or quantised kernels; drift is precision-sensitive.
4. Task accuracy at $L=0$ is near ceiling, so $\Delta$ is not floor-limited. Holds only for easy tasks.

## 3. State of the Art

**Theory (established).** Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models* (ICML 2024): SSMs with the standard parallel-scan formulation lie in $\mathrm{TC}^0$ under log-precision, so they cannot express inherently sequential state tracking — the same expressivity ceiling as Transformers, not a strict advantage. Jelassi et al., *Repeat After Me: Transformers are Better than State Space Models at Copying* (ICML 2024): any recurrent model with state of $S$ bits fails to copy strings longer than $\Theta(S)$ bits, and two-layer Transformers copy with length generalization. These are **memory-capacity** results, not adversarial-robustness results; the reduction from "cannot copy" to "attacker can erase" is intuitive but **not proved**.

**Empirical (established).** Chen et al., *Stuffed Mamba: Oversized States Lead to the Inability to Forget* (2024), reports Mamba-2 models trained at short context collapsing catastrophically when extrapolated well beyond training length, with the failure traced to unbounded state norm growth. Waleffe et al., *An Empirical Study of Mamba-based Language Models* (2024, NVIDIA) trained 8B Mamba, Mamba-2, Mamba-2-Hybrid and Transformer on the same 1.1T-token mixture — the strongest matched control that exists — and found pure Mamba-2 clearly behind on copying/in-context-retrieval tasks (phonebook lookup) while the hybrid matched or beat the Transformer.

**Claimed but unablated.** Vendor and paper claims that hybrid architectures (Jamba; Griffin; Mamba-2-Hybrid) "solve" long-context recall rest on RULER/LongBench/needle-in-a-haystack scores under *benign* distractors. No published work searches over distractors adversarially against a hybrid. These are **benchmark numbers only**.

**Absent.** There is no published adversarial-prefix attack designed specifically for the fixed-state bottleneck, and no certified bound on $\delta(p)$ for a modern SSM. RNN certification exists only at toy scale (POPQORN, Ko et al., ICML 2019).

## 4. What Is Known

- **Copying bound.** State of $S$ bits ⟹ failure beyond $\Theta(S)$ bits of content to reproduce (Jelassi et al., ICML 2024). Demonstrated at 160M scale on synthetic copy; Transformer control succeeds at lengths where Mamba is near 0%.
- **Matched-training gap.** At 8B parameters / 1.1T tokens, Mamba-2 trails the Transformer on in-context retrieval; the 8:1 attention-to-Mamba hybrid closes it (Waleffe et al., 2024). *Scale: 8B.*
- **State collapse is real and length-triggered.** Failure at extrapolated lengths correlates with growing state norm, and increasing state size delays but does not remove it (Chen et al., 2024). *Scale: 130M–2.7B Mamba-2.*
- **Position sensitivity in Transformers.** Liu et al., *Lost in the Middle* (TACL 2024): accuracy varies by tens of points with the position of the gold document — the benign baseline any adversarial-prefix claim must beat.
- **Long benign context already degrades everything.** RULER (Hsieh et al., COLM 2024): most models claiming 32K+ hold accuracy far below their advertised length. So $\Delta$ is large even for $p \sim \mathcal{D}_L$; only $\varepsilon$ is informative.
- **Many-shot jailbreaking** (Anil et al., 2024) shows long benign-looking prefixes shift model behaviour log-linearly in the number of shots — evidence the prefix channel is exploitable, measured on Transformers, **never replicated on SSMs**.

## 5. What Is Not Known

- **Empirically open.** Does $\varepsilon(L)$ grow faster for SSMs than for matched Transformers? Runnable today with 1–3B matched checkpoints; nobody has run it with a real search over $p$.
- **Empirically open.** Does the hybrid ratio (attention layers per recurrent layer) monotonically reduce $\varepsilon$, and at what ratio does it match a pure Transformer?
- **Theoretically open.** No lower bound linking state dimension $d$ to the *minimum fluent prefix length* forcing an $\Omega(1)$ TV shift. Counting gives existence at absurd $L$; nothing gives a constructive, fluency-constrained bound.
- **Theoretically open.** Whether the $\Theta(S)$ copying bound implies any adversarial statement, or is orthogonal (adversarial damage may need only $O(1)$ tokens at the right position).
- **Methodologically blocked.** "Innocuous prefix" has no accepted operationalization. Perplexity filters, LM-judge fluency scores, and human review disagree; without a fixed $\mathcal{P}$, $\varepsilon$ is not comparable across papers.

## 6. Why It Is Hard

**Confounded measurement is the primary obstruction.** Any long prefix degrades any model. Reporting "Mamba drops 22 points under an adversarial prefix" is uninformative unless the same prefix length of natural text is measured on the same instances — and unless the Transformer control shares data, tokenizer, and token budget. Almost no public checkpoint pair satisfies that; the NVIDIA 8B suite is the exception and is not licensed uniformly.

**Second: the max in $\varepsilon(L)$ is unreachable.** Discrete search over $L=32{,}000$ positions with a fluency constraint is combinatorially far worse than GCG's 20-token suffix. Every reported $\varepsilon$ is a lower bound whose tightness depends on the attacker's compute, so a null result ("SSMs are robust") is never conclusive — it may just mean the search was weak. Attack-strength confounding makes negative results unpublishable and positive results non-comparable.

**Third: compute.** A single 8B SSM forward at 128K tokens is cheap in memory but the attack needs $10^3$–$10^5$ forwards per instance, times hundreds of instances, times two architectures. That is a multi-GPU-week bill for one point on the curve.

## 7. Current Research (as of 2026)

- **Hybrid architecture design** (NVIDIA; AI21 with Jamba; Google DeepMind with Griffin/RecurrentGemma) — motivated by benign recall, not adversarial robustness; the robustness implication is untested. *(frontier — verify)*
- **State-capacity and forgetting** (Tsinghua/OpenBMB line following *Stuffed Mamba*; Hazy Research on Based/Zoology) — quantifying how much information the recurrent state holds and when it saturates. Closest existing work to a capacity-based attack.
- **Long-context prompt injection and many-shot attacks** (Anthropic; academic red-teaming groups) — Transformer-only so far; the natural source of $\mathcal{P}$ constructions.
- **Certified robustness for recurrent nets** — dormant since POPQORN-era work; no scaling to selective SSMs. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Matched 1.3B checkpoints trained on identical data and tokenizer: (a) pure Mamba-2, (b) Transformer, (c) Mamba-2-Hybrid at 1 attention layer per 8 recurrent layers. Task: single-key needle retrieval, 500 instances, all at 100% at $L=0$.

**Attack.** Greedy coordinate search over 64 insertion slots of 8-token fluent spans (sampled from a corpus, ranked by gradient of $-\log P_\theta(y^\star)$ w.r.t. the state), $K=256$ restarts, prefix lengths $L \in \{2\mathrm{K}, 8\mathrm{K}, 32\mathrm{K}, 128\mathrm{K}\}$. Fluency gate: prefix perplexity under an independent 7B LM within $1.5\times$ the corpus median.

**Control arm.** Identical instances with $p$ sampled i.i.d. from the same corpus at the same $L$ and the same fluency gate — this is what isolates adversarial excess from generic long-context decay.

**Deciding number.** The ratio

$$\rho = \frac{\varepsilon_{\text{Mamba-2}}(32\mathrm{K})}{\varepsilon_{\text{Transformer}}(32\mathrm{K})}.$$

$\rho > 2$ with non-overlapping bootstrap 95% CIs over the 500 instances establishes a fixed-state-specific vulnerability. $\rho \in [0.8, 1.25]$ says the vulnerability is a long-context property, not an architectural one, and the topic should be folded into general prompt-injection work. Report $\varepsilon$ vs. $K$ so the search-strength confound is visible.

## 9. Key References

- **[Foundational]** Gu, A. & Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[Foundational]** Gu, A., Goel, K. & Ré, C. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR, 2022. — arXiv:2111.00396
- **[SOTA-theory]** Jelassi, S., Brandfonbrener, D., Kakade, S. & Malach, E. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[SOTA-theory]** Merrill, W., Petty, J. & Sabharwal, A. *The Illusion of State in State-Space Models.* ICML, 2024. — arXiv:2404.08819
- **[SOTA-empirical]** Waleffe, R. et al. *An Empirical Study of Mamba-based Language Models.* NVIDIA technical report, 2024. — arXiv:2406.07887
- **[SOTA-empirical]** Chen, Y. et al. *Stuffed Mamba: Oversized States Lead to the Inability to Forget.* 2024.
- **[Benchmark]** Hsieh, C.-P. et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Benchmark]** Liu, N. F. et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Attack]** Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J. Z. & Fredrikson, M. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* 2023. — arXiv:2307.15043
- **[Attack]** Anil, C. et al. *Many-shot Jailbreaking.* NeurIPS, 2024.
- **[Certification]** Ko, C.-Y., Lyu, Z., Weng, L., Daniel, L., Wong, N. & Lin, D. *POPQORN: Quantifying Robustness of Recurrent Neural Networks.* ICML, 2019. — arXiv:1905.07387
- **[Architecture]** De, S. et al. *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.* 2024. — arXiv:2402.19427
- **[Survey]** Peng, B. et al. *RWKV: Reinventing RNNs for the Transformer Era.* Findings of EMNLP, 2023. — arXiv:2305.13048

## 10. Worked Example

Take Mamba-2-2.7B. Its recurrent state is roughly 64 heads × state dim 128 × head dim 64 per layer across 64 layers — order $10^7$ floats, so $S \approx 2\times10^7$ bytes in bf16 at batch 1. Compare a Transformer of the same size at $L=32{,}000$: KV cache is order $10^9$ bytes. The Transformer holds ~50× more context-dependent bytes.

Now the task: retrieve one 6-digit key from a 32K-token document. That needle is $\approx 20$ bits. The copying bound says $S$ is enormous relative to 20 bits, so **capacity is not the binding constraint** — and here is the point the naive story misses. Run the benign control: natural-text prefix at 32K, Mamba-2-2.7B typically scores well below its 2K-context accuracy on this task, while its state is 6 orders of magnitude larger than the information it needs to keep. The failure is not capacity, it is *allocation* — the selection mechanism has already overwritten the needle with irrelevant content.

That is exactly why $\varepsilon$, not $\Delta$, is the quantity. Suppose an adversarial prefix drops accuracy from 100% to 41%. If the natural-text control at the same length gives 55%, the adversarial excess is 14 points, not 59. And if the matched Transformer shows $100 \to 78$ adversarial versus $100 \to 88$ benign — excess 10 points — then $\rho = 1.4$: the SSM is worse in absolute terms and barely worse in adversarial terms. Every number in this paragraph is illustrative; the measured versions do not exist in the literature. The obstruction is visible: without the benign control at matched length and a matched-training Transformer, the 59-point drop looks like an architectural vulnerability when 45 of those points belong to long context in general.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*