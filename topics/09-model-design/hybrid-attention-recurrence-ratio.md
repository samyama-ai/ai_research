---
id: 09-model-design/hybrid-attention-recurrence-ratio
title: "Optimal Attention-to-Recurrence Ratio in Hybrid Architectures"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Attention-to-Recurrence Ratio in Hybrid Architectures

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/hybrid-attention-recurrence-ratio` · **Status:** empirically-open

## 1. Problem Statement

Hybrid language models interleave two kinds of sequence mixer: softmax attention layers, whose state grows linearly with context length and which can address any past token exactly, and recurrent layers (Mamba, Mamba-2, RG-LRU, gated DeltaNet, lightning attention), whose state is a fixed-size matrix and whose cost per token is constant. Every shipped hybrid picks a ratio — Jamba 1:7, Griffin 1:2, Samba 1:1, Nemotron-H ~1:12 — and none of them justifies the number with a scaling law.

The problem: given a parameter budget $N$, a token budget $D$, a target context length $T$, and an inference-time memory ceiling $M$, predict the attention fraction $\rho$ that minimises loss, and predict how $\rho^\star$ moves as each of $N, D, T, M$ changes.

Three variants, of increasing difficulty:

- **Measurement.** Is there a stable, reproducible $\rho^\star$ at fixed $(N, D, T)$, or is the loss basin flat over $\rho \in [0.05, 0.5]$ so that the choice is decided entirely by inference economics?
- **Method.** A fitted law $\rho^\star(N, D, T, M)$ good enough to pick the ratio for a new model without a sweep — plus the *placement* rule (which layer indices get attention), which is a separate degree of freedom.
- **Theory.** A separation result: a task family and a proof that $\rho$ below some threshold costs $\Omega(\cdot)$ in parameters or samples, with a matching upper bound.

Solving it means: publish a law, fit on ≤1B models, and have it predict the held-out optimum at 8B within the noise of seed variance.

## 2. Formal Setting

A model is a layer schedule $\pi \in \{A, R\}^L$ over $L$ sequence-mixing layers ($A$ = softmax attention, $R$ = recurrent), with MLPs interleaved identically in all arms. Define

$$\rho = \frac{1}{L}\left|\{\, l : \pi_l = A \,\}\right|, \qquad \rho \in \{0, 1/L, \dots, 1\}.$$

$\rho$ is measured as a count of layers, not a FLOP share — the two diverge, and papers report both without saying which.

**Loss.** $\mathcal{L}(\pi; N, D, T)$ is next-token cross-entropy in nats on a held-out corpus, measured at the *same* tokenizer and the *same* evaluation context length $T$ for every arm. Downstream accuracy is reported separately; it is not a substitute.

**Compute.** Attention layer FLOPs per token scale as $O(d^2 + dT)$; recurrent (Mamba-2) as $O(d^2 + d \cdot d_s)$ with state dimension $d_s$ independent of $T$. Iso-FLOP means equal total training FLOPs $C \approx 6ND$ *including* the $dT$ term, so at long $T$ an iso-FLOP high-$\rho$ arm has fewer parameters. Most published comparisons are iso-parameter, not iso-FLOP. That is a real confound, not a nitpick.

**Inference memory.** For an attention layer with $n_{kv}$ KV heads of dim $d_h$ in bf16,

$$M_{\text{cache}}(\rho, T) = \rho L \cdot 4 \, n_{kv} d_h T \;+\; (1-\rho) L \cdot 2 d\, d_s ,$$

the first term linear in $T$, the second constant. This is the term $\rho$ is actually chosen to control. Sliding-window attention of width $w$ replaces $T$ with $\min(T, w)$ and changes the problem qualitatively.

**Objective.**
$$\rho^\star(N, D, T, M) = \arg\min_{\rho} \; \mathcal{L}(\pi_\rho; N, D, T) \quad \text{s.t.} \quad M_{\text{cache}}(\rho, T) \le M.$$

**Assumptions, and which fail.**
1. *Loss depends on $\pi$ only through $\rho$.* Known false — placement matters; Jamba puts no attention in the first block, Zamba shares one global attention block across depth.
2. *Recurrent layers are interchangeable.* False — Mamba-2, gated DeltaNet and RG-LRU differ in state capacity and in whether they support delta-rule overwriting.
3. *$\rho^\star$ is stable under post-training.* Unmeasured. Long-context extension, RLHF and quantization all interact with cache size.
4. *Optimum is interior.* Not guaranteed; $\rho^\star = 0$ is plausible at short $T$ for pure perplexity.

## 3. State of the Art

**Empirical SOTA (established, with ablations).**
- *Griffin* (De et al., DeepMind, 2024): a fixed 2-recurrent : 1-local-attention block, scaled to 14B, matching Llama-2 downstream on ~6× fewer training tokens. The 1:2 ratio is ablated against pure-recurrent Hawk, not against a ratio sweep.
- *Jamba* (Lieber et al., AI21, 2024): ablated 1:3 against 1:7 attention-to-Mamba at ~1.3B params / 250B tokens and found no meaningful loss gap; shipped 1:7 because it gives an ~8× smaller KV cache. This is the single most-cited data point in the field and it is one comparison, at one scale, of two ratios.
- *NVIDIA Mamba-2-Hybrid* (Waleffe et al., 2024): 8B params, 3.5T tokens, 4 attention + 24 Mamba-2 + 28 MLP layers. Beat a matched 8B Transformer by 2.65 points averaged over 12 standard tasks. The 4-attention choice came from smaller-scale search; the 8B run is a single point, not a sweep.

**Claimed but unablated.** Nemotron-H, Falcon-H1, MiniMax-01 (1 softmax per 7 lightning-attention layers), Qwen3-Next (~1:3 gated attention to gated DeltaNet) all report benchmark numbers at their chosen ratio with no same-scale alternative-ratio arm. These are existence proofs that a ratio works, not evidence it is optimal.

**Theory SOTA.** Jelassi et al. (ICML 2024) prove a fixed-state recurrent model cannot copy strings longer than its state, while a 2-layer transformer copies strings exponentially longer than its width — a hard separation, but it bounds $\rho > 0$, not $\rho^\star$. Arora et al. (*Zoology*, 2023; *Based*, ICML 2024) establish a recall/state-size trade-off on multi-query associative recall (MQAR). Merrill, Petty & Sabharwal (ICML 2024) show SSMs, like transformers, sit in $\mathsf{TC}^0$ and cannot track state — so the separation is about memory addressing, not expressivity class.

## 4. What Is Known

- **A little attention buys most of the gap.** At 8B / 3.5T tokens, ~7% attention layers recovers and exceeds full-attention quality on short-context tasks (Waleffe et al., 2024). At 0% attention, Mamba-2 8B loses badly on 5-shot MMLU and on phonebook-style retrieval.
- **The loss basin is flat in the middle.** Jamba's 1:3 vs 1:7 at 1.3B/250B: no significant perplexity difference. Griffin at 1:2 and Samba at 1:1 both work at 3–14B. Nothing distinguishes $\rho \in [0.08, 0.5]$ on perplexity at any published scale.
- **Cache savings are large and exactly computable.** Jamba reports ~4 GB KV cache at 256K context for a 52B-total/12B-active model, against tens of GB for a same-active-size transformer.
- **Retrieval is the discriminating axis.** Needle-in-a-haystack and MQAR separate ratios where perplexity does not; pure recurrent models fail MQAR above a state-size-determined key count (Arora et al., 2023, at 100M–1B scale).
- **Placement is not neutral.** Removing attention from the first block, and spacing attention evenly rather than clustering it, both appear in shipped designs; the effect is reported as design lore, not as a controlled ablation at scale.

## 5. What Is Not Known

- **Empirically open (the main gap).** No published iso-FLOP sweep of $\rho \in \{0, 1/16, 1/8, 1/4, 1/2, 1\}$ at ≥1B params and ≥100B tokens with ≥2 seeds and a fixed recurrent primitive. The experiment is entirely runnable — roughly $10^{22}$ FLOPs for the full grid — and nobody has published it. Every ratio in production traces to a 1–2 point comparison.
- **Empirically open.** Whether $\rho^\star$ decreases with $N$ (the "attention is a fixed overhead" hypothesis) or is scale-invariant. Both are consistent with existing data.
- **Methodologically blocked.** $\rho^\star$ as a function of $T$ cannot be measured, because held-out loss at 128K context is dominated by a small number of long documents whose repetition structure is not controlled, and needle benchmarks are saturated or synthetic. There is no agreed long-context loss metric that is comparable across architectures.
- **Theoretically open.** No lower bound of the form: any $\pi$ with $\rho < f(T)$ requires $\Omega(g(N))$ extra parameters to solve task family $\mathcal{F}$. The copying result gives $\rho > 0$; nothing gives a rate.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by cost**. Changing $\rho$ at fixed parameter count changes FLOPs per token, KV cache size, effective depth of exact-recall capability, and optimizer conditioning at once. To hold FLOPs fixed you must change $N$ or $d$, which changes the thing you are measuring. To hold parameters fixed you accept unequal compute, which favours whichever arm has more attention at long $T$.

Second, the signal is small where measurement is cheap and large where it is expensive: at 1B/100B tokens the perplexity spread across $\rho \in [1/16, 1/2]$ is on the order of 0.005–0.02 nats, comparable to seed variance, so the sweep needs multiple seeds per point. The differences that actually matter — long-context retrieval, in-context learning of new formats — only appear at 7B+ and 1T+ tokens, where a 6-point sweep costs a full pretraining budget. That is why the field has 1-point evidence and a folk ratio.

## 7. Current Research (as of 2026)

- **Systematic hybrid design studies.** NVIDIA (Nemotron-H line) and Zyphra (Zamba2) have published the most controlled comparisons; NVIDIA's 2025 work on hybrid design choices is the closest thing to a sweep *(frontier — verify the exact grid before citing it as one)*.
- **Linear-attention hybrids at frontier scale.** MiniMax-01, Qwen3-Next, Falcon-H1, IBM Bamba — all shipping in 2025, all reporting the ratio as a design constant *(frontier — verify)*.
- **Distillation as a cheap probe.** MOHAWK (Bick et al., 2024) and *The Mamba in the Llama* (Wang et al., NeurIPS 2024) convert a trained transformer into a hybrid by replacing a chosen subset of attention layers, which makes ratio sweeps cost distillation FLOPs rather than pretraining FLOPs. Whether the distilled $\rho^\star$ matches the pretrained $\rho^\star$ is untested and is itself a good experiment.
- **Learned or adaptive schedules.** Search over $\pi$ rather than $\rho$, and per-head parallel hybrids (Hymba, NVIDIA 2024). Early, no scaling evidence.

## 8. Concrete Next Experiment

**Scale.** Six arms at $N \approx 1.3$B non-embedding params, $D = 200$B tokens, $T = 8192$, identical data order, identical Mamba-2 recurrent primitive ($d_s = 128$), 2 seeds each. $\rho \in \{0, 1/16, 1/8, 1/4, 1/2, 1\}$, attention layers placed at evenly spaced indices. Total ≈ $1.9\times10^{22}$ FLOPs, about 3k H100-days.

**Control arm.** $\rho = 1$ (pure transformer, GQA, same $N$, same $D$) *and* $\rho = 0$ (pure Mamba-2). Both are needed: one bounds quality, the other bounds cache.

**Iso-FLOP correction.** Run each arm twice at $T = 8192$ and $T = 32768$ with $N$ adjusted so training FLOPs match to within 2%. This is what separates "attention helps" from "attention costs more compute".

**The deciding number.** The held-out cross-entropy gap
$$\Delta(\rho) = \mathcal{L}(\rho) - \mathcal{L}(1)$$
in nats, plotted against $\rho$, with seed-variance error bars. **Decision rule:** if $\Delta(1/8) < 0.01$ nats and the seed standard deviation is $< 0.005$, the perplexity basin is flat and $\rho$ should be set purely by $M_{\text{cache}}$ — the design question is closed for perplexity and moves entirely to retrieval. If $\Delta(1/8) > 0.03$ nats, there is a real quality gradient in $\rho$ and a scaling law is worth fitting.

**Second readout.** MQAR accuracy at 512 key-value pairs and needle-in-a-haystack at 32K, per arm. Expected: these separate arms that $\Delta$ does not — which would localise the whole problem to long-context measurement.

## 9. Key References

- **[Foundational]** Gu, A., Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[Foundational]** Dao, T., Gu, A. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML, 2024. — arXiv:2405.21060
- **[SOTA]** Lieber, O., Lenz, B., Bata, H., et al. *Jamba: A Hybrid Transformer-Mamba Language Model.* 2024. — arXiv:2403.19887
- **[SOTA]** De, S., Smith, S. L., Fernando, A., et al. *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.* 2024. — arXiv:2402.19427
- **[SOTA]** Waleffe, R., Byeon, W., Riach, D., et al. *An Empirical Study of Mamba-based Language Models.* 2024. — arXiv:2406.07887
- **[SOTA]** Ren, L., Liu, Y., Lu, Y., et al. *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.* ICLR, 2025. — arXiv:2406.07522
- **[Theory]** Jelassi, S., Brandfonbrener, D., Kakade, S., Malach, E. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Theory]** Merrill, W., Petty, J., Sabharwal, A. *The Illusion of State in State-Space Models.* ICML, 2024. — arXiv:2404.08819
- **[Theory/Empirical]** Arora, S., Eyuboglu, S., Timalsina, A., et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR, 2024. — arXiv:2312.04927
- **[Empirical]** Park, J., Park, J., Xiong, Z., et al. *Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks.* ICML, 2024. — arXiv:2402.04248
- **[Method]** Wang, J., Paliotta, D., May, A., Rush, A. M., Dao, T. *The Mamba in the Llama: Distilling and Accelerating Hybrid Models.* NeurIPS, 2024. — arXiv:2408.15237
- **[Survey]** Tiezzi, M., Casoni, M., Betti, A., et al. *Back to Recurrent Processing at the Crossroad of Transformers and State-Space Models.* Nature Machine Intelligence, 2025.

## 10. Worked Example

Take the NVIDIA 8B hybrid layout and price the ratio decision explicitly. Layers: 4 attention, 24 Mamba-2, 28 MLP. Model dim $d = 4096$, GQA with $n_{kv} = 8$, $d_h = 128$, Mamba-2 state $d_s = 128$.

Per token, per attention layer, bf16 KV cache: $2 \times 8 \times 128 \times 2\ \text{bytes} = 4096$ B. Across 4 layers: **16 KB/token**. At $T = 128$K: $16\,\text{KB} \times 131072 \approx 2.1$ GB.

A same-shape pure transformer with 28 attention layers: $28 \times 4096 = 114.7$ KB/token, $\approx 15.0$ GB at 128K. Ratio 7.0×. Recurrent state is $24 \times 2 \times 4096 \times 128 \times 2\,\text{B} \approx 50$ MB, constant in $T$ — negligible either way.

Now the obstruction. On an 80 GB H100 with ~16 GB of weights (8B in bf16), the hybrid fits $\lfloor 64/2.1 \rfloor = 30$ concurrent 128K sequences; the transformer fits 4. That is a 7.5× throughput difference — an unambiguous, arithmetic result.

Against it, the quality evidence: +2.65 points averaged over 12 tasks for the hybrid over the matched transformer, from **one run at each ratio, one seed**. There is no measurement of what $\rho = 2/28$ or $\rho = 8/28$ would have given at the same scale. So the decision "4 attention layers" rests on an exactly-computed 7.5× on one side and a single unreplicated point estimate on the other.

Push further: suppose the true $\Delta(\rho)$ curve rises by 0.02 nats as $\rho$ falls from $8/28$ to $4/28$. At 8B scale, 0.02 nats is roughly the loss gained from ~15% more training tokens — several hundred billion tokens of compute, silently spent to save cache. Nobody knows whether that 0.02 is there, because the measurement was never run with the seeds needed to resolve it. That, and not any theoretical difficulty, is why the ratio is still folklore.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*