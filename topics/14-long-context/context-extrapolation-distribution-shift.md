---
id: 14-long-context/context-extrapolation-distribution-shift
title: "Context Length Extrapolation Under Distribution Shift"
topic: 14-long-context
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Context Length Extrapolation Under Distribution Shift

> **Topic:** Long Context · **ID:** `14-long-context/context-extrapolation-distribution-shift` · **Status:** open

## 1. Problem Statement

A model trained on sequences of length $\le L_{\text{train}}$ is deployed on sequences of length $L_{\text{test}} \gg L_{\text{train}}$. Extrapolation is usually reported as perplexity staying flat, or as a synthetic retrieval probe staying near 100%. The open problem is that both metrics are measured *in-distribution*: the long test documents come from the same corpus family as training, and the probe task is one the model has seen the template of.

**The question.** Does length extrapolation survive a simultaneous shift in the *content* distribution — domain, language, code/prose mix, adversarial filler, task format — or is what we call extrapolation actually memorized interaction between position encoding and familiar token statistics?

Three variants, different difficulty:

- **Measurement.** Define a score that separates *positional* failure (the model cannot use index $i > L_{\text{train}}$) from *content* failure (the model is bad at this domain at any length). Requires a length-matched, domain-matched control. Currently ill-posed in most published evaluations.
- **Method.** Build a position scheme + training recipe whose degradation at $4\times L_{\text{train}}$ is bounded and *invariant to the test domain*. Existing recipes (Position Interpolation, YaRN, LongRoPE) are tuned on a calibration set; their transfer is largely unablated.
- **Theory.** Prove a generalization bound over positions: for which attention parameterizations does behavior at index $i$ extend to $i' > L_{\text{train}}$, and under what condition on the input distribution? No such bound exists for a real transformer.

Solving it means: a recipe with a stated guarantee, plus a measurement that a skeptical reviewer accepts as separating the two failure modes.

## 2. Formal Setting

Let $x_{1:n}$ be a token sequence drawn from a distribution $\mathcal{D}$ over $\mathcal{V}^*$. Training samples come from $\mathcal{D}_{\text{tr}}$ truncated at $L_{\text{train}}$; evaluation from $\mathcal{D}_{\text{te}}$ at length $L_{\text{test}}$. Model $f_\theta$ scores next tokens; per-position loss

$$\ell_\theta(i \mid \mathcal{D}, L) = \mathbb{E}_{x \sim \mathcal{D}, |x|=L}\big[-\log p_\theta(x_i \mid x_{<i})\big].$$

**Measured as:** average NLL bucketed by absolute index $i$, over $\ge 200$ documents of *natural* length $\ge L$ (not concatenated shards — concatenation destroys long-range dependency and inflates the metric).

Two shift axes, deliberately crossed:
- length shift $\lambda = L_{\text{test}} / L_{\text{train}}$;
- content shift $\delta = D(\mathcal{D}_{\text{te}} \,\|\, \mathcal{D}_{\text{tr}})$, measured operationally as the NLL gap of a *short-context* reference model $g$ on the same text: $\hat\delta = \ell_g(\mathcal{D}_{\text{te}}) - \ell_g(\mathcal{D}_{\text{tr}})$, in nats/token at $L = L_{\text{train}}$.

The quantity of interest is the **interaction**, not either main effect:

$$\Delta_{\text{int}} = \big[\ell_\theta(\mathcal{D}_{\text{te}}, \lambda L_{\text{train}}) - \ell_\theta(\mathcal{D}_{\text{te}}, L_{\text{train}})\big] - \big[\ell_\theta(\mathcal{D}_{\text{tr}}, \lambda L_{\text{train}}) - \ell_\theta(\mathcal{D}_{\text{tr}}, L_{\text{train}})\big].$$

$\Delta_{\text{int}} = 0$ means extrapolation is domain-independent. $\Delta_{\text{int}} > 0$ means the extrapolation recipe is calibrated to the training domain.

For RoPE (Su et al., 2021) with head dimension $d$, position $i$ enters only through rotation angles $\theta_k = b^{-2k/d}$, $b = 10^4$. Interpolation methods rescale: $i \mapsto i/s$ (PI, Chen et al., 2023) or per-frequency $\theta_k \mapsto \theta_k / s_k$ (NTK/YaRN).

**Assumptions, and which are violated:**
1. *Test documents are naturally long.* Violated: most long-context perplexity numbers use concatenated PG-19 or Books3 shards.
2. *Perplexity tracks task ability.* Violated — measured: models with flat long-context perplexity fail RULER retrieval at the same length (Hsieh et al., 2024).
3. *Position indices are i.i.d.-exchangeable with content.* Violated: document type correlates with length (code files, legal filings, novels), so $\lambda$ and $\delta$ are confounded in any natural corpus.
4. *Attention entropy is scale-free.* Violated: logit sum over $n$ keys grows with $n$, so attention flattens with length independent of any distribution shift (Chiang & Cholak, ACL 2022, on attention-hardness/scaling).

## 3. State of the Art

**Empirical/systems SOTA.**
- Position Interpolation (Chen et al., 2023) and YaRN (Peng et al., ICLR 2024) extend RoPE models to $8$–$32\times$ with $\sim$0.1–1k fine-tuning steps. *Established:* perplexity recovery on the calibration corpus. *Claimed but unablated:* that the chosen scale factor transfers across domains — YaRN's temperature is fit on a validation set and reused.
- LongRoPE (Ding et al., ICML 2024) searches per-dimension rescalings evolutionarily to 2M tokens. Reported as a benchmark number on passkey and book perplexity; no cross-domain ablation of the searched schedule.
- Data-side recipes: Fu et al. (ICML 2024) show 80k-context ability arises from ~5B tokens of *length-upsampled, domain-balanced* continued pretraining, and that domain balance matters as much as length — the closest existing evidence that $\Delta_{\text{int}} \ne 0$.
- Meta's long-context Llama scaling (Xiong et al., NAACL 2024): base-frequency increase plus continued pretraining to 32k.

**Theory SOTA.** Much weaker. Sanford, Hsu & Telgarsky (NeurIPS 2023) give separations for what a bounded-size transformer can represent as a function of input length (sparse averaging requires width growing with $n$) — a *capacity* statement, not a generalization-over-positions statement. Kazemnejad et al. (NeurIPS 2023) show empirically that no explicit positional encoding (NoPE) extrapolates better than RoPE/ALiBi on small reasoning tasks and prove NoPE can express absolute and relative schemes. No bound relates $\Delta_{\text{int}}$ to any property of $\mathcal{D}$.

**Benchmarks.** RULER (Hsieh et al., COLM 2024), $\infty$Bench (Zhang et al., ACL 2024), HELMET (Yen et al., ICLR 2025), LongBench (Bai et al., ACL 2024). All vary length; none holds content distribution fixed while varying it, and none reports the interaction term.

## 4. What Is Known

- **Claimed context $\gg$ effective context.** RULER (2024): of 10 models claiming $\ge$32k, most hold above their 4k-baseline accuracy only to 4k–16k; several drop $>30$ points at their nominal length. Measured at 7B–70B.
- **Perplexity and retrieval dissociate.** Models with monotone-decreasing per-token NLL to 128k still miss multi-key needle retrieval at 32k (Hsieh et al., 2024). So NLL is not a sufficient extrapolation metric.
- **Position order matters more than length.** "Lost in the Middle" (Liu et al., TACL 2024): answer-in-middle accuracy falls up to ~20 points below answer-at-either-end for 20-document QA, at 7B–175B scale, *at fixed length* — a purely positional effect with content held constant.
- **Extrapolation is fragile to format.** Zhou et al. (2024) get length generalization on addition to $2.5\times$ training length with FIRE + reversed-digit + index hints, but variance across random seeds and data order is large enough to flip the conclusion — extrapolation is not a stable property of the checkpoint. Scale: small decoder-only transformers, 25M–150M.
- **Scratchpads help, then break.** Anil et al. (NeurIPS 2022): even with scratchpads, in-context and fine-tuned parity/variable-assignment tasks degrade sharply beyond training length at up to 64B parameters.
- **Naive RoPE extrapolation collapses.** Beyond $L_{\text{train}}$, perplexity rises by orders of magnitude within a few hundred tokens (Chen et al., 2023; Press et al., ICLR 2022 for the ALiBi comparison).

## 5. What Is Not Known

- **Theoretically open.** No generalization bound over position indices. Nothing tells you when $f_\theta$'s behavior at $i \le L_{\text{train}}$ constrains behavior at $i > L_{\text{train}}$, and no theory predicts the sign or size of $\Delta_{\text{int}}$.
- **Empirically open.** The crossed $\lambda \times \delta$ experiment is runnable today with $\le$ 2k GPU-hours and has not been published. Whether YaRN/LongRoPE scale factors fit on English books transfer to code, non-Latin scripts, or multi-turn dialogue is unmeasured.
- **Methodologically blocked.** Separating positional from content failure needs a length-matched control from the *same* distribution. Natural corpora do not supply one: long legal documents differ from short ones in content, not only length. Synthetic padding creates a distribution no model was trained on. There is currently no accepted construction of the control arm — this is the binding obstruction on the measurement variant.

## 6. Why It Is Hard

**The primary obstruction is confounded measurement, and it is structural.** Length is not an independently manipulable variable in natural text. To vary $L$ at fixed $\delta$ you must either (a) truncate long documents — which changes the dependency structure, not just the length; (b) concatenate short ones — which removes long-range dependency, making extrapolation look easy; or (c) pad synthetically — which introduces a third, out-of-distribution content shift. Every option perturbs the thing being controlled.

Secondary: attention-entropy dilution means *any* model degrades with $n$ from logit-scale effects alone, so a nonzero degradation is not evidence of a positional-generalization failure. And there is no ground truth for "the model should have used token $i$" — attribution over 128k positions has no reference answer, so failures cannot be localized to a position.

Compute is real but not the binding constraint: the decisive experiment fits well inside an academic budget.

## 7. Current Research (as of 2026)

- **Length-controlled benchmark design.** HELMET (Princeton, ICLR 2025) and RULER (NVIDIA) push toward controlled length sweeps; neither yet crosses length with domain. *(frontier — verify)* Several groups are reportedly building domain-crossed long-context suites.
- **Positional schemes with explicit inductive bias.** FIRE (Li et al., ICLR 2024), NoPE follow-ups, per-dimension frequency search (LongRoPE).
- **Recurrent / state-space hybrids** (Mamba-2, Griffin, Jamba) shift the question: they have no position index, so their failure mode is state capacity, not extrapolation. Whether $\Delta_{\text{int}}$ is smaller for them is open. *(frontier — verify)*
- **Mechanistic work** on retrieval heads and induction-head reuse at long range, asking whether the same heads fire out-of-domain. *(frontier — verify)*

## 8. Concrete Next Experiment

**The crossed $\lambda \times \delta$ ablation.**

- **Scale.** One 7B base model, $L_{\text{train}} = 8$k, RoPE $b=10^4$. Apply YaRN with scale $s=8$ (target 64k), fine-tuned for 1B tokens on English books only. ~500–1500 A100-hours total including evaluation.
- **Grid.** Lengths $L \in \{8\text{k}, 16\text{k}, 32\text{k}, 64\text{k}\}$ $\times$ domains $\{$English books (in-domain), Python repos, Chinese web text, arXiv LaTeX, multi-turn dialogue$\}$. 300 *naturally* long documents per cell; no concatenation. Report per-index NLL and RULER-style multi-key retrieval per cell.
- **Control arm.** The same base model *without* YaRN, evaluated with a sliding 8k window on the identical documents. This isolates positional extrapolation from raw domain competence: any domain the windowed control also fails is a content failure, not an extrapolation failure.
- **Deciding number.** $\Delta_{\text{int}}$ in nats/token at $\lambda=8$, averaged over the four out-of-domain cells. **If $|\Delta_{\text{int}}| < 0.02$, extrapolation recipes transfer and the problem is mostly closed on the method side. If $\Delta_{\text{int}} > 0.1$, the recipe is a domain-specific calibration and every published $\lambda$ claim needs a domain caveat.** Report per-cell retrieval accuracy alongside; a $\ge 15$-point out-of-domain retrieval gap at fixed $\Delta_{\text{int}} \approx 0$ would show NLL misses the effect entirely.

## 9. Key References

- **[Foundational]** Su, Lu, Pan, Murtadha, Wen, Liu. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing, 2024 (arXiv 2021). — arXiv:2104.09864
- **[Foundational]** Press, Smith, Lewis. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.* ICLR 2022. — arXiv:2108.12409
- **[Foundational]** Anil, Wu, Andreassen, Lewkowycz, Misra, Ramasesh, Slone, Gur-Ari, Dyer, Neyshabur. *Exploring Length Generalization in Large Language Models.* NeurIPS 2022. — arXiv:2207.04901
- **[SOTA]** Chen, Wong, Chen, Tian. *Extending Context Window of Large Language Models via Positional Interpolation.* 2023. — arXiv:2306.15595
- **[SOTA]** Peng, Quesnelle, Fan, Shippole. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR 2024. — arXiv:2309.00071
- **[SOTA]** Ding, Zhang, Zhang, Xia, Dong, Zhu, Zhu, Chen, Yang. *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens.* ICML 2024. — arXiv:2402.13753
- **[SOTA]** Fu, Panda, Niu, Ma, Zhang, Wang, Wang, Chen. *Data Engineering for Scaling Language Models to 128K Context.* ICML 2024. — arXiv:2402.10171
- **[Analysis]** Kazemnejad, Padhi, Natesan Ramamurthy, Das, Reddy. *The Impact of Positional Encoding on Length Generalization in Transformers.* NeurIPS 2023. — arXiv:2305.19466
- **[Analysis]** Zhou, Alon, Chen, Wang, Agarwal, Zhou. *Transformers Can Achieve Length Generalization But Not Robustly.* 2024. — arXiv:2402.09371
- **[Analysis]** Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024. — arXiv:2307.03172
- **[Benchmark]** Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024. — arXiv:2404.06654
- **[Benchmark]** Yen, Gao, Hou, Huang, Xiong, Sun, Chen. *HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly.* ICLR 2025. — arXiv:2410.02694
- **[Theory]** Sanford, Hsu, Telgarsky. *Representational Strengths and Limitations of Transformers.* NeurIPS 2023. — arXiv:2306.02896
- **[Survey]** Li, Zhang, Do, Yue, Chen. *Long-context LLMs Struggle with Long In-context Learning.* TMLR 2024. — arXiv:2404.02060

## 10. Worked Example

Take a 7B RoPE model, $L_{\text{train}} = 8$k, extended by PI with $s = 4$ to 32k, fine-tuned on English books.

Reported result (typical of the PI-style literature): mean NLL on 32k book documents is 2.11 nats/token, versus 2.08 at 8k. Degradation $= 0.03$ nats. Conclusion drawn: extrapolation works.

Now evaluate on 32k-token Python repositories. Mean NLL $= 1.24$ at 32k versus 1.09 at 8k: degradation $= 0.15$ nats. The raw number is *lower* than books at both lengths — code is more predictable — so a naive reading says code long-context is fine. But the interaction is

$$\Delta_{\text{int}} = 0.15 - 0.03 = 0.12 \ \text{nats/token},$$

four times the in-domain degradation. The extrapolation recipe is five times worse on code, and the absolute NLL hides it completely because $\hat\delta \approx -1.0$ nats swamps it.

**Where the obstruction bites.** To claim $0.12$ is positional, you need a control showing the model is not just worse at *long* Python for content reasons. The sliding-8k-window arm gives you that: if the windowed control scores 1.10 nats on the same 32k repos, the extended model is $0.14$ nats *worse than a model that cannot see past 8k at all* — unambiguously a positional failure. If instead the windowed control scores 1.26, the extension gained nothing but lost nothing, and the degradation was content all along.

Without the control arm, both stories fit the same two numbers. That is the measurement problem, not a compute problem — and it is why the published $\lambda$ claims cannot currently be checked.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*