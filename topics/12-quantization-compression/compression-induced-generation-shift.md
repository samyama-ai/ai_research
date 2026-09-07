---
id: 12-quantization-compression/compression-induced-generation-shift
title: "Compression-Induced Distribution Shift in Generation"
topic: 12-quantization-compression
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compression-Induced Distribution Shift in Generation

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/compression-induced-generation-shift` · **Status:** methodologically-blocked

## 1. Problem Statement

Compressing a language model — post-training quantization, pruning, distillation, low-rank factorization — changes the conditional next-token distribution. The standard acceptance test is an aggregate scalar: perplexity on WikiText-2/C4, or accuracy on a benchmark suite. The problem is that these scalars can be preserved while the *generative distribution* moves substantially.

- **Input:** a base model $p_\theta$, a compression operator $C$ producing $p_{\hat\theta} = p_{C(\theta)}$, a prompt distribution $\mathcal{P}$, and a decoding rule $D$.
- **Output:** a certificate that the induced text distributions $D \circ p_\theta$ and $D \circ p_{\hat\theta}$ are close under a divergence that predicts downstream behavioral difference.
- **Decision predicate:** given tolerance $\epsilon$, decide whether $\hat\theta$ is behaviorally substitutable for $\theta$ on $\mathcal{P}$.

Three variants, of very different difficulty:

- **Measurement variant (the blocked one).** Define a divergence over *generations* that is (a) estimable from a feasible number of samples, (b) monotone in real downstream harm, and (c) not dominated by benign paraphrase. No such measure is agreed on.
- **Method variant (partly solved).** Given any fixed divergence, minimize it under a bit budget. This is the GPTQ/AWQ/AQLM line and it works.
- **Theory variant (open).** Bound sequence-level divergence at horizon $T$ from per-token weight or logit perturbation, without vacuous exponential-in-$T$ blowup.

Solving it means: a metric $M$ with a published threshold such that $M < \tau$ empirically implies bounded degradation on held-out agentic, multilingual, safety and long-horizon tasks — validated on tasks that were not used to fit $\tau$.

## 2. Formal Setting

Let $V$ be the vocabulary, $x_{<t} \in V^{*}$ a context, and $p_\theta(\cdot \mid x_{<t}) \in \Delta(V)$ the next-token distribution. Compression gives $\hat\theta = C(\theta)$; for uniform $b$-bit quantization with per-group scale $s$ and zero point $z$, $\hat{W} = s\left(\mathrm{clip}\left(\mathrm{round}(W/s) + z, 0, 2^b - 1\right) - z\right)$.

**Token-level shift.** Measured by sampling $N$ contexts from a corpus and averaging the exact per-token KL (both distributions are available in closed form; no sampling over $V$ needed):

$$\Delta_{\mathrm{tok}} = \mathbb{E}_{x_{<t} \sim \mathcal{D}}\left[ \mathrm{KL}\!\left(p_\theta(\cdot \mid x_{<t}) \,\|\, p_{\hat\theta}(\cdot \mid x_{<t})\right) \right].$$

**Flip rate.** For a task with greedy answers, $\mathrm{Flips} = \Pr[\arg\max p_\theta \neq \arg\max p_{\hat\theta}]$ over items. This is the quantity accuracy hides: flips in both directions cancel in the aggregate.

**Sequence-level shift.** With decoding rule $D$ (temperature, top-$p$, beam) and horizon $T$,

$$\Delta_{\mathrm{seq}}(T) = \mathrm{TV}\!\left(D\!\circ\! p_\theta(\cdot \mid x_0)^{\otimes T},\; D\!\circ\! p_{\hat\theta}(\cdot \mid x_0)^{\otimes T}\right).$$

Under teacher forcing the per-token KLs chain: $\mathrm{KL}$ over full sequences is the sum of conditional KLs. Under free running they do not — the models condition on their own divergent prefixes. The naive bound $\Delta_{\mathrm{seq}}(T) \le \sqrt{\tfrac{1}{2}\sum_{t\le T}\mathrm{KL}_t}$ is vacuous at $T \sim 10^3$ for any $\Delta_{\mathrm{tok}}$ measurable in practice.

**Assumptions, and which fail.**

1. *Calibration data is representative of deployment* — violated: calibration is typically 128 sequences of C4 or WikiText; deployment is chat, code, tool calls, non-English.
2. *Teacher-forced divergence predicts free-running divergence* — violated by construction; exposure bias compounds.
3. *TV/KL over surface tokens tracks semantic difference* — violated: a paraphrase has TV near 1 and semantic distance near 0.
4. *A single scalar suffices* — violated: degradation is heterogeneous across languages, task types and rare facts.
5. *$\Delta_{\mathrm{tok}}$ is stationary in context length* — untested at $T > 32$k.

## 3. State of the Art

**Method SOTA (established).** GPTQ (Frantar et al., ICLR 2023) — second-order layerwise rounding, 3–4 bit, OPT-175B in ~4 GPU-hours. AWQ (Lin et al., MLSys 2024) — activation-aware per-channel scaling. QuIP (Chee et al., NeurIPS 2023) and QuIP\# (Tseng et al., ICML 2024) — incoherence processing plus lattice codebooks, usable 2-bit. AQLM (Egiazarian et al., ICML 2024) — additive quantization, Pareto-dominant below 3 bits. SparseGPT (Frantar & Alistarh, ICML 2023) — 50–60% one-shot sparsity at 175B. All of these optimize a *layerwise proxy* ($\|WX - \hat{W}X\|_2^2$), not any generation-level divergence.

**Measurement SOTA (weak).** "Accuracy is Not All You Need" (Dutta et al., 2024) argues for flip rate and per-token KL over aggregate accuracy. MAUVE (Pillutla et al., NeurIPS 2021) gives a divergence between generated and reference text in embedding space, but was designed for human-vs-model comparison and has known sensitivity to the embedding model and quantization of the histogram. Neither has a validated threshold.

**Claimed but unablated.** "Lossless at 4-bit" claims are near-universal in quantization papers and are supported almost entirely by (i) WikiText-2 perplexity to two decimals and (ii) a fixed zero-shot suite (ARC, HellaSwag, PIQA, WinoGrande, LAMBADA). These are multiple-choice-likelihood tasks, not generation. Long-form generation, tool-use trajectories and non-English output are typically not measured at all. Where they are measured, the picture changes: Jaiswal et al. (ICLR 2024) found compressed models that look intact on perplexity degrade sharply on knowledge-intensive and instruction-following tasks.

**Benchmark-number-only results.** Every reported "≤0.1 perplexity increase" figure is a single-corpus number at one sequence length with one calibration set. Reproductions across calibration seeds are rarely published.

## 4. What Is Known

- **4-bit is Pareto-optimal for accuracy-per-bit.** Dettmers & Zettlemoyer (ICML 2023), >35,000 zero-shot runs over 19M–176B parameters. The finding is about *aggregate zero-shot accuracy*, and does not extend to generation quality.
- **Outlier features emerge with scale.** Dettmers et al. (LLM.int8(), NeurIPS 2022): systematic outlier dimensions appear around 6.7B parameters and, if not handled, collapse int8 inference. SmoothQuant (Xiao et al., ICML 2023) migrates the difficulty from activations to weights.
- **Accuracy parity hides item-level divergence.** Dutta et al. (2024) report cases where a quantized model matches the baseline's benchmark accuracy to within about a point while flipping on the order of 10% of individual items. Measured on 7B–70B open models over standard suites.
- **Degradation is unevenly distributed across languages.** Marchisio et al. (EMNLP Findings 2024) found automatic metrics understate damage relative to human judgment, with non-Latin-script languages hit hardest, at 8B–35B scale.
- **Compression interacts with safety and long-context behavior** in ways not captured by perplexity — reported repeatedly, but with heterogeneous protocols; no independent multi-lab reproduction of a specific effect size.
- **Scaling laws for precision exist.** Kumar et al. (2024) fit degradation as a function of training and inference precision, with the prediction that more training data makes a model *more* sensitive to post-training quantization. Fitted on models up to ~1.7B; unverified at frontier scale.

## 5. What Is Not Known

- **Methodologically blocked (the core).** There is no accepted generation-level divergence with a calibrated threshold. Candidates fail differently: token TV is dominated by benign paraphrase; embedding-space divergences inherit the encoder's blind spots; LLM-judge preference is not a metric (asymmetric, not a distance, drifts with judge version).
- **Theoretically open.** No non-vacuous bound from per-token KL to free-running sequence divergence at $T\sim10^3$ under realistic decoding. The exponential-in-$T$ growth may be an artifact of worst-case analysis; contraction under top-$p$ truncation is plausible and unproven.
- **Empirically open.** Whether $\Delta_{\mathrm{tok}}$ measured on 128 C4 sequences predicts agentic task success at 70B+. Runnable today; nobody has published it with adequate seeds.
- **Empirically open.** Whether compression damage concentrates on rare facts (tail memorization) or spreads uniformly. Requires a per-fact ground-truth probe set.

## 6. Why It Is Hard

**Absent ground truth for "same distribution."** The object we want to preserve is the model's *behavior*, and there is no reference measurement of behavior independent of the benchmark being used to define it. This makes the failure mode circular: each new metric is validated against a benchmark suite that the previous metric already failed to predict.

**Confounded measurement.** Perplexity is an expectation over the corpus; the harm is in the tail. A compression that leaves 99.9% of tokens untouched and destroys a rare capability moves perplexity by less than run-to-run calibration noise.

**Non-identifiability under decoding.** Two models with different conditionals can induce nearly identical top-$p$ generations, and two models with near-identical conditionals can diverge irreversibly after one sampled token. So neither direction of implication between $\Delta_{\mathrm{tok}}$ and $\Delta_{\mathrm{seq}}$ holds.

**Compute.** A sequence-level estimate needs $O(10^3)$ generations $\times$ $O(10^2)$ prompts $\times$ several calibration seeds $\times$ both arms — hundreds of GPU-hours per compression configuration, versus minutes for perplexity. The cheap metric wins by default.

## 7. Current Research (as of 2026)

- **Divergence-aware compression objectives.** Replacing the layerwise $\ell_2$ proxy with a KL-to-teacher objective over model-generated (self-distilled) data. Pursued in the QuIP\#/AQLM lineage (Cornell, IST Austria, Yandex) and in industrial post-training pipelines. *(frontier — verify)*
- **Flip-rate and KL-based acceptance gates** replacing perplexity in release checklists at several labs. *(frontier — verify)*
- **Precision-aware scaling laws** (Harvard/MIT/Databricks line, Kumar et al.) extended to post-training quantization sensitivity as a function of tokens-per-parameter.
- **Rotation/incoherence methods** (QuaRot, SpinQuant) as a way to make the weight distribution Gaussian-like before rounding, reducing outlier damage.
- **Multilingual and safety-specific compression audits** (Cohere Labs and academic groups), pushing toward per-subgroup rather than aggregate reporting.

## 8. Concrete Next Experiment

**Question:** does teacher-forced token KL on standard calibration data predict free-running behavioral divergence?

- **Scale.** One open 70B instruction-tuned model. Compression arms: GPTQ-4bit, AWQ-4bit, AQLM-2bit, SparseGPT-50%, each built with 3 independent calibration seeds (128 sequences of C4) — 12 compressed models.
- **Measurement A (cheap).** $\Delta_{\mathrm{tok}}$ against the FP16 base, exact per-token KL over 200k held-out tokens.
- **Measurement B (expensive).** Free-running divergence on 500 agentic/long-form prompts, 16 samples each at temperature 0.7, scored by: (i) task success against programmatic checkers, (ii) pairwise flip rate on 5k greedy short-answer items.
- **Control arm.** FP16 base against *itself* under a different random seed and a different kernel/batch-size configuration — this fixes the noise floor for both measurements. Any effect smaller than the control arm's spread is not an effect.
- **Deciding number.** The Spearman rank correlation $\rho$ between $\Delta_{\mathrm{tok}}$ and agentic success drop across the 12 models. $\rho > 0.8$: token KL is an adequate gate and the problem downgrades to *method*. $\rho < 0.4$: the standard acceptance test is not measuring what it names, and the field needs a sequence-level metric before any further "lossless" claim is meaningful.

Estimated cost: ~1,500 A100-hours. This is small relative to a single quantization paper's sweep.

## 9. Key References

- **[Foundational]** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[Foundational]** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[SOTA]** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Song Han. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[SOTA]** Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML, 2024.
- **[SOTA]** Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh. *Extreme Compression of Large Language Models via Additive Quantization.* ICML, 2024.
- **[Measurement]** Abhinav Dutta, Sanjeev Krishnan, Nipun Kwatra, Ramachandran Ramjee. *Accuracy is Not All You Need.* NeurIPS, 2024.
- **[Measurement]** Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, Zaid Harchaoui. *MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers.* NeurIPS, 2021.
- **[Empirical]** Ajay Jaiswal, Zhe Gan, Xianzhi Du, Bowen Zhang, Zhangyang Wang, Yinfei Yang. *Compressing LLMs: The Truth is Rarely Pure and Never Simple.* ICLR, 2024.
- **[Empirical]** Kelly Marchisio, Saurabh Dash, Hongyu Chen, Dennis Aumiller, Ahmet Üstün, Sara Hooker, Sebastian Ruder. *How Does Quantization Affect Multilingual LLMs?* Findings of EMNLP, 2024.
- **[Scaling]** Tim Dettmers, Luke Zettlemoyer. *The Case for 4-bit Precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[Scaling]** Tanishq Kumar, Zachary Ankner, Benjamin F. Spector, Blake Bordelon, Niklas Muennighoff, Mansheej Paul, Cengiz Pehlevan, Christopher Ré, Aditi Raghunathan. *Scaling Laws for Precision.* 2024. — arXiv:2411.04330
- **[Pruning]** Elias Frantar, Dan Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* ICML, 2023. — arXiv:2301.00774
- **[Decoding]** Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi. *The Curious Case of Neural Text Degeneration.* ICLR, 2020. — arXiv:1904.09751

## 10. Worked Example

Take a 4-bit GPTQ model whose WikiText-2 perplexity is 5.68 against the FP16 baseline's 5.62 — a 1.1% increase, well inside the range every paper calls "lossless."

Convert that to a per-token divergence. Perplexity is $\exp(H)$, so the cross-entropy gap is

$$\Delta H = \ln 5.68 - \ln 5.62 = 1.7370 - 1.7263 = 0.0107\ \text{nats/token}.$$

Under teacher forcing, sequence KL is additive, so over a 1,000-token generation the accumulated divergence is $\approx 10.7$ nats. Pinsker gives

$$\mathrm{TV} \le \sqrt{\tfrac{1}{2}(10.7)} = 2.31,$$

which exceeds 1 and is therefore vacuous. The bound says nothing after roughly 90 tokens ($\sqrt{0.0107 \cdot T / 2} \ge 1$ at $T \approx 187$; it is already uninformative in practice well before that). **The metric that certified "lossless" cannot certify a single paragraph.**

Now the other direction. Suppose the divergence is concentrated: 99% of tokens carry $\mathrm{KL} \approx 0.002$ and 1% carry $\mathrm{KL} \approx 0.9$. The average is $0.99(0.002) + 0.01(0.9) = 0.0109$ — the same 0.0107 to within rounding. But in the second regime, a 1,000-token generation contains about 10 tokens where the model is nearly making a different choice, and one such token early in a chain-of-thought reroutes the entire trajectory. Perplexity cannot distinguish the two regimes; they are the same number.

The obstruction is visible here in full: the aggregate scalar is (a) an average that erases the tail, and (b) formally unable to bound the generative object anyone cares about. Fixing it requires a new measurement, not a better rounding algorithm — which is why this entry is filed *methodologically blocked* rather than *empirically open*.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*