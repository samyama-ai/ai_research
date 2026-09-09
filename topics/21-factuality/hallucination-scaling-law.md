---
id: 21-factuality/hallucination-scaling-law
title: "Hallucination Rate Scaling Law with Model and Data Size"
topic: 21-factuality
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hallucination Rate Scaling Law with Model and Data Size

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/hallucination-scaling-law` · **Status:** empirically-open

## 1. Problem Statement

Cross-entropy loss follows a clean power law in parameters $N$, tokens $D$, and compute $C$ (Kaplan et al. 2020; Hoffmann et al. 2022). Hallucination — asserting a specific claim that is false — has no such established law. The problem: **does hallucination rate admit a scaling law of the same form, and what are its exponents and its irreducible floor?**

Three variants, of very different difficulty:

- **Measurement variant.** Define a hallucination rate $H$ that is (a) comparable across model scales, (b) not dominated by refusal-policy differences, and (c) stable under paraphrase of the prompt. Currently the hardest of the three.
- **Empirical variant.** Fit $H(N, D)$ over a controlled scaling ladder where the pretraining corpus is *the same corpus*, subsampled, and the evaluation is held fixed. Runnable today; not run at the right scale.
- **Theory variant.** Prove that $H$ is bounded below by a corpus statistic (e.g. the fraction of facts appearing once) irrespective of $N$, so that no amount of parameters removes it. Partially done for a restricted setting.

A solution to the empirical variant is a fitted form with reported confidence intervals on the exponents and the floor, validated by held-out extrapolation across at least one order of magnitude in $N$.

## 2. Formal Setting

Let $p_\theta$ be an autoregressive model with $N$ non-embedding parameters trained on $D$ tokens from corpus $\mathcal{C}$. Let $\mathcal{Q}$ be a query distribution over prompts $q$ with a *verifier* $V(q, a) \in \{\text{correct}, \text{incorrect}, \text{abstain}\}$.

**Measured quantities.** Sample $q \sim \mathcal{Q}$, decode $a \sim p_\theta(\cdot \mid q)$ at temperature $T$, adjudicate with $V$. Then

$$H(N,D) = \Pr_{q,a}\!\left[V(q,a) = \text{incorrect}\right], \qquad A = \Pr[\text{correct}], \qquad R = \Pr[\text{abstain}], \qquad H + A + R = 1.$$

$H$ alone is gameable: a model that always abstains has $H = 0$. The scale-comparable quantity is the **conditional error rate on attempted answers**,

$$\tilde{H} = \frac{H}{H + A},$$

which is what actually needs a scaling law. For long-form generation, replace the ternary verdict with atomic-claim decomposition: generation $a$ yields claims $c_1,\dots,c_m$, and $\tilde{H} = \frac{1}{m}\sum_i \mathbb{1}[c_i \text{ unsupported}]$ — this is FActScore (Min et al. 2023) with the sign flipped.

**Hypothesised form.**

$$\tilde{H}(N,D) = \tilde{H}_\infty + \frac{a}{N^{\alpha}} + \frac{b}{D^{\beta}}$$

with $\tilde{H}_\infty > 0$ the irreducible floor. The claim to test is whether $(\alpha, \beta, \tilde{H}_\infty)$ are stable across query distributions, or whether each $\mathcal{Q}$ induces its own.

**Corpus-side covariate.** For a fact $f$, let $k(f)$ be its occurrence count in $\mathcal{C}$ (measured by entity co-occurrence counting over the pretraining index, as in Kandpal et al. 2023). The **monofact rate** $\mu = \Pr_{f}[k(f) = 1]$ is the key statistic in the theory variant.

**Assumptions, and which are violated.**

1. *$V$ is accurate.* Violated: LLM-judge verifiers agree with humans at roughly 90–95% on short-form, worse on long-form, and their errors correlate with model style — so a judge favours generations resembling its own family.
2. *$\mathcal{Q}$ is fixed across scales.* Usually violated: benchmarks are refreshed as they saturate, so measured "hallucination over time" mixes model change with benchmark change.
3. *$\mathcal{C}$ is held constant while $D$ varies.* Almost always violated: large models are trained on different, better-filtered corpora than small ones.
4. *Query facts are not in the eval-contaminated part of $\mathcal{C}$.* Violated at unknown rate for every public benchmark.
5. *Decoding is fixed.* Violated: temperature, sampling, and RLHF-tuned abstention policy shift $R$ by tens of points independent of knowledge.

## 3. State of the Art

**Established.**

- **Loss scaling laws** (Kaplan et al. 2020; Hoffmann et al. 2022) are reproduced across labs. They govern cross-entropy, not factual correctness.
- **Long-tail knowledge scaling** (Kandpal et al. 2023, ICML): QA accuracy is approximately log-linear in the number of pretraining documents supporting the entity, measured over BLOOM 560M–176B and GPT-Neo 125M–20B on Natural Questions and TriviaQA. Larger models shift the curve up but do not change its shape.
- **Knowledge capacity** (Allen-Zhu & Li 2024, "Physics of Language Models: Part 3.3"): under controlled synthetic biographies with sufficient exposure, transformers store about **2 bits of knowledge per parameter**, roughly constant across architectures. This is the cleanest existing candidate for a mechanism behind a floor.
- **Calibration forces hallucination** (Kalai & Vempala, STOC 2024): a language model calibrated on a corpus of facts must hallucinate at a rate lower-bounded by roughly the monofact rate $\mu$ minus a calibration term. Scale does not appear in the bound.

**Claimed but unablated.**

- That "hallucination decreases with scale" as a general law. What is measured is usually accuracy on a benchmark increasing, with abstention rate uncontrolled — i.e. $A$ up, $\tilde{H}$ unreported.
- That RLHF reduces hallucination. It reliably changes $R$; the effect on $\tilde{H}$ is rarely separated.

**Benchmark numbers only.** SimpleQA (Wei et al. 2024) scores, HaluEval, and hallucination leaderboards are point measurements on frontier checkpoints of unknown $N$ and $D$, with no controlled ladder behind them. They cannot identify an exponent.

## 4. What Is Known

- **Inverse scaling exists for truthfulness.** TruthfulQA (Lin, Hilton & Evans, ACL 2022): across GPT-3 350M→175B, the *largest* model was the *least* truthful; the best model was 58% truthful against 94% for humans. Later work (Wei et al., EMNLP 2023) showed several inverse-scaling tasks become U-shaped at larger scale, so the sign of $\partial \tilde{H}/\partial N$ is not fixed even within one task.
- **Frequency dominates size.** Kandpal et al. (2023): accuracy on entities with $<10$ supporting documents stays near floor even at 176B; the authors' extrapolation is that reaching retrieval-augmented accuracy on the tail by scale alone would need parameter counts far beyond any plausible budget (their figure is on the order of $10^{18}$).
- **Capacity is linear in $N$, knowledge demand is not.** At 2 bits/parameter, a 7B model holds $\approx 1.4 \times 10^{10}$ bits. World-fact demand grows with corpus coverage, so the fraction of queried facts that fit shrinks as $\mathcal{Q}$ moves toward the tail.
- **New knowledge in finetuning increases hallucination.** Gekhman et al. (EMNLP 2024): examples introducing facts absent from pretraining are learned slowly and, once fit, raise hallucination on unrelated questions — a data-side, not size-side, effect.
- **Short-form factuality is far from saturated.** SimpleQA (OpenAI, 2024) was constructed so that frontier models answer well under half of the 4,326 questions correctly, with high overconfidence in stated certainty.

## 5. What Is Not Known

- **Empirically open.** No published fit of $\tilde{H}(N, D)$ on a controlled ladder — same corpus, same tokenizer, same eval, same decoding, $N$ spanning $\geq 3$ decades and $D$ varied independently. The compute is a small fraction of one frontier run; nobody has spent it on this.
- **Empirically open.** Whether $\alpha$ (parameter exponent) differs between head facts ($k > 10^3$) and tail facts ($k \leq 10$). The prediction from capacity arguments is that tail-fact $\alpha \approx 0$; untested at scale.
- **Theoretically open.** Whether the Kalai–Vempala monofact bound extends from the calibrated-density setting to post-trained models with an abstention action, and whether $\tilde{H}_\infty$ is a function of $\mu$ alone.
- **Methodologically blocked.** A hallucination measure invariant to abstention policy. $\tilde{H}$ is the right normalisation only if abstention is not itself knowledge-dependent — but models abstain more on facts they half-know, which makes $\tilde{H}$ a selected quantity with an unmodelled selection mechanism.
- **Methodologically blocked.** Long-form claim decomposition is not scale-invariant: larger models emit longer, more decomposable outputs, so the denominator $m$ co-varies with $N$.

## 6. Why It Is Hard

**The obstruction is non-identifiability under a confounded measurement**, in three named pieces:

1. **Abstention/knowledge confound.** $H$, $A$, $R$ move together under post-training. Two models with identical parametric knowledge can differ by 30 points in $H$ purely through refusal calibration. Without an intervention that fixes the abstention policy, $\partial \tilde H/\partial N$ is not identified.
2. **Corpus co-variation.** In every public model family, $N$ and $\mathcal{C}$ change together. The measured "$N$ effect" includes a data-quality effect of unknown sign and size.
3. **Absent ground truth at the tail.** The interesting regime is $k(f) \leq 10$, exactly where automated verifiers are least reliable — the reference sources are sparse, so an unsupported-but-true claim is scored as a hallucination. Verifier error and the quantity of interest are correlated, which biases the exponent rather than just adding noise.

Compute is *not* the primary obstruction: a ladder of 70M–7B models on a fixed corpus is well within an academic budget. The blocker is that building the frequency-annotated evaluation and the fixed-abstention protocol is unglamorous work with no benchmark to top.

## 7. Current Research (as of 2026)

- **Theory of unavoidable hallucination.** Kalai, Nachum, Vempala & Zhang (2025) reduce generative error to a binary "Is-It-Valid" classification problem and argue that leaderboard scoring, which rewards guessing over abstaining, sustains hallucination after pretraining. The argument is about *incentives*, not scale, and predicts that $\tilde{H}_\infty$ is set by evaluation design.
- **Knowledge-capacity physics.** Allen-Zhu and collaborators continue controlled synthetic-corpus studies isolating storage from retrieval *(frontier — verify current status)*.
- **Frequency-annotated evaluation.** Follow-ups to Kandpal et al. and Head-to-Tail (Sun et al., NAACL 2024) build QA sets stratified by entity popularity — the necessary substrate for a per-stratum scaling law.
- **Abstention and calibration training.** Work on teaching models to say "I don't know" is active at the major labs; from a scaling-law perspective it changes the measured $H$ without changing the underlying knowledge, and so is a confound to control, not a solution.

## 8. Concrete Next Experiment

**Scale.** Train a ladder of dense decoder models at $N \in \{70\text{M}, 160\text{M}, 410\text{M}, 1\text{B}, 2.8\text{B}, 6.9\text{B}\}$ on a *single fixed corpus* (e.g. a deduplicated open web+wiki mix), at $D \in \{20N, 100N\}$ tokens. Reuse an existing open ladder (Pythia-style) if the corpus index is available. Cost: on the order of $10^{22}$ FLOPs total — days on a modest cluster.

**Evaluation.** 5,000 short-form factual questions, each annotated with $k(f)$ from a count over the *actual* pretraining corpus, stratified into four bins: $k \in [1,10)$, $[10, 10^2)$, $[10^2,10^4)$, $\geq 10^4$. Force answering — no abstention option, greedy decoding, fixed prompt — so $R = 0$ and $\tilde{H} = H$.

**Control arm.** The same questions answered by the same models with gold documents in context (retrieval-augmented, oracle retrieval). This separates *storage* failure from *retrieval/formatting* failure: any residual error in the control arm is not a knowledge-capacity effect and must be subtracted.

**The deciding number.** The fitted parameter exponent $\alpha$ in the lowest-frequency bin, $\tilde{H}_{k<10}(N) = \tilde{H}_\infty + aN^{-\alpha}$, with a bootstrap confidence interval. If $\alpha$ is statistically indistinguishable from $0$ while the top bin shows $\alpha > 0.1$, the scaling law is frequency-dependent and tail hallucination is not a scale problem. If $\alpha$ is comparable across all four bins, the single-law hypothesis survives and the exponents are the answer.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Lin, Hilton, Evans. *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* ACL, 2022. — arXiv:2109.07958
- **[SOTA]** Kandpal, Deng, Roberts, Wallace, Raffel. *Large Language Models Struggle to Learn Long-Tail Knowledge.* ICML, 2023. — arXiv:2211.08411
- **[SOTA]** Allen-Zhu, Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* 2024. — arXiv:2404.05405
- **[Theory]** Kalai, Vempala. *Calibrated Language Models Must Hallucinate.* STOC, 2024. — arXiv:2311.14648
- **[Theory]** Kalai, Nachum, Vempala, Zhang. *Why Language Models Hallucinate.* 2025. — arXiv:2509.04664
- **[Measurement]** Min, Krishna, Lyu, et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[Measurement]** Wei, Karina, Chung, et al. *Measuring Short-Form Factuality in Large Language Models (SimpleQA).* 2024. — arXiv:2411.04368
- **[Empirical]** Gekhman, Yona, Aharoni, et al. *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* EMNLP, 2024. — arXiv:2405.05904
- **[Empirical]** McKenzie, Lyzhov, Pieler, et al. *Inverse Scaling: When Bigger Isn't Better.* TMLR, 2023. — arXiv:2306.09479
- **[Survey]** Huang, Yu, Ma, et al. *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions.* ACM TOIS, 2025. — arXiv:2311.05232

## 10. Worked Example

Take one bin of the proposed evaluation: 1,000 questions about entities with $k(f) \in [1,10)$ in the pretraining corpus.

Suppose the raw measurements come back as:

| $N$ | correct $A$ | incorrect $H$ | abstain $R$ | $\tilde H = H/(H{+}A)$ |
|---|---|---|---|---|
| 410M | 0.06 | 0.90 | 0.04 | 0.938 |
| 1.4B | 0.11 | 0.72 | 0.17 | 0.867 |
| 6.9B | 0.14 | 0.51 | 0.35 | 0.785 |

Read as $H$, hallucination fell from 0.90 to 0.51 — a 43% relative reduction, and a headline that says scale fixes hallucination. Read as $\tilde{H}$, the drop is 0.938 → 0.785, and **most of the apparent gain is abstention**: $R$ tripled. Fit $\tilde{H} = \tilde{H}_\infty + aN^{-\alpha}$ on these three points and $\alpha \approx 0.06$ with $\tilde{H}_\infty$ unidentifiable from three points — extrapolating to $N = 10^{12}$ predicts $\tilde{H} \approx 0.6$, still worse than a coin flip on the tail.

Now the obstruction becomes visible. Both readings are computed from the *same* generations. The difference is entirely which normalisation is chosen, and neither is obviously right: if the 6.9B model abstains because it knows it does not know, its abstentions are a success and $H$ is the honest metric; if it abstains because RLHF made it timid, they are a failure to answer and $\tilde{H}$ is honest. Nothing in the measurement distinguishes these. That is why the experiment in §8 forces answering ($R = 0$) — not because forced answering is realistic, but because it is the only setting in which the two readings coincide and the exponent is identified.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*